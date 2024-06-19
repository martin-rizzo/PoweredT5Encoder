"""
  File    : p5.py
  Brief   : t5 tokenizer y encoder implementados con transformers
  Author  : Martin Rizzo | <martinrizzo@gmail.com>
  Date    : Apr 29, 2024
  Repo    : https://github.com/martin-rizzo/ComfyUI-PixArt
  License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                             Powered T5 Encoder
       An enhanced T5 encoder with integrated weighted prompt support

     Copyright (c) 2024 Martin Rizzo

     Permission is hereby granted, free of charge, to any person obtaining
     a copy of this software and associated documentation files (the
     "Software"), to deal in the Software without restriction, including
     without limitation the rights to use, copy, modify, merge, publish,
     distribute, sublicense, and/or sell copies of the Software, and to
     permit persons to whom the Software is furnished to do so, subject to
     the following conditions:

     The above copyright notice and this permission notice shall be
     included in all copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
     CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
     TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE
     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

 File Summary
 ============
   T5Tokenizer
    - from_pretrained(tokenizer_dir, max_length, embedding_dir...)
    - parse_segments_weights(text)
    - tokenize_with_weights(text, padding, padding_max_size, include_word_ids)
    - get_vocab()
    - untokenize(token_weight_pairs, return_word_weights)

   T5EncoderModel
    - from_safetensors(path, model_class, max_length, frozen, device)
    - encode( input_ids )
    - encode_with_weights( batch_of_tokens_with_weights, return_attn_mask )
    - freeze()
    - unfreeze()
    - load_state_dict(state_dict, strict=False)
    - state_dict(..)
    - to(..)

 Documentation
 =============
   Loading a checkpoint reducing compute and memory as much as possible:
    - https://pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html
   T5 V1.1 is an improved version of T5 with some architectural tweaks:
    - https://huggingface.co/docs/transformers/en/model_doc/t5v1.1

"""
import os
import re
import gc
import torch
from typing import Optional, Union, Dict, List

# Hugging Face Safetensors
# - https://huggingface.co/docs/safetensors/index
from safetensors import safe_open

# Hugging Face Transformers
#  - https://huggingface.co/docs/transformers/index
# Note:
#  - T5Tokenizer has a transitive dependency on 'sentencepiece'
#  - T5TokenizerFast has a transitive dependency on 'sentencepiece' and 'protobuf'
from transformers import                   \
    T5Config        as HF_T5Config,        \
    T5TokenizerFast as HF_T5TokenizerFast, \
    T5EncoderModel  as HF_T5EncoderModel


# Google T5 v1.1 models
# =====================
#
#          | #Params | #Params   |
#   Models | encoder | enco+deco | layers | d^model |  d^ff | d^kv | #heads
# ---------+---------+-----------+--------+---------+-------+------+-------
#   small  |    ?    |     77M   |    8   |    512  |  1024 |  64  |   6
#    base  |    ?    |    250M   |   12   |    768  |  2048 |  64  |  12
#   large  |    ?    |    780M   |   24   |   1024  |  2816 |  64  |  16
#      xl  |    ?    |     3B    |   24   |   2048  |  5120 |  64  |  32
#     xxl  |  4.7B   |    11B    |   24   |   4096  | 10240 |  64  |  64

T5PredefinedConfigs={
    'xxl': {
            'vocab_size': 32128,
            'd_model': 4096,
            'd_kv': 64,
            'd_ff': 10240,
            'num_layers': 24,
            'num_decoder_layers': 24,
            'num_heads': 64,
            'relative_attention_num_buckets': 32,
            'relative_attention_max_distance': 128,
            'dropout_rate': 0.1,
            'layer_norm_epsilon': 1e-6,
            'initializer_factor': 1.0,
            'feed_forward_proj': "gated-gelu",
            'is_encoder_decoder': True,
            'use_cache': True,
            'pad_token_id': 0,
            'eos_token_id': 1,
            #'classifier_dropout': 0.0
        }
    }


#============================= WEIGHTS PARSING =============================#

PARENTHESIS_MULTIPLIER = 1.1
BRACKET_MULTIPLIER     = 1 / PARENTHESIS_MULTIPLIER

# divide el prompt en segmentos para ser procesados por 'parse_segments_weights(..)'
RE_SEGMENTS = re.compile(r"""
\\\(|              # cualquier escape escrito por el usuario:
\\\)|              # '\(' '\)' '\[' '\]' y '\\'
\\\[|
\\]|
\\\\|

\\|                # la barra para escapar el siguiente char

\(|                # apertura de segmento con peso: '(' y '['
\[|

:(\s*[+-]?[.\d]+)\s*\)|  # extrae el peso si arranca en ':' y termina en ')'


\)|                # cierre de segmento con peso ')' y ']'
]|

[^\\()\[\]:]+|     # cualquier secuencia de texto sin '\', '(', ')', '[', ']', ni ':'

:                  # el caracter ':' sin representar peso
""", re.X)


#============================== T5-TOKENIZER ===============================#
class T5Tokenizer:

    def __init__(self,
                 tokenizer,
                 max_length,
                 embedding_dir,
                 embedding_key,
                 embedding_size
                 ):

        # tokenizer
        self.tokenizer  = tokenizer
        self.max_length = max_length

        # embeddings
        self.embedding_dir  = embedding_dir
        self.embedding_tag  = 'embedding:'
        self.embedding_key  = embedding_key
        self.embedding_size = embedding_size

        # tokens
        self.pad_token = self.tokenizer('<pad>', add_special_tokens=False)['input_ids'][0]
        self.end_token = self.tokenizer('</s>' , add_special_tokens=False)['input_ids'][0]
        self.inverse_vocab = { v: k for k, v in self.get_vocab().items() }



    @classmethod
    def from_pretrained(
            cls,
            tokenizer_dir: Optional[os.PathLike] = None,
            max_length   : int                   = 300,  # pixart alpha=120 | sigma=300 #
            legacy       : bool                  = None
    ):

        if tokenizer_dir is None:
            _this_file_dir = os.path.dirname(os.path.realpath(__file__))
            tokenizer_dir = os.path.join(_this_file_dir, 't5data')

        tokenizer = HF_T5TokenizerFast.from_pretrained(tokenizer_dir,
                                                       legacy=legacy)
        return T5Tokenizer(tokenizer,
                           max_length     = max_length,
                           embedding_dir  = None,
                           embedding_size = 't5',
                           embedding_key  = 4096)


    def parse_segments_weights(self, text):
        output = []
        parenth_open = []
        bracket_open = []

        def multiply_output_by(multiplier, start_index):
            output[start_index:] = [(t[0], t[1] * multiplier) for t in output[start_index:]]

        for segment in RE_SEGMENTS.finditer(text):
            text   = segment.group(0)
            weight = segment.group(1)

            if text.startswith('\\'):
                output.append( (text[1:], 1.) )
            elif text == '(':
                parenth_open.append( len(output) )
            elif text == '[':
                bracket_open.append( len(output) )
            elif (weight is not None) and len(parenth_open) > 0:
                multiply_output_by( float(weight), start_index=parenth_open.pop() )
            elif text == ')' and len(parenth_open) > 0:
                multiply_output_by( PARENTHESIS_MULTIPLIER, start_index=parenth_open.pop() )
            elif text == ']' and len(bracket_open) > 0:
                multiply_output_by( BRACKET_MULTIPLIER, start_index=bracket_open.pop() )
            else:
                output.append( (text, 1.) )

        # si el usuario tipeo caracteres sin sentido tal vez no se genero nada
        # en dicho caso retornar string vacio con peso 1.0
        if not output:
            return [('', 1.)]

        # merge identical weights
        merged_output = []
        acumulator    = output[0]
        for segment_weight in output[1:]:
            if acumulator[1] == segment_weight[1]:
                acumulator = (acumulator[0] + segment_weight[0], acumulator[1])
            else:
                merged_output.append(acumulator)
                acumulator = segment_weight
        merged_output.append(acumulator)

        return merged_output


    def tokenize_with_weights(self,
                              text            : Union[str, list],
                              padding         : bool = False,
                              padding_max_size: int  = 0,
                              include_word_ids: bool = False,
                              ):
        '''
        Convert a text/prompt into a list of (token, weight, word_id) elements.
        The input text can be a string or a list of strings (for batch tokenization).
        In the output (token, weight, word_id):
         - 'token' can be either integer tokens or pre-computed T5 tensors.
         - 'weight' is the user-assigned weight.
         - 'word_id' is an integer indicating the word to which the token belongs,
            where word_id=0 is reserved for non-word tokens.

        The returned list has dimensions of [batch_size, seq_length].
        '''
        output_batch     = []
        input_batch      = text if isinstance(text,list) else [text]
        end_item         = (self.end_token, 1., 0) if include_word_ids else (self.end_token, 1.)
        pad_item         = (self.pad_token, 1., 0) if include_word_ids else (self.pad_token, 1.)
        padding_max_size = 0 if not padding else padding_max_size
        process_word_by_word = (include_word_ids == True)

        max_number_of_tokens = 0
        for text in input_batch:
            text = text.replace('\n', ' ')
            segments_weights = self.parse_segments_weights(text)

            # inicializar la lista de (token, weight)
            # que sera rellenada con la tokenizacion de segments_weights
            tokens = []

            #- process word by word -------------
            if process_word_by_word:
                segment_word0 = 1
                for segment, weight in segments_weights:
                    words = [word for word in segment.split() if word]
                    for word_idx, word in enumerate(words):
                        # # TODO: procesar embeddings?
                        # if word.startswith(self.embedding_tag):
                        #     _ = self.get_embeddings(self, word)
                        #     continue

                        # tokenize word
                        word_tokens = self.tokenizer(word, add_special_tokens=False)['input_ids']
                        if include_word_ids:
                            tokens.extend([ (token, weight, segment_word0 + word_idx) for token in word_tokens ])
                        else:
                            tokens.extend([ (token, weight) for token in word_tokens ])
                    segment_word0 += len(words)
            #- process segment by segment -------
            else:
                for segment, weight in segments_weights:
                    segment_tokens = self.tokenizer(segment, add_special_tokens=False)['input_ids']
                    tokens.extend([ (token, weight) for token in segment_tokens ])
            #------------------------------------

            # agregar el token de final
            tokens.append( end_item )

            # verificar si es el texto mas largo del batch
            if len(tokens) > max_number_of_tokens:
                max_number_of_tokens = len(tokens)

            # agregar padding si el usuario lo solicito
            if padding:
                tokens_left = self.max_length - len(tokens)
                if tokens_left > 0:
                    tokens.extend( [pad_item] * tokens_left )

            output_batch.append( tokens )

        # si se agrego padding o se excedio el limite (max_length)
        # entonces loopear nuevamente recortando el exceso
        if padding or max_number_of_tokens>self.max_length:
            truncate_at = padding_max_size if padding_max_size>0 else max_number_of_tokens
            if  truncate_at > self.max_length:
                truncate_at = self.max_length
            # truncar todos los tensores del batch
            # y agregar 'end_item' al final de tokens cuando corresponda
            for i, tokens in enumerate(output_batch):
                tokens     = tokens[:truncate_at]
                tokens[-1] = end_item if tokens[-1][0] != self.pad_token else pad_item
                output_batch[i] = tokens

        return output_batch


    # Returns the vocabulary as a dictionary of token to index.
    def get_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()


    def untokenize(self,
                   token_weight_pairs: List,
                   return_word_weights: bool = False
    ) -> List:
        """
        Converts a list of (token, weight) pairs into a list of ((..data..), word) tuples.

        Args:
            token_weight_pairs (List): A list of (token, weight) pairs, where
                token is an integer representing a token, and weight is a
                number representing the weight associated with that token.
            return_word_weights (bool):
                si True entonces retornara una lista con (word, height) pairs

        Returns:
            List: A list of ((..data..), word) tuples, where word is the string
                representation of the token according to the vocabulary.
        """
        if return_word_weights:
            return list(map( lambda tw: (self.inverse_vocab[tw[0]], tw[1]), token_weight_pairs ))
        else:
            return list(map( lambda tw: (tw, self.inverse_vocab[tw[0]]), token_weight_pairs ))


#============================ T5-ENCODER-MODEL =============================#
class T5EncoderModel:


    def __init__(self,
                 t5encoder,
                 max_length = 120, # alpha=120 / sigma=300 !!!
                 num_layers = 24,
                 frozen     = False
                 ):
        self.t5encoder    = t5encoder
        self.max_length   = max_length
        self.num_layers   = num_layers
        self.empty_tokens = [[0] * self.max_length] # <pad> token
        self.empty_vector = None
        if frozen:
            self.freeze()


    @classmethod
    def from_safetensors(self,
                         safetensors_path: Union[os.PathLike, list],
                         model_class     : str  = 'xxl',
                         max_length      : int  = 300,  # alpha=120 / sigma=300 !!!
                         frozen          : bool = True,
                         device          : str  = 'cpu'
                         ):
        model_class = model_class.lower()
        model       = None
        assert model_class == 'xxl', 'xxl es el unico model t5 soportado hasta el momento'

        if not isinstance(safetensors_path, list) and \
           not isinstance(safetensors_path, tuple):
            safetensors_path = [ safetensors_path ]

        config = T5PredefinedConfigs[model_class]
        model_config = HF_T5Config(**config)
        with torch.device('meta'):
            model = HF_T5EncoderModel(model_config)

        for filepath in safetensors_path:
            print(f'Loading {filepath} in {str(device)}')
            state_dict = {}
            with safe_open(filepath, framework='pt', device=device) as f:
                for key in f.keys():
                    # print(f"  - loading tensor {key}")
                    state_dict[key] = f.get_tensor(key)

            result = model.load_state_dict(state_dict, strict=False, assign=True)
            del state_dict
            gc.collect()
            if isinstance(result,tuple):
                print("PixArt T5 unexpected keys:", result.unexpected_keys)

        return T5EncoderModel(model, max_length=max_length, num_layers=24, frozen=frozen)


    def encode( self, input_ids ):
        device    = self.t5encoder.get_input_embeddings().weight.device
        input_ids = torch.LongTensor(input_ids).to(device)
        attention_mask = torch.zeros_like(input_ids)
        max_token = 1 # </s> token
        for x in range(attention_mask.shape[0]):
            for y in range(attention_mask.shape[1]):
                attention_mask[x, y] = 1
                if input_ids[x, y] == max_token:
                    break

        outputs = self.t5encoder(input_ids=input_ids, attention_mask=attention_mask)

        z = outputs['last_hidden_state']
        z.detach().cpu().float()
        return z


    def encode_with_weights(self,
                            batch_of_tokens_with_weights,
                            return_attn_mask: bool = False
                            ):

        # separa tokes y pesos
        input_ids     = []
        input_weights = []
        for element in batch_of_tokens_with_weights:
            input_ids.append(     list(map(lambda a: a[0], element)) )
            input_weights.append( list(map(lambda a: a[1], element)) )

        # cachea el vector de prompt vacio
        if self.empty_vector is None:
            self.empty_vector = self.encode( self.empty_tokens ).squeeze(0)
        empty_vector = self.empty_vector

        # generar la representacion vectorial del bach de tokens suministrado
        #   input_ids.shape    = [batch_size, sequence_length]
        #   output_batch.shape = [batch_size, sequence_length, embedding_size]
        output_batch = self.encode( input_ids = input_ids )

        # Aplica los pesos a los context_embeddings
        # ATTENTION: se estan aplicando los pesos a la SALIDA del encoder T5
        #            y utilizando la implementacion de comfy.
        # Nota:
        #  Los embeddings contextualizados son representaciones vectoriales
        #  de palabras que incorporan información semántica y sintáctica del
        #  contexto circundante.
        #  A diferencia de los embeddings tradicionales, su carácter más
        #  conceptual los hace ideales para procesamiento de lenguaje natural.
        #
        # T5 permitiria aplicar los pesos:
        #    - a los embeddings de cada token (antes de ingresar al encoder T5)
        #    * a los contextualized embeddings (luego de salir del encoder T5)
        # Weight interpretation:
        #    * comfy: vectors are lerped between the prompt and an empty prompt
        #    - A1111: vectors are scaled by their weight
        #
        weighted_batch = []
        for i, context_embeddings in enumerate(output_batch):
            empty_vector = self.empty_vector[:context_embeddings.shape[0]]
            #  context_embeddings.shape = [sequence_length=300, embedding_size=4096]
            weights = torch.Tensor( input_weights[i] )
            weighted_embeddings = ( context_embeddings - empty_vector ) * weights.unsqueeze(1) + empty_vector
            weighted_batch.append( weighted_embeddings )

        # si por algun motivo el batch a encodear no tiene ningun elemento
        # retornar el vector de prompt vacio
        if len(weighted_batch) == 0:
            return empty_vector.cpu()

        if not return_attn_mask:
            return torch.stack(weighted_batch)

        input_ids_batch = input_ids

        attn_mask_batch = []
        for input_ids in input_ids_batch:
            attn_mask = torch.tensor( input_ids )
            attn_mask = attn_mask > 0
            attn_mask_batch.append( attn_mask )

        return torch.stack(weighted_batch), torch.stack(attn_mask_batch)

        # # TODO: mejorar esto. [[ tal vez usando attention_mask (?) ]]
        # keep_index = sum([sum([1 for y in x if y[0] != 0]) for x in batch_of_tokens_with_weights])
        # weighted_batch = weighted_batch[:, :keep_index, :]

        return weighted_batch, "" # first_pooled #


    # Freeze all params for inference.
    def freeze(self) -> None:
        for param in self.t5encoder.parameters():
            param.requires_grad = False
        self.t5encoder.eval()


    # Unfreeze all parameters for training.
    def unfreeze(self) -> None:
        for param in self.t5encoder.parameters():
            param.requires_grad = True
        self.t5encoder.train()


    # Copy parameters and buffers from state_dict into the t5encoder
    #  - state_dict: a dict containing parameters and persistent buffers.
    #  - strict: whether to strictly enforce that the keys in state_dict match the keys in t5encoder
    # Returns: NamedTuple with 'missing_keys' and 'unexpected_keys' fields.
    def load_state_dict(self, state_dict, strict=False):
        return self.t5encoder.load_state_dict(state_dict, strict=strict)


    # Return a dictionary containing references to the whole state of the T5 encoder
    def state_dict(self, *args, **kwargs):
        return self.t5encoder.state_dict(*args, **kwargs)


    # Move and/or cast the parameters and buffers of the T5 encoder
    def to(self, *args, **kwargs):
        self.t5encoder.to(*args, **kwargs)


