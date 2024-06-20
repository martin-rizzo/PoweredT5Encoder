import os
import sys
import torch
import shutil
import argparse
from configparser import ConfigParser
from safetensors.torch import save_file as save_safetensors
from src.t5 import T5Tokenizer, T5EncoderModel

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH    = os.path.join(SCRIPT_DIR, 't5xe.ini')
CONFIG_EXAMPLE = os.path.join(SCRIPT_DIR, 't5xe.example.ini')

DEFAULT_CHECK_TENSOR_PROMPT = """
A ((photo)) of a [small] flying ((hovercraft:1.1) with a pilot inside),
hovering over the [((rugged)) surface of Mars].
[The ((craft:1.2)), made of [reflective metals], (gleams under the (sunlight))].
(In the background, a tunnel [[entrance]] adds a sense of (mystery:1.15)).
(The image, with its [selective focus], captures ((detailed textures)),
and (an atmospheric color palette that emphasizes the Martian landscape)).
"""


#================================= HELPERS =================================#

def fatal_error(message: str):
    """Print an error message and exit the program with status 1."""
    print(message)
    sys.exit(1)


def filter_files(filepaths, extension):
    """Filter filepaths list to include only files with the specified extension."""
    extension = extension.lower()
    return [filepath for filepath in filepaths if filepath.lower().endswith(extension)]

def change_extension(fullpath, new_extension):
    """Return a new path string with the file extension changed."""
    base, ext = os.path.splitext(fullpath)
    return base + '.' + new_extension.lstrip('.')

def get_unique_path(path: str) -> str:
    """Get a unique file path by adding incremental numbers if the file exists."""
    if not os.path.exists(path):
        return path
    counter              = 1
    directory, file_name = os.path.split(path)
    name, extension      = os.path.splitext(file_name)
    while True:
        new_file_name = f"{name}-{counter:02}{extension}"
        new_path = os.path.join(directory, new_file_name)
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def metadata_from_config(config: ConfigParser) -> dict:
    image_config = {}

    for section in config.sections():
        for key, value in config.items(section):
            if value.startswith("'"):
                value = value.strip("'")
            elif value.startswith('"'):
                value = value.strip('"')
            image_config[f"{section}.{key}"] = value

    return image_config


def get_config_section(section_name:str):
    # primero se asegura que el archivo de configuracion exista
    if not os.path.exists(CONFIG_PATH):
        if not os.path.exists(CONFIG_EXAMPLE):
            fatal_error(f"El archivo {CONFIG_EXAMPLE} no fue encontrado")
        shutil.copy(CONFIG_EXAMPLE, CONFIG_PATH)

    # obtiene el nombre del archivo de configuracion que sera mostrado al usuario
    display_name = os.path.basename(CONFIG_PATH)

    # intenta leer la seccion solicitada
    config = ConfigParser()
    config.read(CONFIG_PATH)
    section = config[section_name] if section_name in config else None
    if not section:
        fatal_error(f"The configuration file '{display_name}' no contiene la sección [t5]")

    return section, display_name


#===========================================================================#

def initialize_encoder() -> (T5Tokenizer, T5EncoderModel):
    SECTION_NAME = 't5encoder'
    MODEL_PATH   = 'model_file'
    config_section, config_name = get_config_section(SECTION_NAME)

    # agregar todos los paths configurados en
    # model_file, model_file1, model_file2, model_file3, ...
    safetensors_paths = []
    for i in range(0, 1000):
        key = MODEL_PATH if i==0 else f'{MODEL_PATH}{i}'
        if key in config_section:
            print(f"##(DEBUG) config_section[{key}]: {config_section[key]}")
            safetensors_paths.append( config_section[key] )
        elif i>0:
            break

    # revisar que todo haya funcionado correctamente
    if not safetensors_paths:
        fatal_error(f"Debe configurar correctamente the '{MODEL_PATH}' key in the '{SECTION_NAME}' section of '{config_name}'.")
    for path in safetensors_paths:
        if not os.path.isfile(path):
            fatal_error(f"El archivo {path} no existe, por favor revise la configuracion de '{MODEL_PATH}' in '{config_name}.")

    # initializar y retornar tokenizer y encoder
    tokenizer = T5Tokenizer.from_pretrained(legacy=True)
    encoder   = T5EncoderModel.from_safetensors(safetensors_paths)
    return tokenizer, encoder


def embeds_from_prompts(prompts  : list,
                        tokenizer: T5Tokenizer,
                        encoder  : T5EncoderModel):
    tokens = tokenizer.tokenize_with_weights(prompts,
                                             padding=False,
                                             include_word_ids=False
                                             )
    return encoder.encode_with_weights(tokens)



#============================== SUB-COMMANDS ===============================#

def process_file(filepath : str,  # .ini file
                 tokenizer: T5Tokenizer,
                 encoder  : T5EncoderModel
                 ):

    image_config = ConfigParser()
    image_config.read(filepath)
    image_config = metadata_from_config(image_config)

    positive = image_config.get('prompt.positive', '')
    negative = image_config.get('prompt.negative', '')

    padding = False
    tensors = {}

    if padding:
        _tokens = tokenizer.tokenize_with_weights([positive,negative],
                                                  padding=True,
                                                  padding_max_size=300)
        _embeds, _attn_mask = encoder.encode_with_weights(_tokens,
                                                          return_attn_mask=True
                                                          )
        tensors['prompt.positive']           = _embeds[0]
        tensors['prompt.positive_attn_mask'] = _attn_mask[0]
        tensors['prompt.negative']           = _embeds[1]
        tensors['prompt.negative_attn_mask'] = _attn_mask[1]
    else:
        _tokens = tokenizer.tokenize_with_weights([positive], padding=False)
        tensors['prompt.positive'] = encoder.encode_with_weights(_tokens)
        _tokens = tokenizer.tokenize_with_weights([negative], padding=False)
        tensors['prompt.negative'] = encoder.encode_with_weights(_tokens)

    _, output_filename = os.path.split(filepath)
    output_filename    = change_extension(output_filename, '.safetensors')
    output_filename    = get_unique_path(output_filename)
    print("## output_filename:", output_filename)
    print("## prompt.positive:", tensors['prompt.positive'].shape)
    print("## prompt.negative:", tensors['prompt.negative'].shape)
    save_safetensors(tensors, output_filename, metadata=image_config)
    return None

def test_file(filepath : str,  # .safetensors file
              tokenizer: T5Tokenizer,
              encoder  : T5EncoderModel
              ):
    print(f"Testing file: {filepath}")
    return None

import pickle
import gzip

def compress_tensor(tensor_name: str,
                    input_tokens,
                    output_embeddings,
                    encoder: T5EncoderModel
                    ):

    print("## compress_tensor (begin)")
    tensor = encoder.state_dict().get(tensor_name)

    # Serializar y comprimir el tensor
    print("## converting to float16")
    tensor = tensor.to(torch.float16)
    print("## dumping")
    tensor_bytes     = pickle.dumps(tensor)
    print("## compressing")
    compressed_bytes = gzip.compress(tensor_bytes)
    print("## calculating percent")
    percent          = len(compressed_bytes) * 100. / len(tensor_bytes)
    print(f"Compress tensor {tensor_name}: {len(compressed_bytes)}/{len(tensor_bytes)} {percent}")
    print("## compress_tensor (end)")

    # Para descomprimir y deserializar el tensor
    #decompressed_bytes = gzip.decompress(compressed_bytes)
    #deserialized_tensor = pickle.loads(decompressed_bytes)



def check_tensor(tensor_name: str,
                 input_tokens,
                 output_embeddings,
                 encoder: T5EncoderModel
                 ):
    scale = 0.07
    zero_point = 0

    tensor = encoder.state_dict().get(tensor_name)
    if tensor is None:
        print(f"${tensor_name} no existe")
        return

    original_tensor = tensor.clone()
    fp16_tensor   = original_tensor.to(torch.float16)
    fp16_tensor   = fp16_tensor.to(torch.float32)
    qui8_tensor   = torch.quantize_per_tensor(original_tensor, scale=scale, zero_point=zero_point, dtype=torch.quint8)
    qui8_tensor   = qui8_tensor.dequantize()

    tensor.copy_(fp16_tensor)
    fp16_embeddings = encoder.encode_with_weights(input_tokens)[0]
    assert fp16_embeddings.shape == output_embeddings.shape, f"no matchean las shapes, {fp16_embeddings.shape} vs {output_embeddings.shape}"
    fp16_similarity = torch.cosine_similarity(fp16_embeddings, output_embeddings, dim=1).mean()
    print(f"Similitud con tensor {tensor_name} en fp16: {fp16_similarity}")

    # tensor.copy_(qui8_tensor)
    # qui8_embeddings = encoder.encode_with_weights(input_tokens)[0]
    # assert qui8_embeddings.shape == output_embeddings.shape, f"no matchean las shapes, {qui8_embeddings.shape} vs {output_embeddings.shape}"
    # qui8_similarity = torch.cosine_similarity(qui8_embeddings, output_embeddings, dim=1).mean()
    # print(f"Similitud con tensor {tensor_name} en qui8: {qui8_similarity}")
    #
    # # vuelve el tensor a la normalidad
    # tensor.copy_(original_tensor)
    # original_embeddings = encoder.encode_with_weights(input_tokens)[0]
    # original_similarity = torch.cosine_similarity(original_embeddings, output_embeddings, dim=1).mean()
    # print(f"Similitud con tensor {tensor_name} original: {original_similarity}")

#
#
# def check_tensor(tensor_name: str,
#                  input_tokens,
#                  output_embeddings,
#                  encoder: T5EncoderModel
#                  ):
#     tensor = encoder.state_dict().get(tensor_name)
#     if tensor is None:
#         print(f"${tensor_name} no existe")
#         return
#
#     original_tensor = tensor.clone()
#     zero_tensor     = torch.zeros_like(original_tensor)
#     random_tensor   = torch.rand_like(original_tensor)
#
#     # reemplazar 'tensor_name' en el encoder con un tensor todo en cero
#     tensor.copy_(zero_tensor)
#
#     # encodear 'input_tokens' y verificar que tanto se parece el resultado a output_embedings
#    #zero_embeddings = encoder(input_tokens)[0]
#     zero_embeddings = encoder.encode_with_weights(input_tokens)[0]
#     assert zero_embeddings.shape == output_embeddings.shape, f"no matchean las shapes, {zero_embeddings.shape} vs {output_embeddings.shape}"
#     zero_similarity = torch.cosine_similarity(zero_embeddings, output_embeddings, dim=1).mean()
#     print(f"Similitud con tensor {tensor_name} en cero: {zero_similarity}")
#
#     # reemplazar 'tensor_name' en el encoder con un tensor todo random
#     tensor.copy_(random_tensor)
#
#     # encodear 'input_tokens' y verificar que tanto se parece el resultado a output_embedings
#    #random_embeddings = encoder(input_tokens)[0]
#     random_embeddings = encoder.encode_with_weights(input_tokens)[0]
#     assert random_embeddings.shape == output_embeddings.shape, f"no matchean las shapes, {random_embeddings.shape} vs {output_embeddings.shape}"
#     random_similarity = torch.cosine_similarity(random_embeddings, output_embeddings, dim=1).mean()
#     print(f"Similitud con tensor {tensor_name} aleatorio: {random_similarity}")
#
#     # vuelve el tensor a la normalidad
#     tensor.copy_(original_tensor)
#     original_embeddings = encoder.encode_with_weights(input_tokens)[0]
#     original_similarity = torch.cosine_similarity(original_embeddings, output_embeddings, dim=1).mean()
#     print(f"Similitud con tensor {tensor_name} original: {original_similarity}")


def check_all_tensors(tokenizer: T5Tokenizer,
                      encoder  : T5EncoderModel,
                      prompt=None,
                      tokens=None,
                      embeddings=None ):

    if not prompt:
        prompt = DEFAULT_CHECK_TENSOR_PROMPT

    # if tokens is None:
    #     tokens = tokenizer.tokenize_with_weights([prompt],
    #                                              padding=False)
    #
    # if embeddings is None:
    #     embeddings = encoder.encode_with_weights(tokens)[0]

    tensor_names = list(encoder.state_dict().keys())
    for tensor_name in tensor_names:
        #print(f"Verificando tensor: {tensor_name}")
        compress_tensor(tensor_name, tokens, embeddings, encoder)
        #print("-" * 50)

#===========================================================================#
#////////////////////////////////// MAIN ///////////////////////////////////#
#===========================================================================#

def main():
    parser = argparse.ArgumentParser(
        description='Tool for testing the embedding generation of the powered T5 encoder.',
        add_help=False
    )
    parser.add_argument('filepaths', nargs='*', help='Configuration files to generate the embed')
    parser.add_argument('-t', '--test'        , action='store_true', help='Test the generated embed against the saved one')
    parser.add_argument('-l', '--check-layers', action='store_true', help='Check if all tensors/layers are contributing to the output')
    parser.add_argument('-h', '--help'        , action='help', default=argparse.SUPPRESS, help='Show this help message and exit.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')


    # 1) test y archivos -> no se genera nada, se verifican los .safetensors
    # 2) archivos -> deben ser .ini y se generan los .safetensors
    # 3) test (sin archivos) -> se realiza un test interno (??)

    args = parser.parse_args()

    if args.check_layers:
        command = 'check-layers'
    else:
        command  = 'test'   if args.test      else 'process'
        command += '-files' if args.filepaths else '-internal'

    if command == 'process-files':
        ini_files = filter_files(args.filepaths, '.ini')
        if not ini_files:
            fatal_error('Debe suministrar al menos algún archivo .ini')

        tokenizer, encoder = initialize_encoder()
        for ini_file in ini_files:
            process_file(ini_file, tokenizer, encoder)

    elif command == 'test-files':
        safetensors_files = filter_files(args.filepaths, '.safetensors')
        if not safetensors_files:
            fatal_error('Debe suministrar al menos algún archivo .safetensors')

        tokenizer, encoder = initialize_encoder()
        for safetensors_file in safetensors_files:
            test_file(safetensors_file, tokenizer, encoder)

    elif command == 'test-internal':
        print("Automated testing is not supported yet")
        sys.exit(1)

    elif command == 'check-layers':
        tokenizer, encoder = initialize_encoder()
        check_all_tensors(tokenizer, encoder)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()

