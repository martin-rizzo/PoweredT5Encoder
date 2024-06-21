"""
  File    : p5embed.py
  Brief   : Tool for testing T5 extended encoder embedding generation.
  Author  : Martin Rizzo | <martinrizzo@gmail.com>
  Date    : Jun 1, 2024
  Repo    : https://github.com/martin-rizzo/T5ExtendedEncoder
  License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              T5 Extended Encoder
     Enhanced T5 encoder with weighted prompts and easy safetensors loading

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
"""
import os
import sys
import torch
import json
import struct
import shutil
import argparse
from configparser      import ConfigParser
from safetensors.torch import save_file as save_safetensors
from src.t5            import T5Tokenizer, T5EncoderModel

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


def change_extension(path: str, new_extension: str, overwrite: bool = True) -> str:
    """Return a new path string with the file extension changed."""
    base, ext = os.path.splitext(path)
    new_path  = base + '.' + new_extension.lstrip('.')
    if not overwrite:
        new_path = get_unique_path(new_path)
    return new_path


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


#================================ METADATA =================================#

def load_metadata_from_safetensors(filepath  : str,
                                   size_limit: int = (65536 * 1024)
                                   ) -> dict:
    """
    Load metadata from a SafeTensors file.

    Args:
        filepath   (str): Path to the SafeTensors file.
        size_limit (int): Maximum allowed size for the header in bytes.
                          Default is 64MB (65536 * 1024).
    Returns:
        dict: Dictionary containing the file's metadata.
    Raises:
        ValueError: If the file doesn't have a valid SafeTensors header.
        IOError:    If there are issues opening or reading the file.
    """
    try:
        with open(filepath, "rb") as file:

            # read the header length (first 8 bytes)
            header_length = struct.unpack('<Q', file.read(8))[0]
            if header_length > size_limit:
                raise ValueError(f"Header exceeds the size limit of {size_limit} bytes.")

            # read and extract the metadata
            header_data = file.read(header_length)
            header = json.loads(header_data)
            return header.get('__metadata__', {})

    except (ValueError, json.JSONDecodeError):
        filename = os.path.basename(filepath)
        raise ValueError(f"The file '{filename}' does not have a valid safetensors header.")
    except IOError:
        filename = os.path.basename(filepath)
        raise IOError(f"Error opening or reading the file '{filename}'.")


def load_metadata_from_ini_file(ini_file: str
                                ) -> dict:
    """
    Load configuration from an INI file and convert it to metadata format.

    Args:
        ini_file (str): Path to the INI file.

    Returns:
        dict: Dictionary containing the configuration as metadata.
    Raises:
        ValueError: If there is an issue with the INI file format.
        IOError:    If there are issues opening or reading the file.
    """
    metadata = {}
    config = ConfigParser()
    config.read(ini_file)
    for section in config.sections():
        for key, value in config.items(section):
            if value.startswith("'"):
                value = value.strip("'")
            elif value.startswith('"'):
                value = value.strip('"')
            metadata[f"{section}.{key}"] = value
    return metadata


def save_metadata_to_ini_file(filepath: str,
                              metadata: dict
                              ) -> None:
    """
    Saves metadata to an .INI file.

    Args:
        filepath  (str): The path to the .INI file where metadata will be saved.
        metadata (dict): A dictionary containing metadata.
                         Keys should be in the format "<section>.<key>".
    Raises:
        ValueError: If any key in metadata does not follow the "<section>.<key>" format.
    """
    config = ConfigParser()

    # process each key in the metadata dictionary
    for key, value in metadata.items():
        try:
            section, option = key.split('.', 1)
        except ValueError:
            print(f"Error: metadata key '{key}' does not follow the '<section>.<key>' format.")
            return
        if section not in config:
            config[section] = {}
        config[section][option] = value

    # write the config to the .INI file
    with open(filepath, 'w') as configfile:
        config.write(configfile)


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



#================================ COMMANDS =================================#

def generate_safetensors(ini_files: list[str] | str,
                         tokenizer: T5Tokenizer,
                         encoder  : T5EncoderModel,
                         padding  : bool = False,
                         overwrite: bool = False
                         ) -> None:
    """
    Generate SafeTensors from INI files containing prompts.

    Args:
        ini_files (list[str] | str): Path(s) to INI file(s)
        tokenizer  (T5Tokenizer): T5 tokenizer
        encoder (T5EncoderModel): T5 encoder model
        padding           (bool): Whether to pad the tokens
        overwrite         (bool): Whether to overwrite existing output files
    """
    if isinstance(ini_files, str):
        ini_files = [ini_files]


    for ini_file in ini_files:
        metadata = load_metadata_from_ini_file(ini_file)
        positive = metadata.get('prompt.positive', '')
        negative = metadata.get('prompt.negative', '')

        if not positive and not negative:
            print(f"El archivo {ini_files} fue skippeado porque no contiene ningun prompt")
            continue

        tensors = {}
        if padding:
            tokens = tokenizer.tokenize_with_weights([positive,negative],
                                                     padding=True,
                                                     padding_max_size=300)
            embeds, attn_mask = encoder.encode_with_weights(tokens,
                                                            return_attn_mask=True
                                                            )
            tensors['prompt.positive']           = embeds[0]
            tensors['prompt.positive_attn_mask'] = attn_mask[0]
            tensors['prompt.negative']           = embeds[1]
            tensors['prompt.negative_attn_mask'] = attn_mask[1]
        else:
            tokens = tokenizer.tokenize_with_weights([positive], padding=False)
            tensors['prompt.positive'] = encoder.encode_with_weights(tokens)
            tokens = tokenizer.tokenize_with_weights([negative], padding=False)
            tensors['prompt.negative'] = encoder.encode_with_weights(tokens)

        _, output_filename = os.path.split(ini_file)
        output_filename    = change_extension(output_filename, '.safetensors', overwrite=overwrite)

        print("## output_filename:", output_filename)
        print("## prompt.positive:", tensors['prompt.positive'].shape)
        print("## prompt.negative:", tensors['prompt.negative'].shape)
        save_safetensors(tensors, output_filename, metadata=metadata)


def recreate_inis_from_safetensors(safetensors_files: list[str] | str,
                                   overwrite        : bool = False
                                   ) -> None:
    if isinstance(safetensors_files, str):
        safetensors_files = [safetensors_files]

    for safetensors_file in safetensors_files:

        metadata = load_metadata_from_safetensors(safetensors_file)
        _, ini_filename = os.path.split(safetensors_file)
        ini_filename    = change_extension(ini_filename, '.ini', overwrite=overwrite)
        save_metadata_to_ini_file(ini_filename, metadata=metadata)


def check_safetensors(safetensors_files: list[str] | str
                      ) -> None:
    return None


def internal_test() -> None:
    print("No implementado todavia")



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
        description = "Tool for testing T5 extended encoder embedding generation.",
        add_help    = False
        )
    parser.add_argument('files'               , nargs='*'          , help="List of '.prompt.ini' or '.prompt.safetensors' files to process.")
    parser.add_argument('-c', '--check'       , action='store_true', help="Validate '.prompt.safetensors' files by comparing embeddings.")
    parser.add_argument('-i', '--recreate-ini', action='store_true', help="Recreate '.prompt.ini' from '.prompt.safetensors' files.")
    parser.add_argument('-t', '--test'        , action='store_true', help="Run internal tests to verify T5 encoder and tokenizer functionality.")
    parser.add_argument('-h', '--help'        , action='help', default=argparse.SUPPRESS, help='Show this help message and exit.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')

    args = parser.parse_args()

    if args.test:
        internal_test()
        sys.exit(0)

    prompt_ini_files  = filter_files(args.files, 'prompt.ini')
    safetensors_files = filter_files(args.files, 'prompt.safetensors')

    if args.check:
        if not safetensors_files:
            fatal_error("No '*.prompt.safetensors' files provided for validation.")
        check_safetensors(safetensors_files)
        sys.exit(0)

    if args.recreate_ini:
        if not safetensors_files:
            fatal_error("No '*.prompt.safetensors' files provided to recreate the '*.prompt.ini'.")
        recreate_inis_from_safetensors(safetensors_files)
        sys.exit(0)

    if not prompt_ini_files:
        fatal_error("No 'prompt.ini' files provided for safetensors generation.")
    generate_safetensors(prompt_ini_files)


if __name__ == '__main__':
    main()

