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
from glob              import glob
from configparser      import ConfigParser
from safetensors.torch import safe_open, save_file as save_safetensors
from src.t5            import T5Tokenizer, T5EncoderModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CHECK_TENSOR_PROMPT = """
A ((photo)) of a [small] flying ((hovercraft:1.1) with a pilot inside),
hovering over the [((rugged)) surface of Mars].
[The ((craft:1.2)), made of [reflective metals], (gleams under the (sunlight))].
(In the background, a tunnel [[entrance]] adds a sense of (mystery:1.15)).
(The image, with its [selective focus], captures ((detailed textures)),
and (an atmospheric color palette that emphasizes the Martian landscape)).
"""


#================================= HELPERS =================================#

def get_filename(path: str) -> str:
    """Return the basename of the given file path."""
    return os.path.basename(path)

def get_directory(path: str) -> str:
    """Return the directory name of the given file path."""
    return os.path.dirname(path)

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

def fatal_error(message: str) -> None:
    """Print an error message and exit the program with status 1."""
    print(message)
    sys.exit(1)

def filter_files(filepaths: list[str], extension: str) -> list[str]:
    """Filter filepaths list to include only files with the specified extension."""
    extension = extension.lower()
    return [filepath for filepath in filepaths if filepath.lower().endswith(extension)]

def load_safetensors(filepath: str) -> dict[str, torch.Tensor]:
    """Load tensors from a SafeTensors file."""
    tensors = {}
    with safe_open(filepath, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def compare_tensors(file_tensors     : dict[str, torch.Tensor],
                    generated_tensors: dict[str, torch.Tensor]
                    ) -> None:
    """Compare and print the similarity of tensors in two dictionaries."""
    for key, generated_tensor in generated_tensors.items():
        archived_tensor = file_tensors.get(key)
        if archived_tensor is None:
            print(f"Key '{key}' missing in stored tensors")
        else:
            similarity = torch.cosine_similarity(archived_tensor, generated_tensor, dim=1).mean()
            print(f"Tensor '{key}' similarity = {similarity}")


#============================ MAIN CONFIG FILE =============================#

CONFIG_PATH    = os.path.join(SCRIPT_DIR, 't5xe.ini')
CONFIG_EXAMPLE = os.path.join(SCRIPT_DIR, 't5xe.example.ini')

def read_main_config(section_name:str) -> (dict, str):
    # primero se asegura que el archivo de configuracion exista
    if not os.path.exists(CONFIG_PATH):
        if not os.path.exists(CONFIG_EXAMPLE):
            fatal_error(f"El archivo {CONFIG_EXAMPLE} no fue encontrado")
        shutil.copy(CONFIG_EXAMPLE, CONFIG_PATH)

    # obtiene el nombre del archivo de configuracion que sera mostrado al usuario
    config_filename = os.path.basename(CONFIG_PATH)

    # intenta leer la seccion solicitada
    config = ConfigParser()
    config.read(CONFIG_PATH)
    config_section = config[section_name] if section_name in config else None
    if not config_section:
        fatal_error(f"The configuration file '{config_filename}' no contiene la sección ['{section_name}']")

    return config_section, config_filename


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


def load_metadata_from_ini_file(filepath: str
                                ) -> dict:
    """
    Load image parameters from an INI file and convert it to metadata format.

    Args:
        filepath (str): Path to the INI file.

    Returns:
        dict: Dictionary containing the image parameters as metadata.
    Raises:
        ValueError: If there is an issue with the INI file format.
        IOError:    If there are issues opening or reading the file.
    """
    metadata = {}
    config = ConfigParser()
    config.read(filepath)
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

def generate_tensors_from_metadata(metadata, tokenizer, encoder):
    tensors = {}

    positive = metadata.get('prompt.positive', '')
    negative = metadata.get('prompt.negative', '')
    padding  = metadata.get('pt5tokenizer.padding', 'false')
    padding  = padding.lower() in ('yes', 'true', 'y', '1')

    if not positive and not negative:
        return None

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

    return tensors

#===========================================================================#

def initialize_encoder() -> (T5Tokenizer, T5EncoderModel):
    SECTION_NAME = 't5encoder'
    MODEL_PATH   = 'model_file'
    config_section, config_filename = read_main_config(SECTION_NAME)

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
        fatal_error(f"Debe configurar correctamente the '{MODEL_PATH}' key in the '{SECTION_NAME}' section of '{config_filename}'.")
    for path in safetensors_paths:
        if not os.path.isfile(path):
            fatal_error(f"El archivo {path} no existe, por favor revise la configuracion de '{MODEL_PATH}' in '{config_filename}.")

    # initializar y retornar tokenizer y encoder
    tokenizer = T5Tokenizer.from_pretrained(legacy=True)
    encoder   = T5EncoderModel.from_safetensors(safetensors_paths)
    return tokenizer, encoder



#================================ COMMANDS =================================#

def generate_safetensors_files(ini_files: list[str] | str,
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
        tensors  = generate_tensors_from_metadata(metadata)
        if not tensors:
            print(f"El archivo {ini_files} fue skippeado porque no contiene ningun prompt")
            continue

        output_filename = get_filename(ini_file)
        output_filename = change_extension(output_filename, '.safetensors', overwrite=overwrite)

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
        ini_filename = get_filename(safetensors_file)
        ini_filename = change_extension(ini_filename, '.ini', overwrite=overwrite)
        save_metadata_to_ini_file(ini_filename, metadata=metadata)



def check_safetensors(safetensors_files: list[str] | str,
                      tokenizer: T5Tokenizer,
                      encoder  : T5EncoderModel
                      ) -> None:
    if isinstance(safetensors_files, str):
        safetensors_files = [safetensors_files]

    for safetensors_file in safetensors_files:
        metadata = load_metadata_from_safetensors(safetensors_file)
        file_tensors      = load_safetensors(safetensors_file)
        generated_tensors = generate_tensors_from_metadata(metadata, tokenizer, encoder)
        compare_tensors(file_tensors, generated_tensors)



    return None


def internal_test(tokenizer: T5Tokenizer,
                  encoder  : T5EncoderModel
                  ) -> None:

    test_dir     = os.path.join(SCRIPT_DIR, 'test')
    pattern      = os.path.join(test_dir, '*.prompt.safetensors')
    prompt_files = glob(pattern)

    if not prompt_files:
        fatal_error("No '*.prompt.safetensors' files found in the 'test' directory. These files are required for the project.")

    check_safetensors(prompt_files, tokenizer, encoder)


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
        tokenizer, encoder = initialize_encoder()
        internal_test(tokenizer, encoder)
        sys.exit(0)

    prompt_ini_files  = filter_files(args.files, 'prompt.ini')
    safetensors_files = filter_files(args.files, 'prompt.safetensors')

    #------- CHECK SAFETENSORS -------#
    if args.check:
        if not safetensors_files:
            fatal_error("No '*.prompt.safetensors' files provided for validation.")
        tokenizer, encoder = initialize_encoder()
        check_safetensors(safetensors_files, tokenizer, encoder)
        sys.exit(0)

    #------ RECREATE INI FILES -------#
    if args.recreate_ini:
        if not safetensors_files:
            fatal_error("No '*.prompt.safetensors' files provided to recreate the '*.prompt.ini'.")
        recreate_inis_from_safetensors(safetensors_files)
        sys.exit(0)

    #----- GENERATE SAFETENSORS ------#
    if not prompt_ini_files:
        fatal_error("No '*.prompt.ini' files provided for safetensors generation.")
    generate_safetensors_files(prompt_ini_files)


if __name__ == '__main__':
    main()

