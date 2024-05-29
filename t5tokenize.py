"""
  File    : p5tokenize.py
  Brief   : ???
  Author  : Martin Rizzo | <martinrizzo@gmail.com>
  Date    : May 2, 2024
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
"""
import argparse
from src.t5 import T5Tokenizer

def convert_to_string(tokens_weights: list, weight_format: str = "{:.2f}") -> str:
    if not tokens_weights:
        return tokens_weights
    if len(tokens_weights[0]) == 3:
        tokens_weights = [(token, weight_format.format(weight), word_id) for token, weight, word_id in tokens_weights]
    else:
        tokens_weights = [(token, weight_format.format(weight)) for token, weight in tokens_weights]
    return str(tokens_weights)


def main():
    parser = argparse.ArgumentParser(
        description='??',
        add_help=False
        )
    parser.add_argument('prompt', nargs='*', help='The text prompt. Can be a single parameter or multiple.')
    parser.add_argument('-w', '--wordid', action='store_true', help='en cada token muestra el ID de la palabra a la que pertenece')
    parser.add_argument('-t', '--test', action='store_true', help='run tests for the parse_segments_weights function.')
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    parser.add_argument('--version',  action='version', version='%(prog)s 0.1')

    args      = parser.parse_args()
    tokenizer = T5Tokenizer.from_pretrained(legacy=True)

    prompt = ' '.join(args.prompt) if args.prompt else ''
    tokens_weights = tokenizer.tokenize_with_weights(prompt,
                                                     padding          = False,
                                                     padding_max_size = 0,
                                                     include_word_ids = args.wordid
                                                     )[0]
    tokens_weights = convert_to_string( tokens_weights )
    print( tokens_weights )

if __name__ == '__main__':
    main()
