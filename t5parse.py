"""
  File    : p5parse.py
  Brief   : ???
  Author  : Martin Rizzo | <martinrizzo@gmail.com>
  Date    : May 1, 2024
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

TEST_CASES = [
    ("Normal text",
     "[('Normal text', '1.00')]"
    ),
    ("An (important) word",
     "[('An ', '1.00'), ('important', '1.10'), (' word', '1.00')]"
    ),
    ("The (unnecessary)(parens)",
     "[('The ', '1.00'), ('unnecessaryparens', '1.10')]"
    ),
    ("\[Escaped\] \(text\)",
     "[('[Escaped] (text)', '1.00')]"
    ),
    ("(A great (((house:1.2)) [on] a (hill:0.5) under an intense) (((sun))) \(score:8\).",
     "[('A great ', '1.00'), ('house', '1.45'), (' ', '1.10'), ('on', '1.00'), (' a ', '1.10'), ('hill', '0.55'), (' under an intense', '1.10'), (' ', '1.00'), ('sun', '1.33'), (' (score:8).', '1.00')]"
    )
]

def convert_to_string(segments_weights: list, weight_format: str = "{:.2f}") -> str:
    segments_weights = [(segment, weight_format.format(weight)) for segment, weight in segments_weights]
    return str(segments_weights)

def test(tokenizer:T5Tokenizer):
    """
    Test the 'parse_segments_weights' function with a list of test cases.
    Prints a summary of the test results, including the cases that passed and failed.
    """
    passed_cases = []
    failed_cases = []

    for prompt, expected_output in TEST_CASES:
        actual_output = tokenizer.parse_segments_weights(prompt)
        actual_output = convert_to_string( actual_output, weight_format="{:.2f}" )

        if actual_output == expected_output:
            passed_cases.append(prompt)
        else:
            failed_cases.append((prompt, expected_output, actual_output))

    print("TEST RESULTS\n============\n")
    print("Passed Cases:")
    if not passed_cases:
        print("  None")
    else:
        for prompt in passed_cases:
            print(f"  - {prompt}")

    print("\nFailed Cases:")
    if not failed_cases:
        print("  None")
    else:
        for prompt, expected, actual in failed_cases:
            print(f"  - Prompt: {prompt}")
            print(f"    Expected: {expected}")
            print(f"    Actual: {actual}")

    print(f"\nSummary: {len(passed_cases)} passed, {len(failed_cases)} failed")


def main():
    parser = argparse.ArgumentParser(
        description='??',
        add_help=False
        )
    parser.add_argument('prompt', nargs='*', help='The text prompt. Can be a single parameter or multiple.')
    parser.add_argument('-t', '--test', action='store_true', help='run tests for the parse_segments_weights function.')
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    parser.add_argument('--version',  action='version', version='%(prog)s 0.1')

    args      = parser.parse_args()
    tokenizer = T5Tokenizer.from_pretrained(legacy=True)

    if args.test:
        test(tokenizer)
    else:
        prompt = ' '.join(args.prompt) if args.prompt else ''
        segments_weights = tokenizer.parse_segments_weights(prompt)
        segments_weights = convert_to_string( segments_weights )
        print( segments_weights )

if __name__ == '__main__':
    main()
