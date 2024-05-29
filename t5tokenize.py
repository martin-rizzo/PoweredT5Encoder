import argparse
#from src.t5 import T5Tokenizer

def main():
    parser = argparse.ArgumentParser(description='??')
    parser.add_argument('prompt', nargs='*', help='The text prompt. Can be a single parameter or multiple.')
    parser.add_argument('--version',  action='version', version='%(prog)s 0.1')
    parser.add_argument('--test', '-t', action='store_true', help='Run tests for tokenization.')

    #args      = parser.parse_args()
    #tokenizer = T5Tokenizer.from_pretrained(legacy=True)
    print("Hello world!")



if __name__ == '__main__':
    main()
