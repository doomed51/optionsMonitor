from core import options_analyzer
import sys 

if __name__ == '__main__':
    if sys.argv[1] == '':
        print('Please provide ticker')
    options_analyzer.main(sys.argv[1])   