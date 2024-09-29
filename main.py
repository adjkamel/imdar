from train import main as main_train
from test import main as main_test
from arguments import args


if __name__ == '__main__':

    if args.action=='train':
        main_train() 

    if args.action=='test':
        main_test() 

        