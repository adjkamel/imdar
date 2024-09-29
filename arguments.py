import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-d','--dataset', default='ntu60', type=str, help='dataset: ntur60  ntu120  ucla') 
parser.add_argument('-p','--protocol', default='csub', type=str, help='ntu60: csub  cview, ntu120: csub  csetup') 
parser.add_argument('-a','--action', default='train', type=str, help='train  test') 
parser.add_argument('-despth','--descrip_path', default='./data/descriptors/', type=str) 
parser.add_argument('-dspath','--dataset_path', default='./data/ntu/npy60skel/', type=str, help='./data/npy60skel/  ./data/npy120skel/  ./data/ucla/') 
parser.add_argument('-desc','--descriptor', default='GD', type=str, help='GD  JDD  AAD  LAD  FDD  all') 
parser.add_argument('-gpu','--gpu', default=0, type=int)




args = parser.parse_args() 

