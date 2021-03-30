from train import train
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--batchsize", default=2, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--eval_batchsize", default=256, type=int)
parser.add_argument("--multi_cuda", default=True, type=bool)
parser.add_argument("--model", default='unet', type=str)
args_default = parser.parse_args()
def unet(args):
	args.batchsize = 18
	args.model = 'unet'
	train(args_default)
def deeplab(args):
	args.batchsize = 12
	args.model = 'deeplab'
	train(args) 
if __name__ == '__main__':
	unet(args_default)
	deeplab(args_default)






