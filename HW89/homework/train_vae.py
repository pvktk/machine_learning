
# куски кода взяты из https://github.com/pytorch/examples/blob/master/vae/main.py

import vae.trainer
import vae.vae
import argparse
import torch
import torch.utils.data
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()),
                    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

model = vae.vae.VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
def main():
    
    trainer = vae.trainer.Trainer(model, train_loader, test_loader, optimizer, vae.vae.loss_function, device)    
    
    for epoch in range(args.epochs + 1):
        trainer.train(epoch, args.log_interval)
        trainer.test(epoch, args.batch_size, args.log_interval)
        


if __name__ == '__main__':
    main()
