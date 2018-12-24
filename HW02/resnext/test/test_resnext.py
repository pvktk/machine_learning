
from torch.utils.data import Dataset

import sys
sys.path.append('.')
from resnext.resnext import *
from resnext.trainer import Trainer
import torch

torch.manual_seed(10)

print('forward pass on random data test')

class simple_dataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return torch.randn(1, 28, 28), 1
    

def test_forward_pass():
    net = ResNext(Bottleneck, [3, 4, 6, 3], num_classes=2)
    trainer = Trainer(simple_dataset(), simple_dataset())
    
    trainer.train(net)
    
    assert True

if __name__ == "__main__":
    test_forward_pass()
