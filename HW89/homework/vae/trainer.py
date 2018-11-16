import logging
import os

import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter


class Trainer:

    def __init__(self, model, train_loader, test_loader, optimizer,
                 loss_function, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.writer = SummaryWriter()

    def train(self, epoch, log_interval):
        self.model.train()
        epoch_loss = 0

        for batch_idx, (data, _) in enumerate(self.train_loader):
            # ############################
            
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            
            train_loss = loss.item()
            #######################
            epoch_loss += train_loss
            norm_train_loss = train_loss / len(data)

            self.optimizer.step()
            if batch_idx % log_interval == 0:
                msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    norm_train_loss)
                logging.info(msg)

                batch_size = self.train_loader.batch_size
                train_size = len(self.train_loader.dataset)
                batches_per_epoch_train = train_size // batch_size
                self.writer.add_scalar(tag='data/train_loss',
                                       scalar_value=norm_train_loss,
                                       global_step=batches_per_epoch_train * epoch + batch_idx)

        epoch_loss /= len(self.train_loader.dataset)
        #logging.info(f'====> Epoch: {epoch} Average loss: {epoch_loss:.4f}')
        logging.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, epoch_loss))
        self.writer.add_scalar(tag='data/train_epoch_loss',
                               scalar_value=epoch_loss,
                               global_step=epoch)

    def test(self, epoch, batch_size, log_interval):
        self.model.eval()
        test_epoch_loss = 0

        for batch_idx, (data, _) in enumerate(self.test_loader):
            # #############################################
            
            data = data.to(self.device)
            recon_batch, mu, logvar = self.model(data)
            
            test_loss = self.loss_function(recon_batch, data, mu, logvar).item()
            
            ###############################################
            test_epoch_loss += test_loss

            if batch_idx % log_interval == 0:
                msg = 'Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(self.test_loader.dataset),
                    100. * batch_idx / len(self.test_loader),
                    test_loss / len(data))
                logging.info(msg)

                batches_per_epoch_test = len(self.test_loader.dataset) // batch_size
                self.writer.add_scalar(tag='data/test_loss',
                                       scalar_value=test_loss / len(data),
                                       global_step=batches_per_epoch_test * (epoch - 1) + batch_idx)

        test_epoch_loss /= len(self.test_loader.dataset)
        logging.info('====> Test set loss: {:.4f}'.format(test_epoch_loss))
        self.writer.add_scalar(tag='data/test_epoch_loss',
                               scalar_value=test_epoch_loss,
                               global_step=epoch)
        self.plot_generated(epoch, batch_size)

    def plot_generated(self, epoch, batch_size):
        with torch.no_grad():
            sample = torch.randn(64, 20).to(self.device)
            sample = self.model.decode(sample).cpu()
            vutils.save_image(sample.view(64, 1, 28, 28),
                'vae_results/sample_' + str(epoch) + '.png')
        

    def save(self, checkpoint_path):
        dir_name = os.path.dirname(checkpoint_path)
        os.makedirs(dir_name, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
