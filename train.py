import os

import numpy as np
from torch.optim.lr_scheduler import StepLR

import wandb
import torch
from pathlib import Path
import yaml

import utils
from network import TripletNetwork
from dataset import get_dataset_triplet
from utils import seed_everything
from utils import TripletLoss


class TripletLearningTrainer:
    def __init__(self, config):

        self.checkpoint_path = None
        self.scheduler = None
        self.optimizer = None
        self.config = config
        self.cost_function = TripletLoss()

        # Parameters
        self.max_epochs = config["max_epochs"]
        self.save_checkpoint_every = config["save_checkpoint_every"]
        self.early_stopping_patience = config["early_stopping_patience"]
        self.trigger = 0

    def setup_trainer(self, net, checkpoint_path):
        # Create the checkpoint tree
        self.checkpoint_path = checkpoint_path

        params = [
            {"params": net.feature_extractor.parameters(), "lr": self.config["learning_rate_pretrained"]},
            {"params": net.embedding.parameters(), "lr": self.config["learning_rate_embedding"]}
        ]
        self.optimizer = torch.optim.Adam(params)
        self.scheduler = StepLR(self.optimizer, step_size=self.config["scheduler_step"],
                                gamma=self.config["scheduler_gamma"])

    def save(self, net, epoch, is_best=False):
        if is_best:
            torch.save(net.state_dict(), self.checkpoint_path / f'best.pth')  # in this case we store only the model
        else:
            save_dict = {
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }

            torch.save(save_dict, self.checkpoint_path / f'epoch-{epoch}.pth')

    def train(self, net, train_loader, val_loader):

        # For each epoch, train the network and then compute evaluation results
        best_val_accuracy = 0.0
        best_val_loss = float('inf')
        print(f"\tStart training...")

        for e in range(self.max_epochs):
            train_loss, train_accuracy = self.train_one_epoch(e, net, train_loader)
            val_loss, val_accuracy = self.validation_step(net, val_loader)
            self.scheduler.step()
            print('Epoch: {:d}'.format(e + 1))
            print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
            print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss, val_accuracy))
            print('\t Best Accuracy {:.5f}'.format(best_val_accuracy))
            print('-----------------------------------------------------')

            wandb.log({
                # log training stats
                "train/loss": train_loss,
                "train/accuracy": train_accuracy,
                # log validation stats
                "val/loss": val_loss,
                "val/accuracy": val_accuracy
            })

            # Save the model checkpoints
            if e % self.save_checkpoint_every == 0 or e == (
                    self.max_epochs - 1):  # if the current epoch is in the interval, or is the last epoch -> save
                model_checkpoint = {
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }
                torch.save(model_checkpoint, self.checkpoint_path / f'epoch-{e}.pth')

            # Update the best model so far
            if val_accuracy >= best_val_accuracy:
                model_checkpoint = {
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }
                torch.save(model_checkpoint, self.checkpoint_path / f'best.pth')
                best_val_accuracy = val_accuracy

            # Early Stopping
            if val_loss > best_val_loss:
                self.trigger += 1
                if self.trigger == self.early_stopping_patience:
                    print(
                        f"Validation Accuracy did not improve for {self.early_stopping_patience} epochs. Killing the training...")
                    break
            else:
                # update the best val loss so far
                best_val_loss = val_loss
                self.trigger = 0
            # ===========================================

    # Train the model for one epoch, similar to the one we saw on Lab 03 / 04
    def train_one_epoch(self, epoch, net, data_loader):
        running_loss = []
        running_accuracy = []

        for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(data_loader):
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)
            self.optimizer.zero_grad()
            anchor_out = net(anchor_img)
            positive_out = net(positive_img)
            negative_out = net(negative_img)
            loss = self.cost_function(anchor_out, positive_out, negative_out)
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())

            # Calculate accuracy
            anchor_dist = torch.norm(anchor_out - positive_out, dim=1)
            negative_dist = torch.norm(anchor_out - negative_out, dim=1)
            accuracy = torch.mean((anchor_dist < negative_dist).float())
            running_accuracy.append(accuracy.item())

            if step % 20 == 0:
                print('--- Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}, Accuracy: {:.4f}'.format(
                    epoch + 1, step * len(anchor_img), len(data_loader.dataset),
                    100. * step / len(data_loader), loss.item(), accuracy.item()))

        mean_loss = float(np.mean(running_loss))
        mean_accuracy = float(np.mean(running_accuracy))
        return round(mean_loss, 5), round(mean_accuracy, 5)

    # Validation function
    def validation_step(self, net, data_loader):
        running_loss_test = []
        running_accuracy_test = []

        with torch.no_grad():
            for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(data_loader):
                anchor_img = anchor_img.to(device)
                positive_img = positive_img.to(device)
                negative_img = negative_img.to(device)
                anchor_out = net(anchor_img)
                positive_out = net(positive_img)
                negative_out = net(negative_img)
                loss = self.cost_function(anchor_out, positive_out, negative_out)
                running_loss_test.append(loss.cpu().detach().numpy())

                # Calculate accuracy
                anchor_dist = torch.norm(anchor_out - positive_out, dim=1)
                negative_dist = torch.norm(anchor_out - negative_out, dim=1)
                accuracy = torch.mean((anchor_dist < negative_dist).float())
                running_accuracy_test.append(accuracy.item())

        mean_loss_test = float(np.mean(running_loss_test))
        mean_accuracy_test = float(np.mean(running_accuracy_test))
        return round(mean_loss_test, 5), round(mean_accuracy_test, 5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Training Parser")
    # celeba
    parser.add_argument("--config_path", required=False, type=str, help="Path to the configuration file",
                        default="config/triplet_init.yaml")
    parser.add_argument("--run_name", required=False, type=str, help="Name of this training run", default="Triplet")
    parser.add_argument("--checkpoint_path", required=False, type=str, default="./checkpoints",
                        help="path where the checkpoints will be stored")

    opt = parser.parse_args()  # parse the arguments, this creates a dictionary name : value

    seed_everything()

    # === Load the configuration file
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
        print(f"\tConfiguration file loaded from: {opt.config_path}")

    # === Init the wandb run and log the config files
    wandb.init(
        project="IML-Triplet",
        name=opt.run_name,
        config=config)

    # === Create the model
    model = TripletNetwork()
    device = utils.get_default_device()
    model.to(device)

    # Create checkpoint path
    checkpoint_path = Path(opt.checkpoint_path) / opt.run_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    # save a copy of the config file being used, to be sure. Append the comand line parameters
    config.update({'command_line': vars(opt)})
    with open(checkpoint_path / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Get dataset
    train_loader, val_loader = get_dataset_triplet(config["dataset"])
    trainer = TripletLearningTrainer(config["training"])
    trainer.setup_trainer(model, checkpoint_path)
    trainer.train(model, train_loader, val_loader)
