from utils.checkpoint import Checkpoint
from utils.default import DefaultSetting
from utils.earlystopping import EarlyStopping

import torch
import torch.nn as nn


class TrainingSetup(DefaultSetting):
    def __init__(
        self,
        loss_func=None,
        max_epochs=1000,
        device=None,
        multi_gpus=True,
        early_stopping=True,
        log=100,
    ):
        super().__init__(device, loss_func)
        self.max_epochs = max_epochs
        self.multi_gpus = multi_gpus
        self.early_stopping = early_stopping
        self.log = log

    # train model
    def train(self, model, train_loader, val_loader, optimizer=None):
        if optimizer is None:
            optimizer = self.default_optimizer(model)
        loss_func = self.loss_func

        # model setup
        model.train().to(self.device)
        if self.multi_gpus and torch.cuda.device_count() > 1:
            print(f"*Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        # early stopping instance
        if self.early_stopping:
            early_stopping = EarlyStopping(patience=7)

        # training start!
        for epoch in range(1, self.max_epochs + 1):
            running_loss = 0.0

            for step, data in enumerate(train_loader, start=1):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()

                if step % 100 == 0 or step == len(train_loader):
                    print(
                        f"[{epoch}/{self.max_epochs}, {step}/{len(train_loader)}] loss: {running_loss / step :.3f}"
                    )

            # train & validation loss
            train_loss = running_loss / len(train_loader)
            val_loss = self.validation(model, val_loader)
            print(f"train loss: {train_loss:.3f}, val loss: {val_loss:.3f}")

            if self.early_stopping:
                early_stopping(model, val_loss, optimizer)
                if early_stopping.get_early_stop() == True:
                    print("*Early Stopping.")
                    break

        print("*Finished Training!")
        if self.early_stopping:
            checkpoint = early_stopping.get_checkpoint()
        else:
            checkpoint = Checkpoint()
            checkpoint.tmp_save(model, optimizer, epoch, val_loss)
        return checkpoint

    # %% validation
    def validation(self, model, val_loader):
        model.eval().to(self.device)
        loss_func = self.loss_func
        running_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                running_loss += loss.item()
        return running_loss / len(val_loader)
