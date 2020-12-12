from utils.checkpoint import Checkpoint
from utils.default import DefaultSetting
from utils.earlystopping import EarlyStopping

import torch
import torch.nn as nn


class ModelWrapper(DefaultSetting):
    def __init__(
        self,
        model,
        loss_func=None,
        optimizer=None,
        device=None,
        multi_gpus=True,
        log=100,
    ):
        super().__init__(device, loss_func)
        self.model = model
        if optimizer is None:
            self.optimizer = self.default_optimizer(model)
        else:
            self.optimizer = optimizer
        self.multi_gpus = multi_gpus
        self.log = log
        self.checkpoint = None

    # train model
    def train(
        self, train_loader, val_loader=None, max_epochs=1000, enable_early_stopping=True
    ):
        if val_loader is None:
            enable_early_stopping = False

        print("-" * 2, "Training Setup", "-" * 2)
        print(f"Maximum Epochs: {max_epochs}")
        print(f"Enable Early Stoping: {enable_early_stopping}")
        print("-" * 20)
        print("*Start Training.")
        model = self.model
        optimizer = self.optimizer
        loss_func = self.loss_func

        # model setup
        model.train().to(self.device)
        if self.multi_gpus and torch.cuda.device_count() > 1:
            print(f"*Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        # early stopping instance
        if enable_early_stopping:
            early_stopping = EarlyStopping(patience=5)

        # training start!
        for epoch in range(1, max_epochs + 1):
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
                        f"[{epoch}/{max_epochs}, {step}/{len(train_loader)}] loss: {running_loss / step :.3f}"
                    )

            # train & validation loss
            train_loss = running_loss / len(train_loader)
            if val_loader is None:
                print(f"train loss: {train_loss:.3f}")
            else:
                val_loss = self.validation(model, val_loader)
                print(f"train loss: {train_loss:.3f}, val loss: {val_loss:.3f}")

                if enable_early_stopping:
                    early_stopping(model, val_loss, optimizer)
                    if early_stopping.get_early_stop() == True:
                        print("*Early Stopping.")
                        break

        print("*Finished Training!")
        if enable_early_stopping:
            checkpoint = early_stopping.get_checkpoint()
        else:
            checkpoint = Checkpoint()
            checkpoint.tmp_save(model, optimizer, epoch, val_loss)
        self.checkpoint = checkpoint
        self.model = checkpoint.load(model, optimizer)["model"]
        return self.model

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

    def classification_evaluate(self, test_loader, classes):
        model = self.model
        model.eval().to(self.device)

        total = 0
        correct = 0
        class_correct = list(0.0 for i in range(len(classes)))
        class_total = list(0.0 for i in range(len(classes)))
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        print(
            f"Accuracy of the network on the {len(test_loader)} test inputs: {(100 * correct / total)} %"
        )
        for i in range(len(classes)):
            print(
                f"Accuracy of {classes[i]: >5} : {100 * class_correct[i] / class_total[i]:.0f} %"
            )
