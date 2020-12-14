from utils.checkpoint import Checkpoint
from utils.default import DefaultSetting
from utils.earlystopping import EarlyStopping

import torch
import torch.nn as nn

# TODO: add support for tensorboard & clean code
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
        self.early_stopping = None

    # train model
    def train(
        self, train_loader, val_loader=None, max_epochs=1000, enable_early_stopping=True
    ):
        if val_loader is None:
            enable_early_stopping = False

        print()
        print("-" * 2, "Training Setup", "-" * 2)
        print(f"Maximum Epochs: {max_epochs}")
        print(f"Enable Early Stoping: {enable_early_stopping}")
        print("-" * 20)
        print("*Start Training.")
        loss_func = self.loss_func

        # model setup
        self.model.train().to(self.device)
        if self.multi_gpus and torch.cuda.device_count() > 1:
            print(f"*Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)

        # early stopping instance
        if enable_early_stopping:
            if self.early_stopping is None:
                self.early_stopping = EarlyStopping(patience=5)
            else:
                self.early_stopping.reset_counter()

        # training start!
        for epoch in range(1, max_epochs + 1):
            running_loss = 0.0

            for step, data in enumerate(train_loader, start=1):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = loss_func(outputs, labels)
                loss.backward()
                self.optimizer.step()
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
                # TODO: fixed the problem that first validation is not correct
                val_loss = self.validation(val_loader)
                print(f"train loss: {train_loss:.3f}, val loss: {val_loss:.3f}")

                if enable_early_stopping:
                    self.early_stopping(self.model, val_loss, self.optimizer)
                    if self.early_stopping.get_early_stop() == True:
                        print("*Early Stopping.")
                        break

        print("*Finished Training!")
        if enable_early_stopping:
            checkpoint = self.early_stopping.get_checkpoint()
        else:
            checkpoint = Checkpoint()
            checkpoint.tmp_save(self.model, self.optimizer, epoch, val_loss)
        self.checkpoint = checkpoint
        self.model = checkpoint.load(self.model, self.optimizer)["model"]
        return self.model

    # %% validation
    def validation(self, val_loader):
        self.model.eval().to(self.device)
        loss_func = self.loss_func
        running_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = loss_func(outputs, labels)
                running_loss += loss.item()
        return running_loss / len(val_loader)

    def classification_evaluate(self, test_loader, classes):
        print("-" * 2, "Evaluation Report", "-" * 2)
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
            f"Accuracy of the network on the {total} test inputs: {(100 * correct / total)} %"
        )
        for i in range(len(classes)):
            print(
                f"Accuracy of {classes[i]: >5} : {100 * class_correct[i] / class_total[i]:.0f} %"
            )

    def binary_classification_evaluate(self, test_loader, classes=["0", "1"]):
        print("-" * 2, "Evaluation Report", "-" * 2)
        model = self.model
        model.eval().to(self.device)

        total = 0
        correct = 0
        class_correct = list(0.0 for i in range(len(classes)))
        class_total = list(0.0 for i in range(len(classes)))
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                outputs = model(inputs)
                outputs = torch.round(outputs)

                total += labels.size(0)
                correct += (outputs == labels).sum().item()

                c = (outputs == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        print(
            f"Accuracy of the network on the {total} test inputs: {(100 * correct / total):.3f} %"
        )
        for i in range(len(classes)):
            print(
                f"Accuracy of {classes[i]: >5} : {100 * class_correct[i] / class_total[i]:.1f} % ({class_total[i]:.0f} inputs)"
            )
