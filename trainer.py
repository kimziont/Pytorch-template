import time, wandb
import torch
import numpy as np

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from utils import EarlyStopping, stratified_k_fold


class Trainer():
    def __init__(self, config, device, model, optimizer, criterion, dataset, lr_scheduler):
        self.config = config
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()
        self.lr_scheduler= lr_scheduler
        self.dataset = dataset
        self.epochs = self.config["epochs"]

        self.train_total_loss = 0
        self.valid_total_loss = 0
        self.train_correct = 0
        self.valid_correct = 0

        self.train_loss = []
        self.valid_loss = []
        self.train_predict_list = np.array([])
        self.train_target_list = np.array([])
        self.valid_predict_list = np.array([])
        self.valid_target_list = np.array([])
        self.k_fold_accuracy = []
        self.k_fold_f1_score = []
        self.now = time.localtime()
        self.early_stopping = EarlyStopping(path=self.config["best_model"]+f"{self.now.tm_year}-{self.now.tm_mon}-{self.now.tm_mday}-{self.now.tm_hour}-{self.now.tm_min}-model{self.config['model']}-bs{self.config['batch_size']}-lr{self.config['learning_rate']}-epoch{self.config['epochs']}.pt", 
                                            patience=self.config["patience"], verbose=True)

        
    
    def train_per_epoch(self, epoch, data_loader, train_dataset):
        self.model.train()
        self.train_total_loss = 0
        self.train_correct = 0
        for data, target in tqdm(data_loader, leave=False):
            data = data.to(self.device)
            target = target.to(self.device)
            with torch.cuda.amp.autocast():
                pred = self.model(data)
                loss = self.criterion(pred, target)
                # print([pred.argmax(dim=1), target])

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            # AssertionError: No inf checks were recorded for this optimizer
            # Scaler (optimizer) looks for parameters used in the graph which is empty, hence, the error.
            try:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            except:
                pass
            

            self.train_correct += sum(pred.argmax(dim=1) == target)
            self.train_predict_list = np.concatenate((self.train_predict_list, pred.argmax(dim=1).cpu().numpy()))
            self.train_target_list = np.concatenate((self.train_target_list, target.cpu().numpy()))

            self.train_total_loss += loss.item()
        self.train_total_loss /= len(train_dataset)
        self.train_loss.append(self.train_total_loss)
        accuracy = accuracy_score(y_true=self.train_target_list, y_pred=self.train_predict_list)
        f1 = f1_score(y_true=self.train_target_list, y_pred=self.train_predict_list, average="macro")
        self.lr_scheduler.step()
        print(f"{epoch}/{self.epochs}Epochs {self.train_correct}/{len(train_dataset)} train accuracy: {accuracy:.4f}, f1_score: {f1:.4f}, loss: {self.train_total_loss:.4f}")
 
    
    def valid_per_epoch(self, epoch, data_loader, valid_dataset):
        self.model.eval()

        with torch.no_grad():
            self.valid_total_loss = 0
            self.valid_correct = 0
            loop = tqdm(data_loader, total=len(data_loader))

            for data, target in loop:
                data = data.to(self.device)
                target = target.to(self.device)
                pred = self.model(data)
                self.valid_predict_list = np.concatenate((self.valid_predict_list, pred.argmax(dim=1).cpu().numpy()))
                self.valid_target_list = np.concatenate((self.valid_target_list, target.cpu().numpy()))

                loss = self.criterion(pred, target)

                # print([pred.argmax(dim=1), target])


                
                self.valid_correct += sum(pred.argmax(dim=1) == target)
                self.valid_total_loss += loss.item()
            self.valid_total_loss /= len(valid_dataset)
            self.valid_loss.append(self.valid_total_loss)
            accuracy = accuracy_score(y_true=self.valid_target_list, y_pred=self.valid_predict_list)
            f1 = f1_score(y_true=self.valid_target_list, y_pred=self.valid_predict_list, average="macro")
            print(f"{epoch}/{self.epochs}Epochs {self.valid_correct}/{len(valid_dataset)} valid accuracy: {accuracy:.4f}, f1_score: {f1:.4f}, loss: {self.valid_total_loss:.4f}")

            # valid에서는 best_model을 저장하기 위해 loss값을 리턴한다
            return self.valid_total_loss, accuracy, f1

    

    def epoch_iter(self, epochs, train_per_epoch, valid_per_epoch, train_dataloader, train_dataset, valid_dataloader, valid_dataset):
        for epoch_index in range(1, epochs + 1):
            train_per_epoch(epoch_index, train_dataloader, train_dataset)
            valid_loss, valid_accuracy, valid_f1 = valid_per_epoch(epoch_index, valid_dataloader, valid_dataset)
            self.early_stopping(self.valid_total_loss, self.model, self.optimizer)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            yield valid_loss, valid_accuracy, valid_f1


    def train(self):
        df = self.dataset.get_dataframe()
        X, y = df['file'], df['label']

        if self.config["kfold"]:

            k_fold_iter = stratified_k_fold(self.config, X, y)

            for train_dataset, valid_dataset, train_dataloader, valid_dataloader in k_fold_iter:
                for valid_loss, valid_accuracy, valid_f1 in self.epoch_iter(self.epochs, self.train_per_epoch, self.valid_per_epoch, train_dataloader, train_dataset, valid_dataloader, valid_dataset):
                    wandb.log({'Valid accuracy': valid_accuracy, 'Valid f1 score': valid_f1, 'Valid loss': valid_loss})
                self.k_fold_accuracy.append(valid_accuracy)
                self.k_fold_f1_score.append(valid_f1)
            print(f"교차 검증 정확도: {sum(self.k_fold_accuracy)/len(self.k_fold_accuracy):.4f}, f1 스코어: {sum(self.k_fold_f1_score)/len(self.k_fold_f1_score):.4f}")
        # not k-fold
        else:
            dataset = self.dataset
            train_dataset, valid_dataset = random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)])
            train_dataloader = DataLoader(train_dataset, batch_size=self.config["batch_size"], num_workers=self.config["num_workers"])
            valid_dataloader = DataLoader(valid_dataset, batch_size=self.config["batch_size"], num_workers=self.config["num_workers"])

            for valid_loss, valid_accuracy, valid_f1 in self.epoch_iter(self.epochs, self.train_per_epoch, self.valid_per_epoch, train_dataloader, train_dataset, valid_dataloader, valid_dataset):
                wandb.log({'Valid accuracy': valid_accuracy, 'Valid f1 score': valid_f1, 'Valid loss': valid_loss})
            
            
        
            
        
            
