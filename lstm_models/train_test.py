import os
import time
import numpy as np
from glob import glob

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.functions import dict_to_str, eval_regression

class Train_Test:
    def __init__(self, args, model, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader, logger):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.best_test_loss = float('inf')
        self.best_test_acc = 0
        self.best_epoch = 0
        self.best_model = None
        self.best_model_path = None

        self.tensorboard_path = os.path.join(self.args.res_save_dir, 'tensorboard')
        os.makedirs(self.tensorboard_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tensorboard_path)

    def train_epoch(self, epoch):
        y_pred, y_true = [], []
        self.model.train()
        train_loss = 0.0
        step = 0
        with tqdm(self.train_dataloader, desc="Training", leave=False) as pbar:
            for batch_data in pbar:
                start_time = time.time()

                vision = batch_data['vision'].to(self.args.device)
                audio = batch_data['audio'].to(self.args.device)
                text = batch_data['text'].to(self.args.device)
                labels = batch_data['labels']['M'].view(audio.shape[0], -1).to(self.args.device)

                if not self.args.need_data_aligned:
                    audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                    vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                else:
                    audio_lengths, vision_lengths = 0, 0

                model_loss, batch_pred, true_label = self.model(text, (audio, audio_lengths), (vision, vision_lengths),
                                                                labels)

                model_loss.backward()
                train_loss += float(model_loss)

                y_pred.append(batch_pred.cpu())
                y_true.append(true_label.cpu())
                self.optimizer.step()
                self.optimizer.zero_grad()

                pbar.set_postfix(loss=model_loss.item(), Time=(time.time() - start_time))
                step += 1

        train_loss = train_loss / len(self.train_dataloader)
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        return train_loss, y_pred, y_true

    def do_train(self):
        self.logger.info("Start training...")
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = float('inf') if min_or_max == 'min' else 0

        for epoch in range(self.args.epochs):
            train_loss, y_pred, y_true = self.train_epoch(epoch)

            self.logger.info("Epoch %d/%d Finished, Train Loss: %f", epoch + 1, self.args.epochs,
                             round(np.mean(train_loss), 4))
            self.writer.add_scalar('Loss/train', train_loss, epoch + 1)

            val_results = self.do_test(self.model, mode="VAL")
            test_results = self.do_test(self.model, mode="TEST")
            cur_valid = val_results['Loss']
            self.writer.add_scalar('Valid_Loss', cur_valid, epoch + 1)

            is_better = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if is_better:
                best_valid, best_epoch = cur_valid, epoch
                torch.save(self.model.state_dict(), self.args.model_save_path)

        self.logger.info("Save model at Epoch %d", best_epoch + 1)


    def do_test(self, model, mode="VAL"):
        model = model.to(self.args.device)
        model.eval()
        eval_loss = 0.0
        
        if mode == "TEST":
            dataloader = self.test_dataloader
        elif mode == "VAL":
            dataloader = self.val_dataloader
        
        with torch.no_grad():
            total_pred = []
            total_true_label = []
            eval_loss = 0.0
            criterion = nn.L1Loss()
            
            with tqdm(dataloader, desc=mode, leave=False) as pbar:
                for batch_data in pbar:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    
                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    labels = batch_data['labels']['M'].view(audio.shape[0], -1).to(self.args.device)
                    
                    batch_pred, true_label = self.model(text, (audio, audio_lengths), (vision, vision_lengths),
                                                        groundTruth_labels=labels, training=False)

                    total_pred.append(batch_pred.cpu())
                    total_true_label.append(true_label.cpu())
                    loss = criterion(batch_pred, true_label)
                    eval_loss += loss.item()

                    pbar.set_postfix({"Loss": loss.item()})
        
        total_pred = torch.cat(total_pred, 0)
        total_true_label = torch.cat(total_true_label, 0)
        eval_loss = eval_loss / len(dataloader)

        self.logger.info(mode + "-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        eval_results = eval_regression(total_pred, total_true_label)
        self.logger.info('M: >> ' + dict_to_str(eval_results))
        eval_results['Loss'] = round(eval_loss, 4)
        return eval_results
