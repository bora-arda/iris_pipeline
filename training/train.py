from torch import nn
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from core.model import IrisModel
from core.dataset import IrisDataset
import logging
import os
from core.utils import load_settings, setup_logger
setup_logger()
logger = logging.getLogger(__name__)

class IrisTrainer:
    def __init__(self, settings_path = "settings.json"):
        # Load settings
        self.settings = load_settings(settings_path)
        train_cfg = self.settings['train']
        general_cfg = self.settings['general']
        
        data_dir = general_cfg['processed_data_dir']
        data_file = train_cfg['data_table']
        data_path = os.path.join(data_dir, data_file)
        
        self.model_dir = general_cfg['models_dir']
        model_name = train_cfg['model_name']
        self.model_path = os.path.join(self.model_dir, model_name)
        test_size = train_cfg['test_size']
        random_state = general_cfg['random_state']
        
        self.hidden_dim = train_cfg['hidden_dim']
        self.batch_size = train_cfg['batch_size']
        self.epochs = train_cfg['epochs']
        self.learning_rate = train_cfg['learning_rate']
        self.dropout = train_cfg['dropout']
        
        # Dataset
        df = pd.read_csv(data_path)
        X = df.drop(columns=['target']).values
        y = df["target"].values
        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          test_size=test_size,
                                                          random_state=random_state,
                                                          stratify=y
                                                          )
        
        self.train_dataset = IrisDataset(X_train, y_train)
        self.val_dataset = IrisDataset(X_val, y_val)
        
        num_classes = len(set(y))
        
        # Model
        self.model = IrisModel(
            input_dim = self.train_dataset.num_features,
            hidden_dim = self.hidden_dim,
            num_classes = num_classes,
            dropout = self.dropout
            )
        
        # Dataloader
        self.train_dataloader = self.train_dataset.create_dataloader(batch_size = self.batch_size)
        self.val_dataloader = self.val_dataset.create_dataloader(batch_size=self.batch_size, shuffle = False)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        
        logger.info(f"Running model on {self.device}")
        
    def train(self):
        best_val_loss = float("inf")
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_samples = 0
            
            for X, y in self.train_dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                logits = self.model(X)
                loss = self.criterion(logits, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                batch_size = X.size(0)
                train_loss += loss.item() * batch_size
                
                preds = logits.argmax(dim = 1)
                train_correct += (preds == y).sum().item()
                train_samples += batch_size
                
            train_loss /= train_samples
            train_acc = train_correct / train_samples
            
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_samples = 0
            
            with torch.no_grad():
                for X, y in self.val_dataloader:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    
                    logits = self.model(X)
                    loss = self.criterion(logits, y)
                    
                    batch_size = X.size(0)
                    val_loss += loss.item() * batch_size
                    
                    preds = logits.argmax(dim = 1)
                    val_correct += (preds == y).sum().item()
                    val_samples += batch_size
                    
            val_loss /= val_samples
            val_acc = val_correct / val_samples
            
            logger.info(
                f"Epoch [{epoch + 1} / {self.epochs}] | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
            
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
                logger.info(
                f"New best model saved "
                f"(val_loss={best_val_loss:.4f})"
                )
            
    def save_model(self):
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
            
            
def main():
    try:
        logger.info("Training started.")
        trainer = IrisTrainer()
        trainer.train()
        logger.info("Training finished.")
    except Exception as e:
        logger.exception(f"Training failed due to:\n{e}")
        
if __name__ == "__main__":
    main()