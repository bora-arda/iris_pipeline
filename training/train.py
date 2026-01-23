from torch import nn
import torch
from pathlib import Path
from training.model import IrisModel
from training.dataset import IrisDataset
import logging
import os
from utils import load_settings, setup_logger
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
        
        self.hidden_dim = train_cfg['hidden_dim']
        self.batch_size = train_cfg['batch_size']
        self.epochs = train_cfg['epochs']
        self.learning_rate = train_cfg['learning_rate']
        self.dropout = train_cfg['dropout']
        
        # Dataset
        self.dataset = IrisDataset(data_path)
        
        num_classes = len(set(self.dataset.y))
        
        # Model
        self.model = IrisModel(
            input_dim = self.dataset.num_features,
            hidden_dim = self.hidden_dim,
            num_classes = num_classes,
            dropout = self.dropout
            )
        
        # Dataloader
        self.data_loader = self.dataset.create_dataloader(batch_size = self.batch_size)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        
        logger.info(f"Running model on {self.device}")
        
    def train(self):
        average_loss = 0.0
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            train_loss = 0.0
            
            for X, y in self.data_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                logits = self.model(X)
                loss = self.criterion(logits, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss = loss.item()
                
            train_loss = total_loss / len(self.data_loader)
            logger.info(f"Epoch [{epoch + 1} / {self.epochs}] - Loss: {train_loss:.4f}")
            average_loss += train_loss

        self.save_model()
        logger.info(f"New best model (avg loss={average_loss / self.epochs:.4f}) saved to {self.model_path} ")
            
    def save_model(self):
        save_dir = Path(self.model_dir)
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