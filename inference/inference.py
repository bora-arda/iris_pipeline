import torch
import time
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix
from core.model import IrisModel
from core.dataset import IrisDataset
import logging
import os
from core.utils import load_settings, setup_logger, is_file_exist, is_dir_exists

setup_logger()
logger = logging.getLogger(__name__)

class IrisInference:
    def __init__(self, settings_path = "settings.json"):
        logger.info("Initializing inference pipeline")
        
        # Load settings
        self.settings = load_settings(settings_path)
        train_cfg = self.settings['train']
        general_cfg = self.settings['general']
        inference_cfg = self.settings['inference']
        
        # Dataset
        data_dir = general_cfg['processed_data_dir']
        data_file = inference_cfg['data_table']
        data_path = os.path.join(data_dir, data_file)
        self.batch_size = train_cfg["batch_size"]
        
        data_path = is_file_exist(data_path) # Check if file exist
        df = pd.read_csv(data_path)
        if len(df) == 0:
            raise ValueError("Inference dataset is empty.")
        
        self.validate_inference_dataframe(df) # Check if dataframe valid
        
        X = df.drop(columns=['target']).values
        y = df["target"].values
        
        self.inference_dataset = IrisDataset(X, y)
        if self.inference_dataset.num_features == 0:
            raise ValueError("Dataset has zero features.")
        
        self.inference_dataloader = self.inference_dataset.create_dataloader(batch_size = self.batch_size)
        num_classes = len(set(y))
        
        # Predictions table path
        results_dir = general_cfg['predictions_dir']
        results_table = inference_cfg['pred_table']
        self.results_path = os.path.join(results_dir, results_table)
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Model
        self.model_dir = general_cfg['models_dir']
        model_name = train_cfg['model_name']
        self.model_path = os.path.join(self.model_dir, model_name)
        if not is_file_exist(self.model_path):
            raise FileNotFoundError(f"File not found: {self.model_path}")
        
        self.hidden_dim = train_cfg['hidden_dim']
        self.dropout = train_cfg['dropout']
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Running model on {self.device}")
        
        self.model = IrisModel(
            input_dim = self.inference_dataset.num_features,
            hidden_dim = self.hidden_dim,
            num_classes = num_classes,
            dropout = self.dropout
            )
        
        self.model.load_state_dict(torch.load(self.model_path, map_location = self.device))
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded model from: {self.model_path}")
        
    def run(self):
        start_time = time.time()
        logger.info("Running inference")
        
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for X, y_true in self.inference_dataloader:
                X = X.to(self.device)
                y_true = y_true.to(self.device)
                
                logits = self.model(X)
                
                probs = torch.softmax(logits, dim = 1)
                preds = probs.argmax(dim = 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(y_true.cpu().numpy())
        
        acc = accuracy_score(all_targets, all_preds)
        cm = confusion_matrix(all_targets, all_preds)

        logger.info(f"Inference Accuracy: {acc:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        
                
        results_df = pd.DataFrame(all_probs, columns=['prob_setosa', 'prob_versicolor', 'prob_virginica'])
        results_df["prediction"] = all_preds
        results_df["target"] = all_targets
        
        results_df.to_csv(self.results_path, index = False)
        
        logger.info(
            f"Predictions saved to {self.results_path} "
            f"({len(results_df)} rows)"
        )
        elapsed = time.time() - start_time
        logger.info(f"Inference completed in {elapsed:.2f} seconds")
        
    def validate_inference_dataframe(self, df: pd.DataFrame):
        """
        Validate inference dataframe structure and content.
        Raises ValueError if invalid.
        """
        if df.empty:
            raise ValueError("Inference dataset is empty.")

        if "target" not in df.columns:
            raise ValueError("Inference dataset must contain 'target' column.")

        feature_columns = [c for c in df.columns if c != "target"]
        if len(feature_columns) == 0:
            raise ValueError("Inference dataset must contain at least one feature column.")

        if df.isna().any().any():
            raise ValueError("Inference dataset contains NaN values.")
            
def main():
    try:
        logger.info("Inference started")
        inference = IrisInference()
        inference.run()
        logger.info("Inference completed successfully")
    except Exception as e:
        logger.exception(f"Inference failed due to:\n{e}")
        
if __name__ == "__main__":
    main()