import joblib
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



def save_models(tabnet_model, meta_learner_model, filename="combined_model.joblib"):
    models = {'tabnet': tabnet_model, 'meta_learner': meta_learner_model}
    joblib.dump(models, filename)



def load_models_and_predict(filename, excel_path):
    models = joblib.load(filename)
    tabnet_model = models['tabnet']
    meta_learner_model = models['meta_learner']

    data = pd.read_excel(excel_path)
    X = data.iloc[:, 2:].values
    y = data.iloc[:, 1].values - 1 

    X_meta = tabnet_model.predict_proba(X)

    preds = meta_learner_model.predict(X_meta)

    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds, average='macro', zero_division=1)
    recall = recall_score(y, preds, average='macro')
    f1 = f1_score(y, preds, average='macro')

    print(f"Accuracy: {accuracy:}")
    print(f"Precision: {precision:}")
    print(f"Recall: {recall:}")
    print(f"F1 Score: {f1:}")

    return preds


predictions = load_models_and_predict("path_to_save_model.joblib", r"")

data = pd.read_excel(r"")
results_df = pd.DataFrame(data.iloc[:, :2]) 
results_df['Predicted'] = predictions+1 

results_path = r""
results_df.to_excel(results_path, index=False)

print(f"Full dataset predictions saved to {results_path}")

