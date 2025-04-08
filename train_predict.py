################################################################################
# TRAINING_ROBERTA_MULTILABEL_FIXED.py
#
# This script trains a multi-label RoBERTa classifier using BCEWithLogitsLoss.
# 
################################################################################

import logging
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from datasets import Dataset, DatasetDict
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig
)
from transformers.trainer_callback import EarlyStoppingCallback

#from pytorch.tools.testing.test_selections import THRESHOLD

################################################################################
# 0. CONFIGURE LOGGING
################################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

################################################################################
# 1. LOAD DATA & DOMAIN MAPPING
################################################################################

logger.info("Loading dataset from 'train.xlsx'...")
CSV_PATH = "train.xlsx"
df = pd.read_excel(CSV_PATH, engine="openpyxl").dropna(subset=["text"])
value_threshold = 0.30

# Domain mapping from your single-label script:
subcategory_to_domain = {
    # life_demands
    "acculturation_stress": "life_demands",
    "aging": "life_demands",
    "bullying": "life_demands",
    # .....
    #"building_positive_relationships": "wellbeing",
    "work-life_balance": "life_resources",
}


################################################################################
# 2. MULTI-LABEL BINARIZATION + SMART UPSAMPLING
################################################################################

# Identify all label columns: label, label_2, etc.
label_cols = [col for col in df.columns if col.startswith("label")]

def gather_subcategories(row):
    """Collect subcategory names from label columns into a list."""
    subcats = []
    for c in label_cols:
        val = row.get(c, None)
        if pd.notna(val):
            subcats.append(str(val).strip())
    return list(set(subcats))

df["multi_labels"] = df.apply(gather_subcategories, axis=1)

logger.info("Binarizing labels into multi-hot vectors...")
mlb = MultiLabelBinarizer()
multi_hot = mlb.fit_transform(df["multi_labels"])
subcat_classes = mlb.classes_
logger.info(f"Detected {len(subcat_classes)} unique subcategories.")

# IMPORTANT FIX: Instead of adding columns one by one, let's do one concat:
multi_hot_df = pd.DataFrame(multi_hot, columns=subcat_classes, index=df.index)

# Convert to float to avoid BCEWithLogitsLoss complaining about int -> float
multi_hot_df = multi_hot_df.astype("float32")

# Merge back into original df
df = pd.concat([df, multi_hot_df], axis=1)

################################################################################
# 2.1 CALCULATE POSITIVE CLASS WEIGHTS FOR BCE LOSS
################################################################################

logger.info("Calculating class weights for BCEWithLogitsLoss (pos_weight)...")
pos_weight = []
for sc in subcat_classes:
    positives = df[sc].sum()
    negatives = len(df) - positives
    weight = negatives / positives if positives > 0 else 1.0
    pos_weight.append(weight)

pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
logger.info("Completed pos_weight calculation.")


################################################################################
# 2.3 CUSTOM TRAINER WITH LABEL SMOOTHING + WEIGHTED LOSS
################################################################################
class SmoothedTrainer(Trainer):
    def __init__(self, *args, pos_weight_tensor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight_tensor = pos_weight_tensor

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Label smoothing
        epsilon = getattr(self.args, "label_smoothing_factor", 0.05)
        labels = labels * (1 - epsilon) + (epsilon / 2)

        # Weighted BCE loss
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss



################################################################################
# SMART UPSAMPLING
################################################################################

def smart_resample_multilabel(input_df, subcats, bottom_percentile=50, multiplier=30, seed=42):
    """
    For each subcategory in the bottom percentile count,
    upsample the rows containing that subcategory.
    """
    logger.info(f"Applying multi-label upsampling: bottom {bottom_percentile}% with multiplier {multiplier}...")
    df_out = input_df.copy()

    # Count frequency for each subcat
    subcat_counts = {}
    for sc in subcats:
        subcat_counts[sc] = df_out[sc].sum()  # sum of float values => freq of '1's



    counts_series = pd.Series(subcat_counts).sort_values()
    threshold = np.percentile(counts_series, bottom_percentile)
    logger.info(f"Upsample threshold = {threshold:.2f} (bottom {bottom_percentile}%)")

    underrepresented = counts_series[counts_series <= threshold].index.tolist()
    logger.info(f"Number of underrepresented subcats: {len(underrepresented)}")

    for sc in underrepresented:
        current_count = subcat_counts[sc]
        if current_count == 0:
            continue
        df_subset = df_out[df_out[sc] == 1.0]
        target_count = int(multiplier * threshold)
        if len(df_subset) < target_count:
            df_upsampled = resample(
                df_subset,
                replace=True,
                n_samples=(target_count - len(df_subset)),
                random_state=seed
            )
            df_out = pd.concat([df_out, df_upsampled], axis=0)

    for sc in subcat_classes:
        logger.info(f"{sc}: {df[sc].sum()} examples post-upsampling")

    df_out = df_out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    logger.info(f"After upsampling, dataset size is {len(df_out)} rows.")
    return df_out

df = smart_resample_multilabel(df, subcat_classes, bottom_percentile=30, multiplier=6)

################################################################################
# 3. TRAIN/VAL/TEST SPLIT
################################################################################

logger.info("Splitting dataset into train/val/test...")

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

logger.info(f"Train size = {len(train_df)} | Val size = {len(val_df)} | Test size = {len(test_df)}")

def to_hf_dict(row):
    """
    Convert each row into a dict with:
      - text
      - labels (the float multi-hot vector)
    """
    # Make a float list for each subcat
    label_vector = [row[sc] for sc in subcat_classes]
    return {"text": str(row["text"]), "labels": label_vector}

train_dicts = train_df.apply(to_hf_dict, axis=1).to_list()
val_dicts   = val_df.apply(to_hf_dict, axis=1).to_list()
test_dicts  = test_df.apply(to_hf_dict, axis=1).to_list()

train_dataset = Dataset.from_list(train_dicts)
val_dataset   = Dataset.from_list(val_dicts)
test_dataset  = Dataset.from_list(test_dicts)

hf_dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

################################################################################
# 4. TOKENIZATION (FINAL VERSION)
################################################################################

logger.info("Tokenizing dataset with RoBERTa tokenizer...")

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

# Apply tokenizer to all splits
hf_dataset = hf_dataset.map(tokenize_function, batched=True)

# CRITICAL: Remove "text" field after tokenization
hf_dataset = hf_dataset.remove_columns(["text"])

# CRITICAL: Format everything as PyTorch tensors
# Note: “labels” must already be a list of floats for BCEWithLogitsLoss
hf_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)


################################################################################
# 5. CONFIGURE ROBERTA FOR MULTI-LABEL
################################################################################

logger.info("Configuring RobertaForSequenceClassification for multi-label classification...")

id2label = dict(enumerate(subcat_classes))
label2id = {v: k for k, v in id2label.items()}

config = AutoConfig.from_pretrained(
    "roberta-base",
    num_labels=len(subcat_classes),
    id2label=id2label,
    label2id=label2id,
    problem_type="multi_label_classification",  # => uses BCEWithLogitsLoss internally
    hidden_dropout_prob=0.3,    # Dropout to reduce overfitting
    attention_probs_dropout_prob=0.3
)

model = RobertaForSequenceClassification.from_pretrained("roberta-base", config=config)

################################################################################
# 6. DEFINE METRICS FOR MULTI-LABEL
################################################################################

def compute_multilabel_metrics(eval_preds):
    """
    Using Micro-averaged Precision, Recall, and F1 as well as an exact match ratio.
    """
    logits, labels = eval_preds
    # logits -> shape: [batch_size, num_labels]
    # labels -> shape: [batch_size, num_labels], float32

    preds_prob = torch.sigmoid(torch.tensor(logits))
    preds = (preds_prob > value_threshold).int().numpy()  # convert prob > 0.5 to 1

    labels = labels.astype(int)  # or keep as float then cast to int
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )
    exact_matches = np.all(preds == labels, axis=1).mean()

    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "exact_match_ratio": exact_matches
    }

################################################################################
# 7. TRAINING ARGUMENTS & TRAINER
################################################################################

OUTPUT_DIR = "MULTI_LABEL_PREDICTION"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=25,
    per_device_train_batch_size=25,
    per_device_eval_batch_size=25,
    #eval_steps=200,  # or your preference
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    #logging_steps=200,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    learning_rate=2e-5,
    weight_decay=0.01,
    #label_smoothing_factor=0.0,
    warmup_ratio=0.01,
    save_total_limit=10,
    lr_scheduler_type="cosine",
    fp16=True,  # optional for faster training

)

trainer = SmoothedTrainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset["train"],
    eval_dataset=hf_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_multilabel_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    pos_weight_tensor = pos_weight_tensor
)

################################################################################
# 8. TRAIN
################################################################################

logger.info("Starting model training...")
trainer.train()

################################################################################
# 9. EVALUATE ON VAL AND TEST
################################################################################

logger.info("Evaluating on validation set...")
val_metrics = trainer.evaluate(hf_dataset["validation"])
logger.info(f"Validation metrics: {val_metrics}")

logger.info("Evaluating on test set...")
test_metrics = trainer.evaluate(hf_dataset["test"])
logger.info(f"Test metrics: {test_metrics}")

################################################################################
# 10. SAVE MODEL, TOKENIZER, AND MULTI-LABEL BINARIZER
################################################################################

logger.info("Saving final model and tokenizer...")
final_model_path = os.path.join(OUTPUT_DIR, "FINAL_MODEL")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)


logger.info("Saving MultiLabelBinarizer...")
joblib.dump(mlb, os.path.join(final_model_path, "mlb.pkl"))

with open(os.path.join(final_model_path, "subcat_classes.json"), "w") as f:
    json.dump(list(subcat_classes), f)

# Save subcategory label ↔ ID mapping
label2id = {label: i for i, label in enumerate(mlb.classes_)}
id2label = {i: label for label, i in label2id.items()}
label_map = {"label2id": label2id, "id2label": id2label}

with open(os.path.join(final_model_path, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=4)
logger.info("Saved subcategory label2id/id2label mapping to label_map.json")


# Create and save single-label encoder (for sentiment or domain tasks)
sentiments = ["positive", "neutral", "negative"]  
sentiment_encoder = LabelEncoder()
sentiment_encoder.fit(sentiments)

joblib.dump(sentiment_encoder, os.path.join(final_model_path, "sentiment_encoder.pkl"))

# Also save a JSON map for clarity
sentiment_map = {
    "label2id": {label: int(sentiment_encoder.transform([label])[0]) for label in sentiments},
    "id2label": {int(sentiment_encoder.transform([label])[0]): label for label in sentiments}
}
with open(os.path.join(final_model_path, "sentiment_map.json"), "w") as f:
    json.dump(sentiment_map, f, indent=4)

logger.info("Saved sentiment encoder and mapping")


logger.info("All model artifacts saved!")

################################################################################
# 11. TEST PREDICTIONS ON FULL TRAIN.XLSX
################################################################################

def test_predictions_on_csv(
    csv_path,
    model,
    tokenizer,
    mlb,
    threshold=value_threshold,
    output_excel="predictions_multilabel.xlsx",
    batch_size=16
):
    """
    Predict on all rows of a CSV/XLSX file, batch-wise to avoid OOM, then write subcategory + domain predictions.

    For ThresholdsL
    0.2 → maximizes recall
    0.3 → balances better
    0.5 → too strict right now
    """

    logger.info(f"Loading test file: {csv_path} ...")
    df_test = pd.read_excel(csv_path, engine="openpyxl").dropna(subset=["text"])

    # Recreate true multi-hot if needed
    df_test["multi_labels"] = df_test.apply(gather_subcategories, axis=1)
    true_multi_hot = mlb.transform(df_test["multi_labels"])

    texts = df_test["text"].astype(str).tolist()
    total = len(texts)
    all_preds = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    logger.info(f"Running prediction in batches of {batch_size}...")
    for i in tqdm(range(0, total, batch_size)):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(int)
            all_preds.append(preds)

    # Final shape = (total_examples, num_classes)
    final_preds = np.vstack(all_preds)

    #DEBUGGING. IF IT PRINTS Mean = 0.9, Min = 0, Max = 3 then its too conservative
    label_counts = np.sum(final_preds, axis=1)
    logger.info(
        f"Predicted label count: Mean={label_counts.mean():.2f}, Min={label_counts.min()}, Max={label_counts.max()}")

    # Convert predictions into list of subcategory strings
    pred_subcats = []
    for row in final_preds:
        idxs = np.where(row == 1)[0]
        sc_list = [mlb.classes_[i] for i in idxs]
        pred_subcats.append(sc_list)
    df_test["predicted_subcategories"] = pred_subcats

    # 1) Expand predicted_subcategories into columns
    max_preds = df_test["predicted_subcategories"].apply(len).max()
    pred_cols = [f"pred_subcat_{i + 1}" for i in range(max_preds)]

    df_pred_subcats_expanded = df_test["predicted_subcategories"].apply(pd.Series)
    df_pred_subcats_expanded.columns = pred_cols

    # 2) Merge them back
    df_test = pd.concat([df_test.drop(columns=["predicted_subcategories"]), df_pred_subcats_expanded], axis=1)



    # Map predicted subcats to domains
    predicted_domains = []
    for sc_list in pred_subcats:
        domain_set = {subcategory_to_domain.get(sc, "unknown") for sc in sc_list}
        predicted_domains.append(list(domain_set))
    df_test["predicted_domains"] = predicted_domains

    # Expand true labels into separate columns for Excel-friendliness
    max_labels = df_test["multi_labels"].apply(len).max()
    multi_label_cols = [f"multi_label_{i+1}" for i in range(max_labels)]
    df_labels_expanded = df_test["multi_labels"].apply(pd.Series)
    df_labels_expanded.columns = multi_label_cols
    df_test = pd.concat([df_test.drop(columns=["multi_labels"]), df_labels_expanded], axis=1)

    # Make predicted subcats/domains readable (string, not lists)
    #df_test["predicted_subcategories"] = df_test["predicted_subcategories"].apply(lambda x: ", ".join(x))
    df_test["predicted_subcats_str"] = df_test[pred_cols].apply(lambda row: ", ".join(row.dropna().astype(str)), axis=1)
    df_test["predicted_domains"] = df_test["predicted_domains"].apply(lambda x: ", ".join(x))
    # Also store a single string column for reference


    # EVALUATE
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_multi_hot, final_preds, average="micro", zero_division=0
    )
    exact_match = np.all(true_multi_hot == final_preds, axis=1).mean()

    logger.info(f"Full data Micro-F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Exact Match: {exact_match:.4f}")

    # Save the files  to Excel
    df_test.to_excel(output_excel, index=False)
    logger.info(f"Saved detailed predictions to {output_excel}")

    label_counts = np.sum(final_preds, axis=1)
    logger.info(
        f"Predicted label count — mean: {label_counts.mean():.2f}, min: {label_counts.min()}, max: {label_counts.max()}")

    return df_test


logger.info("Running test_predictions_on_csv to check performance on the entire train.xlsx ...")
test_predictions_on_csv(
    csv_path=CSV_PATH,
    model=model,
    tokenizer=tokenizer,
    mlb=mlb,
    threshold=value_threshold,
    batch_size=16, 
    output_excel=os.path.join(OUTPUT_DIR, "predictions_on_train.xlsx")
)

################################################################################
# 12. DUMMY SCRIPT - TESTING SAMPLE TEXTS
################################################################################

def predict_texts(text_list, model, tokenizer, mlb, threshold=0.5):
    logger.info("Running sample text predictions...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    encodings = tokenizer(text_list, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()

    preds = (probs > threshold).astype(int)

    for text, row in zip(text_list, preds):
        sc_list = []
        dom_set = set()
        for i, val in enumerate(row):
            if val == 1:
                sc = mlb.classes_[i]
                sc_list.append(sc)
                dom_set.add(subcategory_to_domain.get(sc, "unknown"))

        logger.info(
            f"\nText: {text}\n"
            f"Predicted Subcategories: {sc_list}\n"
            f"Predicted Domains: {list(dom_set)}"
        )

# Quick demonstration
sample_texts = [
    "I feel anxious about losing my job, it's so stressful lately.",
    "My team helps me learn new skills. I'm grateful for them.",
    "I'm worried about finances and the cost of living. I'm so burned out."
]
predict_texts(sample_texts, model, tokenizer, mlb, threshold=0.5)

logger.info("All done. Your multi-label pipeline should now run smoothly.")





