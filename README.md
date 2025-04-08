# RoBERTa-Based Multi-Label Wellbeing Classifier (Training + Prediction Pipeline)

This repository contains the training and prediction pipeline for a **multi-label RoBERTa-based text classifier** tailored for wellbeing-related qualitative responses. The model learns to classify open-text responses into multiple relevant wellbeing subcategories and map them to broader psychological domains.

Built for use in organizational wellbeing diagnostics, mental health research, and real-world NLP pipelines that handle complex human data.

---

## 🧠 What This Project Does

This system classifies raw text (e.g., "I’m stressed and burned out from work and struggling to sleep") into **multiple overlapping wellbeing categories** like:

- `burnout`
- `sleep_issues`
- `job_dissatisfaction`

These subcategories are also mapped to broader **psychological domains** such as:

- `life_demands`
- `personal_demands`
- `wellbeing`
- `life_resources`
- `personal_resources`

The classifier supports large-scale survey analysis, post-hoc content tagging, and transformation of free-text into structured data for dashboards or SEM models.

---

## 🔧 Project Structure

- `TRAINING_ROBERTA_MULTILABEL_FIXED.py`: Main script for training and prediction
- `train.xlsx`: Input file with manually  coded categories in multiple label columns (`label`, `label_2`, ..., `label_10`)
- `MULTI_LABEL_PREDICTION/FINAL_MODEL/`: Folder containing the trained model, tokenizer, encoders, and config
- `predictions_on_train.xlsx`: Final predicted subcategories and domains for each input
- `sample_text_predictions`: Dummy prediction examples using live input

---

## 📦 Dependencies

Install the required packages using:

```bash
pip install pandas scikit-learn transformers datasets openpyxl torch joblib numpy
```

---

## 🚀 How to Train the Model

### 📄 Input Format

The `train.xlsx` file should contain:
- A `text` column with raw qualitative responses
- One or more `label_*` columns containing true subcategory labels (multi-label format)

### 🧪 Run Training

```bash
python train_predict.py
```

This will:
- Collect all `label*` columns into a multi-label vector
- Apply smart upsampling on underrepresented subcategories
- Compute class-balanced `BCEWithLogitsLoss` with optional label smoothing
- Train a RoBERTa model for 25 epochs with early stopping and cosine LR
- Evaluate using micro-averaged precision, recall, F1, and exact match
- Save the full model and predicted results

---

### 🧾 Output Files

After training, artifacts are saved in:

```bash
MULTI_LABEL_PREDICTION/FINAL_MODEL/
├── model + tokenizer
├── mlb.pkl                        # MultiLabelBinarizer
├── label_map.json                 # Subcategory ↔ ID mapping
├── sentiment_encoder.pkl          # Optional encoder for sentiment
├── predictions_on_train.xlsx      # Subcategory/domain predictions
└── subcat_classes.json
```

---

## 🔍 Prediction Capabilities

### 🧪 Predict on Full Dataset

The script automatically runs prediction on the entire `train.xlsx` set and saves results with:

- `pred_subcat_*` columns (each predicted subcategory)
- `predicted_domains` column (mapped high-level wellbeing domains)
- `multi_label_*` columns (true labels)
- Overall micro-F1, precision, recall, exact match

### ✨ Predict on New Texts

```python
predict_texts([
    "I feel anxious about losing my job.",
    "My team helps me grow. I'm grateful.",
    "I'm so burned out from work."
], model, tokenizer, mlb, threshold=0.5)
```

Outputs:
- Predicted subcategories
- Predicted wellbeing domains

---

## ⚙️ Adjusting Sensitivity

Prediction confidence threshold (default = 0.3) can be adjusted:

```python
test_predictions_on_csv(..., threshold=0.3)
```

- `0.2` favors **recall**
- `0.3` is a **balanced default**
- `0.5` is strict (fewer predictions)

---

## 📚 Label Schema and Mapping

Each predicted subcategory is mapped to a broader domain:

```python
"burnout"            → "personal_demands"
"gratitude"          → "life_resources"
"self-awareness"     → "personal_resources"
"autonomy"           → "wellbeing"
```

See the full domain map in the script under `subcategory_to_domain`.

---

## 👥 Use Cases

- Classify wellbeing and mental health text from surveys, social media, or interviews
- Converting qualitative responses into structured data
- Supporting sentiment analysis and more advanced analytics
- Building NLP dashboards for psychological analytics

---

## 📌 Notes

- This is a **multi-label** classifier using `BCEWithLogitsLoss`
- Resampling strategies help mitigate subcategory imbalance
- Includes built-in support for class weights, label smoothing, and early stopping
- Optional prediction scripts available for new data and dummy examples

---

## 👤 Author

This project was developed by an [Prof. Llewellyn van Zyl (PhD)](https://www.linkedin.com/in/llewellynvz), an organizational psychologist and data scientist specializing in human wellbeing, applied NLP, and data-sciences.

---

## 📬 Contact

💬 **[Open an Issue](https://github.com/llewellynvz/roberta_training_wellbeing/issues)** on GitHub  
✉️ **Contact me via my website** [psynalytics.com](https://www.psynalytics.com)

---

Feel free to fork, adapt, or collaborate to improve wellbeing classification pipelines.

---
## 🛡 License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute it with attribution.  
© 2025 Prof. Llewellyn van Zyl. See [LICENSE](./LICENSE) for details.

