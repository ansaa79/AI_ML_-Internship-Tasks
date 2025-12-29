#  Task 01: News Topic Classifier Using BERT

## Objective
The objective of this project is to fine-tune a BERT model to classify news headlines into topic categories using the AG News Dataset.  

---

##  Dataset
- **Dataset Name:** AG News Dataset  
- **Source:** Hugging Face Datasets  
- **Training Samples:** 120,000 samples
- **Test Samples:** 7,600 samples

### Categories
1. World  
2. Sports  
3. Business  
4. Sci/Tech  

For faster training on CPU, a subset of **5,000 training samples** and **1,000 test samples** was used.

---

##  Technologies Used
- Python 3.x  
- PyTorch  
- Hugging Face Transformers  
- Hugging Face Datasets  
- Scikit-learn  
- Pandas & NumPy  
- Matplotlib & Seaborn  

---

##  Model Architecture
- **Base Model:** bert-base-uncased  
- **Model Type:** Transformer-based BERT  
- **Total Parameters:** ~110 million  
- **Pre-training Data:** Trained on BookCorpus and English Wikipedia
- **Fine-Tuning:** Classification head added for 4 output classes  

---

##  Methodology

### 1️⃣ Data Loading
- Loaded AG News dataset using Hugging Face Datasets
- Created smaller subset (5k train, 1k test) for CPU training
- Analyzed distribution across categories (balanced dataset)

### 2️⃣ Text Preprocessing
- Used `BertTokenizer` from bert-base-uncased
- Tokenized text with max length of 128 tokens
- Applied padding and truncation
- Created attention masks for variable-length sequences

### 3️⃣ Model Fine-Tuning
- Loaded pre-trained BERT model
- Added classification layer for 4 output classes
- Training configuration:
  - Epochs: 3  
  - Training Batch Size: 16  
  - Evaluation Batch Size: 32  
  - Learning Rate: 5e-5  
  - Warmup Steps: 500  
  - Weight Decay: 0.01  

### 4️⃣ Training Process
- Used Hugging Face `Trainer` API
- Evaluated model after each epoch
- Saved best model based on F1-score
- Training time: ~12 minutes (CPU)

---

##  Evaluation & Results

### Overall Performance
| Metric            | Score |
|------             |------ | 
| Accuracy          | 91.2% |
| Weighted F1 Score | 0.9115|
| Training Loss     | 0.2847|

### Per-Class Performance
| Category | Precision | Recall | F1-Score |
|--------  |---------- |--------|--------- |
| World    | 0.88      | 0.91   | 0.89     |
| Sports   | 0.97      | 0.95   | 0.96     |
| Business | 0.89      | 0.88   | 0.89     |
| Sci/Tech | 0.91      | 0.90   | 0.90     |

### Confusion Matrix Insights
- Sports category achieved highest accuracy
- Some confusion between World and Business news
- Sci/Tech category is well separated
- Overall strong performance across all classes
---

## Key Insights
- BERT achieved strong performance with limited fine-tuning
- Sports news is easiest to classify due to distinct vocabulary
- World and Business categories sometimes overlap
- Model understands context beyond keywords

---

##  Sample Predictions
- **"Biden announces new economic policy for 2024"** → World  
- **"Lakers defeat Warriors in overtime thriller"** → Sports  
- **"Tesla stock surges after earnings report"** → Business  
- **"Scientists discover new planet"** → Sci/Tech  

---


##  Inference (Using Saved Model)
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

model = BertForSequenceClassification.from_pretrained('./news_classifier_bert_model')
tokenizer = BertTokenizer.from_pretrained('./news_classifier_bert_model')

text = "Your news headline here"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()

labels = ['World', 'Sports', 'Business', 'Sci/Tech']
print(f"Category: {labels[prediction]}")

```

## Future Work
 - Deploy with Streamlit/Gradio web interface
 - Add real-time news classification API
 - Expand to more news categories
 - Multi-language support
 - Fine-tune on domain-specific news (tech, finance, sports)

## Contact
**Intern Name:** Ansa Bint E Zia  
**Role:** AI/ML Engineering Intern at DevelopersHub Corporation

**GitHub:** https://github.com/ansaa79
**Email:**  ansabintezia72@gmail.com

**Date:** December 2025




