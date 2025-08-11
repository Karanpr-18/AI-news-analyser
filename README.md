# 📰 News Intelligence Toolkit
- Try live here -> https://ai-news-analyser.streamlit.app/

Three NLP models:
- 🕵️‍♂️ Fake News Detection
- 🚫 Hate Speech Detection
- 🗂️ News Category Classification

## 🎯 Overall accuracy
- Fake News: 0.89  
- Hate Speech: 0.80  
- News Category: 0.89

## 📂 What’s inside
- app.py (run predictions)
- Notebooks: fakenews_detect.ipynb, hate_speach.ipynb, news_category.ipynb
- Models: fakenews_model.pkl, hatespeach_model.pkl, news_category_model.pkl
- Vectorizers: fakenews_vectorizer.pkl, hatespeach_vectorizer.pkl, news_category_vector.pkl
- Sample data: fakenews_detection.xlsx, HateSpeech_Dataset.xlsx, news_cat_data_1.xlsx, news_cat_data_2.xlsx


## Minimal requirements.txt:
- numpy
- pandas
- scikit-learn
- joblib
- jupyter
- streamlit


**Streamlit:**
- streamlit run app.py


## 📊 Detailed results

### 🕵️‍♂️ Fake News (binary 0/1)
- Accuracy: 0.89  
- Class 0 P/R/F1: 0.87 / 0.90 / 0.88  
- Class 1 P/R/F1: 0.91 / 0.87 / 0.89  
- Support: 10,417 / 11,212 (total 21,629)  
- Final printed score: 0.8860  

### 🚫 Hate Speech (binary 0/1)
- Accuracy: 0.80  
- Class 0 P/R/F1: 0.89 / 0.69 / 0.77  
- Class 1 P/R/F1: 0.75 / 0.91 / 0.82  
- Support: 108,278 / 109,558 (total 217,836)  
- Final printed score: 0.7993  

### 🗂️ News Category (4 classes: 1–4)
- Accuracy: 0.89  
- Class 1 P/R/F1: 0.90 / 0.89 / 0.89  
- Class 2 P/R/F1: 0.94 / 0.97 / 0.96  
- Class 3 P/R/F1: 0.87 / 0.83 / 0.85  
- Class 4 P/R/F1: 0.85 / 0.87 / 0.86  
- Support: 9,600 / 9,482 / 9,632 / 9,566 (total 38,280)  
- Final printed score: 0.8918  

## 🗃️ Data schema
- Fake news: [text, label] → {0,1} or {real,fake}  
- Hate speech: [text, label] → {0,1}  
- Category: [text, category] → string labels (1–4 in current set)  

## ⚠️ Notes
- Models are TF‑IDF + linear classifiers (see notebooks).  
- Load the matching vectorizer + model per task.  
- Review fairness and out‑of‑domain performance before production.  

## 📄 License
Add a LICENSE (MIT recommended).  

## 🤝 Contributing
PRs and issues welcome.
