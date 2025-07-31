# Titanic Survival Prediction – LightGBM Model

This project uses the Titanic dataset to build a machine learning model with LightGBM to predict passenger survival.

## 🔍 Overview

We perform:
- Data preprocessing
- Feature engineering (like Title, FamilySize)
- Training with LightGBM
- Evaluation using validation accuracy

## 📁 Project Structure

```
titanic-lightgbm/
│
├── README.md                <- Project description
├── titanic_model.py         <- Cleaned Python code
├── requirements.txt         <- Required libraries
```

## 📦 Libraries Used

- pandas
- lightgbm
- scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Output

Validation accuracy from train/validation split is around 71%.

## 🚀 Future Improvements

- Add new features like `IsAlone`, `HasCabin`
- Try ensembling with XGBoost, CatBoost
- Tune hyperparameters
