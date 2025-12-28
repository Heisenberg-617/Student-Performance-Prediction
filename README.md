# Student Performance Prediction

## Project Overview

This project predicts student performance using machine learning. By analyzing historical student data, the model estimates expected grades or performance scores. This can help educators identify students needing additional support and optimize teaching strategies.

The notebook leverages **XGBoost regression** to model the relationship between student features (demographic, behavioral, or academic) and performance outcomes. The workflow includes data preparation, model training, evaluation, cross-validation, and hyperparameter tuning.

## Dataset

- The dataset contains various student features (numerical and categorical) and the target performance measure.  
- Missing values should be handled appropriately before model training.  
- The dataset is assumed to be in CSV format.

## Project Workflow

1. **Import Libraries**  
   - Libraries: `numpy`, `pandas`, `xgboost`, `scikit-learn`  

2. **Load Dataset**  
   - Split data into features (`X`) and target (`y`).  

3. **Split Data**  
   - Training and testing sets (80/20 split) using `train_test_split`.  

4. **Train Model**  
   - Train `XGBRegressor` with initial hyperparameters.  

5. **Evaluate Model**  
   - Metrics:  
     - **R² Score** – proportion of variance explained by the model  
     - **Mean Absolute Error (MAE)** – average absolute difference between predictions and true values  
     - **Mean Squared Error (MSE)** – average squared difference between predictions and true values  

6. **Cross-Validation**  
   - 10-fold cross-validation to check generalization  
   - Reports mean MSE across folds  

7. **Hyperparameter Tuning**  
   - Uses `GridSearchCV` to optimize key XGBoost parameters:  
     - `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `learning_rate`, `n_estimators`  

## Usage

1. Place the dataset in the notebook path or update the file path.  
2. Run the notebook cells sequentially:  
   - Library imports  
   - Data loading and splitting  
   - Model training and evaluation  
   - Cross-validation and hyperparameter tuning  
3. Review the printed outputs for metrics and best hyperparameters.  

## Key Libraries

- `numpy` – numerical computations  
- `pandas` – data manipulation  
- `xgboost` – gradient boosting regression  
- `scikit-learn` – model evaluation, cross-validation, and hyperparameter tuning  

## Results

- Achieves a reasonably high **R² score** and low error metrics, indicating effective performance prediction.  
- Cross-validation confirms model generalization.  
- Hyperparameter tuning can further improve accuracy depending on dataset specifics.  
