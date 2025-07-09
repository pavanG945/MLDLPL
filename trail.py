import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings("ignore")

def load_and_preprocess(filepath):
  df = pd.read_excel("DATATHON.xlsx")

  df["Team"] = df['Team'].replace({"KRR": "KKR", "RPG": "RPSG"})

  # If there's a Year column in the original data, use it
  # Otherwise, we'll add it in the main function based on the data structure

  # Create team dummies
  team_dummies = pd.get_dummies(df['Team'], prefix='Team')

  # Combine with original dataframe and drop the original Team column
  final = pd.concat([df, team_dummies], axis=1)
  final = final.drop(['Team'], axis="columns")

  # Feature engineering with safety checks for division by zero
  # Boundary percentage (fours and sixes) of total runs
  final['Boundary_Ratio'] = np.where(
    final['Total Runs Scored by Team'] > 0,
    (final['Number of Fours'] * 4 + final['Number of Sixes'] * 6) / final['Total Runs Scored by Team'],
    0
  )

  # Team form/momentum
  final['Win_Momentum'] = final['Total Matches Played'] * final['NRR']

  # Bowling effectiveness
  final['Bowling_Strike_Efficiency'] = np.where(
    final['Total Overs Bowled'] > 0,
    final['Total Wickets'] / final['Total Overs Bowled'],
    0
  )

  # Batting impact
  final['Batting_Impact'] = final['Total Runs Scored by Team']

  # Death overs scoring rate
  # final['Death_Overs_Impact'] = np.where(
  #     final['Total Overs Bowled'] > 0,
  #     (final['Total Runs Scored by Team'] / final['Total Overs Bowled']) * final['Number of Sixes'],
  #     0
  # )

  # # Powerplay performance indicator
  # final['Powerplay_Dominance'] = np.where(
  #     final['Total Balls Faced'] > 0,
  #     (final['Number of Fours'] / final['Total Balls Faced']) * 100,
  #     0
  # )

  # Bowling in pressure situations
  final['Bowling_Clutch'] = np.where(
    (final['Total Overs Bowled'] > 0) & (final['Bowling Economy Rate'] > 0),
    (final['Total Wickets'] / final['Total Overs Bowled']) * (1 / final['Bowling Economy Rate']),
    0
  )

  # Additional feature: Run rate
  # final['Run_Rate'] = np.where(
  #     final['Total Overs Bowled'] > 0,
  #     final['Total Runs Scored by Team'] / final['Total Overs Bowled'],
  #     0
  # )

  # Additional feature: Wicket-taking ability
  # final['Wicket_Taking_Rate'] = np.where(
  #     final['Total Balls Faced'] > 0,
  #     (final['Total Wickets'] / final['Total Balls Faced']) * 100,
  #     0
  # )

  # Performance trend features - these would capture year-to-year changes
  # Adding any trend features would require the Year column to be present

  # Data integrity check - remove any remaining NaN values
  final = final.fillna(0)

  # Save the processed dataframe for future reference


  

  # Drop original features that have been transformed into more meaningful metrics
  drop_cols = [
    'Number of Fours', 'Number of Sixes', "Total Runs Scored by Team",'Team_DCB', 'Team_GL', 'Team_PW', 'Team_RPSG',
    'Total Matches Played', 'NRR', "Total Overs Bowled","Average Bowling Strike Rate",
    "Batting Average", "Average Strike Rate", "Total Balls Faced","Total Runs Conceded","Total Runs Conceded","Total Maiden Overs",'Total Catches Taken'
  ]
  final.drop(columns=[col for col in drop_cols if col in final.columns], inplace=True)
  final.to_pickle("final_processed.pkl")


  

    



  return final

def get_models(random_state=42):
    """Create a dictionary of models with appropriate pipelines"""
    # Define class weights to handle imbalance (alternative to SMOTE)
    class_weight = {0: 1, 1: 5, 2: 4, 3: 3, 4: 3}

    models = {
        "Random Forest": ImbPipeline([
            ('smote', SMOTE(random_state=random_state)),
            ('clf', RandomForestClassifier(random_state=random_state, class_weight=class_weight))
        ]),
        "XGBoost": ImbPipeline([
            ('smote', SMOTE(random_state=random_state)),
            ('clf', XGBClassifier(eval_metric='mlogloss', random_state=random_state))
        ]),
        "Gradient Boosting": ImbPipeline([
            ('smote', SMOTE(random_state=random_state)),
            ('clf', GradientBoostingClassifier(random_state=random_state))
        ]),
        "Logistic Regression": ImbPipeline([
            ('smote', SMOTE(random_state=random_state)),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000, random_state=random_state, class_weight=class_weight))
        ]),
        "SVM": ImbPipeline([
            ('smote', SMOTE(random_state=random_state)),
            ('scaler', StandardScaler()),
            ('clf', SVC(probability=True, random_state=random_state, class_weight=class_weight))
        ])
    }

    return models

def get_param_grids():
    """Define parameter grids for GridSearchCV - focused for time efficiency"""
    return {
        "Random Forest": {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [5, 10, None],
            'clf__min_samples_split': [2, 5]
        },
        "XGBoost": {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [3, 6],
            'clf__learning_rate': [0.05, 0.1],
            'clf__subsample': [0.8, 1.0]
        },
        "Gradient Boosting": {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [3, 5],
            'clf__learning_rate': [0.05, 0.1]
        },
        "Logistic Regression": {
            'clf__C': [0.1, 1, 10],
            'clf__solver': ['liblinear'],
            'clf__penalty': ['l1', 'l2']
        },
        "SVM": {
            'clf__C': [1, 10],
            'clf__kernel': ['linear', 'rbf']
        }
    }

def evaluate_model(name, model, X_test, y_test=None):
    """Evaluate model performance or make predictions for 2025"""
    if y_test is not None:
        # Evaluation mode - calculate metrics against known results
        y_pred = model.predict(X_test)

        # If model provides probability estimates, get them for ROC AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
        else:
            y_prob = None

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Calculate ROC AUC if possible
        try:
            if y_prob is not None:
                auc = roc_auc_score(pd.get_dummies(y_test), y_prob, average='macro', multi_class='ovo')
            else:
                auc = np.nan
        except Exception:
            auc = np.nan

        # Print detailed results
        print(f"\n{name} Results:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")
        print(f"ROC-AUC (macro): {auc:.4f}")
        print("-" * 60)

        # Return dictionary of metrics for comparison
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": mcc,
            "auc": auc
        }
    else:
        # Prediction mode - for 2025 data with unknown results
        # Get predicted results (0-4 ranking)
        y_pred = model.predict(X_test)

        # Get probability estimates if available
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)

            # Create a dataframe with team predictions and probabilities
            results_df = pd.DataFrame({
                'Predicted_Result': y_pred
            })

            # Add probability columns for each class
            for i in range(y_prob.shape[1]):
                results_df[f'Prob_Class_{i}'] = y_prob[:, i]

            # Sort by predicted result (lower is better - 0 is champion, 4 is eliminated)
            results_df = results_df.sort_values('Predicted_Result')

            # Print detailed prediction results
            print(f"\n{name} Predictions for 2025:")
            print(results_df)

            # Identify top 4 teams based on model predictions
            top_4_mask = results_df['Predicted_Result'] <= 3  # Results 0-3 are top 4
            eliminated_mask = results_df['Predicted_Result'] > 3  # Result 4 is eliminated

            print("\nTop 4 Teams (Qualified):")
            print(results_df[top_4_mask])

            print("\nEliminated Teams:")
            print(results_df[eliminated_mask])

            print("-" * 60)

            return results_df

        else:
            # If no probabilities available, just return predictions
            results_df = pd.DataFrame({
                'Predicted_Result': y_pred
            })

            # Sort by predicted result
            results_df = results_df.sort_values('Predicted_Result')

            # Print results
            print(f"\n{name} Predictions for 2025:")
            print(results_df)

            # Identify top 4 teams
            top_4_mask = results_df['Predicted_Result'] <= 3
            eliminated_mask = results_df['Predicted_Result'] > 3

            print("\nTop 4 Teams (Qualified):")
            print(results_df[top_4_mask])

            print("\nEliminated Teams:")
            print(results_df[eliminated_mask])

            print("-" * 60)

            return results_df

def plot_feature_importance(model_name, model, feature_names):
    """Plot feature importance for tree-based models"""
    # Extract the classifier from the pipeline
    if hasattr(model, 'named_steps') and 'clf' in model.named_steps:
        clf = model.named_steps['clf']
    else:
        clf = model

    # Check if the model has feature_importances_ attribute
    if hasattr(clf, 'feature_importances_'):
        # Get feature importances
        importances = clf.feature_importances_

        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Plot the feature importances
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importances - {model_name}')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f"{model_name.replace(' ', '_').lower()}_feature_importance.png")
        plt.close()

def main():
    """Main execution function"""
    start_time = datetime.now()
    print(f"Starting model training at {start_time}")
    models = get_models()
    param_grids = get_param_grids()
    results = {}
    best_estimators = {}

    # Load and preprocess data
    print("Loading and preprocessing data...")
    final = load_and_preprocess('DATATHON.xlsx')

    # Add a year column if it doesn't exist based on variable team count per year
    if 'Year' not in final.columns:
        print("Year column not found. Creating based on the data...")

        # We need to determine how many teams per year
        # This information should be provided or detected from the data
        # Let's try to detect the pattern by looking at the data

        # Check if we have explicit information about which rows belong to which year
        # If not, we'll need to infer it

        # For this example, we'll implement a dynamic detection approach
        # We'll analyze the data to detect patterns that might indicate year boundaries

        # Each year contains team data with a set "Result" pattern
        # We expect each year to have multiple teams competing, with a distribution of results

        # Let's look at patterns in the data that might help us identify year boundaries
        result_sequences = []
        current_sequence = []
        years = []
        current_year = 2008  # Starting year

        # For each row in the dataset
        for i, row in final.iterrows():
            current_sequence.append(row['Result'])

            # Check if we've found a pattern indicating a year boundary
            # For example, after we've seen each Result value at least once
            if len(set(current_sequence)) >= 4:  # Assuming we have at least 4 different Result values
                # Check if the current sequence length is reasonable for a year
                if 8 <= len(current_sequence) <= 12:  # Allow for 8-12 teams per year
                    result_sequences.append(current_sequence)
                    years.extend([current_year] * len(current_sequence))
                    current_sequence = []
                    current_year += 1
                # If sequence is too long, we might need to split it
                elif len(current_sequence) > 12:
                    # Divide into reasonable chunks
                    chunk_size = len(current_sequence) // 2  # Simple approach - just divide in half
                    years.extend([current_year] * chunk_size)
                    years.extend([current_year + 1] * (len(current_sequence) - chunk_size))
                    current_sequence = []
                    current_year += 2

        # Add any remaining sequence
        if current_sequence:
            years.extend([current_year] * len(current_sequence))

        # If our detection logic didn't work well, fall back to fixed chunks
        if len(years) != len(final):
            print("Warning: Year detection failed. Falling back to fixed-size chunks.")
            years = []
            current_year = 2008
            i = 0

            # Ask user for team counts per year
            print("Please provide team counts for each year (separated by commas):")
            print("For example: '8,8,10,10,10,8,10' for 7 years with varying team counts")
            print("Press Enter to use default (10 teams per year)")

            user_input = input().strip()

            if user_input:
                # Parse user input for team counts
                try:
                    team_counts = [int(x) for x in user_input.split(',')]
                    for year, count in enumerate(team_counts, start=2008):
                        years.extend([year] * count)
                except ValueError:
                    print("Invalid input. Using default 10 teams per year.")
                    team_counts = [10] * ((len(final) + 9) // 10)  # Ceil division by 10
                    for year, count in enumerate(team_counts, start=2008):
                        if i + count <= len(final):
                            years.extend([year] * count)
                            i += count
                        else:
                            years.extend([year] * (len(final) - i))
                            break
            else:
                # Default: 10 teams per year
                team_counts = [10] * ((len(final) + 9) // 10)  # Ceil division by 10
                for year, count in enumerate(team_counts, start=2008):
                    if i + count <= len(final):
                        years.extend([year] * count)
                        i += count
                    else:
                        years.extend([year] * (len(final) - i))
                        break

        # Ensure we have exactly the right number of year labels
        if len(years) > len(final):
            years = years[:len(final)]
        elif len(years) < len(final):
            # Fill any missing values with the last year
            years.extend([years[-1]] * (len(final) - len(years)))

        final['Year'] = years
        print(f"Added Year column based on detected team counts per year")

    # Print year distribution to verify
    print("Year distribution:")
    print(final['Year'].value_counts().sort_index())

    # Separate features and target
    X = final.drop(columns=['Result', 'Year'])  # Remove Year from features
    y = final['Result']
    years = final['Year'].values  # Keep track of years

    # Print dataset info
    print(f"Dataset shape: {final.shape}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Target distribution: \n{y.value_counts(normalize=True)}")

    # Time-based split: train on earlier years, test on most recent year
    # For predicting 2025, we use all available data to train
    # If there's prediction data for 2025, we need to handle it differently

    # Check if we have prediction data for 2025
    prediction_data_file = 'Book13.xlsx'
    try:
        prediction_df = pd.read_excel(prediction_data_file)
        has_prediction_data = True
        print(f"Found prediction data for 2025 with {len(prediction_df)} teams")
    except FileNotFoundError:
        has_prediction_data = False
        print("No prediction data file found. Will use the most recent year as test data.")

    if has_prediction_data:
        # In this case, we use all available data for training
        X_train = X
        y_train = y

        # Preprocess prediction data the same way as training data
        # prediction_df["Team"] = prediction_df['Team'].replace({"KRR": "KKR", "RPG": "RPSG"})
        # team_dummies = pd.get_dummies(prediction_df['Team'], prefix='Team')
        # prediction_final = pd.concat([prediction_df, team_dummies], axis=1)
        # prediction_final = prediction_final.drop(['Team'], axis="columns")

        # Apply the same feature engineering to prediction data
        # This is simplified - in practice, you'd reuse the same preprocessing function
        # Ensure all needed columns exist in prediction data

        # Prepare prediction data - we need to align columns with training data
        # Get the list of feature columns from X
        train_columns = X.columns.tolist()

        # Create the prediction feature matrix with same columns as training data
        X_pred = pd.DataFrame(columns=train_columns)

        # Fill in the values where possible
        for col in train_columns:
            if col in prediction_final.columns:
                X_pred[col] = prediction_final[col]
            else:
                X_pred[col] = 0  # Default value for missing features

        # Now we have X_train, y_train for training and X_pred for prediction
        test_idx = []  # No test data in this scenario
    else:
        # Use the most recent year as test data
        test_years = [max(years)]

        # Create train/test indices based on years
        train_idx = final[~final['Year'].isin(test_years)].index
        test_idx = final[final['Year'].isin(test_years)].index

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print(f"Training on years: {sorted(set(years) - set(test_years))}")
        print(f"Testing on years: {test_years}")
        print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    #models  = get_models()
    # Train, tune and evaluate each model
    for name, model in models.items():
        print(f"\n--- Tuning and Training {name} ---")

        if not has_prediction_data:
            # Normal training and testing scenario
            # Define time-series cross-validation to respect the chronological structure
            tscv = TimeSeriesSplit(n_splits=min(5, len(set(years)) - 1))  # n_splits cannot exceed number of years - 1

            # Define grid search with time series cross-validation
            #param_girds=get_param_grids()
            grid = GridSearchCV(
                model, param_grids[name], cv=tscv,
                scoring='f1_weighted', n_jobs=-1, verbose=1
            )

            # Fit grid search to the data
            grid.fit(X_train, y_train)

            # Print grid search results
            print(f"Best params for {name}: {grid.best_params_}")
            print(f"Best CV score: {grid.best_score_:.4f}")

            # Evaluate on test set (most recent year)
            results[name] = evaluate_model(name, grid.best_estimator_, X_test, y_test)

            # Save the model
            best_model_file = f"{name.replace(' ', '_').lower()}_model.joblib"
            joblib.dump(grid.best_estimator_, best_model_file)
            print(f"Model saved as {best_model_file}")

            # Store the best estimator
            best_estimators[name] = grid.best_estimator_

            # Plot feature importance for tree-based models
            if name in ["Random Forest", "XGBoost", "Gradient Boosting"]:
                plot_feature_importance(name, grid.best_estimator_, X.columns)
        else:
            # 2025 prediction scenario - use all data for training
            # We'll still do cross-validation for hyperparameter tuning
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            grid = GridSearchCV(
                model, param_grids[name], cv=cv,
                scoring='f1_weighted', n_jobs=-1, verbose=1
            )

            # Fit grid search to all available data
            grid.fit(X_train, y_train)

            # Print grid search results
            print(f"Best params for {name}: {grid.best_params_}")
            print(f"Best CV score: {grid.best_score_:.4f}")

            # Make predictions for 2025
            prediction_results = evaluate_model(name, grid.best_estimator_, X_pred)

            # Store predictions
            prediction_results.to_csv(f"{name.replace(' ', '_').lower()}_2025_predictions.csv")

            # Save the model
            best_model_file = f"{name.replace(' ', '_').lower()}_model.joblib"
            joblib.dump(grid.best_estimator_, best_model_file)
            print(f"Model saved as {best_model_file}")

            # Store the best estimator
            best_estimators[name] = grid.best_estimator_

            # Plot feature importance for tree-based models
            if name in ["Random Forest", "XGBoost", "Gradient Boosting"]:
                plot_feature_importance(name, grid.best_estimator_, X.columns)

    # Compare models and choose best (only for non-prediction scenario)
    if not has_prediction_data:
        print("\n=== Model Comparison ===")
        comparison_df = pd.DataFrame()

        for name, metrics in results.items():
            print(f"{name}: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, MCC={metrics['mcc']:.4f}, "
                f"AUC={metrics['auc']:.4f}")
            comparison_df[name] = pd.Series(metrics)

        # Save comparison results
        comparison_df.to_csv("model_comparison.csv")

        # Plot comparison results
        plt.figure(figsize=(12, 8))
        comparison_df.T.plot(kind='bar', figsize=(12, 8))
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("model_comparison.png")
        plt.close()

        # Find the best model based on F1 score
        best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
        best_estimator = best_estimators[best_model_name]

        print(f"\nBest Model: {best_model_name} (F1 Score: {results[best_model_name]['f1']:.4f})")

        # Save the best model to a pickle file
        with open("best_model.pkl", "wb") as f:
            pickle.dump(best_estimator, f)
        print(f"Best model saved as best_model.pkl")
    else:
        # For prediction scenario, create an ensemble prediction
        print("\n=== Creating Ensemble Prediction for 2025 ===")

        # Load all prediction CSVs
        ensemble_predictions = []

        for name in models.keys():
            pred_file = f"{name.replace(' ', '_').lower()}_2025_predictions.csv"
            try:
                pred_df = pd.read_csv(pred_file, index_col=0)
                ensemble_predictions.append(pred_df[['Predicted_Result']])
            except FileNotFoundError:
                print(f"Warning: Could not find predictions for {name}")

        if ensemble_predictions:
            # Combine all predictions
            all_preds = pd.concat(ensemble_predictions, axis=1)

            # Create ensemble prediction (average of all model predictions)
            ensemble_df = pd.DataFrame({
                'Ensemble_Result': all_preds.mean(axis=1)
            })

            # Sort by predicted result
            ensemble_df = ensemble_df.sort_values('Ensemble_Result')

            # Save ensemble prediction
            ensemble_df.to_csv("ensemble_2025_prediction.csv")

            # Print results
            print("\nEnsemble Predictions for 2025:")
            print(ensemble_df)

            # Identify top 4 teams based on ensemble predictions
            # Lower values (0-3) indicate higher ranks
            top_4_mask = ensemble_df['Ensemble_Result'] <= 3.5  # Using 3.5 as threshold
            eliminated_mask = ensemble_df['Ensemble_Result'] > 3.5

            print("\nTop 4 Teams (Qualified):")
            print(ensemble_df[top_4_mask])

            print("\nEliminated Teams:")
            print(ensemble_df[eliminated_mask])

            print(f"\nEnsemble prediction saved as ensemble_2025_prediction.csv")

    # Print training duration
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Training completed in {duration}")

if __name__ == "__main__":
    main()