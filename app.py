import pandas as pd
import numpy as np
import pickle
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

def preprocess_prediction_data(filepath):
    """
    Preprocess the 2025 prediction data with the same transformations as training data
    """
    # Load the new data for 2025
    df = pd.read_excel(filepath)
    
    # Fix team name inconsistencies if any
    df["Team"] = df['Team'].replace({"KRR": "KKR", "RPG": "RPSG"})
    
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
    
    # Bowling in pressure situations
    final['Bowling_Clutch'] = np.where(
        (final['Total Overs Bowled'] > 0) & (final['Bowling Economy Rate'] > 0),
        (final['Total Wickets'] / final['Total Overs Bowled']) * (1 / final['Bowling Economy Rate']),
        0
    )
    
    # Fill missing values
    final = final.fillna(0)
    
    # Drop the original features that were transformed into more meaningful metrics
    drop_cols = [
        'Number of Fours', 'Number of Sixes', "Total Runs Scored by Team",
        'Total Matches Played', 'NRR', "Total Overs Bowled", "Average Bowling Strike Rate",
        "Batting Average", "Average Strike Rate", "Total Balls Faced", "Total Runs Conceded", 
        "Total Maiden Overs", 'Total Catches Taken'
    ]
    
    final.drop(columns=[col for col in drop_cols if col in final.columns], inplace=True)
    
    return final, df['Team'].values  # Return the processed data and the team names

def align_features(X_pred, X_train_cols):
    """
    Ensure prediction data has the same features as training data
    """
    # Create a new dataframe with all the training columns
    aligned_X = pd.DataFrame(columns=X_train_cols)
    
    # Fill in values where columns exist in prediction data
    for col in X_train_cols:
        if col in X_pred.columns:
            aligned_X[col] = X_pred[col]
        else:
            aligned_X[col] = 0  # Default value for missing features
    
    # Check for any extra columns in prediction data that aren't in training data
    extra_cols = [col for col in X_pred.columns if col not in X_train_cols]
    if extra_cols:
        print(f"Warning: The following columns in prediction data were not used: {extra_cols}")
    
    return aligned_X

def predict_2025_results(model_file, X_pred, team_names):
    """
    Make predictions for 2025 using the loaded model
    """
    # Load the model
    try:
        # Try joblib first (more common for scikit-learn pipelines)
        model = joblib.load(model_file)
        print(f"Loaded model from {model_file} using joblib")
    except:
        try:
            # Try pickle as fallback
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            print(f"Loaded model from {model_file} using pickle")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    # Make predictions
    y_pred = model.predict(X_pred)
    
    # Get probability estimates if available
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_pred)
        
        # Create results dataframe with team names
        results_df = pd.DataFrame({
            'Team': team_names,
            'Predicted_Result': y_pred
        })
        
        # Add probability columns for each class
        for i in range(y_prob.shape[1]):
            results_df[f'Prob_Class_{i}'] = y_prob[:, i]
    else:
        # If no probabilities available, just return predictions
        results_df = pd.DataFrame({
            'Team': team_names,
            'Predicted_Result': y_pred
        })
    
    # Sort by predicted result (lower is better - 0 is champion, 4 is eliminated)
    results_df = results_df.sort_values('Predicted_Result')
    
    return results_df

def visualize_results(results_df):
    """
    Create visualizations of the prediction results
    """
    # Plot the predicted results
    plt.figure(figsize=(12, 8))
    
    # Create a color map (green for qualified, red for eliminated)
    colors = ['green' if result <= 3 else 'red' for result in results_df['Predicted_Result']]
    
    # Create bar chart
    plt.bar(results_df['Team'], 4 - results_df['Predicted_Result'], color=colors)
    plt.title('IPL 2025 Predictions (Higher = Better Finish)', fontsize=16)
    plt.xlabel('Teams', fontsize=14)
    plt.ylabel('Predicted Performance (4 - Result Category)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Add a horizontal line for qualification cutoff
    plt.axhline(y=1, linestyle='--', color='black')
    plt.text(0, 1.05, 'Qualification Line', fontsize=12)
    
    # Save the visualization
    plt.savefig("ipl_2025_predictions.png")
    plt.close()
    
    # If probability data is available, create a stacked probability chart
    if 'Prob_Class_0' in results_df.columns:
        plt.figure(figsize=(14, 10))
        
        # Prepare data for stacked bar chart
        prob_cols = [col for col in results_df.columns if col.startswith('Prob_Class_')]
        prob_data = results_df[prob_cols].values
        
        # Create labels for the legend - interpret what each class means
        class_labels = [
            'Champion (0)',
            'Runner-up (1)',
            'Third Place (2)',
            'Fourth Place (3)',
            'Eliminated (4)'
        ]
        
        # Create stacked bar chart
        bottom = np.zeros(len(results_df))
        for i, col in enumerate(prob_cols):
            plt.bar(results_df['Team'], results_df[col], bottom=bottom, label=class_labels[i])
            bottom += results_df[col]
        
        plt.title('IPL 2025 Prediction Probabilities by Finish Category', fontsize=16)
        plt.xlabel('Teams', fontsize=14)
        plt.ylabel('Probability', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title='Finish Category')
        plt.tight_layout()
        plt.savefig("ipl_2025_prediction_probabilities.png")
        plt.close()

def main():
    # Path to the 2025 prediction data and model file
    prediction_data_file = 'Book1.xlsx'  # Change this to your file name
    model_file = 'best_model.pkl'  # Change this if your model file has a different name
    
    # Process the prediction data
    print(f"Loading and preprocessing 2025 prediction data from {prediction_data_file}...")
    try:
        X_pred_raw, team_names = preprocess_prediction_data(prediction_data_file)
        print(f"Successfully loaded data for {len(X_pred_raw)} teams")
    except Exception as e:
        print(f"Error loading prediction data: {e}")
        return
    
    # We need to align the features with what the model was trained on
    # Try to load the processed training data to get the feature names
    try:
        processed_data = pd.read_pickle("final_processed.pkl")
        train_columns = [col for col in processed_data.columns if col != 'Result' and col != 'Year']
        print(f"Loaded training feature columns from processed data")
    except:
        # If we can't load the processed data, we'll have to define the features manually
        print("Warning: Couldn't load processed training data. Using manual feature list.")
        # This should match the features from your training data
        # You may need to adjust this based on your actual training data
        train_columns = X_pred_raw.columns.tolist()
    
    # Align features
    X_pred = align_features(X_pred_raw, train_columns)
    print(f"Aligned features with training data. Using {len(X_pred.columns)} features for prediction")
    
    # Make predictions
    print(f"\nMaking predictions using model from {model_file}...")
    results_df = predict_2025_results(model_file, X_pred, team_names)
    
    if results_df is not None:
        # Print results
        print("\nIPL 2025 Predictions (sorted by predicted finish):")
        print(results_df[['Team', 'Predicted_Result']])
        
        # Identify top 4 teams
        top_4_teams = results_df[results_df['Predicted_Result'] <= 3].sort_values('Predicted_Result')
        eliminated_teams = results_df[results_df['Predicted_Result'] > 3].sort_values('Predicted_Result')
        
        print("\n=== IPL 2025 PREDICTION RESULTS ===")
        print("\nTop 4 Teams (Qualified):")
        for i, (_, row) in enumerate(top_4_teams.iterrows()):
            rank_labels = ["Champion", "Runner-up", "Third Place", "Fourth Place"]
            print(f"{i+1}. {row['Team']} - Predicted as {rank_labels[int(row['Predicted_Result'])]}")
        
        print("\nEliminated Teams:")
        for i, (_, row) in enumerate(eliminated_teams.iterrows()):
            print(f"{i+1}. {row['Team']}")
        
        # Save results to CSV
        results_df.to_csv("ipl_2025_predictions.csv", index=False)
        print("\nResults saved to ipl_2025_predictions.csv")
        
        # Create visualizations
        try:
            visualize_results(results_df)
            print("Visualizations saved as ipl_2025_predictions.png and ipl_2025_prediction_probabilities.png")
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
if __name__ == "__main__":
    main()