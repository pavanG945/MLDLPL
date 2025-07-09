import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="IPL Standings & Probabilities", page_icon="🏏", layout="centered")

# Custom CSS for classy look
st.markdown("""
    <style>
    .standing-card {
        background: linear-gradient(90deg, #5DADE2 0%, #154360 100%);
        border-radius: 18px;
        padding: 1.5em 0;
        margin-bottom: 1.5em;
        box-shadow: 0 4px 16px rgba(44, 62, 80, 0.08);
        text-align: center;
    }
    .standing-rank {
        font-size: 4em;
        font-weight: bold;
        color: #fff;
        margin-bottom: 0.2em;
        letter-spacing: 0.05em;
        text-shadow: 2px 2px 8px #154360;
    }
    .standing-label {
        font-size: 1.3em;
        color: #FDFEFE;
        font-weight: 500;
        letter-spacing: 0.03em;
    }
    .prob-metric .stMetric {
        font-size: 0.9em !important;
    }
    .stSelectbox > div {
        font-size: 1.1em;
    }
    .team-category-0 {
        background-color: rgba(255, 99, 71, 0.1);
    }
    .team-category-1 {
        background-color: rgba(50, 205, 50, 0.1);
    }
    .team-category-3 {
        background-color: rgba(30, 144, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; margin-bottom:0.5em;'>🏏 IPL 2025 Team Standings & Probabilities</h1>", unsafe_allow_html=True)

# File uploaders


predictions_file =pd.read_csv("ipl_2025_predictions.csv")

nrr_df =pd.read_excel("Book1.xlsx")


df = predictions_file
    
    # Convert probability columns to numeric
prob_cols = [col for col in df.columns if col.startswith('Prob_Class_')]
for col in prob_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert Predicted_Result to numeric
df['Predicted_Result'] = pd.to_numeric(df['Predicted_Result'], errors='coerce')
    
    

        
        
if 'Team' in nrr_df.columns and 'NRR' in nrr_df.columns:
    nrr_dict = dict(zip(nrr_df['Team'], nrr_df['NRR']))
    df['NRR'] = df['Team'].map(nrr_dict)
else:
    st.warning("Excel file must contain 'Team' and 'NRR' columns. Using default NRR values.")
    df['NRR'] = 0.0

    
    # Sort teams according to the specified rules:
    # - If Result = 1, then it should be first
    # - If Result = 2, then it should be after 1
    # - If Result = 0, they are eliminated (they should be at the last)
    # - If 2 teams have the same result, compare their NRR
    
    # Create a sorting priority column
df['Sort_Priority'] = df['Predicted_Result'].map({
1: 1,    # First place
2: 2,    # Second place
0: 10,   # Eliminated (goes to bottom)
}).fillna(df['Predicted_Result'] + 2)  # Other values come after 1 and 2

# Sort by priority first, then by NRR (descending) for ties
standings = df.sort_values(['Sort_Priority', 'NRR'], ascending=[True, False])

# Add standing column based on sorted order
standings['Standing'] = range(1, len(standings) + 1)

# Display all teams standings
st.markdown("### 🏆 All Teams - Predicted Standings")

# Create a more descriptive standing category for display
def get_result_category(result):
    if result == 0:
        return "Eliminated"
    elif result == 1:
        return "Champion"
    elif result == 2:
        return "Runner-up"
    elif result == 3:
        return "Playoff"
    else:
        return f"Position {int(result)}"

standings['Result_Category'] = standings['Predicted_Result'].apply(get_result_category)

# Create the display DataFrame
display_cols = ['Standing', 'Team', 'Result_Category', 'NRR']
standings_display = standings[display_cols].copy()

# Display the table
st.dataframe(standings_display, hide_index=True, use_container_width=True)

# Team selection
teams = standings['Team'].tolist()
team = st.selectbox("Select Team to View Details", teams, index=0)

if team:
    row = standings[standings['Team'] == team].iloc[0]
    result_category = get_result_category(row['Predicted_Result'])

st.markdown(f"""
    <div class="standing-card">
        <div class="standing-label">Standing (Rank)</div>
        <div class="standing-rank">{int(row['Standing'])}</div>
        <div style="font-size:1.2em; color:#FFFFFF; font-weight:600;">{team}</div>
        <div style="font-size:1.1em; color:#D6EAF8; margin-top:0.5em;">{result_category} | NRR: {row['NRR']:.3f}</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Show probability details
st.markdown("### 📊 Probability Breakdown")

# Create a more readable probability dataframe
prob_cols = [col for col in row.index if col.startswith('Prob_Class_')]
if prob_cols:
    probs = pd.DataFrame({
        'Category': ['Eliminated', 'Champion', 'Runner-up', 'Third Place', 'Fourth Place'][:len(prob_cols)],
        'Probability': [row[col] for col in prob_cols]
    })
    
    # Convert probabilities to percentages
    probs['Percentage'] = probs['Probability'] * 100
    
    # Create a bar chart
    fig = px.bar(
        probs, 
        x='Category', 
        y='Percentage',
        text=probs['Percentage'].apply(lambda x: f'{x:.1f}%'),
        title=f'Outcome Probabilities for {team}',
        color='Category',
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    fig.update_layout(
        xaxis_title="Outcome",
        yaxis_title="Probability (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    # Add metrics for key probabilities
    col1, col2, col3 = st.columns(3)
    with col1:
        champion_prob = row['Prob_Class_1'] if 'Prob_Class_1' in row else 0
        st.metric("Champion Probability", f"{champion_prob*100:.1f}%")
    with col2:
        eliminated_prob = row['Prob_Class_0'] if 'Prob_Class_0' in row else 0
        st.metric("Top 4 Probability", f"{(1-eliminated_prob)*100:.1f}%")
    with col3:
        st.metric("Elimination Probability", f"{eliminated_prob*100:.1f}%")


