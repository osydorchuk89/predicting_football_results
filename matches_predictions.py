import numpy as np
import pandas as pd
import streamlit as st

combined_predictions = pd.read_csv(
    'https://raw.githubusercontent.com/osydorchuk89/predicting_football_results/main/data/combined_predictions.csv'
)
teams_2021 = np.insert(combined_predictions.home_team.unique(), 0, '-')

st.subheader('Predicting results of English Premier League matches in 2021/2022 season')
st.write(
    """This is my app for predicting results of English Premier League matches in 2021/2022 season.
    After choosing home team and away team, you can see the result predicted by the model,
    probabilities of three possible outcomes (home team win, draw, away team win), and the true result
    of the match. If the match between the selected teams hasn't happened yet, the true result will be
    unknown.
    """
)

home_team = st.selectbox('Choose home team', teams_2021)
away_team = st.selectbox('Choose away team', teams_2021)

if (home_team != '-' and away_team != '-') and (home_team != away_team):
    st.write(f'You selected {home_team} and {away_team}')

    predict_result = st.button('Predict result')
    if predict_result:
        predicted_result = combined_predictions[
            (combined_predictions.home_team==home_team) & (combined_predictions.away_team==away_team)
        ].prediction.values[0]
        if predicted_result == 'Away team win':
            st.write(f'Predicted results is {away_team} win')
        elif predicted_result == 'Home team win':
            st.write(f'Predicted results is {home_team} win')
        else:
            st.write(f'Predicted result is draw')

    predict_proba = st.button('Predict probabilities')
    if predict_proba:
        predicted_probabilities_home = combined_predictions[
            (combined_predictions.home_team == home_team) & (combined_predictions.away_team == away_team)
            ].home_win_probability.values[0]
        predicted_probabilities_draw = combined_predictions[
            (combined_predictions.home_team == home_team) & (combined_predictions.away_team == away_team)
            ].draw_probability.values[0]
        predicted_probabilities_away = combined_predictions[
            (combined_predictions.home_team == home_team) & (combined_predictions.away_team == away_team)
            ].away_win_probability.values[0]
        st.write(f'Predicted probability of {home_team} win: {predicted_probabilities_home}%')
        st.write(f'\nPredicted probability of draw: {predicted_probabilities_draw}%')
        st.write(f'\nPredicted probability of {away_team}: {predicted_probabilities_away}%')
    true_result = st.button('See actual result')

    if true_result:
        correct_result = combined_predictions[
            (combined_predictions.home_team==home_team) & (combined_predictions.away_team==away_team)
        ].actual_result.values[0]
        if correct_result == 'Away team win':
            st.write(f'Actual result is {away_team} win')
        elif correct_result == 'Home team win':
            st.write(f'Actual result is {home_team} win')
        elif correct_result == 'Draw':
            st.write(f'Actual result is draw')
        else:
            st.write("The match hasn't happened yet")

elif (home_team == away_team) and home_team !='-':
    st.write('Please select different teams')
else:
    st.write('Please select valid teams')
