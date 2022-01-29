# Predicting matches outcomes in the English Premier League 2021/2022 season

## Summary

* Using data from the past English Premier League (EPL) seasons, I built a model to predict the outcomes of matches in the current season: home team win, draw, or away team win
* Collected and scraped data about teams, players, and managers from various sources
* Cleaned, preprocessed, and combined data into a single dataset
* Engineered features related to home team, away team, and difference between home and away teams
* Built, trained, tuned, and evaluated performance of four models, including LogisticRegression, RandomForest, XGBoost, and LightGBM
* Built a web application for comparing predicted and actual outcomes of EPL 2021/2022 season matches using Streamlit

## Project structure

The project is divided into three Jupyter notebook and one Python script:

* football_predictions_data_collection.ipynb - data collection
* football_predictions_data_cleaning_feature_engineering.ipynb - data cleaning and analysis, feature engineering
* football_predictions_model_building.ipynb - model building, training, and evaluation, and making predictions
* matches_predictions.py - script for the web application 

## Data Collection

Data for the project was collected from various sources and included information about attributes and rankings of EPL teams, players, and managers during the past 20 seasons, starting from 2001/2002 season. The collected data included:
* Results of matches from 2001/2002 to 2021/2022 seasons - [Kaggle dataset](https://www.kaggle.com/josephvm/english-premier-league-game-events-and-results) by Joseph Mohr
* Final standings of teams at the end of each season - [Kaggle dataset](https://www.kaggle.com/josephvm/english-premier-league-game-events-and-results) by Joseph Mohr
* FIFA rankings of players from 2011/2012 to 2021/2022 seasons - Kaggle datasets by [Dhia Elhak Goumri](https://www.kaggle.com/justdhia/fifa-players) and [Alex](https://www.kaggle.com/cashncarry/fifa-22-complete-player-dataset)
* FIFA rankings of teams from 2011/2012 to 2021/2022 seasons - scraped from [FIFA Index](https://www.fifaindex.com/teams/)
* Rankings of managers from 2011/2012 to 2019/2020 seasons - scraped from [Football World Rankings](https://www.clubworldranking.com/ranking-coaches) and [Wikipedia](https://en.wikipedia.org/wiki/List_of_Premier_League_managers)

For scraping data about managers, I modified the code by Gonza Ferreiro Volpi available [here](https://github.com/gonzaferreiro/Market_value_football_players/blob/master/Team_and_national_teams_ranking_scraps-Final.ipynb).

## Data cleaning and preprocessing

Some datasets had missing and incorrect values, so I imputed and/or edited those entries about which I had the correct information. Since information about players was present in different datasets and names of some players differed between these datasets, I also had to correct the names of those players manually. The dataset with rating of managers was rather problematic since it did not have information about all EPL managers and it haven't been updated by its creators since March 2020. I imputed the ratings of managers missing form this dataset according to the overall ranking of the respective teams. Finally, I combined data about all matches, starting from 2011/2012 season, in a single dataframe.
The final dataframe have 4014 rows, which corresponds to the same number of matches played starting from 2001/2002 season and ending on January 23, 2022. The distribution of matches outcomes from 2001/2002 to 2020/2021 seasons is presented below.

**Outcomes of football matches in EPL**

![alt text](https://github.com/osydorchuk89/predicting_football_results/blob/main/images/matches_outcomes.png)

## Feature engineering

From the data I collected, I engineering 56 features in total. These included rankings/attributes of teams, players, and managers, based on either their current ratings or past record. Of 56 features, 18 related to home team, 18 to away team, and 20 to the difference between home and away team. The full breakdown of features are given below.

| Feature | Home team | Away team | Difference between home and away teams |
| - | - | - | - | 
| Team attack rating | V | V | V |
| Team defense rating | V | V | V |
| Team midfield rating | V | V | V |
| Team overall rating | V | V | V |
| Team average points during last 3 seasons | V | V | V |
| Team average goal difference during last 3 seasons | V | V | V |
| Team number of appearances in EPL during last 20 seasons | V | V | V |
| Team average ball possession percentage during last 3 seasons | V | V | V |
| Team average number of shots per game during last 3 seasons | V | V | V |
| Team average number of shots on goal per game during last 3 seasons | V | V | V |
| Team number of consecutive appearances in EPL to date | V | V | V |
| Team best performance in EPL during last 20 seasons | V | V | V |
| Team goalkeeper rating | V | V | V |
| Team manager rating | V | V | V |
| Team players overall rating | V | V | V |
| Team players defense skills rating | V | V | V |
| Team players midfield skills rating | V | V | V |
| Team players attack skills rating | V | V | V |
| Difference of wins of home and away teams in previous encounters |  |  | V |
| Difference of goals scored by home and away teams in previous encounters |  |  | V |

## Evaluation metrics

To evaluate the performance of the model, I used two metrics. On the one hand, I would like to predict outcomes of as many games correctly as possible. On the other hand, it was also important to minimize the number costly mistaked, i.e. predicting home team wins as home team defeats and home team defeats as home team wins. 
1. *Accuracy*. Overall share of predictions that turned out to be correct. *The higher the better*.
2. *Critical errors*. Custom metric calculated as a share of away wins predicted as home wins and home wins predicted as away wins. *The lower the better*. 

## Validation approach

To build and train models, I first divided the dataset into train and test sets:
1. Train - 10 seasons, from 2011/2012 to 2020/2021.
2. Test - season 2021/2022 (ongoing).

Next, I used 5-fold time series-like validation on the train test to evaluate performance of different models. This allowed me to avoid making predictions of matches outcomes based of the data from the future.

**Validation used for evaluating model performance**

![alt text](https://github.com/osydorchuk89/predicting_football_results/blob/main/images/validation.png)

## Model training and evaluation

Using the validation method described above, I trained and evaluated performance of four models with default parameters. The results are provided below.

| Model | Accuracy | Critical errors |
| - | - | - |
| LogisticRegression | 53.84% | 21.95% |
|	RandomForestClassifier | 52.68% | 20.32% |
|	XGBClassifier | 50.58% | 19.47% |
|	LGBMClassifier | 52.11% | 19.11% |

Given that all classification algorithms demonstrated comparable results, I decided to proceed with tuning boosting algorithsm, XGBoost and LightGBM, given their bigger potential for improvement by changing their hyperparameters. After tuning XGBoost and LightGBM, I was able to improve their accuracy scores to 55.84% and 55.74% respectively. 
However, after examining the distribution of predictions and comparing them against actual results, I noticed that the models signigficantly underestimates probabilitis of draw and significantly overestimates probabilities of home wins. To address this, I manually changes the thresholds for predictions, so that the models were slightly more likely to predict draws and slightly less likely to predict home wins. After this, the accuracy of predictions decreased slightly, but the number of critical errors dropped significantly.

## Making predictions

Using the tuned boosting algorithms with customized threshold, I predicted the outcomes of matches in the test dataset, i.e. from the ongoing 2021/2022 season. I compared these predictions against baseline, i.e. predicting that all matches will result in home win, and with predictions made by LogisticRegression, RandomForest, and tuned XGBoost and LightGBM with default thresholds.
Among them, Tuned XGBoost with customized threshold showed the best results with **56.07% accuracy** and **14.95% critical errors**.

| Model |	Accuracy |	Critical errors |	
| - |	- |	- |		
| LogisticRegression |	51.40% |	21.96% |	
| Random Forest |	50.93% |	20.09% |	
| Tuned XGBoost |	54.67% |	18.69% |	
| **Tuned XGboost with customized threshold** |	**56.07%** |	**14.95%** |	
| Tuned LightGBM |	51.40% | 21.96% |	
| Tuned LightGBM with customized threshold |	51.40% |	18.22% |	

![alt text](https://github.com/osydorchuk89/predicting_football_results/blob/main/images/predictions_accuracy.png)

![alt_text](https://github.com/osydorchuk89/predicting_football_results/blob/main/images/predictions_critical_errors.png)

To complete the predictions until the end of the current season, I used the same algorithm to predict outcomes of matches that havent happened yet. However, since I do not know starting lineups of future matches, I only used features related to team ratings for prediction purposes. I then concatenated both sets of predictions together.

## Evaluating results

To analyze results of predictions, I compared the ditributions of predicted and actual outcomes and plotted confusion matrix.

**Comparison of distributions of predicted and actual results**

| Outcome | True | Predicted |
| - | - | - |
| Away win | 70 | 83 |
| Draw | 57 | 29 |
| Home win | 87 | 102 |

**Confusion matrix plot**

![alt text](https://github.com/osydorchuk89/predicting_football_results/blob/main/images/confusion_matrix.png)

The model increased the number of draw predictions and decreased the number of home win predictions, although still underestimated the former outcome and overestimated the latter outcome.
I also plotted the most important features identified by the XGBoost algorithm:

![alt text](https://github.com/osydorchuk89/predicting_football_results/blob/main/images/feature_importance.png)

As can be seen from the plot above, the most importnat features were related to the difference between overall teams and players ratings, as well as the difference between teams defense and midfield skills.

## Web application

To allow easy comparison between predicted and actual outcomes of the EPL 2021/2022 season matches, I created a simple web application using Streamlit. To use it, you need to select the name of home and away teams, after which you can see the predicted result of the match, the probabilities of three possible outcomes, and the actual result (if the match has already happened). The application can be accesed **[here](https://share.streamlit.io/osydorchuk89/predicting_football_results/main/matches_predictions.py)**.

## Next steps

Even after tuning and customizing prediction threshold, the accuracy of the model predictions remained quite low. I belive two main factors can explain this:
1. There is a great deal of randomness in how a given football match unfolds that cannot be captured by any machine learning model.
2. The features I used in my project are quite limited in their explanatory power.

Therefore, some ideas for improvement of the model performance and accuracy of predictions include:
* Add more data, e.g. ratings of substitute players and more robust and up-to-date ratings about managers
* Add more complex features, such as expected goals (xG) of players and teams
* Use other approaches to predicting matches outcomes, i.e. predicting the difference in goals scored or predicting probabilities of win by either team and then setting a threshold to predict draw

## Acknowledgments

This project was created for the [Machine Learning Beginning](https://prjctr.com/course/machine-learning-basics) online course by Projector. I would like to express my gratitude to [Evhen Terpil](https://github.com/terpiljenya) and [Vitalii Radchenko](https://github.com/vitaliyradchenko) for delivering excellent instructuions and advice on the project.
