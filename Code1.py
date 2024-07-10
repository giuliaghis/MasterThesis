# Import relevant libraries.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the dataset.
dataset = pd.read_csv ('Thesis_Survey_Students_July_9.csv')

# Have a quick look of the dataset.
dataset.head()

dataset.columns.tolist()

# Create a new dataset where the columns that are not relevant for the analysis are dropped.
df = dataset.drop(columns=['StartDate','EndDate','Status','Progress','Duration (in seconds)','Finished','RecordedDate','ResponseId','DistributionChannel','UserLanguage','Q_RecaptchaScore','Q_AmbiguousTextPresent','Q_AmbiguousTextQuestions','Q_StraightliningCount','Q_StraightliningPercentage','Q_StraightliningQuestions','Q_UnansweredPercentage','Q_UnansweredQuestions'])

df.head()

# Drop rows that are not relevant for the analysis.
df.drop(index=[0,1], inplace=True)

# Reset the indexes.
df.reset_index(drop=True, inplace=True)

df.shape

dataset.shape

df.info()

# Drop rows that have null values in the last mandatory question, meaning respondents have opened the survey but not finished it.
df.dropna(subset=['Thoughts'], inplace=True)

df.shape

df.info()

df

df['Home country'].unique()

# Map the answers to specific countries.
map_answer_country = {
'ROMA': 'ITALY',
'MILANO': 'ITALY',
'ITALIA, ROMA': 'ITALY',
'ITALIA': 'ITALY',
'ITALY': 'ITALY'
}

def change_with_country(country):
    country = country.strip().upper()
    return map_answer_country.get(country, country)

df['Home country'] = df['Home country'].apply(change_with_country)

df ['Home country'].unique()

# Compute frequency for personal information columns.
personal_information_columns = ['Age', 'Gender', 'Home country', 'Level of education', 'University', 'Academic field']

value_counts = {col: df[col].value_counts() for col in personal_information_columns}

for col, counts in value_counts.items():
    print(counts)
    print("\n")
