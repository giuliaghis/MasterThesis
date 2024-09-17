#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import relevant libraries.
import pandas as pd
import numpy as np
import re # regular expression
import textwrap
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import qualitative
import dash
from dash import dcc, html
from dash.dependencies import Input, Output


# In[2]:


dataset = pd.read_csv('Thesis_Survey_Students_September_1.csv')


# In[3]:


print(dataset.head())


# In[4]:


print(dataset.columns.tolist())


# In[5]:


print(dataset.shape)


# # Exploratory Data Analysis

# ## 1) Preliminary Data Quality Analysis Using Qualtrics Metrics

# Qualtrics automatically analyzes the quality of the data and enables the download of specific columns if anomalies are detected.
# - **Q_RecaptchaScore**: When bot detection is enabled, a score is assigned to each respondent in the embedded data field Q_RecaptchaScore, indicating the likelihood the response came from a bot. If this score is less than 0.5, the response will be flagged as a bot.
# - **Q_AmbiguousTextPresent**: Whether or not there was ambiguous text in the survey response. 0 indicates no, and 1 indicates yes.
# - **Q_AmbiguousTextQuestions**: The specific questions that received an ambiguous response. This is given as an internal QID, not the editable question number.
# - **Q_UnansweredPercentage**: Responses are flagged when 30% or more of the seen questions were left unanswered. The percentage of questions in the survey that the respondent did not provide answer for. Only questions they saw are counted towards this value. This is presented as a ratio. For example, 100% is 1, and 66.7% is 0.667.
# - **Q_UnansweredQuestions**: The specific questions that were not answered. This is given as an internal QID, not the editable question number.

# In[6]:


q_metrics = ['Q_RecaptchaScore', 'Q_AmbiguousTextPresent', 'Q_AmbiguousTextQuestions', 'Q_UnansweredPercentage',
             'Q_UnansweredQuestions']

q_metrics_df = dataset[q_metrics]

print(q_metrics_df.info())


# #### A) SUSPECTED BOT

# In[7]:


# Convert Q_RecaptchaScore to a numeric type.
dataset['Q_RecaptchaScore'] = pd.to_numeric(dataset['Q_RecaptchaScore'], errors='coerce')

# Filter the rows where Q_RecaptchaScore is not equal to 1, indicating suspected bot responses.
bot_dataset = dataset[dataset['Q_RecaptchaScore'] != 1]

print(bot_dataset[['Q_RecaptchaScore']])


# **Two responses (indexes: 65, 70) are flagged as bots, having values of 0.2, which is less than 0.5.**
# 
# Also, I need to investigate the reason behind the null values for responses indexed 13, 16, and 18. Conversely, there is no need to investigate the first two rows (indexes: 0, 1) in the dataset, as they are descriptive and are automatically added by Qualtrics to provide information about the questions.

# In[8]:


# Display the suspected bot responses.
suspected_bot= bot_dataset.loc[[65, 70]]
print(suspected_bot)


# They are incomplete questionnaires, as indicated by the progress columns, which show that only 3% of the survey was completed.

# In[9]:


# Display the responses that have a null Q_RecaptchaScore.
null_rscore= bot_dataset.loc[[13, 16, 18]]
print(null_rscore)


# The responses with null values are complete, suggesting that the issue is likely due to Qualtrics errors.

# #### B) AMBIGUOUS RESPONSES

# In[10]:


# Convert Q_AmbiguousTextPresent to a numeric type.
dataset['Q_AmbiguousTextPresent'] = pd.to_numeric(dataset['Q_AmbiguousTextPresent'], errors='coerce')

# Filter the rows where Q_AmbiguousTextPresent is equal to 1, indicating ambiguity.
ambiguous_dataset = dataset[dataset['Q_AmbiguousTextPresent'] == 1]

print(ambiguous_dataset[['Q_AmbiguousTextPresent', 'Q_AmbiguousTextQuestions']])


# The ambiguous responses are related to the questions QID29 and QID33.

# In[11]:


# Identify the column that contains the internal QID.
match_string_1 = '{"ImportId":"QID29'
matching_column_1 = dataset.columns[dataset.loc[1].astype(str).str.contains(match_string_1, na=False)]

match_string_2 = '{"ImportId":"QID33'
matching_column_2 = dataset.columns[dataset.loc[1].astype(str).str.contains(match_string_2, na=False)]

print('QID29:', matching_column_1)
print('QID33:', matching_column_2)


# In[12]:


# Display the responses that are ambiguous.

print('Ambiguous text in the Home country column:\n', ambiguous_dataset[ambiguous_dataset['Q_AmbiguousTextQuestions'] == 'QID29'][['Home_country', 'Q_AmbiguousTextPresent', 'Q_AmbiguousTextQuestions']])

print('Ambiguous text in the Thoughts column:\n', ambiguous_dataset[ambiguous_dataset['Q_AmbiguousTextQuestions'] == 'QID33'][['Final_thoughts', 'Q_AmbiguousTextPresent', 'Q_AmbiguousTextQuestions']])


# - **Home country:** There is no text that includes “gibberish,” or cases where the respondent typed random letters and/or symbols to respond to a question. However, there is an instance where a respondent wrote the city instead of the country. This requires further investigation.
# 
# - **Thoughts:** 'Bah', a typical Italian expression, is often used to express skepticism or a lack of concern, suggesting the respondent is uncertain or indifferent.

# #### C) UNANSWERED QUESTIONS

# In[13]:


# Convert Q_UnansweredPercentage to a numeric type.
dataset['Q_UnansweredPercentage'] = pd.to_numeric(dataset['Q_UnansweredPercentage'], errors='coerce')

# Filter the rows where Q_UnansweredPercentage is not equal to 0, indicating the percentage of unanswered questions.
unanswered_dataset = dataset[dataset['Q_UnansweredPercentage'] != 0]

print(unanswered_dataset[['Q_UnansweredPercentage', 'Q_UnansweredQuestions']])


# The first two rows (indexes: 0, 1) in the dataset are descriptive and are added automatically by Qualtrics to provide information about the questions.

# In[14]:


# Identify the column that contains the internal QID.
match_string = '{"ImportId":"QID15'
matching_column = dataset.columns[dataset.loc[1].astype(str).str.contains(match_string, na=False)]

print(matching_column)


# In[15]:


print(unanswered_dataset[['Frequency_1', 'Frequency_2', 'Frequency_3', 'Frequency_4', 'Frequency_5', 'Frequency_6',
                          'Frequency_7', 'Frequency_7_TEXT', 'Q_UnansweredPercentage', 'Q_UnansweredQuestions']])


# In[16]:


print(unanswered_dataset[['Progress']])


# This is not a problem because the percentage is below 30% of the seen answers. However, it is curious that the survey appears to be 100% completed, yet there are no answers for the QID15 mandatory question (How frequently do you use GenAI technologies like ChatGPT in your academic work for each of the following purposes?). This need further investigation.

# In[17]:


selected_columns = dataset[['Frequency_1', 'Frequency_2', 'Frequency_3', 'Frequency_4', 'Frequency_5', 'Frequency_6',
                            'Frequency_7', 'Frequency_7_TEXT', 'Progress', 'Q_RecaptchaScore', 'Q_AmbiguousTextPresent',
                            'Q_UnansweredPercentage', 'Q_UnansweredQuestions']]
row_34 = selected_columns.loc[34]
print(row_34)


# The response has a slight suspicion of being from a bot, which could explain why a mandatory question was skipped.

# ### TakeAways
# 
# A) Two responses are flagged as bots (index: 65, 70) and, for this reason, they will be dropped.
# 
# B) There are no ambiguous responses. However, data cleaning and mapping are needed for the Home country column.
# 
# C) There are no unanswered questions. However, there is one minor warning for a response (index 34) that has missing values in a mandatory question. Although there is slight suspicion that this response might be from a bot, it does not justify dropping it.

# ## 2) Data Cleaning

# ### Drop answers flagged as bots

# In[18]:


# Drop rows that are flagged as bots.
dataset.drop(index=[65,70], inplace=True)

# Reset the indexes.
dataset.reset_index(drop=True, inplace=True)


# ### Clean and map the Home country column

# In[19]:


# Clean and map the Home country column.
print(dataset['Home_country'].unique())


# In[20]:


# Remove undesired characters.
def clean_text(x):
    if pd.isnull(x):
        return x
    return re.sub(r'[^a-zA-Z0-9]', '', x)

# Map to specific countries.
map_country = {
'Italia': 'Italy',
'Roma': 'Italy',
'Milano': 'Italy',
'ItaliaRoma': 'Italy',
}

def change_country(x):
    if pd.isnull(x):
        return x
    return map_country.get(x, x)


dataset.loc[2:, 'Home_country'] = dataset.loc[2:, 'Home_country'].str.title()

dataset.loc[2:, 'Home_country'] = dataset.loc[2:, 'Home_country'].apply(clean_text)

dataset.loc[2:, 'Home_country'] = dataset.loc[2:, 'Home_country'].apply(change_country)

print(dataset['Home_country'].unique())


# In[21]:


null_country = dataset['Home_country'].isnull().sum()
print(null_country)


# The responses in the Home country column are now mapped correctly to a specific country. However, there are 27 null values, meaning that 27 respondents have not completed the survey.

# ### Drop columns and rows that are not relevant for the analysis

# In[22]:


print(dataset.columns.tolist())


# In[23]:


# Drop columns that are not relevant for the analysis.
df = dataset.drop(columns=['StartDate', 'EndDate', 'Status', 'Duration (in seconds)', 'Finished', 'RecordedDate',
                           'ResponseId', 'DistributionChannel', 'UserLanguage', 'Q_RecaptchaScore', 'Q_AmbiguousTextPresent',
                           'Q_AmbiguousTextQuestions', 'Q_UnansweredPercentage', 'Q_UnansweredQuestions'])
print("Original dataset's shape:", dataset.shape)
print("New dataset's shape:", df.shape)


# In[24]:


# Drop rows that are not relevant for the analysis.
df.drop(index=[0,1], inplace=True)

# Reset the indexes.
df.reset_index(drop=True, inplace=True)

print("Original dataset's shape:", dataset.shape)
print("New dataset's shape:", df.shape)


# In[25]:


print(df.info())


# In[26]:


# Drop columns that refer to 'Other' options in the questions if no one filled them in.
for col in df.columns:
    if 'TEXT' in col and df[col].isna().all():
        df.drop(columns=[col], inplace=True)

print(df.info())


# ### Check the data type

# In[27]:


print(df.info())


# In[28]:


# Convert the data type of the columns.
columns_to_transform = [
    'Progress', 'Familiarity_1', 'Openness_1', 'Comfort_1', 'Study_method_1', 'Study_method_2', 'Study_method_3',
    'Study_method_4', 'Study_method_5', 'Study_method_6', 'Study_method_7', 'In_person_vs_online_1', 'Human_vs_AI_1',
    'Professors_aspects_1', 'Professors_aspects_2', 'Professors_aspects_3', 'Professors_aspects_4',
    'Professors_aspects_5', 'Professors_aspects_6', 'Professors_aspects_7', 'Professors_aspects_8',
    'Professors_aspects_9', 'Professors_aspects_10', 'Frequency_1', 'Frequency_2', 'Frequency_3', 'Frequency_4',
    'Frequency_5', 'Frequency_6', 'Frequency_7', 'Likelihood_1', 'Likelihood_2', 'Likelihood_3', 'Likelihood_4',
    'Likelihood_5', 'Concern_1', 'ChatGPT_effects_1', 'ChatGPT_effects_2', 'ChatGPT_effects_3',
    'ChatGPT_effects_4', 'ChatGPT_effects_5', 'ChatGPT_effects_6', 'ChatGPT_effects_7', 'Impact_1',
    'Performance_1', 'Satisfaction_1', 'Comparison_1', 'Comparison_2', 'Comparison_3', 'Comparison_4',
    'Comparison_5', 'Comparison_6', 'Comparison_7'
]

for x in columns_to_transform:
    df[x] = pd.to_numeric(df[x], errors='coerce')

print(df.info())


# There are few columns related to a specific question where multiple options are available. Without considering those columns, there are still mandatory questions that have null values, which require further investigation.

# ### Check the surveys still in progress

# In[29]:


# Count how many survey are still in progress.

surveys_in_progress = df[df['Progress'] < 100].shape[0]

surveys_completed = df[df['Progress'] == 100].shape[0]

print("Number of surveys still in progress:", surveys_in_progress)
print("Number of surveys completed:", surveys_completed)

# Create a dictionary to store the counts of surveys in progress and completed.
progress_counts = {
    'In Progress': surveys_in_progress,
    'Completed': surveys_completed
}

# Create a new plotly figure object.
fig = go.Figure(data=[go.Bar(name='Survey Status',x=list(progress_counts.keys()), y=list(progress_counts.values()),
                             marker_color=['orange', 'limegreen'])])

# Update the layout of the figure.
fig.update_layout(
    title='Survey Progress Status',
    xaxis_title='Survey Status',
    yaxis_title='Number of Surveys'
)

# Display the plot.
fig.show()


# In[30]:


progress_counts = df[df['Progress'] != 100]['Progress'].value_counts().sort_index()
print(progress_counts)


# The incomplete surveys have missing values. Although the initial questions have been answered, making the responses still valuable, it would not be feasible to retain them because the comparison between the 3P phases would be compromised. Without complete data across all phases, drawing accurate comparisons or conclusions would be challenging, as the missing values could lead to skewed or inconsistent results. Therefore, despite the usefulness of the initial responses, the analysis will proceed considering only complete surveys. 60 responses seems a fair number for esnuring robust and reliable results.

# In[31]:


# Drop incomplete surveys.
progress_threshold = 100
df = df[df['Progress'] == progress_threshold]

df.reset_index(drop=True, inplace=True)


# In[32]:


print(df.info())


# In[33]:


print(df.shape)


# ### Check the Grade question

# In[34]:


print(df['Grade'].unique())


# In[35]:


# Convert the Grade column to a single format.
def convert_grade(grade):
    if isinstance(grade, str):
        if '/' in grade:
            before_slash = grade.split('/')[0]
            if grade.endswith('/10'):
                return float(before_slash) * 3
            return float(before_slash.replace(',', '.'))
        elif '-' in grade:
            return float(grade.split('-')[0].replace(',', '.')) 
        elif any(char.isalpha() for char in grade):
            return np.nan
        else:
            return float(grade.replace(',', '.'))
    return grade

df['Grade'] = df['Grade'].apply(convert_grade)


# In[36]:


print(df['Grade'])


# # Visualizing Data

# The study is based on the Bigg's 3P model theoretical framework, and accordingly, the analysis will align with this framework as well.

# ## Summary of Key Patterns and Trends

# ### 1) Analyze Presage Factors

# In[37]:


# Define the columns of interest
columns = ['Age', 'Gender', 'Home_country', 'Level_of_education', 'University', 'Academic_field']

# Define a diverse color sequence
def get_color_sequence_pie(n):
    return px.colors.qualitative.Set2[:n]  # Use Plotly's qualitative color sequence

# Function to get the question text for dropdown.
def get_question_text_pie(column_name):
    return f"What is your {column_name.lower()}?"

# Initialize the Dash app
app1 = dash.Dash(__name__)

app1.layout = html.Div([
    html.H4('Select an Option:'),
    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': col, 'value': col} for col in columns],
        value='Age'  # default value
    ),
    dcc.Graph(id='pie-chart')
])

@app1.callback(
    Output('pie-chart', 'figure'),
    [Input('column-dropdown', 'value')]
)
def update_pie_chart(selected_column):
    data = df[selected_column]
    counts = data.value_counts()
    color_sequence = get_color_sequence_pie(len(counts))  # Get a diverse color sequence

    fig = px.pie(
        names=counts.index, 
        values=counts.values, 
        labels={name: name for name in counts.index},
        title=get_question_text_pie(selected_column),
        color_discrete_sequence=color_sequence  # Apply the diverse color sequence
    )
    
    fig.update_layout(
    legend=dict(
        font=dict(size=16),  # Adjust the font size of the legend items
        title=dict(
            text='Category',  # Set a title for the legend
            font=dict(size=18)  # Adjust the font size of the legend title
        )
    )
)
    
    fig.update_traces(
    textinfo='percent',  # Display labels and percentage
    textfont=dict(size=13)     # Font size for slice labels and values
)

    return fig

if __name__ == '__main__':
    app1.run_server(debug=True, port=8051)


# In[38]:


import plotly.graph_objects as go
import pandas as pd

# Assuming your DataFrame is named 'df' and contains the specified columns
columns = ['Age', 'Gender', 'Home_country', 'Level_of_education', 'University', 'Academic_field']

# Create a dictionary to store counts of unique values for each column
counts_data = {}

for col in columns:
    counts_data[col] = df[col].value_counts()  # Get counts of each unique value in the column

# Preparing the table data: Flattening the counts dictionary and computing total counts per category
table_data = {
    'Category': [],
    'Value': [],
    'Percentage': [],
    'Count': []
}

# Populate the table data and compute total counts per category
total_counts = {}

for col, counts in counts_data.items():
    total_counts[col] = counts.sum()
    first_entry = True  # Flag to keep track of the first entry for each category
    for value, count in counts.items():
        
        if first_entry:
            table_data['Category'].append(col)  # Add category name only once
            first_entry = False
        else:
            table_data['Category'].append('')  # Leave category name empty for subsequent entries
        
        table_data['Value'].append(value)
        
        table_data['Count'].append(count)
        
        percentage = round((count / total_counts[col]) * 100, 2)
        table_data['Percentage'].append(f"{percentage} %")

# Determine if all categories have the same total count
all_same_count = len(set(total_counts.values())) == 1

# Fixing count label to display correct total or different
if all_same_count:
    total_value = list(total_counts.values())[0]
    count_label = f"Count of Respondents (Total: {total_value})"
else:
    count_label = "Count (Total: different)"

# Define colors for each category
category_colors = {
    'Age': 'LightCyan',
    'Gender': 'MintCream',
    'Home_country': 'LavenderBlush',
    'Level_of_education': 'AliceBlue',
    'University': 'Honeydew',
    'Academic_field': 'Seashell'
}


# Generate colors for the cells based on the category
cell_colors = []
current_color = None  # Initialize a variable to keep track of the current color

for cat in table_data['Category']:
    if cat:  # If the category is not an empty string, update the current color
        current_color = category_colors[cat]
    cell_colors.append(current_color)  # Append the current color to the list

# Create the table
fig = go.Figure(data=[go.Table(
    header=dict(values=['Category', 'Value', count_label, 'Percentage (%) of Respondents'],
                fill_color='paleturquoise',
                align='left',
                font=dict(size=15)),  # Set font size for header
    cells=dict(values=[table_data['Category'], table_data['Value'], table_data['Count'], table_data['Percentage']],
               fill_color=[cell_colors, cell_colors, cell_colors],  # Apply colors per category
               align='left',
               font=dict(size=13),
               height=25)  # Set font size for cells
)])

# Update the layout to improve appearance
fig.update_layout(
    title='Personal Information Data',
    width=800,
    height=600
)

# Show the plot
fig.show()


# In[39]:


print(df.loc[df['Home_country'] == 'Vietnam', 'University'])


# In[40]:


# Convert the table data to a DataFrame
personal_info_export = pd.DataFrame(table_data)

# Export to Excel
personal_info_export.to_excel('personal_information_data.xlsx', index=False)


# ### TakeAways
# - Age: a majority of respondents, 63.33% (38 out of 60 respondents), fall within the 23-26 age range. This is followed by 31.67% (19 respondents) in the 18-22 age group, while the 27-30 age group represents the smallest proportion at 5% (3 respondents).
# - Gender: 70% (42 respondents) are female, while 30% (18 respondents) are male. No participants selected other gender categories.
# - Home_country: nearly all respondents, 59 out of 60 respondents, are from Italy, with only one respondent from Vietnam, who is studying at an Italian university as an international student.
# - Level_of_education: 58.33% (35 respondents) are pursuing a master’s degree, followed by 31.67% (19 respondents) pursuing a bachelor’s degree. A smaller number are in high school (6.67%, 4 respondents) or enrolled in an MBA program (3.33%, 2 respondents).
# - University: 73.33% (44 respondents) are studying at an Italian university other than Luiss University, while 23.33% (14 respondents) are studying at Luiss University. Additionally, one respondent is attending a university abroad, and another is in college.
# - Academic_field: the largest group of respondents, 35% (21 respondents), comes from the field of Economic and Statistical Sciences, followed by 13.33% (8 respondents) in Political and Social Sciences. The fields of Mathematical and Computer Science and Historical, Philosophical, Educational, and Psychological Sciences each account for 10% (6 respondents per field). The remaining 31.67% of participants are studying various other subjects.

# In[41]:


# Create an interactive histogram using Dash for slider questions.

# Specify the columns of interest in a list.
columns_of_interest = ['Familiarity_1', 'Openness_1', 'Comfort_1']

# Create a new df that contains only the desired columns.
df_selected = df[columns_of_interest]

# Create a dictionary to map each column name to its title, using the first row of the original dataset.
column_titles = {col: dataset[col].iloc[0] for col in columns_of_interest}

# Define labels for x axes.
ticktext_map = {'Familiarity_1': ['Extremely<br>Unfamiliar','20','Unfamiliar','40','Neutral','60','Familiar','80','Extremely<br>Familiar','100'],
                'Openness_1': ['Extremely<br>Negative','20','Negative','40','Neutral','60','Positive','80','Extremely<br>Positive','100'],
                'Comfort_1': ['Extremely<br>Uncomfortable','20','Uncomfortable','40','Neutral','60','Comfortable','80','Extremely<br>Comfortable','100']}

# Initialize the Dash app.
app2 = dash.Dash(__name__)

app2.layout = html.Div([   # contains the layout of the app
    html.H4('Select a Question:'),
    dcc.Dropdown(   # creates a dropdown menu
        id='column-dropdown',
        options=[{'label': title, 'value': col} for col, title in column_titles.items()],  # populates the options from column titles
        value=columns_of_interest[0]),  # specifies that the default value is the first column in columns_of_interest
    dcc.Graph(id='histogram-plot')])  # is a placeholder for the histogram that will be updated based on the selected dropdown value

# Define callback function.
@app2.callback(  # is a decorator that specifies the input and the output of the callback function
    Output('histogram-plot', 'figure'),  # indicates that the output of the function will update the figure property of the histogram-plot component
    [Input('column-dropdown', 'value')])  # indicates that the function will be triggered when the value of the column-dropdown component changes

# Implement the function.
def update_histogram(selected_column):  # is the currently selected column from the dropdown menu
    title = column_titles[selected_column]
    
    wrapped_title = "<br>".join(textwrap.wrap(title, width=100))   # breaks the title into multiple lines if it's too long

    bin_edges = list(range(0, 105, 5))   # calculates the edges of the bins used in the histogram
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]   # calculates the bin centers for color mapping

    fig = go.Figure()   # initializes a new plotly figure

    fig.add_trace(go.Histogram(    # adds a histogram trace to the figure with specified properties
        x=df_selected[selected_column],
        xbins=dict(start=0, end=105, size=5),
        name='Histogram',
        marker=dict(color=bin_centers,  # maps the bin centers to colors
                    colorscale='RdYlGn',
                    cmin=0,   # sets the min value for the color scale
                    cmax=100),   # sets the max value for the color scale
        opacity=1))

    ticktext = ticktext_map.get(selected_column)   # gets custom x-axis labels specific to the selected question

    fig.update_layout(   # updates the layout of the figure
        title=wrapped_title,
        xaxis_title="Value",
        yaxis_title="Count",
        xaxis=dict(range=[0, 105],
                   tickmode='array',
                   tickvals=[10,20,30,40,50,60,70,80,90,100],
                   ticktext=ticktext,
                   tickfont=dict(size=12)),
        yaxis=dict(range=[0, 30],
                  tickfont=dict(size=12)),  # sets y-axis to a max value
        bargap=0.01)
     
    for x_value in [20, 40, 60, 80]:
        fig.add_vline(x=x_value, line_dash="dash", line_color="gray", line_width=1) # adds vertical lines for each category
    
    return fig  # returns and displays the figure

if __name__ == '__main__':  # checks if the script is run directly (not imported as a module)
    app2.run_server(debug=True, port=8052)  # starts the Dash server with debugging enabled on port 8051


# ### TakeAways
# - A significant portion of respondents are familiar with GenAI technologies like ChatGPT, with 73.33% (44 respondents) indicating familiarity, including 41.67% (25 respondents) who are extremely familiar, suggesting frequent use of these technologies. Only a small number remain neutral (18.33%, 11 respondents), and just 8.33% (5 respondents) are unfamiliar.
# - The majority of respondents hold a positive attitude towards using technology in educational settings, with 85% (51 respondents) expressing positivity, including 51.67% (31 respondents) being extremely positive. However, 3.33% (2 respondents) exhibit less enthusiasm, while 11.67% (7 respondents) remain neutral.
# - This positivity is likely due to the fact that all respondents feel comfortable using digital tools for academic purposes, with no one expressing neutrality or discomfort.

# In[42]:


# Create an interactive multiple stacked bar chart using Dash for ranking questions.

# Define the column sets.
study_methods = ['Study_method_1', 'Study_method_2', 'Study_method_3', 'Study_method_4', 'Study_method_5', 'Study_method_6', 'Study_method_7']
professors_aspects = ['Professors_aspects_1', 'Professors_aspects_2', 'Professors_aspects_3', 'Professors_aspects_4', 'Professors_aspects_5', 'Professors_aspects_6', 'Professors_aspects_7', 'Professors_aspects_8', 'Professors_aspects_9', 'Professors_aspects_10']

# Define a function to get the question text for dropdown.
def get_question_text_bar(group_name):
    if group_name == 'study_methods':  # checks the group name
        return dataset[study_methods[0]].iloc[0].split('?')[0] + '?'  # extracts the first question text from the respective column and formats it
    else:
        return dataset[professors_aspects[0]].iloc[0].split('?')[0] + '?'

# Create a dictionary to map each column name to its title, using the first row of the original dataset.
def get_column_titles_bar(columns):
    return {col: dataset[col].iloc[0].split('- ')[-1] for col in columns}  # iterates through each column in the list and extracts the title by splitting the string at - and taking the last part

# Wrap labels.
def wrap_labels_bar(labels, max_length=15):
    wrapped_labels = []
    for label in labels:
        words = label.split(' ')  # Split the label into words
        wrapped_label = ''
        line = ''
        for word in words:
            # Check if adding the next word would exceed the max_length
            if len(line) + len(word) + 1 <= max_length:
                line += (word + ' ')  # Add the word to the current line
            else:
                wrapped_label += line.strip() + '<br>'  # Add the current line to the wrapped label and start a new line
                line = word + ' '  # Start the new line with the current word
        wrapped_label += line.strip()  # Add the last line
        wrapped_labels.append(wrapped_label)
    return wrapped_labels

# Initialize the Dash app.
app3 = dash.Dash(__name__)

app3.layout = html.Div([
    html.H4('Select a Question:'),
    dcc.Dropdown(
        id='column-set',
        options=[  # uses get_question_text to provide options for the dropdown menu
            {'label': get_question_text_bar('study_methods'), 'value': 'study_methods'},
            {'label': get_question_text_bar('professors_aspects'), 'value': 'professors_aspects'}],
        value='study_methods'),  # sets the default selected value to study_methods
    dcc.Graph(id='rank-plot')])  # is the placeholder for the bar chart that will be updated based on the dropdown selection

# Define the callback function.
@app3.callback(
    Output('rank-plot', 'figure'),  # specifies that the output of the callback will update the figure property of the rank-plot component
    [Input('column-set', 'value')]) # specifies that the callback is triggered by changes to the value property of the column-set dropdown

def update_stacked_bar_graph(selected_set):  # callback function, where selected_set is the value selected from the dropdown menu
    
    columns_of_interest = study_methods if selected_set == 'study_methods' else professors_aspects   # decides which colums to use based on the selected dropdown value
    df_selected = df[columns_of_interest].copy()   # creates a copy of the df, selecting the relevant columns from the df
    column_titles = get_column_titles_bar(columns_of_interest)   # gets a mapping of columns to their titles
    df_selected.rename(columns=column_titles, inplace=True)   # renames the columns in the df_selected using the titles.
    
    df_melted = df_selected.melt(var_name='Category', value_name='Rank')   # transforms the df from wide to long format for easier plotting (each original column becomes a row with the category and rank)
    df_count = df_melted.groupby(['Category', 'Rank']).size().reset_index(name='Count')   # groups by category and rank, then count the occurrences
    
    df_count['Category'] = wrap_labels_bar(df_count['Category'], max_length=15)
    
    x_axis_title = 'Study Method' if selected_set == 'study_methods' else 'Professors Aspects'  
    title_text = get_question_text_bar(selected_set)    
    
    fig = px.bar(df_count,
                 x='Category',
                 y='Count',
                 color='Rank',
                 barmode='group',
                 title=title_text,
                 labels={'Count': 'Count', 'Rank': 'Rank', 'Category': x_axis_title})    
    
    fig.update_layout(yaxis=dict(range=[0, 80]))
    
    return fig

if __name__ == '__main__':
    app3.run_server(debug=True, port=8053)


# In[43]:


print('The "other" option for study method include:', df['Study_method_7_TEXT'].unique())


# ### TakeAways
# - The top three study methods favored by students are reading, highlighting, and summarizing. This preference is not surprising, as these methods complement each other to enhance learning: reading provides the content, highlighting identifies key information, and summarizing reinforces understanding.
# - Similarly, the top three qualities that student value most in a professor are expertise and knowledge, effective communication skills, and a passion for teaching. These preferences reflect students desire for an engaging learning experience, where the material is both clearly explained and inspiring.

# In[44]:


# Create an interactive histogram using Dash for slider questions.

# Specify the columns of interest in a list.
columns_needed = ['In_person_vs_online_1', 'Human_vs_AI_1']

# Create a new df that contains only the desired columns.
df_selected4 = df[columns_needed]

# Create a dictionary to map each column name to its title, using the first row of the original dataset.
names_titles = {col: dataset[col].iloc[0] for col in columns_needed}

# Define labels for x axes.
ticktext_map4 = {'In_person_vs_online_1': ['Strongly prefer<br>in-person<br>learning',
                                           '20',
                                           'Somewhat prefer<br>in-person<br>learning',
                                           '40',
                                           'Neutral',
                                           '60',
                                           'Somewhat prefer<br>online<br>learning',
                                           '80',
                                           'Strongly prefer<br>online<br>learning',
                                           '100'],
                'Human_vs_AI_1': ['Strongly<br>prefer human<br>teachers',
                                  '20',
                                  'Somewhat<br>prefer human<br>teachers',
                                  '40',
                                  'Neutral',
                                  '60',
                                  'Somewhat<br>prefer AI<br>teachers',
                                  '80',
                                  'Strongly<br>prefer AI<br>teachers',
                                  '100']}

# Initialize the Dash app.
app4 = dash.Dash(__name__)

app4.layout = html.Div([   # contains the layout of the app
    html.H4('Select a Question:'),
    dcc.Dropdown(   # creates a dropdown menu
        id='column-dropdown',
        options=[{'label': title, 'value': col} for col, title in names_titles.items()],  # populates the options from column titles
        value=columns_needed[0]),  # specifies that the default value is the first column in columns_of_interest_2
    dcc.Graph(id='histogram-plot')  # is a placeholder for the histogram that will be updated based on the selected dropdown value
])

# Define callback function.
@app4.callback(  # is a decorator that specifies the input and the output of the callback function
    Output('histogram-plot', 'figure'),  # indicates that the output of the function will update the figure property of the histogram-plot component
    [Input('column-dropdown', 'value')])  # indicates that the function will be triggered when the value of the column-dropdown component changes

def update_bipolar_histogram(selected_column):  # is the currently selected column from the dropdown menu
    title = names_titles[selected_column]
    
    wrapped_title = "<br>".join(textwrap.wrap(title, width=100))   # breaks the title into multiple lines if it's too long

    bin_edges = list(range(0, 105, 5))   # calculates the edges of the bins used in the histogram
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]   # calculates the bin centers for color mapping

    fig = go.Figure()   # initializes a new plotly figure

    fig.add_trace(go.Histogram(    # adds a histogram trace to the figure with specified properties
        x=df_selected4[selected_column],
        xbins=dict(start=0, end=105, size=5),
        name='Histogram',
        marker=dict(color=bin_centers,  # maps the bin centers to colors
                    colorscale='bluered',
                    cmin=0,   # sets the min value for the color scale
                    cmax=100),   # sets the max value for the color scale
        opacity=1))

    ticktext = ticktext_map4.get(selected_column)   # gets custom x-axis labels specific to the selected question

    fig.update_layout(   # updates the layout of the figure
        title=wrapped_title,
        xaxis_title="Value",
        yaxis_title="Count",
        xaxis=dict(range=[0, 105],
                   tickmode='array',
                   tickvals=[10,20,30,40,50,60,70,80,90,100],
                   ticktext=ticktext,
                   tickfont=dict(size=12)),
        yaxis=dict(range=[0, 30], # sets y-axis to a max value
                   tickfont=dict(size=12)),
        bargap=0.01)
    
    for x_value in [20, 40, 60, 80]:
        fig.add_vline(x=x_value, line_dash="dash", line_color="gray", line_width=1) # adds vertical lines for each category
    
    return fig  # returns and displays the figure

if __name__ == '__main__':  # checks if the script is run directly (not imported as a module)
    app4.run_server(debug=True, port=8054)  # starts the Dash server with debugging enabled on port 8053


# ### TakeAways
# - The preference for an engaging learning experience is further underscored by the fact that 66.67% (40 participants) prefer in-person learning, with 53.33% (32 respondents) strongly favoring it. Meanwhile, 23.33% (14 respondents) are neutral, and only 10% (6 respondents) favor online learning.
# - Furthermore, nearly all respondents (90%, 54 out of 60 respondents) prefer human teachers, reinforcing the preference for in-person over online formats and highlighting their desire for engagement and passion in lessons. Only 10% (6 respondents) are neutral, while no one prefers AI instructors.

# In[ ]:





# In[ ]:





# ### 2) Analyze Process Factors
# 

# In[45]:


# Create an interactive bar chart using Dash for slider questions.

column_groups = {'Frequency': ['Frequency_1', 'Frequency_2', 'Frequency_3', 'Frequency_4', 'Frequency_5', 'Frequency_6', 'Frequency_7'],
                 'Likelihood': ['Likelihood_1', 'Likelihood_2', 'Likelihood_3', 'Likelihood_4', 'Likelihood_5'],
                 'ChatGPT_effects': ['ChatGPT_effects_1', 'ChatGPT_effects_2', 'ChatGPT_effects_3', 'ChatGPT_effects_4', 'ChatGPT_effects_5', 'ChatGPT_effects_6', 'ChatGPT_effects_7']}

group_palettes = {'Frequency': qualitative.Set1,
                  'Likelihood': qualitative.Dark24,
                  'ChatGPT_effects': qualitative.Bold}

# adds the question mark for dropdown menu
def get_question_text_bar_chart(group_name):
    if group_name in ['Frequency', 'Likelihood']:
        return dataset[column_groups[group_name][0]].iloc[0].split('?')[0] + '?'
    else:
        return dataset[column_groups[group_name][0]].iloc[0].split('-')[0]

# maps columns to their titles from row index 0 from the original dataset
#def get_column_titles_bar(columns):
    #return {col: dataset[col].iloc[0].split('- ')[-1] for col in columns}

# Initialize the Dash app
app5 = dash.Dash(__name__)

app5.layout = html.Div([
    html.H4('Select a Question:'),
    dcc.Dropdown(
        id='group-dropdown',
        options=[{'label': get_question_text_bar_chart(key), 'value': key} for key in column_groups.keys()],
        value='Frequency'
    ),
    dcc.Graph(id='histogram')
])

# wraps the text of both title and categories if it is too long
def wrap_text(text, wrap_length):
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line) + len(word) + 1 <= wrap_length:
            current_line += (word + " ")
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    lines.append(current_line.strip())
    
    return "<br>".join(lines)

@app5.callback(
    Output('histogram', 'figure'),
    [Input('group-dropdown', 'value')]
)
def update_bar_chart(selected_group):
    column_titles = get_column_titles_bar(column_groups[selected_group])
    color = group_palettes[selected_group]

    # sets the graph title to the wrapped version of the selected question
    title = wrap_text(get_question_text_bar_chart(selected_group), wrap_length=70)  # Adjust wrap_length as needed
    
    # calculates mean ratings for each column in the selected group
    means = {wrap_text(column_titles[col], wrap_length=15): df[col].mean() for col in column_groups[selected_group]}
    means_sorted = dict(sorted(means.items(), key=lambda item: item[1], reverse=True))
        
    colours = group_palettes[selected_group][:len(means_sorted)]  # Limit colors to the number of bars
        
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(means_sorted.keys()),
        y=list(means_sorted.values()),
        name='Average Ratings',
        marker_color=colours
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Category",
        yaxis_title="Average Rating",
        bargap=0.2,
        yaxis=dict(range=[0, 80])
    )
    
    return fig

if __name__ == '__main__':
    app5.run_server(debug=True, port=8055)


# In[46]:


print('The "other" option for frequent use include:', df['Frequency_7_TEXT'].unique())


# In[47]:


# Create an interactive histogram using Dash to display the specific distributions of the above categories.


column_groups = {'Frequency': ['Frequency_1', 'Frequency_2', 'Frequency_3', 'Frequency_4', 'Frequency_5', 'Frequency_6', 'Frequency_7'],
                 'Likelihood': ['Likelihood_1', 'Likelihood_2', 'Likelihood_3', 'Likelihood_4', 'Likelihood_5'],
                 'ChatGPT_effects': ['ChatGPT_effects_1', 'ChatGPT_effects_2', 'ChatGPT_effects_3', 'ChatGPT_effects_4', 'ChatGPT_effects_5', 'ChatGPT_effects_6', 'ChatGPT_effects_7']}

group_palettes = {'Frequency': qualitative.Set1,
                  'Likelihood': qualitative.Dark24,
                  'ChatGPT_effects': qualitative.Bold}

# adds the question mark for dropdown menu
def get_question_text_bar_chart(group_name):
    if group_name in ['Frequency', 'Likelihood']:
        return dataset[column_groups[group_name][0]].iloc[0].split('?')[0] + '?'
    else:
        return dataset[column_groups[group_name][0]].iloc[0].split('-')[0]

# maps columns to their titles from row index 0 from the original dataset
#def get_column_titles_bar(columns):
    #return {col: dataset[col].iloc[0].split('- ')[-1] for col in columns}


app5b = dash.Dash(__name__)

app5b.layout = html.Div([
    html.H4('Select a Question:'),
    dcc.Dropdown(
        id='group-dropdown',
        options=[{'label': get_question_text_bar_chart(key), 'value': key} for key in column_groups.keys()],
        value='Frequency'
    ),
    html.H4('Select a Specific Option:'),
    dcc.Dropdown(
        id='column-dropdown'
    ),
    dcc.Graph(id='histogram')
])

@app5b.callback(
    Output('column-dropdown', 'options'),
    [Input('group-dropdown', 'value')]
)
def set_columns_options(selected_group):
    column_titles = get_column_titles_bar(column_groups[selected_group])
    # Remove the 'All' option
    options = [{'label': title, 'value': col} for col, title in column_titles.items()]
    return options

@app5b.callback(
    Output('column-dropdown', 'value'),
    [Input('column-dropdown', 'options')]
)
def set_columns_value(available_options):
    return available_options[0]['value']

# Further customization for histogram will be implemented here
@app5b.callback(
    Output('histogram', 'figure'),
    [Input('group-dropdown', 'value'), Input('column-dropdown', 'value')]
)
def update_histogram_2(selected_group, selected_column):
    column_titles = get_column_titles_bar(column_groups[selected_group])

    # Histogram customization based on requirements
    if selected_group == 'Frequency':
        title = f"I frequently use ChatGPT for - {column_titles[selected_column]}"
    elif selected_group == 'Likelihood':
        title = f"It is likely that - {column_titles[selected_column]}"
    else:
        title = f"When I use ChatGPT - {column_titles[selected_column]}"

    data = df[selected_column]
    
    bin_edges = list(range(0, 105, 5))   # calculates the edges of the bins used in the histogram
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]   # calculates the bin centers for color mapping

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=data,
        xbins=dict(start=0, end=101, size=5),
        name='Histogram',
        marker=dict(color=bin_centers,  # maps the bin centers to colors
                    colorscale='RdYlGn',
                    cmin=0,   # sets the min value for the color scale
                    cmax=100),   # sets the max value for the color scale
        opacity=1
    ))

    # Set x-axis labels and vertical lines as per requirements
    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Count",
        xaxis=dict(
            tickmode='array',
            tickvals=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            ticktext=['Extremely low', '20', 'Low', '40', 'Neutral', '60', 'High', '80', 'Extremely high', '100']
        ),
        yaxis=dict(
            range=[0, 50],
            tickfont=dict(size=12)
        )
    )
    
    for x_value in [20, 40, 60, 80]:
        fig.add_vline(x=x_value, line_dash="dash", line_color="gray", line_width=1) # adds vertical lines for each category
    
    return fig

if __name__ == '__main__':
    app5b.run_server(debug=True, port=8045)


# ### TakeAways
# - The analysis revealed that most respondents frequently use GenAI tools for clarifying concepts and correcting minor details, with an average rating above 50 points. In contrast, these tools are not commonly used for completing tasks entirely, receiving a lower average rating of 25. The "other" category also received an average rating of 25, with respondents mentioning activities such as programming tasks, answering questions, and double-checking for understanding. The single response distribution graph shows that respondents gave relatively balanced responses across most categories, except for “completion of entire tasks”, which received significantly more negative ratings. Respondents also tended to give more “extreme low” ratings compared to the “extreme high” ratings.
# - Most respondents use GenAI outputs either as starting points that they expand upon themselves or by carefully reviewing the outputs and retaining only relevant information, often rephrasing it in their own words. These two categories scored 65 and 60 average points, respectively. In contrast, using the outputs without even reading them received only 7 average points. This was the only category where respondents gave extreme low ratings, whereas the distribution for other categories was more balanced.
# - The analysis also explored the effects of using ChatGPT on students. The results show minimal variation among the categories, with average ratings from 51 points for encouraging the exploration of additional resources to 36 points for enhancing the ability to retain information. In these cases, respondents also tended to give more “extreme low” ratings compared to the “extreme high” ones.

# In[48]:


# Create an interactive histogram using Plotly for a slider question.

column = ['Concern_1']
df_subset = df[column]

column_title = {col: dataset[col].iloc[0] for col in column}

ticktext_mapping = {'Concern_1': ['Completely<br>Unconcerned', '20', 'Unconcerned', '40', 'Neutral', '60', 'Concerned', '80', 'Extremely<br>Concerned', '100']}

def create_histogram():
    selected_column = 'Concern_1'
    title = column_title[selected_column]

    wrapped_title = "<br>".join(textwrap.wrap(title, width=70))
    
    bin_edges = list(range(0, 105, 5))
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df_subset[selected_column],
        xbins=dict(start=0, end=105, size=5),
        name='Histogram',
        marker=dict(color=bin_centers,
                    colorscale='Reds',
                    cmin=0,
                    cmax=100),
        opacity=1
    ))

    ticktext = ticktext_mapping.get(selected_column)

    fig.update_layout(
        title=wrapped_title,
        xaxis_title="Value",
        yaxis_title="Count",
        xaxis=dict(
            range=[0, 105],
            tickmode='array',
            tickvals=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            ticktext=ticktext,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            range=[0, 30],
            tickfont=dict(size=12)
        ),
        bargap=0.01
    )

    for x_value in [20, 40, 60, 80]:
        fig.add_vline(x=x_value, line_dash="dash", line_color="gray", line_width=1)

    fig.show()

create_histogram()


# ### TakeAways
# The results indicate no strong consensus, with responses being fairly evenly distributed. The two most selected categories “Concerned” and “Completed Unconcerned” account for 25% each (15 respondents each). However, there is a slight skew towards concern, with 40% (24 respondents) falling above neutrality compared to 38.33% (23 respondents) below neutrality, indicating some level of worry about data privacy.

# In[49]:


# Create an interactive bar chart using Plotly for a single choice question.

field = 'Inputs'

field_titles = {col: dataset[col].iloc[0] for col in [field]}
header = field_titles[field]
wrapped_header = "<br>".join(textwrap.wrap(header, width=100, ))

value_counts = df[field].value_counts().sort_index()

fig = px.bar(
    x=value_counts.index,
    y=value_counts.values,
    labels={'x': 'Number of Inputs', 'y': 'Count'},
    title=wrapped_header,
    color=value_counts.index,
    color_discrete_sequence=px.colors.qualitative.Plotly,
    range_y=[0, 30])

fig.update_layout(showlegend=False)

fig.show()


# ### TakeAways
# The study also examined how respondents interact with ChatGPT in conversation. When using ChatGPT to complete a task, users can either provide their input at the beginning or add more inputs based on the responses they receive. The analysis found that the majority of respondents (58.33%, 35 respondents) provide 2 to 5 additional inputs to ChatGPT. Interestingly, an equal number of participants (11.67%, 7 respondents each) do not provide any additional inputs or provide more than 10.

# In[ ]:





# In[ ]:





# In[ ]:





# ### 3) Analyze Product Factors

# In[50]:


# Create an interactive histogram using Dash for slider questions.

# Specify the columns of interest in a list.
key_columns = ['Impact_1', 'Performance_1', 'Satisfaction_1']

# Create a new df that contains only the desired columns.
df_extracted = df[key_columns]

# Create a dictionary to map each column name to its title, using the first row of the original dataset.
key_column_titles = {col: dataset[col].iloc[0] for col in key_columns}

# Define labels for x axes.
ticktext_map6 = {'Impact_1': ['Extremely<br>negative','20','Negative','40','Neutral','60','Positive','80','Extremely<br>Positive','100'],
                'Performance_1': ['Much<br>Worse','20','Worse','40','Neutral','60','Better','80','Much<br>Better','100'],
                'Satisfaction_1': ['Extremely<br>Dissatisfied','20','Dissatisfied','40','Neutral','60','Satisfied','80','Extremely<br>Satisfied','100']}

# Initialize the Dash app.
app6 = dash.Dash(__name__)

app6.layout = html.Div([   # contains the layout of the app
    html.H4('Select a Question:'),
    dcc.Dropdown(   # creates a dropdown menu
        id='column-dropdown',
        options=[{'label': title, 'value': col} for col, title in key_column_titles.items()],  # populates the options from column titles
        value=key_columns[0]),  # specifies that the default value is the first column in columns_of_interest
    dcc.Graph(id='histogram-plot')])  # is a placeholder for the histogram that will be updated based on the selected dropdown value

# Define callback function.
@app6.callback(  # is a decorator that specifies the input and the output of the callback function
    Output('histogram-plot', 'figure'),  # indicates that the output of the function will update the figure property of the histogram-plot component
    [Input('column-dropdown', 'value')])  # indicates that the function will be triggered when the value of the column-dropdown component changes

# Implement the function.
def update_histogram_3(selected_column):  # is the currently selected column from the dropdown menu
    title = key_column_titles[selected_column]
    
    wrapped_title = "<br>".join(textwrap.wrap(title, width=90))   # breaks the title into multiple lines if it's too long

    bin_edges = list(range(0, 105, 5))   # calculates the edges of the bins used in the histogram
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]   # calculates the bin centers for color mapping

    fig = go.Figure()   # initializes a new plotly figure

    fig.add_trace(go.Histogram(    # adds a histogram trace to the figure with specified properties
        x=df_extracted[selected_column],
        xbins=dict(start=0, end=105, size=5),
        name='Histogram',
        marker=dict(color=bin_centers,  # maps the bin centers to colors
                    colorscale='RdYlGn',
                    cmin=0,   # sets the min value for the color scale
                    cmax=100),   # sets the max value for the color scale
        opacity=1))

    ticktext = ticktext_map6.get(selected_column)   # gets custom x-axis labels specific to the selected question

    fig.update_layout(   # updates the layout of the figure
        title=wrapped_title,
        xaxis_title="Value",
        yaxis_title="Count",
        xaxis=dict(range=[0, 105],
                   tickmode='array',
                   tickvals=[10,20,30,40,50,60,70,80,90,100],
                   ticktext=ticktext,
                   tickfont=dict(size=12)),
        yaxis=dict(range=[0, 30],
                  tickfont=dict(size=12)),  # sets y-axis to a max value
        bargap=0.01)
     
    for x_value in [20, 40, 60, 80]:
        fig.add_vline(x=x_value, line_dash="dash", line_color="gray", line_width=1) # adds vertical lines for each category
    
    return fig # returns the figure

if __name__ == '__main__': 
    app6.run_server(debug=True, port=8056) 


# ### TakeAways
# - More than half of the respondents believe that GenAI has had a positive impact on their academic experience.
# - The majority of respondents have not noticed a significant impact on their grades, remaining neutral. This is surprising, considering that most participants felt GenAI had a positive impact. However, a considerable number did report a positive increase in their grades.
# - Almost all participants would not be satisfied with lessons conducted solely by AI, reinforcing their prior preferences for in-person learning and human teachers over AI, as indicated in the Presage phase of the study.

# In[51]:


# Create a histogram using Plotly for a short open-ended question.

variable = 'Grade'

df[variable] = df[variable].apply(lambda x: int(x) if pd.notna(x) and x % 1 < 0.50 else (int(x) + 1 if pd.notna(x) else x))

fig = px.histogram(
    df,
    x=variable,
    nbins=20,
    labels={'x': 'Grade', 'y': 'Count'},
    title='Distribution of Grades'
)

fig.update_yaxes(range=[0, 30])

fig.show()


# ### TakeAways
# The majority of respondents have a great grade point average.

# In[52]:


# Create an interactive bar chart using Plotly for a slider question.

# Define the comparison columns and color palette
comparison_columns = ['Comparison_1', 'Comparison_2', 'Comparison_3', 'Comparison_4', 'Comparison_5', 'Comparison_6', 'Comparison_7']
colors = qualitative.Pastel

# Choose the first column to determine the title for the graph
column_1 = comparison_columns[0]

# Extract titles for each column
column_titles = {col: dataset[col].iloc[0] for col in comparison_columns}

# Get the graph title from the first column, keeping only text before '-'
title_graph = column_titles[column_1].split('-')[0]

# Wrap the graph title to ensure readability
wrapped_title_graph = "<br>".join(textwrap.wrap(title_graph, width=100))

# Define a function to map columns to their titles from the dataset
def get_column_titles_bar_graph(columns):   
    return {col: dataset[col].iloc[0].split('- ')[-1] for col in columns}

# Get column titles for comparison columns
comparison_titles = get_column_titles_bar_graph(comparison_columns)

# Function to wrap text for better readability in the plot
def wrap_text_bar_graph(text, wrap_length):
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line) + len(word) + 1 <= wrap_length:
            current_line += (word + " ")
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    lines.append(current_line.strip())
    
    return "<br>".join(lines)

# Calculate mean ratings for each column in the Comparison group and sort them
means = {wrap_text_bar_graph(comparison_titles[col], wrap_length=15): df[col].mean() for col in comparison_columns}
means_sorted = dict(sorted(means.items(), key=lambda item: item[1], reverse=True))

# Limit colors to the number of bars
colors = colors[:len(means_sorted)]   

# Custom y-axis labels
custom_y_labels = ['0',
                   wrap_text('Human professors perform definitely better',20),
                   '20',
                   wrap_text('Human professors perform somewhat better',20),
                   '40',
                   wrap_text('Neutral',20),
                   '60',
                   wrap_text('AI performs somewhat better',20),
                   '80',
                   wrap_text('AI performs definitely better',20),
                   '100']

# Create the bar chart
fig = go.Figure()
fig.add_trace(go.Bar(
    x=list(means_sorted.keys()),
    y=list(means_sorted.values()),
    name='Average Ratings',
    marker_color=colors
))

# Update the layout of the figure with the dynamically retrieved and wrapped title
fig.update_layout(
    title=wrapped_title_graph,
    xaxis_title="Category",
    yaxis_title="Average Rating",
    bargap=0.2,
    yaxis=dict(
        range=[0, 100],  # Adjust the y-axis range as needed
        tickvals=[0,10,20,30,40,50,60,70,80,90,100],  # Positions for custom labels
        ticktext=custom_y_labels
    )
)

# Show the plot
fig.show()


# In[53]:


app5c = dash.Dash(__name__)

# Get titles for the comparison columns
comparison_titles = get_column_titles_bar_graph(comparison_columns)

app5c.layout = html.Div([
    html.H4('Select a Specific Option:'),
    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': title, 'value': col} for col, title in comparison_titles.items()],
        value=comparison_columns[0]  # Default value set to the first column
    ),
    dcc.Graph(id='histogram')
])

# Callback to update the histogram based on the selected column
@app5c.callback(
    Output('histogram', 'figure'),
    [Input('column-dropdown', 'value')]
)
def update_histogram_4(selected_column):
    # Fetch the column titles again if needed for displaying in the title
    column_titles = get_column_titles_bar_graph(comparison_columns)

    # Histogram customization based on selected column
    title = f"In terms of - {column_titles[selected_column]} ... "
    
    data = df[selected_column]
    
    bin_edges = list(range(0, 105, 5))  # calculates the edges of the bins used in the histogram
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]  # calculates the bin centers for color mapping

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=data,
        xbins=dict(start=0, end=101, size=5),
        name='Histogram',
        marker=dict(color=bin_centers,  # maps the bin centers to colors
                    colorscale='bluered',
                    cmin=0,  # sets the min value for the color scale
                    cmax=100),  # sets the max value for the color scale
        opacity=1
    ))

    # Set x-axis labels and vertical lines as per requirements
    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Count",
        xaxis=dict(
            range=[0, 104],
            tickmode='array',
            tickvals=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            ticktext=[
                   wrap_text('Human professors perform definitely better',20),
                   '20',
                   wrap_text('Human professors perform somewhat better',20),
                   '40',
                   wrap_text('Neutral',20),
                   '60',
                   wrap_text('AI performs somewhat better',20),
                   '80',
                   wrap_text('AI performs definitely better',20),
                   '100']
        ),
        yaxis=dict(
            range=[0, 30],
            tickfont=dict(size=12)
        )
    )
    
    for x_value in [20, 40, 60, 80]:
        fig.add_vline(x=x_value, line_dash="dash", line_color="gray", line_width=1)  # adds vertical lines for each category
    
    return fig

if __name__ == '__main__':
    app5c.run_server(debug=True, port=8046)


# ### TakeAways
# The average ratings indicate that students highly appreciate human professors for qualities such as inspiration, enjoyment of lectures, and teaching quality. Although none of the averages suggest that AI outperforms human professors, a few categories, such as fairness in grading and availability for consultation, fall into the neutral range. This suggests that some students may have perceived AI as performing better in these specific aspects.

# In[54]:


# Create an interactive bar chart using Dash.

# Define the column groups.
benefits_columns = ['Benefits_1', 'Benefits_2', 'Benefits_3', 'Benefits_4', 'Benefits_5','Benefits_6',
                    'Benefits_7', 'Benefits_8', 'Benefits_9', 'Benefits_10', 'Benefits_11']
concerns_columns = ['Concerns_1', 'Concerns_2', 'Concerns_3', 'Concerns_4', 'Concerns_5', 'Concerns_6',
                    'Concerns_7', 'Concerns_8', 'Concerns_9', 'Concerns_10', 'Concerns_11', 'Concerns_12']

# Combine the column groups into a dictionary for the dropdown options.
column_groups = {'Benefits': benefits_columns,'Concerns': concerns_columns}

# Function to get the question text for dropdown.
def get_question_text_bar_plot(group_name):
   return dataset[column_groups[group_name][0]].iloc[0].split('?')[0] + '?'

# Initialize the Dash app.
app7 = dash.Dash(__name__)

app7.layout = html.Div([
    html.H4('Select a Question:'),  # is a simple header prompting the user to select a question
    dcc.Dropdown(
        id='question-dropdown',
        options=[{'label': get_question_text_bar_plot(key), 'value': key} for key in column_groups.keys()],
        value='Benefits'
    ),
    dcc.Graph(id='bar-chart')
])

@app7.callback(
    Output('bar-chart', 'figure'),
    [Input('question-dropdown', 'value')]
)

def update_bar_plot(selected_question):
    
    # Concatenate the data from the selected group of columns.
    data = pd.concat([df[col] for col in column_groups[selected_question]], ignore_index=True)
    
    # Get the value counts for the unique values in the data.
    value_counts = data.value_counts().sort_values(ascending=False)
    
    # Set the color based on the selected question.
    color = 'limegreen' if selected_question == 'Benefits' else 'tomato'
    
    # Create the bar chart
    fig = px.bar(x=value_counts.index, 
                 y=value_counts.values, 
                 labels={'x': 'Value', 'y': 'Count'},
                 title=get_question_text_bar_plot(selected_question),
                 color_discrete_sequence=[color])
    
    fig.update_layout(
    xaxis=dict(tickmode='array',
               tickvals=list(range(len(value_counts.index))),
               ticktext=[f'<br>'.join(label.split()) for label in value_counts.index],
               tickfont=dict(size=10)),
    yaxis=dict(range=[0, 80]))

    return fig

if __name__ == '__main__':
    app7.run_server(debug=True, port=8057)


# ### TakeAways
# - The most valued benefits of ChatGPT are its unlimited access to resources anytime and anywhere, which helps save time and aids in research. Interestingly, students do not consider it particularly useful for improving digital skills, despite it being a digital tool.
# - On the concerns side, accuracy and reliability are the top issues, followed by lack of human interaction and the risk of over-reliance on technology. Notably, concerns about bias, privacy, ethics, and technical issues received the fewest votes. This aligns with the respondents' previous responses of not being overly worried about sharing personal data.

# In[ ]:





# In[ ]:




