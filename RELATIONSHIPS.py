#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import relevant libraries.
import pandas as pd
import numpy as np
import re # regular expression
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import qualitative


# In[2]:


dataset = pd.read_csv('Thesis_Survey_Students_September_1.csv')


# ## 2) Data Cleaning

# ### Drop answers flagged as bots

# In[3]:


# Drop rows that are flagged as bots.
dataset.drop(index=[65,70], inplace=True)

# Reset the indexes.
dataset.reset_index(drop=True, inplace=True)


# ### Clean and map the Home country column

# In[4]:


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


# ### Drop columns and rows that are not relevant for the analysis

# In[5]:


# Drop columns that are not relevant for the analysis.
df = dataset.drop(columns=['StartDate', 'EndDate', 'Status', 'Duration (in seconds)', 'Finished', 'RecordedDate',
                           'ResponseId', 'DistributionChannel', 'UserLanguage', 'Q_RecaptchaScore', 'Q_AmbiguousTextPresent',
                           'Q_AmbiguousTextQuestions', 'Q_UnansweredPercentage', 'Q_UnansweredQuestions'])
print("Original dataset's shape:", dataset.shape)
print("New dataset's shape:", df.shape)


# In[6]:


# Drop rows that are not relevant for the analysis.
df.drop(index=[0,1], inplace=True)

# Reset the indexes.
df.reset_index(drop=True, inplace=True)

print("Original dataset's shape:", dataset.shape)
print("New dataset's shape:", df.shape)


# In[7]:


# Drop columns that refer to 'Other' options in the questions if no one filled them in.
for col in df.columns:
    if 'TEXT' in col and df[col].isna().all():
        df.drop(columns=[col], inplace=True)

print(df.shape)


# ### Check the data type

# In[8]:


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

# Apply transformation
for x in columns_to_transform:
    df[x] = pd.to_numeric(df[x], errors='coerce')


# ### Check the surveys still in progress

# In[9]:


# Drop incomplete surveys.
progress_threshold = 100
df = df[df['Progress'] == progress_threshold]


df.reset_index(drop=True, inplace=True)


# In[10]:


print(df.shape)


# ### Check the Grade question

# In[11]:


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


# # Visualizing Data

# The study is based on the Bigg's 3P model theoretical framework, and accordingly, the analysis will align with this framework as well.

# ## Relationships Between Key Variables

# In[12]:


print(df.columns)


# In[13]:


# Create a new df where continuous values are converted into ordinal.

# Define ranges
bins = [0, 20, 40, 60, 80, 100]
# Define corresponding ordinal labels for each range
labels = [1, 2, 3, 4, 5]

# Create a new df
new_df = pd.DataFrame()

# Add colums to the new df
new_df['Familiarity level'] = pd.cut(df['Familiarity_1'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Openness level'] = pd.cut(df['Openness_1'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Comfort level'] = pd.cut(df['Comfort_1'], bins=bins, labels=labels, include_lowest=True, right=True)

new_df['In person vs online preference'] = pd.cut(df['In_person_vs_online_1'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Human vs AI preference'] = pd.cut(df['Human_vs_AI_1'], bins=bins, labels=labels, include_lowest=True, right=True)

new_df['Highlighting'] = df['Study_method_1']
new_df['Summarizing'] = df['Study_method_2']
new_df['Mind mapping'] = df['Study_method_3']
new_df['Reading'] = df['Study_method_4']
new_df['Repeating'] = df['Study_method_5']
new_df['Group study'] = df['Study_method_6']
new_df['Other methods'] = df['Study_method_7']
new_df['Other methods TEXT'] = df['Study_method_7_TEXT']

new_df['Expertise and knowledge'] = df['Professors_aspects_1']
new_df['Effective communication skills'] = df['Professors_aspects_2']
new_df['Passion'] = df['Professors_aspects_3']
new_df['Ethical conduct'] = df['Professors_aspects_4']
new_df['Motivation'] = df['Professors_aspects_5']
new_df['Precision'] = df['Professors_aspects_6']
new_df['Charisma'] = df['Professors_aspects_7']
new_df['Approachability and availability'] = df['Professors_aspects_8']
new_df['Fairness'] = df['Professors_aspects_9']
new_df['Other aspects'] = df['Professors_aspects_10']

new_df['Idea generation'] = pd.cut(df['Frequency_1'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Concept clarification'] = pd.cut(df['Frequency_2'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Detail correction'] = pd.cut(df['Frequency_3'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Feedback'] = pd.cut(df['Frequency_4'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Text summarization'] = pd.cut(df['Frequency_5'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Task completition'] = pd.cut(df['Frequency_6'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Other uses'] = pd.cut(df['Frequency_7'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Other uses TEXT'] = df['Frequency_7_TEXT']

new_df['Expand inputs'] = pd.cut(df['Likelihood_1'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Read and rephrase info'] = pd.cut(df['Likelihood_2'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Read and retain info'] = pd.cut(df['Likelihood_3'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Skim and retain info'] = pd.cut(df['Likelihood_4'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Retain everything'] = pd.cut(df['Likelihood_5'], bins=bins, labels=labels, include_lowest=True, right=True)

new_df['Inputs count'] = df['Inputs']

new_df['Concern level'] = pd.cut(df['Concern_1'], bins=bins, labels=labels, include_lowest=True, right=True)

new_df['Curiosity'] = pd.cut(df['ChatGPT_effects_1'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Learning motivation'] = pd.cut(df['ChatGPT_effects_2'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Confidence'] = pd.cut(df['ChatGPT_effects_3'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Preparation'] = pd.cut(df['ChatGPT_effects_4'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Resource_exploration'] = pd.cut(df['ChatGPT_effects_5'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Enjoyment'] = pd.cut(df['ChatGPT_effects_6'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Info retention'] = pd.cut(df['ChatGPT_effects_7'], bins=bins, labels=labels, include_lowest=True, right=True)

new_df['Impact on experience'] = pd.cut(df['Impact_1'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Impact on performance'] = pd.cut(df['Performance_1'], bins=bins, labels=labels, include_lowest=True, right=True)

new_df['Grade'] = df['Grade']

new_df['Satisfaction level'] = pd.cut(df['Satisfaction_1'], bins=bins, labels=labels, include_lowest=True, right=True)

new_df['Quality of education'] = pd.cut(df['Comparison_1'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Fairness in grading'] = pd.cut(df['Comparison_2'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Accuracy in responding questions'] = pd.cut(df['Comparison_3'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Enjoyment of the lecture'] = pd.cut(df['Comparison_4'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Inspiration for the course topic'] = pd.cut(df['Comparison_5'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Availability for consultation'] = pd.cut(df['Comparison_6'], bins=bins, labels=labels, include_lowest=True, right=True)
new_df['Usefulness of feedbacks'] = pd.cut(df['Comparison_7'], bins=bins, labels=labels, include_lowest=True, right=True)

new_df['Personalizing learning experience'] = df['Benefits_1']
new_df['Accessing resources anytime, anywhere'] = df['Benefits_2']
new_df['Managing time better'] = df['Benefits_3']
new_df['Saving time'] = df['Benefits_4']
new_df['Maintaining anonymity'] = df['Benefits_5']
new_df['Receiving instant feedback'] = df['Benefits_6']
new_df['Enhancing understanding'] = df['Benefits_7']
new_df['Assisting with research'] = df['Benefits_8']
new_df['Improving digital skills'] = df['Benefits_9']
new_df['Saving money'] = df['Benefits_10']
new_df['Other benefits'] = df['Benefits_11']

new_df['Accuracy and reliability'] = df['Concerns_1']
new_df['Bias, unfairness, and insensitivity'] = df['Concerns_2']
new_df['Limited contextual understanding'] = df['Concerns_3']
new_df['Privacy and data security'] = df['Concerns_4']
new_df['Ethical considerations'] = df['Concerns_5']
new_df['Lack of human interaction and socialization'] = df['Concerns_6']
new_df['Lack of motivation'] = df['Concerns_7']
new_df['Reduction of university value'] = df['Concerns_8']
new_df['Risk of inadequate learning outcomes'] = df['Concerns_9']
new_df['Over-reliance on technology'] = df['Concerns_10']
new_df['Technical issues'] = df['Concerns_11']
new_df['Other concerns'] = df['Concerns_12']

new_df['Age'] = df['Age']
new_df['Gender'] = df['Gender']
new_df['Home country'] = df['Home_country']
new_df['University'] = df['University']
new_df['Level of education'] = df['Level_of_education']
new_df['Academic field'] = df['Academic_field']

new_df['Final thoughts'] = df['Final_thoughts']


print(new_df.shape)


# In[14]:


new_df.info()


# In[15]:


new_df.columns


# **The variables do not follow a normal distribution, the dataset is relatively small, and outliers are present. Additionally, the new dataset has been created by converting continuous variables to ordinal ones to better investigate whether there are relationships between variables. Given these characteristics, the Spearman Rank Correlation Coefficient is a more suitable choice. It effectively measures monotonic relationships, is less sensitive to outliers, and does not require the assumption of normality.**

# ### How does the background influence students' perception of GenAI technologies integration in educational settings?

# In[16]:


# Create a heatmap using Plotly to explore the correlation between attitude towards using GenAI in education and
# the other presage factors.


# Invert the study methods and professors aspects ranking to align with the ranking scale used for openness,
# where 1 is "extremely negative". This ensures that both scales are now aligned, with lower values representing
# more negative attitudes/preferences.
new_df['Highlighting_r'] = new_df['Highlighting'].max() - new_df['Highlighting']
new_df['Summarizing_r'] = new_df['Summarizing'].max() - new_df['Summarizing']
new_df['Mind mapping_r'] = new_df['Mind mapping'].max() - new_df['Mind mapping']
new_df['Reading_r'] = new_df['Reading'].max() - new_df['Reading']
new_df['Repeating_r'] = new_df['Repeating'].max() - new_df['Repeating']
new_df['Group study_r'] = new_df['Group study'].max() - new_df['Group study']
new_df['Other methods_r'] = new_df['Other methods'].max() - new_df['Other methods']

new_df['Expertise and knowledge_r'] = new_df['Expertise and knowledge'].max() - new_df['Expertise and knowledge']
new_df['Effective communication skills_r'] = new_df['Effective communication skills'].max() - new_df['Effective communication skills']
new_df['Passion_r'] = new_df['Passion'].max() - new_df['Passion']
new_df['Ethical conduct_r'] = new_df['Ethical conduct'].max() - new_df['Ethical conduct']
new_df['Motivation_r'] = new_df['Motivation'].max() - new_df['Motivation']
new_df['Precision_r'] = new_df['Precision'].max() - new_df['Precision']
new_df['Charisma_r'] = new_df['Charisma'].max() - new_df['Charisma']
new_df['Approachability and availability_r'] = new_df['Approachability and availability'].max() - new_df['Approachability and availability']
new_df['Fairness_r'] = new_df['Fairness'].max() - new_df['Fairness']

# Specify relevant columns
columns_corr =['Openness level',
               
               'Comfort level','Familiarity level',
               
               'In person vs online preference','Human vs AI preference',
               
               'Highlighting_r','Summarizing_r','Mind mapping_r','Reading_r','Repeating_r','Group study_r',
               'Other methods_r',
               
               'Expertise and knowledge_r','Effective communication skills_r','Passion_r','Ethical conduct_r',
               'Motivation_r','Precision_r','Charisma_r','Approachability and availability_r','Fairness_r']


# Calculate correlation
spearman_corr = new_df[columns_corr].corr(method='spearman')

# Create the figure: heatmap
fig1 = go.Figure(data=go.Heatmap(
    
    z=spearman_corr.values,
    x=spearman_corr.columns, 
    y=spearman_corr.index,
    
    colorscale='RdBu_r',
    
    zmin=-1,
    zmax=1,
    
    colorbar=dict(title='Spearman<br>Corr. Coef.', tickfont=dict(size=10)),
    
    text=spearman_corr.round(2).values,
    texttemplate="%{text}",
    textfont=dict(color='black', size=8)  
))

# Update figure's layout
fig1.update_layout(
    title=dict(text="How does background influence students' perceptions of integrating GenAI tools in education?",
               font=dict(size=15)),
    xaxis=dict(tickfont=dict(size=10)),
    yaxis=dict(tickfont=dict(size=10)),
    height=800,
    width=1000
)


fig1.show()


# ### TakeAways
# **Correlations between attitude and presage factors:**
# - A correlation coefficient of 0.67 between comfort in using digital tools for academic purposes and attitude towards using GenAI tools in education suggests a moderately strong positive relationship. This indicates that as comfort and proficiency with digital tools increase, there is a corresponding positive shift in attitudes towards adopting GenAI tools in educational settings. Therefore, efforts to build digital literacy and confidence could be an effective strategy for promoting the acceptance and use of innovative educational technologies like GenAI.
# 
# - There is no correlation between attitude and preferences for in-person versus online learning or human-led versus AI-led teaching. However, visualizing the distribution of these variables could still be insightful to explore any underlying patterns or trends.
# 
# **Relevant correlations between presage factors:**
# - There is a moderate positive correlation between preference for in-person versus online learning and human-led versus AI-led teaching (0.39). This suggests that people who prefer in-person learning are also more likely to prefer human-led teaching, or vice versa.
# 
# The remaining relationships are less significant as they mainly reflect varying preferences for study methods and qualities in professors. It is logical that some factors are positively or negatively correlated, as participants were required to rank these according to their preferences.
# 
# These findings suggest the need for further investigation to better understand the nuances and implications of these correlations.
# 

# In[36]:


# Create a scatterplot using Plotly to to visualize each participant's responses regarding their openness to
# integrating GenAI tools in education and their preference for in person versus online learning.
# Use the original df to provide a more detailed visualization.

# Sort the df by 'In_person_vs_online_1' in ascending order
df_sorted_a = df.sort_values(by='In_person_vs_online_1').reset_index(drop=True)

# Create the figure: scatterplot
fig2a = go.Figure()

# Add scatter for In_person_vs_online_1
fig2a.add_trace(go.Scatter(
    x=df_sorted_a.index,
    y=df_sorted_a['In_person_vs_online_1'],
    mode='markers',
    name='In_person_vs_online_1',
    marker=dict(symbol='star', size=10, color='red')
))

# Add scatter for Openness_1
fig2a.add_trace(go.Scatter(
    x=df_sorted_a.index,
    y=df_sorted_a['Openness_1'],
    mode='markers',
    name='Openness_1',
    marker=dict(symbol='diamond', size=8, color='darkgreen')
))

# Update figure's layout
fig2a.update_layout(
    width=1050,
    title="Participants' attitude towards the academic use of GenAI and preference for learning modes",
    xaxis_title='Participants (Ordered by In Person vs Online Learning Preference)',
    yaxis_title='Values',
    xaxis=dict(
        tickmode='array',
        tickvals=df_sorted_a.index, 
        range=[df_sorted_a.index.min() - 1, df_sorted_a.index.max() + 1],
        tickfont=dict(size=9) 

    ),
    legend=dict(
        title='Categories:',
        orientation='h',
        y=1.01,
        x=0.5,
        xanchor='center',
        yanchor='bottom',
        font=dict(size=12)
    )
)

# Show the plot
fig2a.show()


# ### TakeAways
# The plot reveals that participants generally have a positive attitude towards integrating GenAI tools in education, regardless of their preference for in-person or online learning. Even those who strongly favor in-person learning (indicated by stars at the 0 value) show a favorable disposition towards using GenAI.

# In[37]:


# Create a scatterplot using Plotly to to visualize each participant's responses regarding their openness to
# integrating GenAI tools in education and their preference for human-led versus AI-led teaching.
# Use the original df to provide a more detailed visualization.

# Sort the df by 'Human_vs_AI_1' in ascending order
df_sorted_b = df.sort_values(by='Human_vs_AI_1').reset_index(drop=True)

# Create the figure: scatterplot
fig2b = go.Figure()

# Add scatter for Human_vs_AI_1
fig2b.add_trace(go.Scatter(
    x=df_sorted_b.index,
    y=df_sorted_b['Human_vs_AI_1'],
    mode='markers',
    name='Human_vs_AI_1',
    marker=dict(symbol='circle', size=8, color='blue')
))

# Add scatter for Openness_1
fig2b.add_trace(go.Scatter(
    x=df_sorted_b.index,
    y=df_sorted_b['Openness_1'],
    mode='markers',
    name='Openness_1',
    marker=dict(symbol='diamond', size=8, color='darkgreen')
))

# Update figure's layout
fig2b.update_layout(
    width=1050,
    title="Participants' attitude towards the academic use of GenAI and preference for teaching modes",
    xaxis_title='Participants (Ordered by Human-led vs AI-led Teaching Preference)',
    yaxis_title='Values',
    xaxis=dict(
        tickmode='array',
        tickvals=df_sorted_b.index, 
        range=[df_sorted_b.index.min() - 1, df_sorted_b.index.max() + 1],
        tickfont=dict(size=9) 
    ),
    legend=dict(
        title='Categories:',
        orientation='h',
        y=1.01,
        x=0.5,
        xanchor='center',
        yanchor='bottom',
        font=dict(size=12)
    )
)

fig2b.show()


# ### TakeAways
# The plot shows that participants generally have a positive attitude towards integrating GenAI tools in education, regardless of their preference for huamn-led or AI-led teaching. Even those who strongly favor huamn-led teaching (indicated by circles at the 0 value) show a favorable disposition towards using GenAI.

# In[19]:


# Create a combined plot using Plotly to explore relationships between participants' preferences for in-person vs
# online learning and human-led vs AI-led teaching and their attitude towards using GenAI tools in education.
# Use the original df to provide a more detailed visualization.

# Create the figure: multiple scatter plots
fig2 = go.Figure()

# Add scatterplot for In person vs AI
fig2.add_trace(go.Scatter(
    x=df['In_person_vs_online_1'], 
    y=df['Openness_1'],
    mode='markers',
    name='In-person vs Online Learning',
    marker=dict(color='red', size=10, symbol='star'),
))

# Add scatterplot for Huamn vs AI
fig2.add_trace(go.Scatter(
    x=df['Human_vs_AI_1'], 
    y=df['Openness_1'],
    mode='markers',
    name='Human-led vs AI-led Teaching',
    marker=dict(color='blue', size=6),
))

# Update figure's layout
fig2.update_layout(
    title="Students' attitude towards GenAI academic use based on their preferences for:",
    xaxis_title='Preference',
    yaxis_title='Openness Level',
    legend=dict(orientation='h',
                y=1.01,
                x=0.5,
                xanchor='center',
                yanchor='bottom',
                font=dict(size=12)
    ),
    template='plotly',
    yaxis=dict(range=[0, 105],
               tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
               ticktext=[0,
                         'Extremely<br>Negative',
                         '20',
                         'Negative',
                         '40',
                         'Neutral',
                         '60',
                         'Positive',
                         '80',
                         'Extremely<br>Positive',
                         '100']),
    xaxis=dict(tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
               ticktext=[0,
                         'Strongly<br>prefer<br>in person<br>learning /<br>human-led<br>teaching',
                         '20',
                         'Somewhat<br>prefer<br>in person<br>learning /<br>human-led<br>teaching',
                         '40',
                         'Neutral',
                         '60',
                         'Somewhat<br>prefer<br>online<br>learning /<br>AI-led<br>teaching',
                         '80',
                         'Strongly<br>prefer<br>online<br>learning /<br>AI-led<br>teaching',
                         '100'])
)

fig2.show()


# ### TakeAways
# The plot shows that most participants have a generally positive attitude toward using GenAI tools in education, regardless of whether they prefer in-person or online learning, or human-led versus AI-led teaching, aligning with the Spearman results.
# - Red Stars: Participants who strongly prefer in-person learning exhibit a range of attitudes toward GenAI tools, from neutral to highly positive, with only a few showing lower scores. As preferences shift toward online learning, this variability remains consistent, suggesting a broad acceptance of GenAI tools. This indicates that attitudes towards GenAI tools are relatively stable, regardless of the preferred learning environment, although some participants may view them as less essential or effective.
# - Blue dots: Participants who favor human-led teaching methods also show predominantly positive attitudes toward GenAI tools, as indicated by the clustering of blue dots in the "Extremely Positive" and "Positive" categories. This suggests that a preference for human-led instruction does not conflict with a favorable view of GenAI tools; rather, these participants may see GenAI as a valuable complement to human teaching. Even when preferences shift toward a more neutral stance regarding teaching methods, attitudes toward GenAI tools generally remain positive. Notably, while no participants preferred AI-led teaching alone, their positive attitudes suggest an openness to incorporating GenAI tools alongside human-led approaches.

# In[22]:


# Create a combined bar and scatter plots to explore relationships between the preferred professors' qualities and
# their preference for in person vs online learning and for human-led vs AI-led teaching.
# Use the original df to provide a more detailed visualization.

# Define relevant columns
columns_bar_scatter = ['Professors_aspects_1', 'Professors_aspects_2', 'Professors_aspects_3',
                       'Professors_aspects_4', 'Professors_aspects_5', 'Professors_aspects_6',
                       'Professors_aspects_7', 'Professors_aspects_8', 'Professors_aspects_9',
                       'Professors_aspects_10', 'In_person_vs_online_1', 'Human_vs_AI_1']

df_bar_scatter = df[columns_bar_scatter].copy()

# Identify the most preferred study method (ranked as '1st place') for each respondent
df_bar_scatter.loc[:, 'Preferred_Quality'] = df_bar_scatter[columns_bar_scatter[:-2]].idxmin(axis=1)

def get_prof_qualities(columns):
    return {col: dataset[col].iloc[0].split('- ')[-1] for col in columns}  # iterates through each column in the list and extracts the title by splitting the string at - and taking the last part

# Map the study method titles
prof_qualities_titles = get_prof_qualities(['Professors_aspects_1', 'Professors_aspects_2', 'Professors_aspects_3',
                                            'Professors_aspects_4', 'Professors_aspects_5', 'Professors_aspects_6',
                                            'Professors_aspects_7', 'Professors_aspects_8', 'Professors_aspects_9',
                                            'Professors_aspects_10'])
df_bar_scatter['Preferred_Quality'] = df_bar_scatter['Preferred_Quality'].map(prof_qualities_titles)

# Calculate the count of each preferred professor quality
prof_quality_counts = df_bar_scatter['Preferred_Quality'].value_counts().reset_index()
prof_quality_counts.columns = ['Preferred Quality', 'Count']

# Create the figure: bar plot and scatter plot
fig5 = go.Figure()

# Add bar plot for the count of each preferred professor quality
fig5.add_trace(
    go.Bar(
        x=prof_quality_counts['Preferred Quality'],
        y=prof_quality_counts['Count'],
        name='Count of Votes as 1st',
        marker=dict(color=px.colors.qualitative.Plotly, opacity=0.8),
        yaxis='y2',  # Assign to the secondary y-axis
        showlegend=False,
        opacity=0.3
    )
)

# Add scatter plot for 'In_person_vs_online_1'
fig5.add_trace(
    go.Scatter(
        x=df_bar_scatter['Preferred_Quality'],
        y=df_bar_scatter['In_person_vs_online_1'],
        mode='markers',
        name='In Person vs Online Learning',
        marker=dict(size=10, color='red', symbol='star'),
        yaxis='y1',  # Assign to the primary y-axis
        showlegend=True
    )
)

# Add scatter plot for 'Human_vs_AI_1'
fig5.add_trace(
    go.Scatter(
        x=df_bar_scatter['Preferred_Quality'],
        y=df_bar_scatter['Human_vs_AI_1'],
        mode='markers',
        name='Human-led vs AI-led Teaching',
        marker=dict(size=6, color='blue'),
        yaxis='y1',  # Assign to the primary y-axis
        showlegend=True
    )
)

# Update layout with customized y-axis
fig5.update_layout(
    height=700,
    title_text="Students' learning and teaching preferences based on their preferred professor's quality",
    xaxis=dict(
        title='Preferred Quality',
        tickmode='array',
        tickvals=df_bar_scatter['Preferred_Quality'].unique(),
        ticktext=["<br>".join(q.split()) for q in df_bar_scatter['Preferred_Quality'].unique() if isinstance(q, str)],
        showgrid=False
    ),
    yaxis=dict(
        title='Learning and Teaching Preferences (Scatter)',
        range=[-5, 105],
        tickvals=[10, 30, 50, 70, 90],
        ticktext=['Strongly prefer<br>in person learning /<br>human-led teaching',
                  'Somewhat prefer<br>in person learning /<br>human-led teaching',
                  'Neutral',
                  'Somewhat prefer<br>online learning /<br>AI-led teaching',
                  'Strongly prefer<br>online learning /<br>AI-led teaching'],
        showgrid=True
    ),
    yaxis2=dict(
        title='Count of Votes per Quality (Bar)',
        overlaying='y',  # overlays on the same x-axis
        side='right',
        range=[-5,105],  # Adjust the range dynamically
        showgrid=True
    ),
    legend=dict(orientation='h',
                y=1.01,
                x=0.5,
                xanchor='center',
                yanchor='bottom',
                font=dict(size=12)
    )
)

fig5.show()


# ### TakeAways
# - Only a few respondents who highly value expertise, knowledge, and effective communication skills prefer online learning, suggesting that they believe a professor's abilities are not hindered by a digital format.
# - Some respondents who prioritize passion showed a neutral stance regarding human-led versus AI-led teaching. This is unexpected, given that passion is typically not associated with AI.

# In[ ]:





# In[ ]:





# In[ ]:





# ### How do presage factors influence students' use of GenAI tools?

# In[23]:


# Create a heatmap using Plotly to explore the correlation between presage factors and process factors.

# Define relevan columns
columns_to_corr=['Openness level', 'Familiarity level', 'Comfort level',
                 
                 'In person vs online preference', 'Human vs AI preference',
                 
                 'Highlighting_r', 'Summarizing_r', 'Mind mapping_r', 'Reading_r', 'Repeating_r', 'Group study_r',
                 'Other methods_r', # use inverted study methods rank so that 1 represents the last voted
                 
                 'Expertise and knowledge_r', 'Effective communication skills_r', 'Passion_r', 'Ethical conduct_r',
                 'Motivation_r', 'Precision_r', 'Charisma_r', 'Approachability and availability_r', 'Fairness_r', # use inverted professors' qualities rank so that 1 represents the last voted
                 
                 'Idea generation', 'Concept clarification', 'Detail correction', 'Feedback', 'Text summarization',
                 'Task completition', 'Other uses',
                 
                 'Expand inputs', 'Read and rephrase info', 'Read and retain info', 'Skim and retain info',
                 'Retain everything',
                 
                 'Concern level',
                 
                 'Curiosity', 'Learning motivation', 'Confidence', 'Preparation', 'Resource_exploration',
                 'Enjoyment','Info retention']

# Calculate Spearman correlation
spearman_corr = new_df[columns_to_corr].corr(method='spearman')

# Create the figure: heatmap
fig6 = go.Figure(data=go.Heatmap(
    z=spearman_corr.values,
    x=spearman_corr.columns,
    y=spearman_corr.columns,
    colorscale='RdBu_r', 
    zmin=-1, zmax=1, 
    colorbar=dict(title='Spearman<br>Corr. Coef.',
                  orientation='h', 
                  x=0.5,
                  y=1,
                  xanchor='center',
                  yanchor='bottom'
                 ),
    text=spearman_corr.round(2).values,
    #text=np.where((spearman_corr.values >= 0.35) | (spearman_corr.values <= -0.35), 
    #             spearman_corr.round(2).values, 
    #             ""),  
    texttemplate="%{text}",
    textfont=dict(color='black', size=6) 
))

# Update figure's layout
fig6.update_layout(
    title='Correlations between process factors',
    xaxis=dict(tickangle=45, tickfont=dict(size=10)),  
    yaxis=dict(tickangle=0, tickfont=dict(size=10)),
    height=800,
    width=1000
)

fig6.show()


# ### TakeAways
# Correlations of 0.40 and above are considered significant because they suggest a moderate to strong relationship between variables.
# 
# **Correlation between presage and process factors**
# 
# Overall, there is no strong correlation between presage factors and the use of GenAI tools. However, few noteworthy patterns emerge:
# 
# - Individuals with greater familiarity tend to use GenAI tools for detail correction (0.48) and are more likely to read ChatGPT outputs and rephrase relevant information (0.42).
# 
# - There are no strong correlations between study methods and the purposes of GenAI use. However, a weak negative correlation (-0.36) suggests that participants who prefer repetition as a study method are less likely to use GenAI tools for text summarization (or vice versa). Conversely, participants who prefer mind mapping show a weak positive correlation (0.35) with the use of GenAI tools for text summarization. This could be due to mind mapping involving organizing and condensing information. Although these correlations are weak, they highlight an area that may benefit from further investigation to understand the underlying reasons behind these patterns.
# 
# **Correlations among process variables**
# 
# Among the process-phase variables, several strong correlations are observed:
# 
# - GenAI Usage for Idea Generation is strongly associated with its use for task completion (0.51), text summarization (0.50), skimming through outputs to retain relevant information (0.49), and detail correction (0.40). Using GenAI for idea generation also makes respondents feel more prepared on the topic of study (0.42).
# - GenAI Usage for Concept Clarification is closely linked to receiving feedback (0.64) and detail correction (0.54). It also positively affects feelings of motivation to learn (0.43), engagement or enjoyment in studying (0.41), and preparation on the topic of study (0.40).
# - GenAI Usage for Feedback is associated with detail correction (0.49).
# - GenAI usage for Text Summarization is strongly linked to skimming through ChatGPT's output and retaining relevant information (0.49). It also positively correlates with enjoyment of the topic of study (0.47), better information retention (0.44), and feeling prepared for class (0.42).
# - GenAI use for other purposes, such as programming or double-checking, show a positive relationship with skimming and retaining information (0.45). These uses are also positively correlated with curiosity (0.66) and resource exploration (0.41). 
# 
# - Expand inputs is positively correlated with read and rephrase information (0.40).
# - Read and retain information has a positive correlation with retain everything (0.48) and skim and retain information (0.42). It is also positively correlated with enjoyment (0.44).
# 
# - The effects of ChatGPT on students (in the upper-right corner of the chart) are generally positively correlated, indicating that high ratings in one area often correspond to high ratings in others (or vice versa).
# 
# - Although some low negative correlations are present, they are worth mentioning. For example, people who are not concerned about data privacy tend to use GenAI for detail correction (-0.32) and feel more confident about the subject (-0.29).

# In[24]:


# Create a mulitple box plot to explore relationships between the preferred study methods and how
# respondents use GenAI tools.

# Define relevant columns
columns_to_plot = ['Highlighting', 'Summarizing', 'Mind mapping', 'Reading', 'Repeating', 'Group study',
                   'Other methods',
                   'Idea generation', 'Concept clarification', 'Detail correction', 'Feedback', 'Text summarization',
                   'Task completition', 'Other uses']

df_to_plot = new_df[columns_to_plot].copy()

# Consider only the preferred study method by participants
df_to_plot['Preferred_method'] = df_to_plot[['Highlighting', 'Summarizing', 'Mind mapping', 'Reading','Repeating',
                                             'Group study','Other methods']].idxmin(axis=1)   # the preferred method is ranked 1

# Merge the preferred method with frequency of GenAI use
preferred_frequency = pd.DataFrame({
    'Preferred_method': df_to_plot['Preferred_method'],
    'Idea generation': df_to_plot['Idea generation'],
    'Concept clarification': df_to_plot['Concept clarification'],   # holds information on both the preferred study method and the 
    'Detail correction': df_to_plot['Detail correction'],           # frequency of GenAI usage across different tasks for each 
    'Feedback': df_to_plot['Feedback'],                             # participant. Each row corresponds to a participant, with columns 
    'Text summarization': df_to_plot['Text summarization'],         # indicating their study method and the frequency with which they 
    'Task completition': df_to_plot['Task completition'],           # use GenAI tools for various tasks.
    'Other uses': df_to_plot['Other uses']
})

# Melt the df for plotting
melted_df = pd.melt(preferred_frequency,
                    id_vars=['Preferred_method'], 
                    value_vars=['Idea generation', 'Concept clarification', 'Detail correction', 'Feedback', 'Text summarization', 'Task completition', 'Other uses'],
                    var_name='Purpose',
                    value_name='Frequency')

# Create the figure: multiple box plots
fig7 = px.box(melted_df,
             x='Preferred_method',
             y='Frequency',
             color='Purpose',
             title='GenAI Usage by Study Methods',
             labels={'Preferred_method': 'Preferred Study Method', 'Frequency': 'Frequency of GenAI Tools Usage'}
            )

# Update figure's layout
fig7.update_layout(
    legend_title_text='Purposes',
    yaxis=dict(range=[0,6],
               tickvals=[0,1,2,3,4,5,6],
               ticktext=[0,'Never','Rarely','Sometimes','Often','Always',6]
    ),
)

fig7.show()


# ### TakeAways
# - Participants who prefer reading as their study method tend to use GenAI tools *often* for correcting minor details. In contrast, they use these tools *rarely* for summarizing texts and completing entire tasks. Also, the majority *never* use GenAI tools for other purposes. However, there is considerable variability in their responses across all purposes, as indicated by the wide range of the whiskers.
# 
# - Participants who prefer highlighting use GenAI tools *often* for correcting minor details and generating ideas. Notably, the majority of those who favor highlighting *never* use GenAI tools for completing tasks entirely. Similar to those who prefer reading, there is variability in how these tools are applied across different purposes.
# 
# - An interesting observation for participants who prefer summarizing is the large interquartile range (IQR), indicating diverse usage patterns, and a negatively skewed median. This could likely result from a smaller number of participants choosing summarizing as their preferred method and using GenAI tools for varying purposes. The only slightly higher median is for using GenAI tools *often* to clarify concepts and *sometimes* for other purposes. Interestingly, there is no strong preference for using GenAI tools to summarize texts; its median usage is closer to *never*. This suggests that those who prefer summarizing tend to do it themselves. Additionally, these participants *never* use GenAI tools for completing tasks entirely.
# 
# - Only a few respondents selected other methods (e.g., Mind Mapping, Group Study, Repetition) as their preferred study approach, making it difficult to draw definitive conclusions. However, based on the available data, participants who prefer mind mapping tend to use GenAI tools *often* for summarizing and generating, while those who favor group study *often* use them for clarifying concepts. These patterns make sense, as mind mapping inherently involves organizing and condensing information, aligning well with the summarization and generation capabilities of GenAI tools. Similarly, group study often involves discussing and understanding various topics, which can be enhanced by using GenAI tools to clarify concepts and provide diverse perspectives or explanations.

# In[ ]:





# In[ ]:





# In[ ]:





# ### How do process factors influence learning outcomes?

# In[25]:


# Create a heatmap using Plotly to show the correlation between the process factors and product ones. Also, analyze
# correlation between product variables.

# Define relevan columns
columns_to_corr=['Idea generation', 'Concept clarification', 'Detail correction', 'Feedback', 'Text summarization',
                 'Task completition', 'Other uses',
                 
                 'Expand inputs', 'Read and rephrase info', 'Read and retain info', 'Skim and retain info',
                 'Retain everything',
                 
                 'Concern level',
                 
                 'Curiosity', 'Learning motivation', 'Confidence', 'Preparation', 'Resource_exploration',
                 'Enjoyment','Info retention',
                 
                 'Impact on experience', 'Impact on performance','Satisfaction level',
                 
                 'Quality of education','Fairness in grading', 'Accuracy in responding questions',
                 'Enjoyment of the lecture', 'Inspiration for the course topic', 'Availability for consultation',
                 'Usefulness of feedbacks']

# Calculate Spearman correlation
spearman_corr = new_df[columns_to_corr].corr(method='spearman')

# Create the figure: heatmap
fig10 = go.Figure(data=go.Heatmap(
    z=spearman_corr.values,
    x=spearman_corr.columns,
    y=spearman_corr.columns,
    colorscale='RdBu_r', 
    zmin=-1, zmax=1, 
    colorbar=dict(title='Spearman<br>Corr. Coef.'),
    text=spearman_corr.round(2).values,
    #text=np.where((spearman_corr.values >= 0.35) | (spearman_corr.values <= -0.35), 
                  #spearman_corr.round(2).values, 
                  #""),  
    texttemplate="%{text}",
    textfont=dict(color='black', size=8) 
))

# Update figure's layout
fig10.update_layout(
    title='How does the use of GenAI influence learning outcomes?',
    xaxis=dict(tickangle=45, tickfont=dict(size=10)),  
    yaxis=dict(tickangle=0, tickfont=dict(size=10)),
    height=800,
    width=1000
)

fig10.show()


# ### TakeAways
# Using Generative AI for purposes like detail correction, concept clarification, and idea generation shows a strong positive correlation with enhancing academic experience, with correlation coefficients of 0.54, 0.54, and 0.43, respectively. Additionally, leveraging AI tools for academic tasks can boost information retention, preparation, and motivation. These factors, in turn, positively impact the overall academic experience, with correlation values of 0.52 for information retention, 0.51 for preparation, and 0.45 for motivation.
# 
# While the direct impact of AI use on academic performance is less evident, likely because respondents already had high GPAs, there is evidence that using tools like ChatGPT can enhance information retention, which subsequently improves academic performance (correlation of 0.46). Furthermore, there is a notable positive correlation (0.48) between an enhanced academic experience and improved academic performance, suggesting that improvements in one area may lead to gains in the other.

# In[26]:


# Create a heatmap using Plotly to show the correlation between variables regarding participants' perceptions
# of human professors vs AI.

# Define relevant columns
columns_to_corr = ['Human vs AI preference', 'In person vs online preference', 'Openness level',
                   
                   'Satisfaction level',
                   
                   'Quality of education','Fairness in grading', 'Accuracy in responding questions',
                   'Enjoyment of the lecture', 'Inspiration for the course topic', 'Availability for consultation',
                   'Usefulness of feedbacks']

# Calculate Spearman correlation
spearman_corr = new_df[columns_to_corr].corr(method='spearman')

# Creeate the figure: heatmap
fig11 = go.Figure(data=go.Heatmap(
    z=spearman_corr.values,
    x=spearman_corr.columns,
    y=spearman_corr.columns,
    colorscale='RdBu_r', 
    zmin=-1, zmax=1, 
    colorbar=dict(title='Spearman<br>Corr. Coef.'),
    text=spearman_corr.round(2).values,  
    texttemplate="%{text}",
    textfont=dict(color='black', size=10) 
))

# Update figure's layout
fig11.update_layout(
    title='Spearman Correlation Between Presage and Process Factors',
    xaxis=dict(tickangle=45, tickfont=dict(size=10)),  
    yaxis=dict(tickangle=0, tickfont=dict(size=10)),
    height=800
)

fig11.show()


# ### TakeAways
# *Remember: A value closer to 0 indicates a preference for human-led approaches, while a value closer to 100 suggests a preference for AI-led methods.*
# 
# There is a positive correlation between satisfaction with AI and inspiration for the course topic (0.55), perceived quality of education (0.51), and enjoyment of lectures (0.51), suggesting that as people perceive humans to perform better in these areas, they are more likely to be dissatisfied with lectures conducted solely by AI (or vice versa). Moreover, the quality of education is positively correlated with the enjoyment of lectures (0.55), and enjoyment of lectures is strongly correlated with inspiration for the course topic (0.75). It is important to note that participants ranked inspiration for the course topic, perceived quality of education, and enjoyment of lectures as the top three areas where they believe human performance exceeds that of AI.

# In[27]:


# Concatenate the relevant columns from both DataFrames
combined_df = pd.concat([new_df['Openness level'], df['Satisfaction_1']], axis=1)

# Calculate counts for each category in 'Openness level'
counts = combined_df['Openness level'].value_counts().sort_index()

# Update x-axis labels to include counts
x_labels = [f'Extremely<br>Negative ({counts.get(1,0)})', 
            f'Negative ({counts.get(2,0)})', 
            f'Neutral ({counts.get(3,0)})', 
            f'Positive ({counts.get(4,0)})', 
            f'Extremely<br>Positive ({counts.get(5,0)})']

# Create a box plot using Plotly Graph Objects
fig = go.Figure()

# Add a box trace for each category
fig.add_trace(go.Box(
    x=combined_df['Openness level'],
    y=combined_df['Satisfaction_1'],
    boxpoints='all',
    pointpos=0,
    jitter=0.3,
    marker=dict(color='navy'),
    line=dict(color='navy'),
    showlegend=False, 
    boxmean=True
))

# Update layout to include custom x-axis labels with counts
fig.update_layout(
    title='Box Plot: Openness Level vs. Satisfaction_1 (with Mean)',
    xaxis=dict(
        title='Openness Level',
        range=[0.5, 5.5],
        tickvals=[1, 2, 3, 4, 5],
        ticktext=x_labels
    ),
    yaxis=dict(
        title='Satisfaction Level',
        range=[-5, 105],
        tickvals=[0,10,20,30,40,50,60,70,80,90,100],
        ticktext=[0, 'Extremely dissatisfied', 20, 'Dissatisfied', 40, 'Neutral', 60, 'Satisfied', 80, 
                  'Extremely satisfied', 100]
    )
)

# Show the plot
fig.show()


# ### TakeAways
# Participants with "Extremely Negative" or "Negative" openness levels tend to report low satisfaction with lectures conducted solely by AI, with scores clustering around the "Extremely Dissatisfied" and "Dissatisfied" ranges. As openness levels become more positive (moving from "Neutral" to "Extremely Positive"), there is an increase in the range of satisfaction levels, suggesting more variability in how these groups perceive AI-led lectures. Notably, those in the "Extremely Positive" category display a wide range of satisfaction, from "Extremely Dissatisfied" to "Extremely Satisfied," indicating that even among those with high openness to GenAI tools, satisfaction with AI-led lectures varies significantly.
# Overall, the plot suggests that while a higher openness to GenAI tools may be associated with a broader range of satisfaction levels with AI-led lectures, satisfaction is not uniformly high even among the most receptive participants. This variability indicates that other factors may also influence satisfaction with AI-led instruction. 

# In[ ]:




