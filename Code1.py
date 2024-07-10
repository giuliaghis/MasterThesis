#!/usr/bin/env python
# coding: utf-8

# In[95]:


# Import relevant libraries.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[46]:


# Import the dataset.
dataset = pd.read_csv ('Thesis_Survey_Students_July_9.csv')


# In[47]:


# Have a quick look of the dataset.
dataset.head()


# In[48]:


dataset.columns.tolist()


# In[49]:


# Create a new dataset where the columns that are not relevant for the analysis are dropped.
df = dataset.drop(columns=['StartDate','EndDate','Status','Progress','Duration (in seconds)','Finished','RecordedDate','ResponseId','DistributionChannel','UserLanguage','Q_RecaptchaScore','Q_AmbiguousTextPresent','Q_AmbiguousTextQuestions','Q_StraightliningCount','Q_StraightliningPercentage','Q_StraightliningQuestions','Q_UnansweredPercentage','Q_UnansweredQuestions'])


# In[50]:


df.head()


# In[51]:


# Drop rows that are not relevant for the analysis.
df.drop(index=[0,1], inplace=True)


# In[52]:


# Reset the indexes.
df.reset_index(drop=True, inplace=True)


# In[53]:


df.shape


# In[54]:


dataset.shape


# In[55]:


df.info()


# In[56]:


# Drop rows that have null values in the last mandatory question, meaning respondents have opened the survey but not finished it.
df.dropna(subset=['Thoughts'], inplace=True)


# In[57]:


df.shape


# In[59]:


df.info()


# In[60]:


df


# In[80]:


df['Home country'].unique()


# In[84]:


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


# In[85]:


df ['Home country'].unique()


# In[94]:


# Compute frequency for personal information columns.
personal_information_columns = ['Age', 'Gender', 'Home country', 'Level of education', 'University', 'Academic field']

value_counts = {col: df[col].value_counts() for col in personal_information_columns}

for col, counts in value_counts.items():
    print(counts)
    print("\n")


# In[ ]:




