#!/usr/bin/env python
# coding: utf-8

# # Power ratings

# ## Introduction
# 
# We'll build the power ratings using the following data, which contains scores from UEFA women's soccer teams in the last three years of Euro qualification and World Cup qualification.  There are 388 matches in this dataset.  The dataset also contains odds for most of those games from [oddspedia.com](https://oddspedia.com/football/world/world-cup-women-qualification#odds), but we won't use those odds to make these power ratings.

# In[1]:


import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import factorial


# In[2]:


df = pd.read_csv("data/scores_and_odds.csv")


# In[3]:


df


# There are 24 teams in the 2022 Women's Euro tournament, but our data also includes European teams which did not qualify.  There are 51 total teams in our data.

# In[4]:


df["home_team"].unique().size


# We are going to estimate a value for each of the following parameters.
# 
# * A rating `r_team` for each team.
# * A home field advantage rating `hfa`.
# * An additive constant `b`.
# 
# The standard power ratings for American football teams (at least the power ratings I know) can be immediately interpreted in terms of spreads in games.  The values here are less easy to interpet, but as a first impression, the higher the rating, the stronger the team is considered.

# ## Reshaping the data
# 
# It will be easier to get equations from our data if we reshape it, so that there is a column for each team.  Each game will also correspond to two rows: one row from the perspective of the home team, and one row from the perspective of the away team.

# In[5]:


df["date"] = pd.to_datetime(df["date"])

df_data1 = pd.get_dummies(df.home_team, dtype=np.int64) - pd.get_dummies(df.away_team, dtype=np.int64)
df_data1["HFA"] = 1
df_data1["const"] = 1
df_data1["days"] = (pd.to_datetime("2022-07-06")-df.date).dt.days
df_data1["Goals"] = df["home_score"]

df_data2 = -df_data1.copy()
df_data2["HFA"] = 0
df_data2["const"] = 1
df_data2["days"] = (pd.to_datetime("2022-07-06")-df.date).dt.days
df_data2["Goals"] = df["away_score"]

df_data = pd.concat([df_data1, df_data2])


# Here is the DataFrame we just made.  (The code above might be difficult to understand, but the same thing could be made for example using a for loop, where you iterate over each match from `df`.)

# In[6]:


df_data


# Let's focus on two of the rows, both from the same game.  In this game, Moldova was home against Romania.  (**Warning**. It's possible that in this dataset, some teams are incorrectly identified as home teams when it was at a neutral location.)
# 
# In the two rows displayed below, the top row is from Moldova's perspective.  The match was played on June 24, 2022, and the 12 in the "days" column refers to the number of days before the start of the Euro tournament.  This match ended 0-4, with Moldova losing to Romania.

# In[7]:


with pd.option_context('display.max_columns', None):
    display(df_data.loc[380])


# ## Setting up the data
# 
# In the eventual formula, we will always subtract the opponent's power rating, so there are not uniquely determined "best" power ratings (if you add 4 to each power rating, you will end up with the exact same predictions).  To normalize things, we will remove Albania from the computation (which is equivalent to forcing Albania to have a power rating of 0).
# 
# We will also remove the "days" column (because we will not solve for that), and we will remove the "goals" column, because that is our target.

# In[8]:


ind_cols = df_data.columns[1:-2]


# We are going to estimate a value for each of the following.

# In[9]:


ind_cols


# Our computation method is based on NumPy objects, not pandas DataFrames, so we take the desired columns from `df_data` and convert the result to a NumPy array.

# In[10]:


A = df_data[ind_cols].to_numpy()


# As our initial guess, we will initialize all of the values to 1.

# In[11]:


X = np.ones_like(ind_cols, dtype=np.float64)


# In[12]:


X


# We removed the "days" column and the "Goals" column from our DataFrame above.  Let's get access to these values, and again convert to NumPy objects.

# In[13]:


days = df_data["days"].to_numpy()
goals = df_data["Goals"].to_numpy()


# ## Computing the power ratings
# 
# Our computation is of the parameter values is based on Section 3.3 of [Prediction of the FIFA World Cup 2018](https://www.researchgate.net/publication/325683270_Prediction_of_the_FIFA_World_Cup_2018_-_A_random_forest_approach_with_an_emphasis_on_estimated_team_ability_parameters) by Groll, Ley, Schauberger, Eetvelde.  (**Warning**.  The point of that paper is to suggest a better approach than what we are doing here.)
# 
# The following function represents (the negative logarithm of) equation (5) from that section.  (The formula looks very complicated, but most of the complication comes from the Poisson distribution.  Getting from the Poisson distribution to this formula is not as difficult.)  We initially parametrized `X` to be an array of all 1s.  We want to find the value of `X` which minimizes the following.
# 
# The parameter `h` represents how much we emphasize recent games.  The smaller the value of `h`, the more we emphasize recent games.  This value of 60 is quite small.  Even a value like 1000 would be reasonable.  Among smallish values (less than one year, 365), 60 seemed to perform the best in my testing, but it is admittedly not a well-motivated choice.  Large values like 1000 also performed similarly well.  You could get a more "pure" rating and a slightly simpler equation by removing this dependence on days completely.

# In[14]:


h = 60

def log_likelihood(X):
    X1 = A.dot(X)
    return -((1/2)**(days/h)*(X1*goals - np.log(factorial(goals)) - np.exp(X1))).sum()


# With this setup, it is surprisingly easy to get parameter estimates for `X`.
# 
# We imported the `minimize` function from SciPy at the top of this notebook.  (Unfortunately I can't locate the blog post where I learned how to do this in Python.  I know [this article](https://journals.sagepub.com/doi/full/10.1177/1471082X18817650) by several of the same authors explicitly mentions the BFGS algorithm that we specify.)

# In[15]:


res = minimize(log_likelihood, X, method='BFGS', options={'disp': True})


# Here are the computed power ratings.  Notice that the home-field advantage parameter and the additive constant are mixed in with the country ratings.

# In[16]:


ratings = pd.Series(res.x, index=ind_cols).sort_values(ascending=False)


# In[17]:


ratings


# (example-computation)=
# ## Example computation
# 
# The above calculations are based on modeling the number of goals scored (by a single team) using a [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution).  To describe a Poisson distribution, all that is needed is to specify the mean $\lambda$.  The above power ratings relate to this $\lambda$ by the following formula, which is equation (4) in the above mentioned paper [Prediction of the FIFA World Cup 2018](https://www.researchgate.net/publication/325683270_Prediction_of_the_FIFA_World_Cup_2018_-_A_random_forest_approach_with_an_emphasis_on_estimated_team_ability_parameters).
# 
# $$
# \log(\lambda) = b+r_1-r_2+\text{hfa}.
# $$
# 
# Here is a quick example of how to use these ratings.  Say we want to estimate how many goals will be scored if the Netherlands plays at home against Denmark. 

# In[18]:


b = ratings["const"]
r1 = ratings["Netherlands"]
r2 = ratings["Denmark"]
hfa = ratings["HFA"]


# Here is our estimate for the expected number of goals scored by the Netherlands.

# In[19]:


lam_neth = np.exp(b+r1-r2+hfa)


# In[20]:


lam_neth


# The same thing for Denmark.  (In this computation, we leave out homefield advantage.)

# In[21]:


lam_dk = np.exp(b+r2-r1)


# In[22]:


lam_dk


# If instead we want to know, what is the probability that the Netherlands wins against Denmark at home, we can simulate for example 10,000,000 matches as follows.

# In[23]:


rng = np.random.default_rng(seed=0)

n = 10**7
scores = rng.poisson(lam=(lam_neth, lam_dk), size=(n,2))


# For example, in the first 3 matches in our simulation, the Netherlands won all three, with scores of 2-0, 7-1, and 3-0.

# In[24]:


scores[:3]


# Here is the probability corresponding to all ten million matches in the simulation.

# In[25]:


np.count_nonzero(scores[:,0] > scores[:,1])/n


# In[26]:


np.count_nonzero(scores[:,0] < scores[:,1])/n


# In[27]:


np.count_nonzero(scores[:,0] == scores[:,1])/n


# Our simulation suggests an 80% chance that the Netherlands wins, a 6% chance that Denmark wins, and a 14% chance that the match ends in a draw.

# ## Saving the ratings
# 
# We'll save the ratings in a csv file.

# In[28]:


ratings.to_csv("data/ratings.csv")

