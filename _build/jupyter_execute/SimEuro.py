#!/usr/bin/env python
# coding: utf-8

# # Simulation
# 
# We will use the group odds (which have nothing to do with our power ratings... they are implied from betting odds) to simulate the outcome of the group stage, and then we will use the power ratings to simulate the knockout stage.

# ## Group stage
# 
# These outcomes are based on the implied odds from the betting odds, as we computed [here](GetGroupOdds.ipynb).

# In[1]:


import pandas as pd
import numpy as np

rng = np.random.default_rng()


# In[2]:


df_odds = pd.read_csv("data/gp_odds.csv")


# In[4]:


df_odds.head(4)


# Here is an example of one simulation.

# In[5]:


gp_top2 = {}

for gp, df_sub in df_odds.groupby("Group"):
    gp_top2[gp] = rng.choice(df_sub["Team"], size=2, replace=False, p=df_sub["Implied"])


# In[6]:


gp_top2


# ## Simulate match
# 
# Once we know the simulated outcomes of the group stage, we will simulate individual matches as we progress through the knockout stage.  To simulate the group stage, we only used the betting odds.  For the individual matches, we will use a completely different strategy, and only use the power ratings computed in the [power ratings section](PowerRatings.ipynb).
# 
# For each pair of teams, we will use the method from the [](example-computation) section to estimate the probability that Team 1 defeats Team 2.  (For now, we will just ignore draws.)
# 
# In matches involving England, we will include the home-field advantage.  We will not use home-field advantage for any of the other teams.

# In[7]:


ratings = pd.read_csv("data/ratings.csv", index_col=0).squeeze("columns")


# Here are the five highest ranked teams.

# In[8]:


ratings[:5]


# In[9]:


match_probs = {}


# In[10]:


teams = df_odds["Team"].unique()


# In[11]:


from itertools import combinations


# In[90]:


match_dict = {}

b = ratings["const"]
hfa = ratings["HFA"]

n = 10**7

for pair in combinations(teams,2):
    r0 = ratings[pair[0]]
    r1 = ratings[pair[1]]
    h0 = 0
    h1 = 0
    if pair[0] == "England":
        h0 = hfa
    elif pair[1] == "England":
        h1 = hfa
    lam = (np.exp(b+r0-r1+h0), np.exp(b+r1-r0+h1))
    scores = rng.poisson(lam, size=(n,2))
    wins = np.count_nonzero(scores[:,0] > scores[:,1])
    losses = np.count_nonzero(scores[:,0] < scores[:,1])
    match_dict[pair] = (wins/(wins+losses), losses/(wins+losses))


# As a reality check, let's check the probability that Denmark beats Austria.  In that order, the tuple doesn't appear.

# In[91]:


match_dict[("Denmark", "Austria")]


# So we use the opposite order.

# In[92]:


match_dict[("Austria", "Denmark")]


# The above is saying that Denmark has about a 20% chance of beating Austria.  (Because of the randomness in these simulations, the results might change when this is run again.)

# For convenience, we'll make a new dictionary `full_matches` that also contains each "reversed" tuple.

# In[93]:


full_matches = match_dict.copy()

for tup in match_dict.keys():
    res = match_dict[tup]
    full_matches[(tup[1], tup[0])] = (res[1], res[0])


# In[94]:


full_matches[("Denmark", "Austria")]


# We'll save the results so we don't need to make the computation again.

# In[125]:


pd.Series(full_matches).to_csv("data/match_probs.csv")


# ## Knockout stage
# 
# Here is the tournament bracket, taken from [Wikipedia](https://en.wikipedia.org/wiki/UEFA_Women%27s_Euro_2022#Knockout_stage).
# 
# ![Tournament bracket](images/knockout.png)

# Let's see how to feed the results of the group stage into this sort of bracket.  Since there are only 8 teams, we'll just specify the matchups manually.  For example, "C0" refers to the winner of Group C and "C1" refers to the runner-up in Group C.

# In[96]:


bracket_abbr = "C0D1A0B1D0C1B0A1"
bracket_str = [bracket_abbr[i:i+2] for i in range(0,len(bracket_abbr),2)]


# In[97]:


bracket


# In[98]:


gp_top2


# In[99]:


# s is a length-2 string like "C0"
def get_team(s, gp_top2):
    letter, place = s
    return gp_top2[f"Group {letter}"][int(place)]


# In[100]:


get_team("C0", gp_top2)


# In[101]:


bracket = [get_team(s, gp_top2) for s in bracket_str]


# In[102]:


bracket


# We can get the probabilities associated to one of these matches using our `full_matches` dictionary from above. 

# In[107]:


match = (bracket[0], bracket[1])


# In[108]:


match


# In[109]:


full_matches[match]


# If we want to simulate this match 10 times using those probabilities, we can use `rng.choice`.  (In our overall simulation, we will only simulate such a map once per bracket.  This example is just to show how the weights work.)

# In[110]:


rng.choice(match, size=10, p=full_matches[match])


# Here is an example of finding the winner after the knockout stage.

# In[111]:


while len(bracket) > 1:
    temp_list = []
    for i in range(0,len(bracket),2):
        match = (bracket[i], bracket[i+1])
        winner = rng.choice(match, p=full_matches[match])
        temp_list.append(winner)
    bracket = temp_list
    
print(f"The winner is {bracket[0]}")


# ## Full simulation
# 
# Here we put together the two steps above.  First we run the simulation 1 time and time how long it takes.

# In[112]:


get_ipython().run_cell_magic('time', '', '\ngp_top2 = {}\n\nfor gp, df_sub in df_odds.groupby("Group"):\n    gp_top2[gp] = rng.choice(df_sub["Team"], size=2, replace=False, p=df_sub["Implied"])\n    \nbracket = [get_team(s, gp_top2) for s in bracket_str]\n\nwhile len(bracket) > 1:\n    temp_list = []\n    for i in range(0,len(bracket),2):\n        match = (bracket[i], bracket[i+1])\n        winner = rng.choice(match, p=full_matches[match])\n        temp_list.append(winner)\n    bracket = temp_list\n    \nprint(f"The winner is {bracket[0]}")\n')


# Here we simulate the tournament two million times.  We have repeated this a few times, and the results come out very similar each time, so there is probably not much to be gained by running the simulation more times.

# In[127]:


n = 2*10**6

results_dict = {team:0 for team in teams}

for _ in range(n):
    gp_top2 = {}

    for gp, df_sub in df_odds.groupby("Group"):
        gp_top2[gp] = rng.choice(df_sub["Team"], size=2, replace=False, p=df_sub["Implied"])

    bracket = [get_team(s, gp_top2) for s in bracket_str]

    while len(bracket) > 1:
        temp_list = []
        for i in range(0,len(bracket),2):
            match = (bracket[i], bracket[i+1])
            winner = rng.choice(match, p=full_matches[match])
            temp_list.append(winner)
        bracket = temp_list
        
    results_dict[bracket[0]] += 1
    
# sorting by probability
prob_dict = {team: (win/n) for team, win in sorted(results_dict.items(), key=lambda x: x[1], reverse=True)}


# In[128]:


prob_dict


# ## Convert to odds
# 
# So that the numbers are easier to compare to betting odds, we display what would be fair odds according to the simulation for each team to win the tournament.
# 
# **Warning**.  The England odds are so different from the current (as of July 3rd) market odds, that our simulation is certainly over-estimating England's chances.

# In[123]:


def prob_to_odds(p):
    if p < .000001:
        return np.nan
    if p > .999999:
        return np.nan
    if p > 0.5:
        x = 100*p/(p-1)
        return f"{x:.0f}"
    elif p <= 0.5:
        x = 100*(1-p)/p
        return f"+{x:.0f}"


# In[130]:


pd.Series({team:prob_to_odds(prob) for team,prob in prob_dict.items()})

