#!/usr/bin/env python
# coding: utf-8

# # Group odds from betting lines

# ## Getting the odds
# 
# Here is an example of getting odds to win groups from Bet Online using Beautiful Soup and regular expressions.  The html is saved in a file "bo73.html".

# In[1]:


import pandas as pd
import re

from bs4 import BeautifulSoup


# In[2]:


with open("bo73.html", "r") as f:
    html = f.read()


# In[3]:


soup = BeautifulSoup(html)


# Using the "Inspect Elements" option in Chrome, we find the css class that contains the headings we're interested in: `"offering-contests__table-description-text"`

# In[4]:


divs = soup.find_all("div", class_="offering-contests__table-description-text")


# Here is an example of what is in this list of divs.  We will just focus on the four group betting sections.

# In[5]:


for div in divs:
    print(div.text)


# In[6]:


group_divs = [div for div in divs if "Group Betting" in div.text]


# We'll use regular expressions to get the corresponding group letter.  (This is probably overkill, but we'll use regular expressions throughout, so this is just practice.)
# 
# Here is an example with just one of the objects.

# In[7]:


div = group_divs[1]
re.search("Group [A-D]", div.text)[0]


# The element `div` is a Beautiful Soup element, and we can use Beautiful Soup to find the next sections within the html code after these group divs.

# In[8]:


type(div)


# In[9]:


odds_raw = div.find_next().text


# In[10]:


odds_raw


# We get rid of the initial time portion.

# In[11]:


re.search("\d\d:\d\d (?:AM|PM)", odds_raw)


# In[12]:


re.search("\d\d:\d\d (?:AM|PM)", odds_raw).span()[1]


# In[13]:


odds2 = odds_raw[re.search("\d\d:\d\d (?:AM|PM)", odds_raw).span()[1]:]


# In[14]:


odds2


# I've never met anyone who can read regular expressions comfortably, so don't worry if the following looks very difficult to understand.

# In[15]:


matches = re.finditer("(?P<Team>[A-Za-z ]+)(?P<Odds>[+-]\d+)", odds2)


# In[16]:


for match in matches:
    print(match["Team"])
    print(match["Odds"])


# We can use this method to get the odds for the various groups.  We will put the results into a DataFrame.

# In[17]:


odds_list = []
for div in group_divs:
    group = re.search("Group [A-D]", div.text)[0]
    odds_raw = div.find_next().text
    odds2 = odds_raw[re.search("\d\d:\d\d (?:AM|PM)", odds_raw).span()[1]:]
    matches = re.finditer("(?P<Team>[A-Za-z ]+)(?P<Odds>[+-]\d+)", odds2)
    for match in matches:
        team = match["Team"].strip()
        odds = match["Odds"].strip()
        odds_list.append((group, team, odds))


# In[18]:


df_odds = pd.DataFrame(odds_list, columns=["Group", "Team", "Odds"])


# In[19]:


df_odds


# ## Converting to probabilities
# 
# We convert the odds into break even probabilities, and then rescale those probabilities for each individual group so that the probabilities sum to 1.

# In[20]:


def odds_to_prob(s):
    x = int(s)
    if x < 0:
        y = -x
        return y/(100+y)
    else:
        return 100/(100+x)


# In[21]:


odds_to_prob("-110")


# In[22]:


odds_to_prob("+300")


# In[23]:


df_odds["Prob"] = df_odds["Odds"].map(odds_to_prob)


# In[24]:


gp_totals = df_odds.groupby("Group").sum()["Prob"]


# For example, if you add up all the break-even probabilities for Group C, the total is approximately 1.09.

# In[25]:


gp_totals


# We rescale the probabilities to get the implied probabilities.

# In[26]:


df_odds["Implied"] = df_odds["Prob"]/df_odds["Group"].map(lambda x: gp_totals[x])


# In[27]:


df_odds


# As a reality check, let's make sure the implied probabilities always sum to 1.

# In[28]:


df_odds.groupby("Group").sum()["Implied"]


# In[29]:


df_odds.Team


# Let's save these results.  We will also record the site and the date.

# In[30]:


df_odds["Site"] = "Bet Online"
df_odds["Date"] = "2022-07-03"


# In[31]:


df_odds.to_csv("data/gp_odds.csv", index=False)

