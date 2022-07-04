# UEFA Women's Euro 2022

**Warning**.  These notebooks are meant for practice with sports models using Python.  We haven't done any fine-tuning of these results, and they should only be considered as a demonstration of the methods.  When there are large discrepancies between these results and the market odds, the market odds should be assumed correct!

There are three notebooks.

* The [Power ratings](PowerRatings.ipynb) notebook shows how to use the `optimize` function from SciPy to compute (one version of) power ratings, based on a Poisson distribution.
* The [Group odds](GetGroupOdds.ipynb) notebook shows an example of scraping an html file using Beautiful Soup to get betting odds, and then converting those betting odds into implied probabilities.
* The [Simulation](SimEuro.ipynb) notebook combines the results from the other sections and shows the results of simulating the tournament 2,000,000 times.

The simulation suggests the following are fair odds to win the 2022 Women's Euro.

| Team             | Odds      |
|------------------|-----------|
| England          | -104      |
| Spain            | +294      |
| Sweden           | +1234     |
| Netherlands      | +1899     |
| Italy            | +2913     |
| Belgium          | +4231     |
| Norway           | +5479     |
| Germany          | +8898     |
| Iceland          | +9613     |
| France           | +13774    |
| Austria          | +19663    |
| Switzerland      | +54680    |
| Denmark          | +531815   |
| Northern Ireland | +4081533  |
| Finland          | +12499900 |
| Portugal         | NaN       |