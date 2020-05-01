# Explaining Aggregate for Large-Scale Data Exploration

This is a repository to accompany paper : [*Explaining Aggregates for Exploratory Analytics*](https://arxiv.org/abs/1812.11346)

![alt text](https://i.imgur.com/EI1aMBo.png)

The goal is to build regression functions that can explain how aggregates (such as AVG/COUNT/SUM) change with respect to different 
inputs. By treating the aggregate functions used in SQL queries as black box functions we approximate their behavior using well-known 
regression functions and observations (ie past executed queries). This is shown in the figure above.

We can then used these local regression functions to explain data subspaces that indicate which regions in a data set might be more important.

![alt text](https://i.imgur.com/sWP0NnX.png)
