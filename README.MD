# Immo Eliza Data Analysis

This challenge was done by Sam Hoste as a part of the BEcode AI Data Engineer bootcamp. 
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sam-hoste-15610945/)

We will be using the data scraped in part 1 of the project (https://github.com/nikolaaswillaert/ImmoEliza)

## Mission objectives

- Be able to use `pandas`.
- Be able to use Data visualization libraries.(`matplotlib` or `seaborn`).
- Be able to clean a dataset for analysis.
- Be able to use colors in visualizations correctly.
- Be able to establish conclusions about a dataset.
- Be able to find and answer creative questions about data.
- Be able to think outside the box.

## The Mission

The real estate company "ImmoEliza" wants to establish itself as the biggest one in all of Belgium. To pursue this goal, it needs to create a machine learning model to predict prices on Belgium's sales. That way, they can pick out the properties that are the most valuable to them.

But for this, it needs to do a preliminary analysis to gather some information. 

We will be cleaning up the data and drawing conclusions from it. 

### Requirements
#### Run Locally

Clone the project

```bash
  git clone https://github.com/Riddis/-ImmoEllizaDataAnalysis
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

![python version](https://img.shields.io/badge/python-3.10.6+-blue) 
![Static Badge](https://img.shields.io/badge/matplotlib.pyplot-8A2BE2) 
![Static Badge](https://img.shields.io/badge/seaborn-e32e0e) 
![Static Badge](https://img.shields.io/badge/pandas-9ef542) 
![Static Badge](https://img.shields.io/badge/numpy-24d3f2) 
![Static Badge](https://img.shields.io/badge/pathlib-e8f224)



Open the notebook

### Duration

I spent 2 working days on cleaning the data. 
Plotting the graphs and drawing conclusions took me another 1.5 working days. 

### Must-have features

##### Step 1 : Data Cleaning

- No duplicates
- No blank spaces (ex: `" I love python "` => `"I love python"`)
- No errors
- No empty values

I started out by dropping all empty rows and duplicates. 
After that, I looked at each column and decided how to clean it up on a case by case basis. 

This exercise was mostly about representing the data so I was very rough with the cleaning. We are making almost no assumptions and drop the row if we're missing data. 

The exception is the surface_land property for appartments, I assumed this was equal to the living_area property. 

I started converting strings like SEMI_EQUIPPED to integers to prepare for the model training. 

In the end, only 7.000 rows remain out of the almost 20.000 we started with.

##### Step 2 : Data Analysis

Now that the data has been collected and cleaned, it is time for the analysis using tools such as `matplotlib`/`seaborn`/`plotly`!

Answer the following questions with a vizualization if appropriate:

- How many rows and columns?
- What is the correlation between the variables and the price? (Why might that be?)
- How are variables correlated to each other? (Why?)
- Which variables have the greatest influence on the price?
- Which variables have the least influence on the price?
- How many qualitative and quantitative variables are there? How would you transform these values into numerical values?
- Percentage of missing values per column?

##### Step 3 : Data Interpretation

After analyzing the data, it's finally time to interpret our results. I have to communicate my analysis using simple words and a table or graph, then use the results to decide on your best course of action.

I try to answer the following points: 

- Plot the outliers.
- Which variables would you delete and why ?
- Represent the number of properties according to their surface using a histogram.
- In your opinion, which 5 variables are the most important and why?
- What are the **most** expensive municipalities in Belgium? (Average price, median price, price per square meter)
- What are the **most** expensive municipalities in Wallonia? (Average price, median price, price per square meter)
- What are the **most** expensive municipalities in Flanders? (Average price, median price, price per square meter)
- What are the **less** expensive municipalities in Belgium? (Average price, median price, price per square meter)
- What are the **less** expensive municipalities in Wallonia? (Average price, median price, price per square meter)
- What are the **less** expensive municipalities in Flanders? (Average price, median price, price per square meter)