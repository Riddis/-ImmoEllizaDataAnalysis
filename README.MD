# Immo Eliza Data Analysis

This challenge was done by Sam Hoste as a part of the BEcode AI Data Engineer bootcamp. 

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sam-hoste-15610945/)

We will be using the data scraped in part 1 of the project (https://github.com/nikolaaswillaert/ImmoEliza)

## The Mission

The real estate company "ImmoEliza" wants to establish itself as the biggest one in all of Belgium. To pursue this goal, it needs to create a machine learning model to predict prices on Belgium's sales. That way, they can pick out the properties that are the most valuable to them.

But for this, it needs to do a preliminary analysis to gather some information. 

We will be cleaning up the data and drawing conclusions from it.

After cleaning and analysing the dataset we will create a machine learning model to predict prices on Belgium's real estate sales.

We will then write an API for this model and deploy it in a Docker container. 

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
![Static Badge](https://img.shields.io/badge/pandas-9ef542) 
![Static Badge](https://img.shields.io/badge/numpy-24d3f2) 
![Static Badge](https://img.shields.io/badge/pathlib-e8f224)
![Static Badge](https://img.shields.io/badge/joblib-e32e0e)
![Static Badge](https://img.shields.io/badge/fastapi-8A2BE2)
![Static Badge](https://img.shields.io/badge/jsonschema-lightgrey)
![Static Badge](https://img.shields.io/badge/Requests-red)
![Static Badge](https://img.shields.io/badge/scikit_learn-orange)
![Static Badge](https://img.shields.io/badge/xgboost-yellow)

You can run the following code to easily install all requirements: 

```bash
  pip install -r requirements.txt
```

Open the notebook for the analyses.

run main.py to train the models and see the scores.

### Duration

I spent 2 working days on cleaning the data. 

Plotting the graphs and drawing conclusions took me another 1.5 working days. 

Building the initial pipeline for the model training took me an hour. 

I then spent the rest of the time (4 working days) on adding different regression models and optimising the training results. 

Building the API took 2 days. 

I spent the first day writing the code, and the second solving errors and finalizing the output.

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

After analyzing the data, it's finally time to interpret our results. I have to communicate my analysis using simple words and a table or graph, then use the results to decide on my best course of action.


##### Step 4: Data formatting

Now that the dataset is ready, I have to format it for machine learning:

- I divide my dataset for training and testing. (`X_train, y_train, X_test, y_test`)
- If needed, I apply scaling to my training data, [read more](https://www.digitalocean.com/community/tutorials/normalize-data-in-python)

##### Step 5: Model selection

The dataset is ready. Now, let's select a model.

We started out with the simplest LinearRegression. 
After succesfully building a pipeline I also started playing with other models: 

* [DecisionTreeRegressor (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor) - Fits a set of if-else rules to the data, [read more](https://quantifyinghealth.com/regression-tree-vs-linear-regression/).
* [XGBoost (xgboost)](https://www.geeksforgeeks.org/xgboost-for-regression/), another optimized tree based algorithm.
* [Stochastic Gradient Descent (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor) - A different implementation of gradient descent.
* [Neural Network (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) - A neural network.

##### Step 6: Train your model

I trained my model on my data.

The results were pretty consistent, except with the DecisionTree model. 

##### Step 7: Model evaluation

I am getting decent results but they could be better. 

Due to data scarcity on some features I'm having trouble improving it further. 

I saw pretty high scores without optimisation thanks to a very clean dataset, but I did not see as much improvement by optimising as I would have expected. 

##### Step 8: Building the API

I started building an API so we can put our model to use. 

For this project, I am using FastAPI. 

The API will have 2 routes: 

One GET route which will specify the format needed for the POST route. 

One POST route that will return a price prediction.

##### Step 9: Creating a docker image



##### Step 10: Deploying to Render

