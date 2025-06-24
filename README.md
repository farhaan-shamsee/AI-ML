# AI ML

## Day 1

* supervised learning: 

### Use cases

• disease predction
    ○ based on diff sym of person…we predict if person is sufering from the disease
    ○ models will predict the probability if u r going to have the disease
    ○ based on previous blodd test etc.
• Fintech domain
    ○ credit card industry
    ○ types of transaciotn to detect fraud or not
    ○ looking at diff fetures of transaction to check if it is fraud
    ○ fraud detection
cyber security

![ai-ml-venn-diagram](https://github.com/user-attachments/assets/5e6cd873-17a4-4b51-99f9-bc275dfc68ef)


### Mean and Median

1. Mean and median
1. summary statistics, descriptive statistics
1. percentiles, outlier
1. exploratory data analysis (EDA)
   1. looking at data sets and figuring out patterns
1. MEAN: average of salaries
1. MEDIAN: central value of data
1. sort data in ascending and look at middle value

1. Outlier: data point that is significantly different from others
   1. If there is outlier, it can skew the mean
   2. Median is more robust to outliers
   3. Never trust the mean/averages. For example, never trust the average salary of the company.
   4. ![mean-median](https://github.com/user-attachments/assets/b06c6208-d2a1-4103-b4c5-c8ecdb4b7870)

1. Histogram: bar chart that shows the distribution of data
   1. X-axis: range of values
   2. Y-axis: frequency of values
   3. ![histogram](https://github.com/user-attachments/assets/beb28768-66e9-48a9-b44e-97300f572cc2)
      1. this is a right skewed distribution
      2. happens when there are outliers on the right side
   4. ![histogram](https://github.com/user-attachments/assets/33a5d34d-872e-4dd5-bd16-83f0aa8c09e8)
      1. this is a left skewed distribution
      2. happens when there are outliers on the left side
      3. most of students have secured high marks and less number of students have secured low marks
   5. Why skew is important?
      1. it make the ML biased
   6. Uni variate analysis
      1. able to look at a distribution of a single variable
      2. histogram is a good way to visualize this
   7. Bi variate analysis
      1. able to look at a distribution of two variables
      2. For example: country wise per capita income
   8. transform skewness: transform the data to make it more normal distribution
  
### Examples

1. google gorilla blunder
   1. google gorilla blunder: google image search for "gorilla" returned images of humans.
   2. original data curation was wrong
   3. reason: the model was trained on a dataset that had a lot of images of humans of a specific skin tone

### data labelling

1. data labelling: process of adding labels to data
2. scale.ai, appen

### pearson correlation coefficient

Correlation is a statistical measure that describes the strength and direction of a relationship between two variables. It is represented by a value between -1 and 1.

Correlation doest not imply causation. It only indicates that there is a relationship between two variables.

![image](https://github.com/user-attachments/assets/b5f0db4c-e0c5-4b06-9ae8-f0acde6956e5)


1. Direction: +ve or -ve
   1. +ve: as one variable increases, the other variable also increases
   2. -ve: as one variable increases, the other variable decreases
2. Magnitude: how strong the relationship is
   1. strong relation between temp and ice cream sales, this is the inference.
   2. closer the points stronger the correlation
   3. more spread out the points weaker the correlation. for example in spending and income.
   4. For example: ice cream sales and temperature
      1. 0.9: it shows strong positive correlation
   5. For example: temp and jacket sales
      1. -0.9: it shows strong negative correlation
   6. For example: temp and ice cream sales
      1. 0.4: it shows weak positive correlation
   7. For example: age and health
      1. -0.2: it shows weak negative correlation


### Workflow of ML

* what it means to train a model

![image](https://github.com/user-attachments/assets/92dda6cf-0706-407a-b8da-e548bf0c595a)

1. Training accuracy: how well the model performs on the training data
2. Testing accuracy: how well the model performs on the testing data
3. Once we are confident that the model is performing well, we can deploy it to production.


#### Pairplot

![image](https://github.com/user-attachments/assets/0a99cfb7-9458-4fa1-ace7-d187eb9db7ad)

![image](https://github.com/user-attachments/assets/b36e44b2-be1a-4ae5-a2d9-014959e67971)


1. Pairplot is a way to visualize the relationship between multiple variables in a dataset.
2. It creates a grid of scatter plots, where each scatter plot shows the relationship between two variables.
3. It is useful for exploring the relationships between multiple variables in a dataset.
4. It can also be used to identify patterns and trends in the data.
5. Non diagonal are the scatter plots.
6. Explain diagram:
   1. Out of the 3 inputs which are the inputs are strongest predictor of sales. TV is best, Radio is second best, newspaper is worst.
   2. Tv has best correlation with sales.
7. Pairplot can also use to identify outliers in the data.
8. Diagonals can be studied to find skewness of the data.
9. **Multi collinearity**
   1. When two or more independent variables(features/inputs) are highly correlated with each other, it can cause problems in the model.
   1. It can lead to overfitting and make the model less interpretable.
   1. It can also lead to multicollinearity, which can cause problems in the model.
   1. Pairplot can be used to identify multicollinearity in the data.
   1. If two variables are highly correlated, it can be seen in the pairplot as a straight line.
   1. If two variables are not correlated, it can be seen in the pairplot as a scatter of points.


```pyton
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=1)
from sklearn.linear_model import LinearRegression
model2 = LinearRegression() 
model2.fit(x_train, y_train)
print("Model Parameters")
print(model2.coef_)
print(model2.intercept_)
#With TV, Radio, Newspaper we get a more overfit model
print("Training R2")
print(model2.score(x_train,y_train))
print("Testing R2")
print(model2.score(x_test,y_test))
```


* Feature selection
* Feature engineering

### Models

![image](https://github.com/user-attachments/assets/60bc7cf8-2fee-475c-8d3e-7af0e59ed47a)

* Overfit model
  * model is too complex and fits the training data too well
  * it performs poorly on the testing data
  * it has high variance and low bias
  * it can be identified by looking at the training and testing accuracy
* Underfit model
  * model is too simple and does not fit the training data well
  * it performs poorly on both training and testing data
  * it has high bias and low variance
  * it can be identified by looking at the training and testing accuracy
* Good fit model
  * model is just right and fits the training data well
  * it performs well on both training and testing data
  * it has low bias and low variance
  * it can be identified by looking at the training and testing accuracy
  
* The training accuracy and testing accuracy should be close to each other. For example, if the training accuracy is 0.9 and the testing accuracy is 0.8, it is a good fit model. Like the system is behaving on both training and testing data. But if training was low and testing is high, it is an underfit model. If training is high and testing is low, it is an overfit model.

Example: if we did not learn anything but performed well in tests, then it is dangerous.  Then it means we memorized the training data and did not learn anything.


### Regression Learning

what is the line that best fits the points in the scatter plot?
the best fit line might not always pass through the points, but it should be as close to the points as possible.

![image](https://github.com/user-attachments/assets/3b6e45a1-4958-47fa-8633-2b7e7339291c)

* RandomForestRegressor
  * it is an ensemble method that combines multiple decision trees to make predictions
  * it is used for regression tasks
  * it can handle non-linear relationships between the features and the target variable
  * it can also handle missing values and categorical variables
  * it is less prone to overfitting than a single decision tree.
 
## Day 2

### Data science lifecycle

![image](https://github.com/user-attachments/assets/dbea31bd-4d7d-4477-b43a-6dd9712dd89b)

