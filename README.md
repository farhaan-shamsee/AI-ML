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




