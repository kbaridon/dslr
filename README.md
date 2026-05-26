First, we need to create a venv:
```shell
python -m venv .venv
source .venv/bin/activate
pip install argparse pandas matplotlib
```

Here is some explanations:

## 1. describe

> The goal with describe is to reproduce the pandas describe function, we will just do basics maths (mean, std, min, max...)

## 2. Histogram

> The goal is to print with matplotlib the distribution of grades between each houses in each course.

## 3. Scatter plot

> This time, we aim to spot a correlation between two courses (if a student have a good grade in x course, he (almost) always have a bad or good grade in another specific course)

## 4. Pair plot

> This time, we will print all scatter plots and all histograms, it will help us to comprehend which courses will have a bigger impact on the result. (ex: if you have good results in flying, you are most likely to be Gryffindor)

# Main subject: Logistic Regression (OvR)

> A Logistic Regression One vs Rest means:

We will do 4 logistic regressions, each time, we will say if:
- Is {student} in {house} or in {3 other houses}.

--> We will get a probability (ex: 0.3 Gryffindor, 0.7 not Gryffindor).

Then, we will do this for each houses, and we will take the result with the highest probability as the prediction.

> To get this probability, we will use a gradient descent, here is a graph to understand how it works:

- Gradient Descent is a method that starts with an initial value (often random). At each step, the direction in which the error increses the most (the gradient) is calculated, and the parameters are adjusted in the opposite direction. The goal is to minimize the lost (or cost here in the graph). The size of the step is controlled by the learning rate. This process is repeated until the loss barely decreases anymore (epsilon) or when then umber of epochs is exceeded.

![Gradient Descent](https://editor.analyticsvidhya.com/uploads/97106gd4.jpeg)

# Bonuses

Here are all the bonuses done:

- Add more fields to describe.py: unique, top, freq and dtype.
- Stochastic gradient descent
- Batch GD (default)
- mini-batch GD
- Early stopping