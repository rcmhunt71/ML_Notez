## **Basis Documents**:
#### **Primary:** [Machine Learning for Humans](https://medium.com/machine-learning-for-humans/why-machine-learning-matters-6164faf1df12)
#### **Secondary:** [Supervised Machine Learning](https://onclick360.com/supervised-machine-learning)

## **Definitions:** (In order of increasing complextity)
* **ANI** - Artificial Narrow Intelligence -   effectively perform a narrowly defined task.
* **AGI** - Artificial General Intelligence (Strong AI) -  The definition of an AGI is an artificial intelligence that can successfully perform any intellectual task that a human being can, including learning, planning and decision-making under uncertainty, communicating in natural language, making jokes, manipulating people, trading stocks, or… reprogramming itself.
* **ASI** - Artificial Super Intelligence


# Types of Learning
* **Supervised Learning** - Start with data set containing training examples with associated correct labels. Take a set of known data nad the corresponding answer, evaluate and statistically determine the differences/deviations. Then try on unknown data to see accuracy of model (check-reading is a good example).
* **Unsupervised Machine Learning Algorithms**
* **Reinforcement Machine Learning Algorithms**


## Supervised Learning
:: *Taken from*: [Machine Learning for Humans, Part 2.1: Supervised Learning](https://medium.com/machine-learning-for-humans/supervised-learning-740383a2feab)


| BASIS FOR COMPARISON | CLASSIFICATION | REGRESSION |
| -------------------- | -------------- | ---------- |
| Basic | The model or algorithm when the mapping of data is done into predefined classes(labeled). | The model in which the mapping of data is done into values. |
| Includes prediction of Algorithms | Discrete data values logistic regression, SVC, Decision tree, etc. | Continuous data values Linear regression, Regression tree |
| Nature of the predicted data | Unordered | Ordered |
| Mode of calculation | Measuring accuracy | Measurement of root mean square (RMS) error |

### REGRESSION
* Regression - predict a continuous numerical value (Y) based X (Tensor). --> How much will that house sell for?

* *Possible Algorithms:*
  1. Linear Regression (LR)
  1. Logistic Regression
  1. Polynomial Regression
  1. Random forest
  1. Stepwise Regression
  1. Support Vector Regression (SVR)
  1. Ridge Regression
  1. Lasso Regression

#### *LINEAR REGRESSION*
1. **CONTINUOUS** means there aren't any gaps (discontinuities) in the value that Y can assume; versus **DISCRETE** variables, on the other hand, can only take on a finite number of values.

1. _**X**_ can be a TENSOR with an any number of dimensions.
   * A 1D tensor is a vector (1 row, many columns),
   * A 2D tensor is a matrix (many rows, many columns), and
   * Tensors can have 3, 4, 5 or more dimensions (e.g. a 3D tensor with rows, columns, and depth).
   * **Review**: [tensor review](http://www.deeplearningbook.org/contents/linear_algebra.html)

1. **DATA**
  Your input data X includes all relevant information about individuals in the data set that can be used to predict income:
   * Features of the data:
     * numerical (e.g.  - years of work experience)
     * categorical (e.g. - job title or field of study)
     * **NOTE**: More data = more accurate model (obvious)

1. **DATA SETS**
   * **Training data** = Labeled data to build model
   * **Test data** = unlabeled data to validate model

#### ALGORITHMS
 * **Linear Regression** (Ordinary Least Squares, OLS)
   + LR is a parametric method --> it makes an assumption about the form of the function relating X & Y.

   ![alt_text][OLS_Cost]

   * TO FIND THE BEST PARAMETERS:
     1. Define a cost function, or loss function, that measures how inaccurate our model’s predictions are.
     2. Find the parameters that minimize loss, i.e. make our model as accurate as possible.

**NOTE**: Dimensionality: our example is two-dimensional for simplicity, but you’ll typically have more features (x’s) and coefficients (betas) in your model; e.g. when adding more relevant variables to improve the accuracy of your model predictions. The same principles generalize to higher dimensions, though things get much harder to visualize beyond three dimensions.

        OLS = ordinary least squares
        COST = R^2 = SUM((value of y from model - actual y)^2)/2n
        COST_LR = (SUM[(i=1,n)((β1x_i + β0) - y_i))^2]/2*n =

For a simple problem like this, we can compute a closed form solution using calculus to find the optimal beta parameters that minimize our loss function. But as a cost function grows in complexity, finding a closed form solution with calculus is no longer feasible.
     * This is the motivation for an iterative approach called gradient descent, which allows us to minimize a complex loss function.


## **GRADIENT DESCENT**
The goal of gradient descent is to find the minimum of our model’s loss function by iteratively getting a better and better approximation of it.

![alt text][gradient_image]

[Code implementation of gradient descent in Python](https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/)

 The function is 
 
    f(β0, β1) = z
    
so to begin gradient descent, you make some guess of the parameters β0 and β1 that minimize the function.
  + Next, you find the partial derivatives of the loss function with respect to each beta parameter: [dz/dβ0, dz/dβ1].
    + A partial derivative indicates how much total loss is increased or decreased if you increase β0 or β1 by a very small amount.

## **OVERFITTING**
* Overfitting happens when a model overlearns from the training data to the point that it starts picking up idiosyncrasies that aren’t representative of patterns in the real world.

* Underfitting is a related issue where your model is not complex enough to capture the underlying trend in the data.

**EXAMPLE:**
  1. _OVERFITTING_: “Sherlock, your explanation of what just happened is too specific to the situation.”
  1. _REGULARIZATION_: “Don’t overcomplicate things, Sherlock. I’ll punch you for every extra word.”
  1. _HYPERPARAMETER_ (λ): “Here’s the strength with which I will punch you for every extra word.”

### **BIAS-VARIANCE TRADE-OFF**
**Bias** is the amount of error introduced by approximating real-world phenomena with a simplified model.

**Variance** is how much your model's test error changes based on variation in the training data. It reflects the model's sensitivity to the idiosyncrasies of the data set it was trained on.

As a model increases in complexity and it becomes more wiggly (flexible), its bias decreases (it does a good job of explaining the training data), but variance increases (it doesn't generalize as well).

**KEY TAKEAWAY**
 
     Ultimately, in order to have a good model, you need one with low bias and low variance.

![alt text][overfit_image]

 ## **Combatting Overfitting**
   + Use more training data.
   + Use regularization: add in a penalty in the loss function for building a model that assigns too much explanatory power to any one feature or allows too many features to be taken into account.

![alt text][regularization_image]

         
E.g. determine the cost function (LR), but add a regulation term to the equation: 

    R(x:xO) = λ * SUM[(βi)^2]
          
  * where lambda (λ) is a hyper-parameter: a general setting of your model that can be increased or decreased (i.e. tuned) in order to improve performance.
    * A higher λ value will more harshly penalize large beta coefficients that could lead to potential overfitting.
    * To decide the best value of λ, you’d use a method called cross-validation which involves holding out a portion of the training data during training, and then seeing how well your model explains the held-out portion.

**ADDITIONAL INFORMATION**: [Gradient Descent - from a mathematical POV](https://eli.thegreenplace.net/2016/understanding-gradient-descent/)
        
Gradient descent is a standard tool for optimizing complex functions iteratively within a computer program. Its goal is: given some arbitrary function, find a minimum. For some small subset of functions - those that are convex - there's just a single minimum which also happens to be global.

#### BACJKGROUND MATHEMATICS REVIEW
  * Dot-product = multiplying the magnitude of the vectors to determine the contributions of the magnitudes
  
           a . b = |a| * |b| * cos(T)
  
  * Cross Products = finding an orthogonal vector that is the product of the magnitude
           
           given 2 vectors a & b:
                a X b = |a| * |b| * sin(T) * n, where n is the unit vector orthogonal to both a & b.

    + For 3 dimensions, _**IF**_ the points are based from the origin (which is the case in absolute coordinate systems)
           
           given a=(x1,y1,z1) & b=(x2,y2,z2):
               c_x = ay*bz - az*by
               c_y = az*bx - ax*bz
               c_z = ax*by - bx*ax
           Therefore a x b = (c_x, c_y. c_z)



   * (Chain Rule)[https://eli.thegreenplace.net/2016/the-chain-rule-of-calculus/]:
   
           given a function h(x) --> Decompose into h(x) = f(X)*g(x)
              The derivative of H(x): h'(x) = g'(f(x))*f'(x)
              
           This allows for stepwise differentiation, which is iterative in nature, hence can be programmed.

#### Classification - assign a label.  --> Is that a cat or a dog?
    ::Taken from: [Machine Learning for Humans: Supervised Learning 2](https://medium.com/machine-learning-for-humans/supervised-learning-2-5c1c23f3560d)

The main goal of **classification** is to predict the target class (Yes/No), in which the algorithm effort to label each data by selecting between two or more, unlike classes.
  + In a _binary_ classification model selecting between two classes such as finding whether an email is spam or not, or whether or not person will loan defaulter, predict whether the student will pass or fail, predict whether the customer will buy the new product or not, the patient has cancer or not, image contains a dog or not have only two possible outcomes (Yes/ No)
  + Classification algorithms are applied whenever the desired output is separated by the label. Many use cases, image and audio categorization, customer segmentation, and text analysis for mining customer sentiment.

Classification predicts a discrete target label Y. Classification is the problem of assigning new observations to the class to which they most likely belong, based on a classification model built from labeled training data.

## **Possible Algorithms**
  + Logistic Regression
  + Decision Trees
  + K Nearest Neighbors (K-NN)
  + Naive Bayes
  + Linear SVC (SVC = Support Vector Classifier)

## **Logistic Regression**
  + Logistic regression is a method of classification: the model outputs the probability of a categorical target variable Y belonging to a certain class.

### **LOGIT Model**
The **logit model** is a modification of linear regression that makes sure to output a probability between 0 and 1 by applying the sigmoid function, which, when graphed, looks like the characteristic S-shaped curve that you’ll see a bit later.

**Sigmoid Function**: A mathematical function having a characteristic "S"-shaped curve or sigmoid curve.

    S(x) = 1/(1 + e^-x) ==> Sigmoid function, which squashes values between 0 and 1.

FOR LR:

    LR: g(x) = β0 + β1*x
    Sigmoid: F(x) = 1/(1 + e^-x)

    Through composition:
          P(Y=1) = F(g(x)) = 1/(1 + e^-(β0 + β1*x))

    Solve for P:
          ln(p/(1-p)) = β0 + β1*x + ϵ

    where p/(1-p) is the odds ratio.

To predict the Y label — spam/not spam, cancer/not cancer, fraud/not fraud, etc. — you have to set a probability cutoff, or threshold, for a positive result. For example: “If our model thinks the probability of this email being spam is higher than 70%, label it spam. Otherwise, don’t.”

    COST (R^2) = SUM((value of y from model - actual y)^2)/2n
    Cost = SUM(i=1,n)[(y^i*log(h_β(x^i)) + (1-y^i)*log(1-h_β(x^i)))]/2n + λ*(SUM(i=1,2) β_i^2)

### SUPPORT VECTOR MACHINES (SVMs)
  + It typically solves the same problem as logistic regression — classification with two classes — and yields similar performance. It’s worth understanding because the algorithm is geometrically motivated in nature, rather than being driven by probabilistic thinking.

A few examples of the problems SVMs can solve:
  * Is this an image of a cat or a dog?
  * Is this review positive or negative?
  * Are the dots in the 2D plane red or blue? (Use as an example.)

We would like to classify new, unclassified points in this plane. To do this, SVMs use a separating line (or, in more than two dimensions, a multi-dimensional hyperplane) to split the space into a red zone and a blue zone.

The distance to the nearest point on either side of the line is called the margin, and SVM tries to maximize the margin. You can think about it like a safety space: the bigger that space, the less likely that noisy points get misclassified.

It turns out there’s a clean mathematical way to do this maximization, but the specifics are beyond our scope. To explore it further, here’s a video lecture that shows how it works using Lagrangian Optimization.

![alt text][Lagrangian_Optimization]

More info about [Lagrangian Optimization](https://en.wikipedia.org/wiki/Lagrange_multiplier)

### What happens if you can’t separate the data cleanly?
  + Soften the definition of "separate"
  + Allow a few mistakes, meaning we allow some blue points in the red zone or some red points in the blue zone. We do that by adding a cost C for misclassified examples in our loss function. Basically, we say it’s acceptable but costly to misclassify a point.
  + Throw the data into higher dimensions

### Non-Parametric Learners
Non-parametric learners do not assume a distribution model a priori. These models are more flexible to the shape of the training data, but this sometimes comes at the cost of interpretability.

#### **K-NEAREST NEIGHBORS** (k-NN) - [Implementation Example in Python](http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch)
+ k-NN seems almost too simple to be a machine learning algorithm. The idea is to label a test data point x by finding the mean (or mode) of the k closest data points’ labels. The fact that k-NN doesn’t require a pre-defined parametric function f(X) relating Y to X makes it well-suited for situations where the relationship is too complex to be expressed with a simple linear model.
+ You look at the "k" closest data points and take the average of their values if variables are continuous (like housing prices), or the mode if they’re categorical (like cat vs. dog).
  - How to use k-NN to predict housing prices:
      1) Store the training data, a matrix X of features like zip code, neighborhood, # of bedrooms, square feet, distance from public transport, etc., and a matrix Y of corresponding sale prices.
      2) Sort the houses in your training data set by similarity to the house in question, based on the features in X. We’ll define “similarity” below.
      3) Take the mean of the k closest houses. That is your guess at the sale price (i.e. ŷ)

+ **Distance Metrics**:
  - Euclidean distance (straight line distance via Pythagorean Theorem)
  - Manhattan distance (distance while staying as close to Euclidean distance, but path constrained. e.g. - point A to point B, but must stay on roads.
  - In n-dimensional space:

              d(p,q) = SQRT(SUM[i=0-n] (q_i - p_i)^2)
              
              PYTHON: In NumPy or SciPy: euclidean_dist = numpy.linalg.norm(p-q)

+ **Choosing _k_** - Tuning hyper-parameters with cross validation
   1) Split your training data into segments, and train your model on all but one of the segments; use the held-out segment as the “test” data.
   2) See how your model performs by comparing your model’s predictions (ŷ) to the actual values of the test data (y).
   3) Pick whichever yields the lowest error, on average, across all iterations

+ Higher k prevents overfitting but if the value of k is too high your model will be very biased and inflexible.

   + **Use Cases:**
     + **Classification**: fraud detection. The model can update virtually instantly with new training examples since you’re just storing more data points, which allows quick adaptation to new methods of fraud.
     + **Regression**: predicting housing prices. In housing price prediction, literally being a “near neighbor” is actually a good indicator of being similar in price. k-NN is useful in domains where physical proximity matters.
     + Imputing missing training data. If one of the columns in your .csv has lots of missing values, you can impute the data by taking the mean or mode. k-NN could give you a somewhat more accurate guess at each missing value.

#### [DECISION TREES](http://www-bcf.usc.edu/~gareth/ISL/)
[Tutorial](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/)
- Making a good decision tree is like playing a game of “20 questions”.
- You want to separate the data as cleanly as possible, thereby maximizing INFORMATION GAIN from that split.
- Choosing splits in a decision tree:
  + There are ways to quantify information gain so that you can essentially evaluate every possible split of the training data and maximize information gain for every split. This way you can predict every label or value as efficiently as possible.
  + **Entropy**: Entropy is the amount of disorder in a set
    + **[Gini Index](https://en.wikipedia.org/wiki/Gini_coefficient)** -  a measure of statistical dispersion intended to represent the income or wealth distribution of a nation's residents, and is the most commonly used measurement of inequality.
    + **[Cross Entropy](https://en.wikipedia.org/wiki/Cross_entropy)** - In information theory, the cross entropy between two probability distributions p and q over the same underlying set of events measures the average number of bits needed to identify an event drawn from the set, if a coding scheme is used that is optimized for an "artificial" probability distribution q, rather than the "true" distribution p.
    + EXAMPLE:
      + If we were using decision trees for regression — say, to predict housing prices — we would create splits on the most important features that determine housing prices. 
        + How many square feet: more than or less than _x_? 
        + How many bedrooms & bathrooms: more than or less than _x_?
Then, during testing, you would run a specific house through all the splits and take the average of all the housing prices in the final leaf node (bottom-most node) where the house ends up as your prediction for the sale price.

+ **Tuning Descision Trees**:
   + There are a few hyperparameters you can tune with decision trees models, including [max_depth and max_leaf_nodes](https://scikit-learn.org/stable/modules/tree.html).

+ **ADVANTAGES/DISADVANTAGES**
  + Decision trees are effective because they are easy to read, powerful even with messy data, and computationally cheap to deploy once after training. Decision trees are also good for handling mixed data (numerical or categorical).

  + That said, decision trees are computationally expensive to train, carry a big risk of overfitting, and tend to find local optima because they can’t go back after they have made a split. To address these weaknesses, we turn to a method that illustrates the power of combining many decision trees into one model. (Often a log_n algorithm)

### RANDOM FOREST: An ensemble of decision trees
  + A model comprised of many models is called an ensemble model, and this is usually a winning strategy.
  + A random forest is a meta-estimator that aggregates many decision trees, with some helpful modifications:
     1)   The number of features that can be split on at each node is limited to some percentage of the total (this is a hyperparameter you can choose — see scikit-learn documentation for details). This ensures that the ensemble model does not rely too heavily on any individual feature, and makes fair use of all potentially predictive features.
     2) Each tree draws a random sample from the original data set when generating its splits, adding a further element of randomness that prevents overfitting.
 + These modifications also prevent the trees from being too highly correlated. Without #1 and #2 above, every tree would be identical, since recursive binary splitting is deterministic.

Another clever ensemble model is XGBoost [Extreme Gradient Boosting](http://xgboost.readthedocs.io/en/latest/model.html)


[gradient_image]: https://cdn-images-1.medium.com/max/800/0*ZaEKARNxNgB7-H3F. "Gradient"
[regularization_image]: https://cdn-images-1.medium.com/max/800/1*rFT6mtU45diT0OJhlgDcBg.png "Regularization"
[overfit_image]: https://cdn-images-1.medium.com/max/800/1*lb7lEh2Ob5PAJLtnAyGSBA.png "Fitting Examples"
[OLS_Cost]: https://cdn-images-1.medium.com/max/800/0*4YosVQ8oGBg6ZAWv. "OLS Cost"
[Lagrangian_Optimization]: https://www.youtube.com/watch?v=_PwhiWxHK8o "Lagrangian Video"
