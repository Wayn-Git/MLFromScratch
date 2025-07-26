

# Mean Squared Error 

Formula: 
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

MSE = (1 / n) * sum of (actual - predicted) squared

Mainly used in Regression Analysis 
Optimization 
Model Evaluation

 
## Pros: 
 - Comprehensive Measure of the model accuracy 
 - Easy to calcaulate 
 - It penalizes large errors more heavily

## Cons: 

 - Not Good When The Data Contains Outliers
 - Too Sensitive to outliers a single large error can affect the total error

---

# Mean Absolute Erorr 

Formula: 
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)
$$

MAE = (1 / n) * sum of (actual - predicted) 

## Pros: 

 - A good choice when outliers are present as it isn't really that sensitive outliers
 - A large error doesn't really affect the total error as it's not squared

 ## Cons: 
 - All Errors are treated Linearly 
 - Less sensitive to Variance

---


# Root Mean Squared Error

Formula

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

RMSE = sqrt((1 / n) * sum of (actual - predicted) squared)


## Pros

 -Same units as the target (unlike MSE).
 - Still penalizes larger errors more strongly than MAE.
 - More interpretable compared to MSE.

## Cons

 - Sensitive to outliers (like MSE).
 - Usually optimized indirectly (by minimizing MSE instead).