## Example 1: 
#############################################################
#eigenvalues and eigenvectors
############################################

import math    
from scipy.stats import shapiro
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import scipy.stats
from sklearn.decomposition import FastICA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from numpy.random import seed
from numpy.random import randn
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy.stats import skew
from scipy.stats import skew
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from numpy.linalg import eig  
    
   
 # Generate a sample data (replace this with your actual data)
#data = np.random.normal(0, 1, 100)
    
mean = np.array([0.766, 0.506, 0.438, 0.161]) 
 #A = np.random.rand(5, 5)
cov=np.array([[ 0.220588235, -148.253676, -2.53676471, -1.25000000,-1.13235294],
 [-148.253676,  3343690.26, 28424.4485, -37368.7500, 19935.2022],
 [-2.53676471,  28424.4485, 538.985294, -393.500000,195.272059],
 [-1.25000000, -37368.7500, -393.500000,  983.000000,-312.125000],
 [-1.13235294,  19935.2022,  195.272059, -312.125000 ,469.404412]])

w,v=eig(cov)
print('E-value:', w)
print('E-vector', v)





#############################################################################


##################################################################################
# Table 1:
 ## Ridge regression for PCs:
    
    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Data
y = np.array([3389, 1101, 1131, 596, 896, 1767, 807, 1111, 645, 628, 1360, 652, 860, 500, 781, 1070, 1754])
X = np.array([
    [1, 7500, 220, 0, 140],
    [1, 1975, 200, 0, 100],
    [0, 3600, 205, 60, 111],
    [1, 675, 160, 60, 120],
    [1, 750, 185, 70, 83],
    [1, 2500, 180, 60, 80],
    [1, 350, 154, 80, 98],
    [0, 1500, 200, 70, 93],
    [1, 375, 137, 60, 105],
    [1, 1050, 167, 60, 74],
    [1, 3000, 180, 60, 80],
    [1, 450, 160, 64, 60],
    [1, 1750, 135, 90, 79],
    [0, 2000, 160, 60, 80],
    [0, 4500, 180, 0, 100],
    [0, 1500, 170, 90, 120],
    [1, 3000, 180, 0, 129]
])

# Fit Ridge Regression model
ridge_model = Ridge(alpha=0.1)  # Alpha is the regularization parameter, can be adjusted
ridge_model.fit(X, y)
y_pred = ridge_model.predict(X)

# Display R-squared, intercept, and coefficients
print("R-squared (Ridge Regression):", ridge_model.score(X, y))
print("Intercept:", ridge_model.intercept_)
print("Coefficients:", ridge_model.coef_)

# Calculate error metrics
mse = mean_squared_error(y, y_pred)  # Mean Squared Error
mre = np.mean(np.abs(y - y_pred) / y) * 100  # Mean Relative Error in percentage
print("Mean Squared Error:", mse)
print("Mean Relative Error (%):", mre)

# Plot Actual vs Predicted values
plt.figure(figsize=(6,5))
plt.scatter(y, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)  # 45-degree reference line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted (Ridge Regression)")
plt.grid(True)
plt.show()

# Plot Residuals
residuals = y - y_pred
plt.figure(figsize=(6,5))
plt.scatter(y_pred, residuals, color='green', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # Zero line for residuals
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot (Ridge Regression)")
plt.grid(True)
plt.show()


#############################################################
## Lasso regression for PCs:
###############################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Dependent variable
y = np.array([3389, 1101, 1131, 596, 896, 1767, 807, 1111, 645, 628, 1360, 652, 860, 500, 781, 1070, 1754])

# Original features (example: 5 features)
X = np.array([
    [1, 7500, 220, 0, 140],
    [1, 1975, 200, 0, 100],
    [0, 3600, 205, 60, 111],
    [1, 675, 160, 60, 120],
    [1, 750, 185, 70, 83],
    [1, 2500, 180, 60, 80],
    [1, 350, 154, 80, 98],
    [0, 1500, 200, 70, 93],
    [1, 375, 137, 60, 105],
    [1, 1050, 167, 60, 74],
    [1, 3000, 180, 60, 80],
    [1, 450, 160, 64, 60],
    [1, 1750, 135, 90, 79],
    [0, 2000, 160, 60, 80],
    [0, 4500, 180, 0, 100],
    [0, 1500, 170, 90, 120],
    [1, 3000, 180, 0, 129]
])

# Perform PCA and take the first principal component
pca = PCA(n_components=1)
pc = pca.fit_transform(X)  # Shape: (n_samples, 1)

# Fit Lasso regression on the first principal component
lasso_model = Lasso(alpha=0.1)  # Regularization parameter
lasso_model.fit(pc, y)
y_pred_lasso = lasso_model.predict(pc)

# R-squared, intercept, and coefficient
r_squared_lasso = lasso_model.score(pc, y)
print("R-squared (Lasso Regression):", r_squared_lasso)
print("Intercept (Lasso Regression):", lasso_model.intercept_)
print("Coefficient (Lasso Regression):", lasso_model.coef_)

# Error metrics
mse = mean_squared_error(y, y_pred_lasso)
mre = np.mean(np.abs(y - y_pred_lasso) / y) * 100
print("Mean Squared Error:", mse)
print("Mean Relative Error (%):", mre)

# Plot Actual vs Predicted
plt.figure(figsize=(6,5))
plt.scatter(y, y_pred_lasso, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)  # 45-degree reference line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted (Lasso Regression)")
plt.grid(True)
plt.show()

# Plot Residuals
residuals = y - y_pred_lasso
plt.figure(figsize=(6,5))
plt.scatter(y_pred_lasso, residuals, color='green', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot (Lasso Regression)")
plt.grid(True)
plt.show()


#Table1:
##### Ridge Regression for ICs and y:
############################################################

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA
#from sklearn.metrics import r2_score, mean_squared_error

# Define the data matrix
data = np.array([
    [1, 7500, 220, 0, 140],
    [1, 1975, 200, 0, 100],
    [0, 3600, 205, 60, 111],
    [1, 675, 160, 60, 120],
    [1, 750, 185, 70, 83],
    [1, 2500, 180, 60, 80],
    [1, 350, 154, 80, 98],
    [0, 1500, 200, 70, 93],
    [1, 375, 137, 60, 105],
    [1, 1050, 167, 60, 74],
    [1, 3000, 180, 60, 80],
    [1, 450, 160, 64, 60],
    [1, 1750, 135, 90, 79],
    [0, 2000, 160, 60, 80],
    [0, 4500, 180, 0, 100],
    [0, 1500, 170, 90, 120],
    [1, 3000, 180, 0, 129]
])


# Center the data
data_centered = data - np.mean(data, axis=0)

# Calculate the covariance matrix
cov_matrix = np.cov(data_centered, rowvar=False)

# Perform eigenvalue decomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Whitening transformation
whitening_matrix = np.dot(eigenvectors, np.dot(np.diag(1.0/np.sqrt(eigenvalues + 1e-5)), eigenvectors.T))

# Whiten the data
w_d = np.dot(data_centered, whitening_matrix)
Tw_d = np.transpose(w_d)
print(w_d)
correlation_coefficients = []


for i in range(2):
   # Generate some sample data
   s1 = Tw_d[0]  # Signal 1 : sinusoidal signal
   s2 = Tw_d[1]  # Signal 2 : square signal
   s3 = Tw_d[2] 
   s4 = Tw_d[3]
   s5 = Tw_d[4]
 
  
   ica = FastICA(n_components=5)
   A = ica.fit_transform(Tw_d)

   # Mix the signals into a single observed signal
   #n=1030
   S = np.c_[s1, s2, s3 , s4, s5]
   X1 = np.dot(S, A.T)  # Observed signal
   
   import numpy as np
   import matplotlib.pyplot as plt
   # Perform ICA using the FastICA algorithm
   ica = FastICA(n_components=5)
   S_ = ica.fit_transform(X1)  # Reconstruct signals
   

   # Compare the reconstructed signals to the original signals
   print(np.allclose(X1, np.dot(S_, ica.mixing_.T)))


   TS_ = np.transpose(S_)
   X = np.array([TS_[0], TS_[1] ,TS_[2], TS_[3], TS_[4]])
   XT= np.transpose(X)
   # Fit the model 
   #XT = sm.add_constant(XT) 
###########################################


from sklearn.linear_model import Ridge
y = np.array([3389, 1101, 1131, 596, 896, 1767, 807, 1111, 645, 628, 1360, 652, 860, 500, 781, 1070, 1754])
# Fit the Ridge regression model
ridge_regression_model = Ridge(alpha=0.1)  # Adjust the alpha parameter as needed
ridge_regression_model.fit(XT, y)

# Calculate predicted values
y_pred_ridge = ridge_regression_model.predict(XT)

# Calculate R-squared for Ridge regression
r_squared_ridge = ridge_regression_model.score(XT, y)
print("R-squared (Ridge Regression):", r_squared_ridge)

# Print intercept and coefficients for Ridge regression
print("Intercept (Ridge Regression):", ridge_regression_model.intercept_)
print("Coefficients (Ridge Regression):", ridge_regression_model.coef_)

# Plot actual vs. predicted values
plt.scatter(y, y_pred_ridge)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values (Ridge Regression)")
plt.show()

# Plot residuals vs. predicted values
residuals_ridge = y - y_pred_ridge
plt.scatter(y_pred_ridge, residuals_ridge)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot (Ridge Regression)")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0
plt.show()

####################

 # Calculate mean squared error
mse = mean_square_error(y, y_pred_ridge)
print("Mean Square Error:", mse)



    # Calculate mean relative error
mre = mean_relative_error(y, y_pred_ridge)
print("Mean Relative Error:", mre)


# Table1:
##### Lasso Regression in ICs,y components:
################################################################

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso
import math    
from scipy.stats import shapiro
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import scipy.stats
from sklearn.decomposition import FastICA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from numpy.random import seed
from numpy.random import randn
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy.stats import skew
from scipy.stats import skew
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


# Define the dependent variable y
y = np.array([3389, 1101, 1131, 596, 896, 1767, 807, 1111, 645, 628, 1360, 652, 860, 500, 781, 1070, 1754])
# Define the data matrix
data = np.array([
    [1, 7500, 220, 0, 140],
    [1, 1975, 200, 0, 100],
    [0, 3600, 205, 60, 111],
    [1, 675, 160, 60, 120],
    [1, 750, 185, 70, 83],
    [1, 2500, 180, 60, 80],
    [1, 350, 154, 80, 98],
    [0, 1500, 200, 70, 93],
    [1, 375, 137, 60, 105],
    [1, 1050, 167, 60, 74],
    [1, 3000, 180, 60, 80],
    [1, 450, 160, 64, 60],
    [1, 1750, 135, 90, 79],
    [0, 2000, 160, 60, 80],
    [0, 4500, 180, 0, 100],
    [0, 1500, 170, 90, 120],
    [1, 3000, 180, 0, 129]
])


# Center the data
data_centered = data - np.mean(data, axis=0)

# Calculate the covariance matrix
cov_matrix = np.cov(data_centered, rowvar=False)

# Perform eigenvalue decomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Whitening transformation
whitening_matrix = np.dot(eigenvectors, np.dot(np.diag(1.0/np.sqrt(eigenvalues + 1e-5)), eigenvectors.T))

# Whiten the data
w_d = np.dot(data_centered, whitening_matrix)
Tw_d = np.transpose(w_d)
print(w_d)
correlation_coefficients = []


for i in range(2):
   # Generate some sample data
   s1 = Tw_d[0]  # Signal 1 : sinusoidal signal
   s2 = Tw_d[1]  # Signal 2 : square signal
   s3 = Tw_d[2] 
   s4 = Tw_d[3]
   s5 = Tw_d[4]
 
  
   ica = FastICA(n_components=5)
   A = ica.fit_transform(Tw_d)

   # Mix the signals into a single observed signal
   #n=1030
   S = np.c_[s1, s2, s3 , s4, s5]
   X1 = np.dot(S, A.T)  # Observed signal
   
   import numpy as np
   import matplotlib.pyplot as plt
   # Perform ICA using the FastICA algorithm
   ica = FastICA(n_components=5)
   S_ = ica.fit_transform(X1)  # Reconstruct signals
   

   # Compare the reconstructed signals to the original signals
   print(np.allclose(X1, np.dot(S_, ica.mixing_.T)))


   TS_ = np.transpose(S_)
   X = np.array([TS_[0], TS_[1] ,TS_[2], TS_[3], TS_[4]])
   XT= np.transpose(X)
#############################    
  
# mre  #####################
#calculate MRE:

def mean_relative_error(actual1, predicted1):
    
   actual1, predicted1 = np.array(actual1), np.array(predicted1)
   return np.mean(np.abs((actual1 - predicted1) / actual1))

#########################
 # Calculate mean squared error
def mean_square_error(actual_values, predicted_values):
   
    # Check if actual_values and predicted_values have the same length
   if len(actual_values) != len(predicted_values):
    raise ValueError("Length of actual_values and predicted_values must be the same.")
    
    # Calculate MSE
   mse = sum((actual - predicted) ** 2 for actual, predicted in zip(actual_values, predicted_values)) / len(actual_values)
    
   return mse


############################
    
###############################    
# Fit the Lasso regression model
lasso_regression_model = Lasso(alpha=0.1)  # Adjust the alpha parameter as needed
lasso_regression_model.fit(XT, y)

# Calculate predicted values
y_pred_lasso = lasso_regression_model.predict(XT)

# Calculate R-squared for Lasso regression
r_squared_lasso = lasso_regression_model.score(XT, y)
print("R-squared (Lasso Regression):", r_squared_lasso)

# Print intercept and coefficients for Lasso regression
print("Intercept (Lasso Regression):", lasso_regression_model.intercept_)
print("Coefficients (Lasso Regression):", lasso_regression_model.coef_)

# Plot actual vs. predicted values
plt.scatter(y, y_pred_lasso)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values (Lasso Regression)")
plt.show()

# Plot residuals vs. predicted values
residuals_lasso = y - y_pred_lasso
plt.scatter(y_pred_lasso, residuals_lasso)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot (Lasso Regression)")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0
plt.show()

 # Calculate mean squared error
mse = mean_square_error(y, y_pred_lasso)
print("Mean Square Error:", mse)



    # Calculate mean relative error
mre = mean_relative_error(y, y_pred_lasso)
print("Mean Relative Error:", mre)

## Simulation##  Table 2 (PCR)

#(True)  Pcr for 4 component (random coefficient):# for witening data: for simulation data

###################################################################################################

import math    
from scipy.stats import shapiro
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import scipy.stats
from sklearn.decomposition import FastICA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from numpy.random import seed
from numpy.random import randn
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy.stats import skew
from scipy.stats import skew
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
   
    
   
 # Generate a sample data (replace this with your actual data)
#data = np.random.normal(0, 1, 100)
    
mean = np.array([0.04568769, 0.03916346, 0.10603323, 0.60737812, 0.21861233]) 
 #A = np.random.rand(5, 5)
A=np.array([[0.54407526, 0.01804927 ,0.95803021 ,0.53246229, 0.98903702],
     [0.85178268, 0.27275685 ,0.8007421  ,0.53891002, 0.07267785],
     [0.0977354 , 0.86782946 ,0.09019516 ,0.59581037, 0.97433572],
     [0.69733138, 0.27297179 ,0.69968404 ,0.33733182, 0.96135071],
     [0.98289599, 0.01509631 ,0.150057   ,0.346731  , 0.85385711]])
    
cov = np.dot(A, A.T)
    # Generate random samples from the multivariate normal distribution
num_samples = 5000

def root_mean_squared_error(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("Length of actual and predicted lists must be the same.")
    squared_errors = [(actual[k] - predicted[k]) ** 2 for k in range(len(actual))]
    mean_squared_error = sum(squared_errors) / len(actual)
    rmse = math.sqrt(mean_squared_error)
    return rmse


def mean_relative_error(actual1, predicted1):
    
    actual1, predicted1 = np.array(actual1), np.array(predicted1)
    return np.mean(np.abs((actual1 - predicted1) / actual1))

rmse_sum = 0
mre_sum = 0
mse_sum = 0


mse_sum = 0
for j in range(5000):
      samples = np.random.multivariate_normal(mean, cov, num_samples)
    
    # Define the coefficients for the linear combination
      coefficients = np.random.rand(5)
    #coefficients =np.array([0.89931776, 0.35102325, 0.55151473, 0.21779235, 0.89916957])
    # Generate the response variable as a linear combination of variables with noise
      noise = np.random.normal(0, 1, num_samples)  # Additive noise term
      response = np.dot(samples, coefficients) + noise
   
    # Display the first few response values
    #print(response[:5])
    #print(coefficients)
    #plt.hist(samples[:500])
    #plt.hist(response[:500])
    # Perform the Shapiro-Wilk test
      statistic, p_value = shapiro(response)
    # Display the test statistic and p-value
      print("Test statistic:", statistic)
      print("P-value:", p_value)
   
      if p_value > 0.05:
        print("The data is normally distributed (fail to reject H0)")
      else:
        print("The data is not normally distributed (reject H0)")
      
   
    # Split the real data into independent and dependent variables
      X_real =samples
      y_real = response
   
    # Perform principal component analysis
      pca = PCA(n_components=5)  # Replace 8 with the desired number of principal components
      X_pca = pca.fit_transform(X_real)
   
    # Calculate the correlation coefficients of the principal components with the original variables
      correlation_coefficients = np.corrcoef(X_pca.T, X_real.T)
      print("Correlation Coefficients of Principal Components with Original Variables:")
      print(correlation_coefficients)
   
      pc = np.zeros((len(samples), pca.n_components_))
    # Individual principal components
      print("Individual Principal Components:")
      for i in range(pca.n_components_):
        print(f"Principal Component {i+1}: {pca.components_[i]}")
        pc[:, i] = np.dot(samples, pca.components_[i])
      print("pc:")
      print(pc)
   
    # Eigenvalues of the principal components
      print("Eigenvalues of Principal Components:")
      print(pca.explained_variance_)
   
    # Matrix of correlation coefficients of the principal components with the original variables
      print("Matrix of Correlation Coefficients of Principal Components with Original Variables:")
      print(correlation_coefficients)
    #print(pca.components_[1])
    #############################################################
    #pcr for test , train data:
    #X_real_reshaped = X_real.reshape(-1,1)
      X_real = np.array([pc[:,0], pc[:,1], pc[:,2], pc[:,3] ])
      TX_real= np.transpose(X_real)
      y_real = response
      mse_list = []
    #for i in range(2):
    # Perform principal component analysis\n",
      pca = PCA(n_components=4)  # Replace 2 with the desired number of principal components
    #X_pca = pca.fit_transform(X_real)
      X_pca = pca.fit_transform(TX_real)
    
      X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca, y_real, test_size=0.2, random_state=42)

    # Perform linear regression on the PCA data
      X_pca_train = sm.add_constant(X_pca_train)  # Add a constant term to the independent variables
      reg_pca = sm.OLS(y_pca_train, X_pca_train).fit()  # Fit the linear regression model
      X_pca_test = sm.add_constant(X_pca_test)
      y_pred = reg_pca.predict(X_pca_test)

     # Calculate mean squared error
      mse = mean_squared_error(y_pca_test, y_pred)
      print("Mean Squared Error:", mse)
      mse_sum += mse

    # Calculate root mean squared error
      rmse = root_mean_squared_error(y_pca_test, y_pred)
      print("Root Mean Squared Error:", rmse)
      rmse_sum += rmse

    # Calculate mean relative error
      mre = mean_relative_error(y_pca_test, y_pred)
      print("Mean Relative Error:", mre)
      mre_sum += mre
    
avg_mse = mse_sum / 5000
avg_rmse = rmse_sum / 5000
avg_mre = mre_sum / 5000

print("Average Mean Squared Error:", avg_mse)
print("Average Root Mean Squared Error:", avg_rmse)
print("Average Mean relative Error:", avg_mre)



## Simulation## Table 2 (ICA):

##  Icr for 4 component (random coefficient): for simulation data 

##########################################################################################

import math   
from scipy.stats import shapiro
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import FastICA
from sklearn.metrics import r2_score

mean = np.array([0.04568769, 0.03916346, 0.10603323, 0.60737812, 0.21861233]) 
A = np.array([[0.54407526, 0.01804927, 0.95803021, 0.53246229, 0.98903702],
              [0.85178268, 0.27275685, 0.8007421,  0.53891002, 0.07267785],
              [0.0977354,  0.86782946, 0.09019516, 0.59581037, 0.97433572],
              [0.69733138, 0.27297179, 0.69968404, 0.33733182, 0.96135071],
              [0.98289599, 0.01509631, 0.150057,   0.346731,   0.85385711]])

cov = np.dot(A, A.T)
num_samples = 5000

def root_mean_squared_error(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("Length of actual and predicted lists must be the same.")
    squared_errors = [(actual[k] - predicted[k]) ** 2 for k in range(len(actual))]
    mean_squared_error = sum(squared_errors) / len(actual)
    rmse = math.sqrt(mean_squared_error)
    return rmse


def mean_relative_error(actual1, predicted1):
    
    actual1, predicted1 = np.array(actual1), np.array(predicted1)
    return np.mean(np.abs((actual1 - predicted1) / actual1))

rmse_sum = 0
mre_sum = 0
mse_sum = 0
for j in range(5000):
    samples = np.random.multivariate_normal(mean, cov, num_samples)
    coefficients = np.random.rand(5)
    noise = np.random.normal(0, 1, num_samples)  # Additive noise term
    response = np.dot(samples, coefficients) + noise

    X_real = samples
    y_real = response
    
    X_real_centered = X_real - np.mean(X_real, axis=0)
    cov_matrix = np.cov(X_real_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    whitening_matrix = np.dot(eigenvectors, np.dot(np.diag(1.0 / np.sqrt(eigenvalues + 1e-5)), eigenvectors.T))
    w_d = np.dot(X_real_centered, whitening_matrix)
    Tw_d = np.transpose(w_d)
    
    s1 = Tw_d[0]
    s2 = Tw_d[1]
    s3 = Tw_d[2]
    s4 = Tw_d[3]
    s5 = Tw_d[4]
    
    ica = FastICA(n_components=5)
    A = ica.fit_transform(Tw_d)
    
    S = np.c_[s1, s2, s3, s4, s5]
    X1 = np.dot(S, A.T)
    S_ = ica.fit_transform(X1)
    TS_ = np.transpose(w_d)
    
    Y = y_real
    X = np.array([TS_[0], TS_[1], TS_[2], TS_[3], TS_[4]])
    XT = np.transpose(X)
    X_train, X_test, y_train, y_test = train_test_split(XT, Y, test_size=0.2, random_state=42)
    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train).fit()
    X_test = sm.add_constant(X_test)
    y_pred = model.predict(X_test)
    
    r_squared = r2_score(y_test, y_pred)
    print("R-squared:", r_squared)
    print(model.summary())
    
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    mse_sum += mse
    
    rmse = root_mean_squared_error(y_test, y_pred)
    print("Root Mean Squared Error:", rmse)
    rmse_sum += rmse
    
    mre = mean_relative_error(y_test, y_pred)
    print("Mean Relative Error:", mre)
    mre_sum += mre
    
avg_mse = mse_sum / 5000
avg_rmse = rmse_sum / 5000
avg_mre = mre_sum / 5000

print("Average Mean Squared Error:", avg_mse)
print("Average Root Mean Squared Error:", avg_rmse)
print("Average Mean relative Error:", avg_mre)
   

## Simulation## Table 3:

## True AIC, BIC, MSE, RMSE, MRE FOR ICR,PcR

#############################################################################

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Error metrics functions
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_relative_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

# AIC and BIC Calculation functions
def calculate_aic(n, rss, k):
    return n * np.log(rss / n) + 2 * k

def calculate_bic(n, rss, k):
    return n * np.log(rss / n) + k * np.log(n)

# Function to apply Kaiser Criterion
def kaiser_criterion(eigenvalues):
    return sum(eigenvalues > 1)

# Function to apply Explained Variance Criterion
def explained_variance_criterion(pca, threshold=0.90):
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    return np.argmax(explained_variance >= threshold) + 1

# Function to perform Horn’s Parallel Analysis
def horns_parallel_analysis(X, num_simulations=5000):
    num_samples, num_features = X.shape
    eigenvalues = np.linalg.eigvals(np.cov(X.T))
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    
    # Generate random data and compute eigenvalues for multiple simulations
    random_eigenvalues = np.zeros((num_simulations, num_features))
    for i in range(num_simulations):
        random_data = np.random.normal(size=(num_samples, num_features))
        random_eigenvalues[i] = np.sort(np.linalg.eigvals(np.cov(random_data.T)))[::-1]
    
    # Compute the critical eigenvalue threshold
    threshold = np.percentile(random_eigenvalues, 95, axis=0)
    
    # Count the number of eigenvalues greater than the threshold
    return sum(eigenvalues_sorted > threshold[0])

# Initialize metrics for ICA and PCA models
metrics_sums = {
    "ICA-Lasso": {"MSE": 0, "RMSE": 0, "MRE": 0, "AIC": 0, "BIC": 0},
    "ICA-Ridge": {"MSE": 0, "RMSE": 0, "MRE": 0, "AIC": 0, "BIC": 0},
    "ICA-OLS": {"MSE": 0, "RMSE": 0, "MRE": 0, "AIC": 0, "BIC": 0},
    "PCA-Lasso": {"MSE": 0, "RMSE": 0, "MRE": 0, "AIC": 0, "BIC": 0},
    "PCA-Ridge": {"MSE": 0, "RMSE": 0, "MRE": 0, "AIC": 0, "BIC": 0},
    "PCA-OLS": {"MSE": 0, "RMSE": 0, "MRE": 0, "AIC": 0, "BIC": 0},
}

# Simulation parameters
num_simulations = 5000
num_samples = 5000
mean = np.zeros(5)
cov = np.eye(5)

for _ in range(num_simulations):
    # Generate synthetic data
    samples = np.random.multivariate_normal(mean, cov, num_samples)
    coefficients = np.random.rand(5)
    noise = np.random.normal(0, 1, num_samples)
    response = np.dot(samples, coefficients) + noise

    # Outlier detection and removal using IsolationForest
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    outlier_labels = isolation_forest.fit_predict(samples)
    clean_indices = outlier_labels == 1
    samples_cleaned = samples[clean_indices]
    response_cleaned = response[clean_indices]

    # Standardize data
    scaler = StandardScaler()
    samples_standardized = scaler.fit_transform(samples_cleaned)

    # Determine the number of components for PCA and ICA based on different criteria
    pca = PCA()
    pca.fit(samples_standardized)
    
    # Apply Kaiser Criterion to PCA
    num_components_pca_kaiser = kaiser_criterion(pca.explained_variance_)

    # Apply Explained Variance Criterion to PCA
    num_components_pca_variance = explained_variance_criterion(pca, threshold=0.90)
    
    # Apply Horn's Parallel Analysis to PCA
    num_components_pca_horn = horns_parallel_analysis(samples_standardized)
    
    # Perform ICA with determined number of components
    ica = FastICA(n_components=num_components_pca_variance, random_state=42, max_iter=1000, tol=0.001)
    X_ica = ica.fit_transform(samples_standardized)

    # Perform PCA with determined number of components
    pca = PCA(n_components=num_components_pca_kaiser)
    X_pca = pca.fit_transform(samples_standardized)

    # Split data for ICA and PCA
    X_ica_train, X_ica_test, y_train, y_test = train_test_split(
        X_ica, response_cleaned, test_size=0.2, random_state=42
    )
    X_pca_train, X_pca_test, _, _ = train_test_split(
        X_pca, response_cleaned, test_size=0.2, random_state=42
    )

    # Regression models for ICA and PCA
    for method_prefix, X_train, X_test in zip(
        ["ICA", "PCA"],
        [X_ica_train, X_pca_train],
        [X_ica_test, X_pca_test],
    ):
        # Lasso Regression
        lasso = Lasso(alpha=0.1, random_state=42)
        lasso.fit(X_train, y_train)
        y_pred_lasso = lasso.predict(X_test)
        rss_lasso = np.sum((y_test - y_pred_lasso) ** 2)
        aic_lasso = calculate_aic(len(y_test), rss_lasso, lasso.coef_.shape[0] + 1)
        bic_lasso = calculate_bic(len(y_test), rss_lasso, lasso.coef_.shape[0] + 1)

        # Ridge Regression
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        rss_ridge = np.sum((y_test - y_pred_ridge) ** 2)
        aic_ridge = calculate_aic(len(y_test), rss_ridge, ridge.coef_.shape[0] + 1)
        bic_ridge = calculate_bic(len(y_test), rss_ridge, ridge.coef_.shape[0] + 1)

        # OLS Regression
        ols = LinearRegression()
        ols.fit(X_train, y_train)
        y_pred_ols = ols.predict(X_test)
        rss_ols = np.sum((y_test - y_pred_ols) ** 2)
        aic_ols = calculate_aic(len(y_test), rss_ols, ols.coef_.shape[0] + 1)
        bic_ols = calculate_bic(len(y_test), rss_ols, ols.coef_.shape[0] + 1)

        # Calculate and accumulate metrics
        for method_suffix, y_pred, aic, bic in zip(
            ["Lasso", "Ridge", "OLS"], [y_pred_lasso, y_pred_ridge, y_pred_ols],
            [aic_lasso, aic_ridge, aic_ols], [bic_lasso, bic_ridge, bic_ols]
        ):
            method = f"{method_prefix}-{method_suffix}"
            metrics_sums[method]["MSE"] += mean_squared_error(y_test, y_pred)
            metrics_sums[method]["RMSE"] += root_mean_squared_error(y_test, y_pred)
            metrics_sums[method]["MRE"] += mean_relative_error(y_test, y_pred)
            metrics_sums[method]["AIC"] += aic
            metrics_sums[method]["BIC"] += bic

# Calculate averages
metrics_avgs = {
    method: {metric: total / num_simulations for metric, total in metrics.items()}
    for method, metrics in metrics_sums.items()
}

# Print results
for method, metrics in metrics_avgs.items():
    print(f"{method}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

# Visualization of results
methods = list(metrics_avgs.keys())
mse_values = [metrics["MSE"] for metrics in metrics_avgs.values()]
rmse_values = [metrics["RMSE"] for metrics in metrics_avgs.values()]
mre_values = [metrics["MRE"] for metrics in metrics_avgs.values()]
aic_values = [metrics["AIC"] for metrics in metrics_avgs.values()]
bic_values = [metrics["BIC"] for metrics in metrics_avgs.values()]

plt.figure(figsize=(20, 8))
plt.subplot(1, 5, 1)
plt.bar(methods, mse_values, color=['blue', 'orange', 'green', 'red', 'purple', 'brown'])
plt.xticks(rotation=45)
plt.title('Average MSE Comparison')
plt.ylabel('MSE')

plt.subplot(1, 5, 2)
plt.bar(methods, rmse_values, color=['blue', 'orange', 'green', 'red', 'purple', 'brown'])
plt.xticks(rotation=45)
plt.title('Average RMSE Comparison')
plt.ylabel('RMSE')

plt.subplot(1, 5, 3)
plt.bar(methods, mre_values, color=['blue', 'orange', 'green', 'red', 'purple', 'brown'])
plt.xticks(rotation=45)
plt.title('Average MRE Comparison')
plt.ylabel('MRE')

plt.subplot(1, 5, 4)
plt.bar(methods, aic_values, color=['blue', 'orange', 'green', 'red', 'purple', 'brown'])
plt.xticks(rotation=45)
plt.title('Average AIC Comparison')
plt.ylabel('AIC')

plt.subplot(1, 5, 5)
plt.bar(methods, bic_values, color=['blue', 'orange', 'green', 'red', 'purple', 'brown'])
plt.xticks(rotation=45)
plt.title('Average BIC Comparison')
plt.ylabel('BIC')

plt.tight_layout()
plt.show()

## Figure 1

############################################################################

from sklearn.linear_model import LinearRegression
from statsmodels.graphics.gofplots import qqplot
import numpy as np
from sklearn.decomposition import FastICA
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pandas as pd
import scipy.stats
import statsmodels.api as sm
from numpy.random import seed
from numpy.random import randn
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.stats import skew
from sklearn.preprocessing import PolynomialFeatures

class CustomFastICA:
    def __init__(self, n_components, df, loc, scale, max_iter=200, tol=1e-4):
        self.n_components = n_components
        self.df = df  # degrees of freedom for the t-distribution
        self.loc = loc  # mean
        self.scale = scale  # standard deviation
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n, m = X.shape
        W = np.random.rand(self.n_components, m)

        for iteration in range(self.max_iter):
            W_old = W.copy()

            # Calculate new weights
            Y = np.dot(X, W.T)
            g = self._tanh(Y)
            g_dot = self._tanh_derivative(Y)
            W = (1 / n) * np.dot(g.T, X) - np.mean(g_dot) * W

            # Symmetric orthogonalization
            W = self._symmetric_orthogonalization(W)

            # Decorrelation
            W = self._decorrelation(W)

            # Check convergence
            if iteration > 0 and np.linalg.norm(W - W_old) < self.tol:
                break

        self.components_ = W
        return self

    def transform(self, X):
        return np.dot(X, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def _symmetric_orthogonalization(self, W):
        u, s, vh = np.linalg.svd(W, full_matrices=False)
        return np.dot(u, vh)

    def _decorrelation(self, W):
        WTW = np.dot(W, W.T)
        _, WTW_eigenvecs = np.linalg.eigh(WTW)
        return np.dot(WTW_eigenvecs.T, W)
    ###########################################################################

# Load dataset
df=pd.read_csv('data1.csv')
X = df.values
# Center the data
X_centered = X - np.mean(X, axis=0)

# Calculate the covariance matrix
cov_matrix = np.cov(X_centered, rowvar=False)

# Perform eigenvalue decomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Whitening transformation
whitening_matrix = np.dot(eigenvectors, np.dot(np.diag(1.0/np.sqrt(eigenvalues + 1e-5)), eigenvectors.T))

# Whiten the data
w_d = np.dot(X_centered, whitening_matrix)
Tw_d = np.transpose(w_d)

#############################################################
# Defining signals (input variables)
s11 = df['Cement ']  
s22 = df['Blast Furnace Slag']  
s33 = df['Fly Ash ']  
s44 = df['Water  ']
s55 = df['Superplasticize']
s66 = df['Coarse Aggregate  \n\n']
s77 = df['Fine Aggregate']
s88 = df['Age ']
s99 = df['Concrete compressive strength']

ica = FastICA(n_components=8)
A = ica.fit_transform(df)
S = np.c_[s11, s22, s33 , s44 , s55, s66, s77, s88 ]
X1 = np.dot(S, A.T) 
X1T= np.transpose(X1)

##########################################################
correlation_coefficients = []

s1 = Tw_d[0]  
s2 = Tw_d[1]  
s3 = Tw_d[2] 
s4 = Tw_d[3]
s5 = Tw_d[4]
s6 = Tw_d[5] 
s7 = Tw_d[6] 
s8 = Tw_d[7] 
s9 = Tw_d[8] 

S = np.c_[s1, s2, s3 , s4 , s5, s6, s7, s8]

##################
# Instantiate and fit the CustomFastICA model
custom_ica = CustomFastICA(n_components=8, df=8.869463345651127, loc=-0.0364849978662904, scale=0.8750974313065253)
S_ = custom_ica.fit_transform(S)

y = df['Concrete compressive strength']
X = np.array([S_[:, 0],S_[:, 1],S_[:, 2], S_[:, 3],S_[:, 4],S_[:, 5],S_[:, 6],S_[:, 7]])
XT= np.transpose(X)
XT = sm.add_constant(XT)  
model = sm.OLS(y, XT).fit()
########################################################

# ... (middle parts of the code remain unchanged)

#########################################3

###############################################
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Independent components (ICs): S_
df_IC = pd.DataFrame(S_, columns=[f"IC{i+1}" for i in range(S_.shape[1])])

# Scatter function with nonlinear polynomial fitting
def scatter_with_polyfit(x, y, degree=2, **kwargs):
    plt.scatter(x, y, alpha=0.5, color='gray')
    # Polynomial fitting
    x_poly = x.values.reshape(-1, 1) if hasattr(x, 'values') else x.reshape(-1, 1)
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(x_poly)
    model = LinearRegression()
    model.fit(X_poly, y)
    # Plot fitted curve
    x_fit = np.linspace(x_poly.min(), x_poly.max(), 200).reshape(-1, 1)
    y_fit = model.predict(poly.transform(x_fit))
    plt.plot(x_fit, y_fit, color='red', linewidth=3)

# Pairplot
pairplot_fig = sns.PairGrid(df_IC)
pairplot_fig.map_offdiag(scatter_with_polyfit)
pairplot_fig.map_diag(plt.hist, color='gray', edgecolor='black', alpha=0.7)

# Label ICs on axes
for ax, col in zip(pairplot_fig.axes[-1], df_IC.columns):
    ax.set_xlabel(col, fontsize=12, fontweight='bold')
for ax, row in zip(pairplot_fig.axes[:,0], df_IC.columns):
    ax.set_ylabel(row, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

##################################################

##################################################
# Scatter plots of each IC vs dependent variable with nonlinear fitting

fig, axes = plt.subplots(2, 4, figsize=(20, 10))  
axes = axes.flatten()  

for i in range(S_.shape[1]):
    x = S_[:, i]
    axes[i].scatter(x, y, alpha=0.5)
    
    # Nonlinear polynomial fitting (degree=2)
    x_poly = x.reshape(-1, 1)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(x_poly)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    x_fit = np.linspace(x_poly.min(), x_poly.max(), 200).reshape(-1, 1)
    y_fit = model.predict(poly.transform(x_fit))
    axes[i].plot(x_fit, y_fit, color='red', linewidth=3)
    
    axes[i].set_xlabel(f"IC{i+1}", fontsize=20, fontweight='bold')
    axes[i].set_ylabel("Y", fontsize=20, fontweight='bold')

# Hide unused plots if ICs < 8
for j in range(S_.shape[1], 8):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

#######################################

##################################################
# Scatter plots of each PC vs dependent variable with nonlinear fitting

# Make sure X_pca exists
df_PC = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])

fig, axes = plt.subplots(2, 4, figsize=(20, 10))  
axes = axes.flatten()  

for i in range(df_PC.shape[1]):
    x = df_PC.iloc[:, i].values
    axes[i].scatter(x, y, alpha=0.5)
    
    # Nonlinear polynomial fitting (degree=2)
    x_poly = x.reshape(-1, 1)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(x_poly)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    x_fit = np.linspace(x_poly.min(), x_poly.max(), 200).reshape(-1, 1)
    y_fit = model.predict(poly.transform(x_fit))
    axes[i].plot(x_fit, y_fit, color='red', linewidth=3)
    
    axes[i].set_xlabel(f"PC{i+1}", fontsize=12, fontweight='bold')
    axes[i].set_ylabel("Concrete compressive strength", fontsize=12, fontweight='bold')

# Hide unused plots if PCs < 8
for j in range(df_PC.shape[1], 8):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

## Table 4 
##################################################################

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA, FastICA
import statsmodels.api as sm

# -------------------------------
# 1) Load data
# -------------------------------
data = pd.read_csv("data1.csv")
y = data['Concrete compressive strength'].values
feature_names = [col for col in data.columns if col != 'Concrete compressive strength']
X = data[feature_names].values

# -------------------------------
# Function to compute indices
# -------------------------------
def compute_indices(Xmat, y, labels):
    results = []
    for i in range(Xmat.shape[1]):
        xi = Xmat[:, i]

        # Compute correlations
        pearson_corr = pearsonr(xi, y)[0]
        spearman_corr = spearmanr(xi, y).correlation

        # Mutual Information
        mi = mutual_info_regression(xi.reshape(-1, 1), y, random_state=0)[0]

        # Simple univariate regression
        r2_single = sm.OLS(y, sm.add_constant(xi)).fit().rsquared

        results.append({
            "Variable": labels[i],
            "Skewness": skew(xi),
            "Kurtosis": kurtosis(xi),
            "PearsonCorr": pearson_corr,
            "SpearmanCorr": spearman_corr,
            "MI": mi,
            "R2_single": r2_single
        })
    return pd.DataFrame(results)

# -------------------------------
# 2) Indices for original data
# -------------------------------
df_original = compute_indices(X, y, feature_names)
df_original["Type"] = "Original"
df_original["AbsContribution"] = np.nan

# -------------------------------
# 3) Run PCA
# -------------------------------
pca = PCA()
X_pca = pca.fit_transform(X)
pca_labels = [f"PC{i+1}" for i in range(X_pca.shape[1])]
df_pca = compute_indices(X_pca, y, pca_labels)
df_pca["Type"] = "PCA"

# Multivariate regression for all PCs
X_pca_const = sm.add_constant(X_pca)
model_pcr = sm.OLS(y, X_pca_const).fit()
coef_pcr = model_pcr.params[1:]
share_pcr = np.abs(coef_pcr) / np.sum(np.abs(coef_pcr))
df_pca["AbsContribution"] = share_pcr
R2_pcr = model_pcr.rsquared

# -------------------------------
# 4) Run ICA
# -------------------------------
ica = FastICA(random_state=0, max_iter=1000)
X_ica = ica.fit_transform(X)
ica_labels = [f"IC{i+1}" for i in range(X_ica.shape[1])]
df_ica = compute_indices(X_ica, y, ica_labels)
df_ica["Type"] = "ICA"

# Multivariate regression for all ICs
X_ica_const = sm.add_constant(X_ica)
model_icr = sm.OLS(y, X_ica_const).fit()
coef_icr = model_icr.params[1:]
share_icr = np.abs(coef_icr) / np.sum(np.abs(coef_icr))
df_ica["AbsContribution"] = share_icr
R2_icr = model_icr.rsquared

# -------------------------------
# 5) Combine all tables
# -------------------------------
df_all = pd.concat([df_original, df_pca, df_ica], ignore_index=True)
df_all = df_all[[
    "Type", "Variable", "Skewness", "Kurtosis",
    "PearsonCorr", "SpearmanCorr", "MI",
    "R2_single", "AbsContribution"
]]

# -------------------------------
# 6) Add overall R² values of models
# -------------------------------
summary_rows = pd.DataFrame([
    {"Type": "PCR", "Variable": "Overall", "R2_single": R2_pcr},
    {"Type": "ICR", "Variable": "Overall", "R2_single": R2_icr}
])
df_final = pd.concat([df_all, summary_rows], ignore_index=True)

# -------------------------------
# 7) Save output
# -------------------------------
print(df_final)
df_final.to_csv("combined_indices_full.csv", index=False)
df_final.to_excel("combined_indices_full.xlsx", index=False)

## Linear regression results for ICR and PCR  &  Figures 2,3,4 

##########################################################################################################

#برنامه ی نمودارهای مقاله ordered fast ica with replace t distributon instead of normal distribution:###True
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.gofplots import qqplot
import numpy as np
from sklearn.decomposition import FastICA
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pandas as pd
import scipy.stats
import statsmodels.api as sm
from numpy.random import seed
from numpy.random import randn
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.stats import skew
from sklearn.preprocessing import PolynomialFeatures

class CustomFastICA:
    def __init__(self, n_components, df, loc, scale, max_iter=200, tol=1e-4):
        self.n_components = n_components
        self.df = df  # degrees of freedom for the t-distribution
        self.loc = loc  # mean
        self.scale = scale  # standard deviation
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n, m = X.shape
        W = np.random.rand(self.n_components, m)

        for iteration in range(self.max_iter):
            W_old = W.copy()

            # Calculate new weights
            Y = np.dot(X, W.T)
            g = self._tanh(Y)
            g_dot = self._tanh_derivative(Y)
            W = (1 / n) * np.dot(g.T, X) - np.mean(g_dot) * W

            # Symmetric orthogonalization
            W = self._symmetric_orthogonalization(W)

            # Decorrelation
            W = self._decorrelation(W)

            # Check convergence
            if iteration > 0 and np.linalg.norm(W - W_old) < self.tol:
                break

        self.components_ = W
        return self

    def transform(self, X):
        return np.dot(X, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def _symmetric_orthogonalization(self, W):
        u, s, vh = np.linalg.svd(W, full_matrices=False)
        return np.dot(u, vh)

    def _decorrelation(self, W):
        WTW = np.dot(W, W.T)
        _, WTW_eigenvecs = np.linalg.eigh(WTW)
        return np.dot(WTW_eigenvecs.T, W)
    ###########################################################################
    
# Replace this with your actual data
df=pd.read_csv('data1.csv')
X = df.values
# Center the data
X_centered = X - np.mean(X, axis=0)

# Calculate the covariance matrix
cov_matrix = np.cov(X_centered, rowvar=False)

# Perform eigenvalue decomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Whitening transformation
whitening_matrix = np.dot(eigenvectors, np.dot(np.diag(1.0/np.sqrt(eigenvalues + 1e-5)), eigenvectors.T))

# Whiten the data
w_d = np.dot(X_centered, whitening_matrix)
Tw_d = np.transpose(w_d)


#############################################################
s11 = df['Cement ']  # Signal 1 : sinusoidal signal
s22 = df['Blast Furnace Slag']  # Signal 2 : square signal
s33 = df['Fly Ash ']  
s44=df['Water  ']
s55=df['Superplasticize']
s66=df['Coarse Aggregate  \n\n']
s77=df['Fine Aggregate']
s88=df['Age ']
s99=df['Concrete compressive strength']
ica = FastICA(n_components=8)
A = ica.fit_transform(df)
S = np.c_[s11, s22, s33 , s44 , s55, s66, s77, s88 ]
X1 = np.dot(S, A.T) 
X1T= np.transpose(X1)

##########################################################
correlation_coefficients = []

#for i in range(2):
   # Generate some sample data
s1 = Tw_d[0]  # Signal 1 : sinusoidal signal
s2 = Tw_d[1]  # Signal 2 : square signal
s3 = Tw_d[2] 
s4 = Tw_d[3]
s5 = Tw_d[4]
s6 = Tw_d[5] 
s7 = Tw_d[6] 
s8 = Tw_d[7] 
s9 = Tw_d[8] 
#ica = FastICA(n_components=8)
#A = ica.fit_transform(Tw_d)

   # Mix the signals into a single observed signal
   #n=1030
S = np.c_[s1, s2, s3 , s4 , s5, s6, s7, s8]
#X1 = np.dot(S, A.T)  # Observed signal
   


##################
# Center the data
#X_centered = X - np.mean(X, axis=0)

# Instantiate and fit the CustomFastICA model
custom_ica = CustomFastICA(n_components=8, df=8.869463345651127, loc=-0.0364849978662904, scale=0.8750974313065253)
S_ = custom_ica.fit_transform(S)
 # Fit the model 
y = df['Concrete compressive strength']
X = np.array([S_[:, 0],S_[:, 1],S_[:, 2], S_[:, 3],S_[:, 4],S_[:, 5],S_[:, 6],S_[:, 7]])
XT= np.transpose(X)
XT = sm.add_constant(XT)  
model = sm.OLS(y, XT).fit()
# S_ now contains the independent components computed using CustomFastICA with the t-distribution
########################################################

pvalues = model.pvalues
print(pvalues)

plt.figure(figsize=(12, 8))
plt.subplot(2, 4,1)
plt.plot(X1T[0])
plt.subplot(2, 4,2)
plt.plot(X1T[1])
plt.subplot(2, 4,3)
plt.plot(X1T[2])
plt.subplot(2, 4,4)
plt.plot(X1T[3])
plt.subplot(2, 4,5)
plt.plot(X1T[4])
plt.subplot(2, 4,6)
plt.plot(X1T[5])
plt.subplot(2, 4,7)
plt.plot(X1T[6])
plt.subplot(2, 4,8)
plt.plot(X1T[7])
plt.show()
######################
plt.figure(figsize=(12, 8))
plt.subplot(2, 4,1)
plt.hist(X1T[0])
plt.subplot(2, 4,2)
plt.hist(X1T[1])
plt.subplot(2, 4,3)
plt.hist(X1T[2])
plt.subplot(2, 4,4)
plt.hist(X1T[3])
plt.subplot(2, 4,5)
plt.hist(X1T[4])
plt.subplot(2, 4,6)
plt.hist(X1T[5])
plt.subplot(2, 4,7)
plt.hist(X1T[6])
plt.subplot(2, 4,8)
plt.hist(X1T[7])
plt.show()


#######################################

TS_ = np.transpose(S_)
pvalues = model.pvalues
plt.figure(figsize=(12, 8))
print(pvalues)
plt.subplot(2, 4,1)
plt.plot(TS_[0])
plt.subplot(2, 4,2)
plt.plot(TS_[1])
plt.subplot(2, 4,3)
plt.plot(TS_[2])
plt.subplot(2, 4,4)
plt.plot(TS_[3])
plt.subplot(2, 4,5)
plt.plot(TS_[4])
plt.subplot(2, 4,6)
plt.plot(TS_[5])
plt.subplot(2, 4,7)
plt.plot(TS_[6])
plt.subplot(2, 4,8)
plt.plot(TS_[7])
plt.show()

#########################
plt.figure(figsize=(12, 8))
plt.subplot(2, 4,1)
plt.hist(S_[:,0])
plt.subplot(2, 4,2)
plt.hist(S_[:,1])
plt.subplot(2, 4,3)
plt.hist(S_[:,2])
plt.subplot(2, 4,4)
plt.hist(S_[:,3])
plt.subplot(2, 4,5)
plt.hist(S_[:,4])
plt.subplot(2, 4,6)
plt.hist(S_[:,5])
plt.subplot(2, 4,7)
plt.hist(S_[:,6])
plt.subplot(2, 4,8)
plt.hist(S_[:,7])
plt.show()

###########################
   # Print summary 
print(model.summary())
# Calculate the residuals
residuals = y - model.predict(XT)

# Calculate the standard deviation of the residuals
noise_std = np.std(residuals)

print("Noise level (standard deviation of residuals):", noise_std)
TS_ = np.transpose(S_)
pvalues = model.pvalues


   # calculate correlation coefficient
from numpy.random import seed
from numpy.random import randn
from scipy.stats import pearsonr
   # calculate Pearson's correlation
corr1 = pearsonr(TS_[0], s1)
   
corr2 = pearsonr(TS_[1], s2)
corr3 = pearsonr(TS_[2], s3)
corr4 = pearsonr(TS_[3], s4)
corr5 = pearsonr(TS_[4], s5)
corr6 = pearsonr(TS_[5], s6)
corr7 = pearsonr(TS_[6], s7)
corr8 = pearsonr(TS_[7], s8)
   
  

correlation_coefficients.append([corr1, corr2, corr3, corr4,corr5,corr6,corr7,corr8])  
   

  # Assuming vector is your list of numbers
  #X = np.array([TS_[0], TS_[1] ,TS_[2], TS_[3],TS_[4],TS_[5],TS_[6],TS_[7],TS_[8]])
#XT= np.transpose(X)
kurtosis_values = scipy.stats.kurtosis(S_)

print(correlation_coefficients)
print("Kurtosis of the vector:", kurtosis_values)
#print(w_d[0])

# Create scatter plot
plt.subplot(3, 3,1)
plt.scatter(S_[:, 0], S_[:, 1])
plt.subplot(3, 3,2)
plt.scatter(S_[:, 1], S_[:, 2])
plt.subplot(3, 3,3)
plt.scatter(S_[:, 2], S_[:, 3])
plt.subplot(3, 3,4)
plt.scatter(S_[:, 3], S_[:, 4])
plt.subplot(3, 3,5)
plt.scatter(S_[:, 4], S_[:, 5])
plt.subplot(3, 3,6)
plt.scatter(S_[:, 5], S_[:, 6])
plt.subplot(3, 3,7)
plt.scatter(S_[:, 6], S_[:, 7])
plt.show()

plt.subplot(3, 3,1)
plt.scatter(S_[:, 0], y)
plt.subplot(3, 3,2)
plt.scatter(S_[:, 1], y)
plt.subplot(3, 3,3)
plt.scatter(S_[:, 2], y)
plt.subplot(3, 3,4)
plt.scatter(S_[:, 3], y)
plt.subplot(3, 3,5)
plt.scatter(S_[:, 4], y)
plt.subplot(3, 3,6)
plt.scatter(S_[:, 5], y)
plt.subplot(3, 3,7)
plt.scatter(S_[:, 6], y)
plt.subplot(3, 3,8)
plt.scatter(S_[:, 7], y)
plt.show()
############################################################
import numpy as np

def count_outliers(data):
    """
    Count the number of outliers in a dataset using the interquartile range (IQR) method.

    Parameters:
        data (array-like): Input data (list or numpy array).

    Returns:
        num_outliers (int): Number of outliers.
    """
    # Convert to numpy array if input is list
    data = np.array(data)
    
    # Calculate the first and third quartiles
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Define the lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Count the number of outliers
    num_outliers = np.sum((data < lower_bound) | (data > upper_bound))
    return num_outliers

# Example usage:
data = S_
num_outliers = count_outliers(data)
print("Number of outliers:", num_outliers)


###################################################

# Show the plot
#plt.xlabel('Third Principal Component')
#plt.ylabel('Sixth Principal Component')
#plt.title('Scatter Plot of Third vs Sixth Principal Components')
#plt.show()


############ correlation between y and S_  for noise:
corr11 = pearsonr(TS_[0], y)
   
corr12 = pearsonr(TS_[1], y)
corr13 = pearsonr(TS_[2], y)
corr14 = pearsonr(TS_[3], y)
corr15 = pearsonr(TS_[4], y)
corr16 = pearsonr(TS_[5], y)
corr17 = pearsonr(TS_[6], y)
corr18 = pearsonr(TS_[7], y)
   
  

correlation_coefficients.append([corr11, corr12, corr13, corr14,corr15,corr16,corr17,corr18])  
   

  # Assuming vector is your list of numbers
  #X = np.array([TS_[0], TS_[1] ,TS_[2], TS_[3],TS_[4],TS_[5],TS_[6],TS_[7],TS_[8]])
#XT= np.transpose(X)

print(correlation_coefficients)

skewness_values = scipy.stats.skew(S_, axis=0)
print("Skewness of the independent components:")
print(skewness_values)


########################3
#qqplot for cosider symmetric of S_

# Calculate the residuals
residuals = y - model.predict(XT)

# Generate Q-Q plots for the residuals
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i in range(8):
    ax = axes[i // 4, i % 4]
    qqplot(residuals, line='s', ax=ax)
    ax.set_title(f"Residuals Q-Q Plot for Component {i+1}")

plt.tight_layout()
plt.show()

##########################################
#Anova for realize nonlinear relationship between y and S_:

# Generate polynomial features for the independent variables
poly = PolynomialFeatures(degree=2)  # You can adjust the degree as needed
S_poly = poly.fit_transform(S_)

# Fit the ANOVA model with polynomial terms
model_anova = sm.OLS(y, S_poly).fit()

# Print summary of ANOVA model
print(model_anova.summary())


##################################################
# plot of residual against of predicted value:
#####################################
# Assuming S_ contains the independent components computed using CustomFastICA with the t-distribution

# Reshape the first column of S_ to a 2D array with a single feature
X_first_column = S_[:, 0].reshape(-1, 1)
X_two_column = S_[:, 1].reshape(-1, 1)
X_three_column = S_[:, 2].reshape(-1, 1)
X_four_column = S_[:, 3].reshape(-1, 1)
X_five_column = S_[:, 4].reshape(-1, 1)
X_six_column = S_[:, 5].reshape(-1, 1)
X_seven_column = S_[:, 6].reshape(-1, 1)
X_eight_column = S_[:, 7].reshape(-1, 1)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_first_column, y)
model.fit(X_two_column, y)
model.fit(X_three_column, y)
model.fit(X_four_column, y)
model.fit(X_five_column , y)
model.fit(X_six_column, y)
model.fit(X_seven_column, y)
model.fit(X_eight_column, y)

# Calculate residuals
residuals1 = y - model.predict(X_first_column)
residuals2 = y - model.predict(X_two_column)
residuals3 = y - model.predict(X_three_column)
residuals4 = y - model.predict(X_four_column)
residuals5 = y - model.predict(X_five_column )
residuals6 = y - model.predict(X_six_column)
residuals7 = y - model.predict(X_seven_column)
residuals8= y - model.predict(X_eight_column)

# Display the model coefficients
print("Model Coefficients:", model.coef_)

# Display the residuals
#print("Residuals:")
#for value in residuals:
#    print(value)
    
# Plot residuals vs. predicted values
predicted_values1 = model.predict(X_first_column)
predicted_values2 = model.predict(X_two_column)
predicted_values3 = model.predict(X_three_column)
predicted_values4 = model.predict(X_four_column)
predicted_values5 = model.predict(X_five_column)
predicted_values6 = model.predict(X_six_column)
predicted_values7 = model.predict(X_seven_column)
predicted_values8 = model.predict(X_eight_column)

plt.subplot(3, 3,1)
plt.scatter(predicted_values1, residuals1)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,2)
plt.scatter(predicted_values2, residuals2)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,3)
plt.scatter(predicted_values3, residuals3)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,4)
plt.scatter(predicted_values4, residuals4)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,5)
plt.scatter(predicted_values5, residuals5)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,6)
plt.scatter(predicted_values6, residuals6)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,7)
plt.scatter(predicted_values7, residuals7)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,8)
plt.scatter(predicted_values8, residuals8)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0
plt.show()

### تشخیص هم خطی
  # calculate correlation coefficient
from numpy.random import seed
from numpy.random import randn
from scipy.stats import pearsonr
# calculate Pearson's correlation
corr1 = pearsonr(TS_[0], df['Cement '])
corr11 = pearsonr(TS_[0],df['Blast Furnace Slag'] )
corr12 = pearsonr(TS_[0],df['Fly Ash '] )
corr13 = pearsonr(TS_[0], df['Water  '])
corr14 = pearsonr(TS_[0], df['Superplasticize'])
corr15 = pearsonr(TS_[0], df['Coarse Aggregate  \n\n'])
corr16 = pearsonr(TS_[0],df['Fine Aggregate'])
corr17 = pearsonr(TS_[0], df['Age '])
#corr18 = pearsonr(TS_[0], 'Concrete compressive strength')
#corr2 = pearsonr(TS_[1], s1)
#corr21 = pearsonr(TS_[1], s2)
#corr22 = pearsonr(TS_[1], s3)
#corr23 = pearsonr(TS_[1], s4)
#corr24 = pearsonr(TS_[1], s5)
#corr25 = pearsonr(TS_[1], s6)
#corr26 = pearsonr(TS_[1], s7)
#corr27 = pearsonr(TS_[1], s8)
#corr28 = pearsonr(TS_[1], s9)
#corr3 = pearsonr(TS_[2], s1)
#corr31 = pearsonr(TS_[2], s2)
#corr32 = pearsonr(TS_[2], s3)
#corr33 = pearsonr(TS_[2], s4) 
#corr34 = pearsonr(TS_[2], s5)
#corr35 = pearsonr(TS_[2], s6)
#corr36 = pearsonr(TS_[2], s7) 
#corr37 = pearsonr(TS_[2], s8)
#corr38 = pearsonr(TS_[2], s9) 
#corr4 = pearsonr(TS_[3], s1)
#corr41 = pearsonr(TS_[3], s2) 
#corr42 = pearsonr(TS_[3], s3) 
#corr43 = pearsonr(TS_[3], s4) 
#corr44 = pearsonr(TS_[3], s5) 
#corr45 = pearsonr(TS_[3], s6) 
#corr46 = pearsonr(TS_[3], s7)
#corr47 = pearsonr(TS_[3], s8) 
#corr48 = pearsonr(TS_[3], s9)
#corr5 = pearsonr(TS_[4], s1)
#corr51 = pearsonr(TS_[4], s2)
#corr52 = pearsonr(TS_[4], s3)
#corr53 = pearsonr(TS_[4], s4)
#corr54 = pearsonr(TS_[4], s5)
#corr55 = pearsonr(TS_[4], s6)
#corr56 = pearsonr(TS_[4], s7)
#corr57 = pearsonr(TS_[4], s8)
#corr58 = pearsonr(TS_[4], s9)
#corr6 = pearsonr(TS_[5], s1)
#corr61 = pearsonr(TS_[5], s2)
#corr62= pearsonr(TS_[5], s3)
#corr63= pearsonr(TS_[5], s4)
#corr64= pearsonr(TS_[5], s5)
##corr65= pearsonr(TS_[5], s6)
#corr66= pearsonr(TS_[5], s7)
#corr67= pearsonr(TS_[5], s8)
#corr68= pearsonr(TS_[5], s9)
#corr7 = pearsonr(TS_[6], s1)
#corr71 = pearsonr(TS_[6], s2)
#corr72 = pearsonr(TS_[6], s3)
#corr73= pearsonr(TS_[6], s4)
#corr74 = pearsonr(TS_[6], s5)
#corr75 = pearsonr(TS_[6], s6)
#corr76 = pearsonr(TS_[6], s7)
#corr77 = pearsonr(TS_[6], s8)
#corr78 = pearsonr(TS_[6], s9)
#corr8 = pearsonr(TS_[7], s1)
#corr81 = pearsonr(TS_[7], s2)
#corr82 = pearsonr(TS_[7], s3)
#corr83 = pearsonr(TS_[7], s4)
#corr84 = pearsonr(TS_[7], s5)
#corr85 = pearsonr(TS_[7], s6)
#corr86 = pearsonr(TS_[7], s7)
#corr87 = pearsonr(TS_[7], s8)
#corr88 = pearsonr(TS_[7], s9)
#corr9 = pearsonr(TS_[8], s1)
#corr91 = pearsonr(TS_[8], s2)
#corr92= pearsonr(TS_[8], s3)
#corr93 = pearsonr(TS_[8], s4)
#corr94 = pearsonr(TS_[8], s5)
#corr95 = pearsonr(TS_[8], s6)
#corr96 = pearsonr(TS_[8], s7)
#corr97 = pearsonr(TS_[8], s8)
#corr98 = pearsonr(TS_[8], s9)

correlation_coefficients.append([corr1,corr11,corr12,corr13,corr14,corr15,corr16,corr17])#, corr2,corr21,corr22,corr23,corr24,corr25,corr26,corr27,corr3,corr31,corr32,corr33,corr34,corr35, corr36, corr37, corr4,corr41,corr42,corr43,corr44,corr45,corr46,corr47,corr5,corr51,corr52,corr53,corr54,corr55,corr56,corr57, corr6,corr61,corr62,corr63,corr64,corr65,corr66,corr67,corr7,corr71,corr72,corr73,corr74,corr75,corr76,corr77,corr8, corr81,corr82,corr83,corr84,corr85,corr86,corr87]) 



plt.subplot(3, 3,1)
plt.hist(s11)
plt.subplot(3, 3,2)
plt.hist(s22)
plt.subplot(3, 3,3)
plt.hist(s33)
plt.subplot(3, 3,4)
plt.hist(s44)
plt.subplot(3, 3,5)
plt.hist(s55)
plt.subplot(3, 3,6)
plt.hist(s66)
plt.subplot(3, 3,7)
plt.hist(s77)
plt.subplot(3, 3,8)
plt.hist(s88)
#plt.subplot(3, 3,9)
#plt.hist(s99)
plt.show()




# concrete data, Results for Lasso regression: 

# ordered fast ica with replace t distributon instead on normal distribution.... Lasso Regression   :###True
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.gofplots import qqplot
import numpy as np
from sklearn.decomposition import FastICA
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pandas as pd
import scipy.stats
import statsmodels.api as sm
from numpy.random import seed
from numpy.random import randn
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.stats import skew
from sklearn.preprocessing import PolynomialFeatures

class CustomFastICA:
    def __init__(self, n_components, df, loc, scale, max_iter=200, tol=1e-4):
        self.n_components = n_components
        self.df = df  # degrees of freedom for the t-distribution
        self.loc = loc  # mean
        self.scale = scale  # standard deviation
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n, m = X.shape
        W = np.random.rand(self.n_components, m)

        for iteration in range(self.max_iter):
            W_old = W.copy()

            # Calculate new weights
            Y = np.dot(X, W.T)
            g = self._tanh(Y)
            g_dot = self._tanh_derivative(Y)
            W = (1 / n) * np.dot(g.T, X) - np.mean(g_dot) * W

            # Symmetric orthogonalization
            W = self._symmetric_orthogonalization(W)

            # Decorrelation
            W = self._decorrelation(W)

            # Check convergence
            if iteration > 0 and np.linalg.norm(W - W_old) < self.tol:
                break

        self.components_ = W
        return self

    def transform(self, X):
        return np.dot(X, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def _symmetric_orthogonalization(self, W):
        u, s, vh = np.linalg.svd(W, full_matrices=False)
        return np.dot(u, vh)

    def _decorrelation(self, W):
        WTW = np.dot(W, W.T)
        _, WTW_eigenvecs = np.linalg.eigh(WTW)
        return np.dot(WTW_eigenvecs.T, W)
    ###########################################################################
    
# Replace this with your actual data
df=pd.read_csv('data1.csv')
X = df.values
# Center the data
X_centered = X - np.mean(X, axis=0)

# Calculate the covariance matrix
cov_matrix = np.cov(X_centered, rowvar=False)

# Perform eigenvalue decomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Whitening transformation
whitening_matrix = np.dot(eigenvectors, np.dot(np.diag(1.0/np.sqrt(eigenvalues + 1e-5)), eigenvectors.T))

# Whiten the data
w_d = np.dot(X_centered, whitening_matrix)
Tw_d = np.transpose(w_d)


correlation_coefficients = []

#for i in range(2):
   # Generate some sample data
s1 = Tw_d[0]  # Signal 1 : sinusoidal signal
s2 = Tw_d[1]  # Signal 2 : square signal
s3 = Tw_d[2] 
s4 = Tw_d[3]
s5 = Tw_d[4]
s6 = Tw_d[5] 
s7 = Tw_d[6] 
s8 = Tw_d[7] 
s9 = Tw_d[8] 
#ica = FastICA(n_components=8)
#A = ica.fit_transform(Tw_d)

   # Mix the signals into a single observed signal
   #n=1030
S = np.c_[s1, s2, s3 , s4 , s5, s6, s7, s8]
#X1 = np.dot(S, A.T)  # Observed signal
   


##################
# Center the data
#X_centered = X - np.mean(X, axis=0)

# Instantiate and fit the CustomFastICA model
custom_ica = CustomFastICA(n_components=8, df=8.869463345651127, loc=-0.0364849978662904, scale=0.8750974313065253)
S_ = custom_ica.fit_transform(S)
 # Fit the model 
y = df['Concrete compressive strength']
X = np.array([S_[:, 0],S_[:, 1],S_[:, 2], S_[:, 3],S_[:, 4],S_[:, 5],S_[:, 6],S_[:, 7]])
XT= np.transpose(X)
XT = sm.add_constant(XT)  
# S_ now contains the independent components computed using CustomFastICA with the t-distribution

###################################################
# Fit the model using Lasso regression
alpha = 1.0  # You can adjust the regularization strength as needed
lasso_model = Lasso(alpha=alpha)
lasso_model.fit(XT, y)

# Calculate predicted values
y_pred_lasso = lasso_model.predict(XT)

# Calculate R-squared manually for Lasso regression
ss_residual_lasso = np.sum((y - y_pred_lasso) ** 2)
r_squared_lasso = 1 - (ss_residual_lasso / ss_total)
print("R-squared (Lasso):", r_squared_lasso)

# Print intercept and coefficients for Lasso regression
print("Intercept (Lasso):", lasso_model.intercept_)
print("Coefficients (Lasso):", lasso_model.coef_)

########################################################

TS_ = np.transpose(S_)
#pvalues = ridge_model.pvalues
#print(pvalues)
plt.subplot(3, 3,1)
plt.plot(TS_[0])
plt.subplot(3, 3,2)
plt.plot(TS_[1])
plt.subplot(3, 3,3)
plt.plot(TS_[2])
plt.subplot(3, 3,4)
plt.plot(TS_[3])
plt.subplot(3, 3,5)
plt.plot(TS_[4])
plt.subplot(3, 3,6)
plt.plot(TS_[5])
plt.subplot(3, 3,7)
plt.plot(TS_[6])
plt.subplot(3, 3,8)
plt.plot(TS_[7])
plt.show()

   # Print summary 

# Calculate the residuals
residuals = y - lasso_model.predict(XT)

# Calculate the standard deviation of the residuals
noise_std = np.std(residuals)

print("Noise level (standard deviation of residuals):", noise_std)
TS_ = np.transpose(S_)
#pvalues = ridge_model.pvalues
plt.subplot(3, 3,1)
plt.hist(TS_[0])
plt.subplot(3, 3,2)
plt.hist(TS_[1])
plt.subplot(3, 3,3)
plt.hist(TS_[2])
plt.subplot(3, 3,4)
plt.hist(TS_[3])
plt.subplot(3, 3,5)
plt.hist(TS_[4])
plt.subplot(3, 3,6)
plt.hist(TS_[5])
plt.subplot(3, 3,7)
plt.hist(TS_[6])
plt.subplot(3, 3,8)
plt.hist(TS_[7])
plt.show()

   # calculate correlation coefficient
from numpy.random import seed
from numpy.random import randn
from scipy.stats import pearsonr
   # calculate Pearson's correlation
corr1 = pearsonr(TS_[0], s1)
   
corr2 = pearsonr(TS_[1], s2)
corr3 = pearsonr(TS_[2], s3)
corr4 = pearsonr(TS_[3], s4)
corr5 = pearsonr(TS_[4], s5)
corr6 = pearsonr(TS_[5], s6)
corr7 = pearsonr(TS_[6], s7)
corr8 = pearsonr(TS_[7], s8)
   
  

correlation_coefficients.append([corr1, corr2, corr3, corr4,corr5,corr6,corr7,corr8])  
   

  # Assuming vector is your list of numbers
  #X = np.array([TS_[0], TS_[1] ,TS_[2], TS_[3],TS_[4],TS_[5],TS_[6],TS_[7],TS_[8]])
#XT= np.transpose(X)
kurtosis_values = scipy.stats.kurtosis(S_)

print(correlation_coefficients)
print("Kurtosis of the vector:", kurtosis_values)
#print(w_d[0])

# Create scatter plot
plt.subplot(3, 3,1)
plt.scatter(S_[:, 0], S_[:, 1])
plt.subplot(3, 3,2)
plt.scatter(S_[:, 1], S_[:, 2])
plt.subplot(3, 3,3)
plt.scatter(S_[:, 2], S_[:, 3])
plt.subplot(3, 3,4)
plt.scatter(S_[:, 3], S_[:, 4])
plt.subplot(3, 3,5)
plt.scatter(S_[:, 4], S_[:, 5])
plt.subplot(3, 3,6)
plt.scatter(S_[:, 5], S_[:, 6])
plt.subplot(3, 3,7)
plt.scatter(S_[:, 6], S_[:, 7])
plt.show()

plt.subplot(3, 3,1)
plt.scatter(S_[:, 0], y)
plt.subplot(3, 3,2)
plt.scatter(S_[:, 1], y)
plt.subplot(3, 3,3)
plt.scatter(S_[:, 2], y)
plt.subplot(3, 3,4)
plt.scatter(S_[:, 3], y)
plt.subplot(3, 3,5)
plt.scatter(S_[:, 4], y)
plt.subplot(3, 3,6)
plt.scatter(S_[:, 5], y)
plt.subplot(3, 3,7)
plt.scatter(S_[:, 6], y)
plt.subplot(3, 3,8)
plt.scatter(S_[:, 7], y)
plt.show()
############################################################
import numpy as np

def count_outliers(data):
    """
    Count the number of outliers in a dataset using the interquartile range (IQR) method.

    Parameters:
        data (array-like): Input data (list or numpy array).

    Returns:
        num_outliers (int): Number of outliers.
    """
    # Convert to numpy array if input is list
    data = np.array(data)
    
    # Calculate the first and third quartiles
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Define the lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Count the number of outliers
    num_outliers = np.sum((data < lower_bound) | (data > upper_bound))
    return num_outliers

# Example usage:
data = S_
num_outliers = count_outliers(data)
print("Number of outliers:", num_outliers)


###################################################

# Show the plot
#plt.xlabel('Third Principal Component')
#plt.ylabel('Sixth Principal Component')
#plt.title('Scatter Plot of Third vs Sixth Principal Components')
#plt.show()


############ correlation between y and S_  for noise:
corr11 = pearsonr(TS_[0], y)
   
corr12 = pearsonr(TS_[1], y)
corr13 = pearsonr(TS_[2], y)
corr14 = pearsonr(TS_[3], y)
corr15 = pearsonr(TS_[4], y)
corr16 = pearsonr(TS_[5], y)
corr17 = pearsonr(TS_[6], y)
corr18 = pearsonr(TS_[7], y)
   
  

correlation_coefficients.append([corr11, corr12, corr13, corr14,corr15,corr16,corr17,corr18])  
   

  # Assuming vector is your list of numbers
  #X = np.array([TS_[0], TS_[1] ,TS_[2], TS_[3],TS_[4],TS_[5],TS_[6],TS_[7],TS_[8]])
#XT= np.transpose(X)

print(correlation_coefficients)

skewness_values = scipy.stats.skew(S_, axis=0)
print("Skewness of the independent components:")
print(skewness_values)


########################3
#qqplot for cosider symmetric of S_

# Calculate the residuals
residuals = y -  lasso_model.predict(XT)

# Generate Q-Q plots for the residuals
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i in range(8):
    ax = axes[i // 4, i % 4]
    qqplot(residuals, line='s', ax=ax)
    ax.set_title(f"Residuals Q-Q Plot for Component {i+1}")

plt.tight_layout()
plt.show()

##########################################
#Anova for realize nonlinear relationship between y and S_:

# Generate polynomial features for the independent variables
poly = PolynomialFeatures(degree=2)  # You can adjust the degree as needed
S_poly = poly.fit_transform(S_)

# Fit the ANOVA model with polynomial terms
model_anova = sm.OLS(y, S_poly).fit()

# Print summary of ANOVA model
print(model_anova.summary())


##################################################
# plot of residual against of predicted value:
#####################################
# Assuming S_ contains the independent components computed using CustomFastICA with the t-distribution

# Reshape the first column of S_ to a 2D array with a single feature
X_first_column = S_[:, 0].reshape(-1, 1)
X_two_column = S_[:, 1].reshape(-1, 1)
X_three_column = S_[:, 2].reshape(-1, 1)
X_four_column = S_[:, 3].reshape(-1, 1)
X_five_column = S_[:, 4].reshape(-1, 1)
X_six_column = S_[:, 5].reshape(-1, 1)
X_seven_column = S_[:, 6].reshape(-1, 1)
X_eight_column = S_[:, 7].reshape(-1, 1)

# Fit a linear regression model
#model = LinearRegression()
lasso_model.fit(X_first_column, y)
lasso_model.fit(X_two_column, y)
lasso_model.fit(X_three_column, y)
lasso_model.fit(X_four_column, y)
lasso_model.fit(X_five_column , y)
lasso_model.fit(X_six_column, y)
lasso_model.fit(X_seven_column, y)
lasso_model.fit(X_eight_column, y)

# Calculate residuals
residuals1 = y - lasso_model.predict(X_first_column)
residuals2 = y - lasso_model.predict(X_two_column)
residuals3 = y - lasso_model.predict(X_three_column)
residuals4 = y - lasso_model.predict(X_four_column)
residuals5 = y - lasso_model.predict(X_five_column )
residuals6 = y - lasso_model.predict(X_six_column)
residuals7 = y - lasso_model.predict(X_seven_column)
residuals8= y - lasso_model.predict(X_eight_column)

# Display the model coefficients
print("Model Coefficients:", lasso_model.coef_)

# Display the residuals
#print("Residuals:")
#for value in residuals:
#    print(value)
    
# Plot residuals vs. predicted values
predicted_values1 = lasso_model.predict(X_first_column)
predicted_values2 = lasso_model.predict(X_two_column)
predicted_values3 = lasso_model.predict(X_three_column)
predicted_values4 = lasso_model.predict(X_four_column)
predicted_values5 = lasso_model.predict(X_five_column)
predicted_values6 = lasso_model.predict(X_six_column)
predicted_values7 = lasso_model.predict(X_seven_column)
predicted_values8 = lasso_model.predict(X_eight_column)

plt.subplot(3, 3,1)
plt.scatter(predicted_values1, residuals1)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,2)
plt.scatter(predicted_values2, residuals2)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,3)
plt.scatter(predicted_values3, residuals3)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,4)
plt.scatter(predicted_values4, residuals4)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,5)
plt.scatter(predicted_values5, residuals5)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,6)
plt.scatter(predicted_values6, residuals6)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,7)
plt.scatter(predicted_values7, residuals7)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,8)
plt.scatter(predicted_values8, residuals8)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0
plt.show()




## concrete data ## results for ridge regression

##########################################################################
 
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.gofplots import qqplot
import numpy as np
from sklearn.decomposition import FastICA
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pandas as pd
import scipy.stats
import statsmodels.api as sm
from numpy.random import seed
from numpy.random import randn
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.stats import skew
from sklearn.preprocessing import PolynomialFeatures

class CustomFastICA:
    def __init__(self, n_components, df, loc, scale, max_iter=200, tol=1e-4):
        self.n_components = n_components
        self.df = df  # degrees of freedom for the t-distribution
        self.loc = loc  # mean
        self.scale = scale  # standard deviation
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n, m = X.shape
        W = np.random.rand(self.n_components, m)

        for iteration in range(self.max_iter):
            W_old = W.copy()

            # Calculate new weights
            Y = np.dot(X, W.T)
            g = self._tanh(Y)
            g_dot = self._tanh_derivative(Y)
            W = (1 / n) * np.dot(g.T, X) - np.mean(g_dot) * W

            # Symmetric orthogonalization
            W = self._symmetric_orthogonalization(W)

            # Decorrelation
            W = self._decorrelation(W)

            # Check convergence
            if iteration > 0 and np.linalg.norm(W - W_old) < self.tol:
                break

        self.components_ = W
        return self

    def transform(self, X):
        return np.dot(X, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def _symmetric_orthogonalization(self, W):
        u, s, vh = np.linalg.svd(W, full_matrices=False)
        return np.dot(u, vh)

    def _decorrelation(self, W):
        WTW = np.dot(W, W.T)
        _, WTW_eigenvecs = np.linalg.eigh(WTW)
        return np.dot(WTW_eigenvecs.T, W)
    ###########################################################################
    
# Replace this with your actual data
df=pd.read_csv('data1.csv')
X = df.values
# Center the data
X_centered = X - np.mean(X, axis=0)

# Calculate the covariance matrix
cov_matrix = np.cov(X_centered, rowvar=False)

# Perform eigenvalue decomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Whitening transformation
whitening_matrix = np.dot(eigenvectors, np.dot(np.diag(1.0/np.sqrt(eigenvalues + 1e-5)), eigenvectors.T))

# Whiten the data
w_d = np.dot(X_centered, whitening_matrix)
Tw_d = np.transpose(w_d)


correlation_coefficients = []

#for i in range(2):
   # Generate some sample data
s1 = Tw_d[0]  # Signal 1 : sinusoidal signal
s2 = Tw_d[1]  # Signal 2 : square signal
s3 = Tw_d[2] 
s4 = Tw_d[3]
s5 = Tw_d[4]
s6 = Tw_d[5] 
s7 = Tw_d[6] 
s8 = Tw_d[7] 
s9 = Tw_d[8] 
#ica = FastICA(n_components=8)
#A = ica.fit_transform(Tw_d)

   # Mix the signals into a single observed signal
   #n=1030
S = np.c_[s1, s2, s3 , s4 , s5, s6, s7, s8]
#X1 = np.dot(S, A.T)  # Observed signal
   


##################
# Center the data
#X_centered = X - np.mean(X, axis=0)

# Instantiate and fit the CustomFastICA model
custom_ica = CustomFastICA(n_components=8, df=8.869463345651127, loc=-0.0364849978662904, scale=0.8750974313065253)
S_ = custom_ica.fit_transform(S)
 # Fit the model 
y = df['Concrete compressive strength']
X = np.array([S_[:, 0],S_[:, 1],S_[:, 2], S_[:, 3],S_[:, 4],S_[:, 5],S_[:, 6],S_[:, 7]])
XT= np.transpose(X)
XT = sm.add_constant(XT)  
# S_ now contains the independent components computed using CustomFastICA with the t-distribution

###################################################
# Fit the model using Ridge regression

import numpy as np

# Fit the model using Ridge regression
alpha = 1.0  # You can adjust the regularization strength as needed
ridge_model = Ridge(alpha=alpha)
ridge_model.fit(XT, y)

# Calculate predicted values
y_pred = ridge_model.predict(XT)

# Calculate R-squared manually
ss_residual = np.sum((y - y_pred) ** 2)
ss_total = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (ss_residual / ss_total)
print("R-squared:", r_squared)

# Print intercept and coefficients
print("Intercept:", ridge_model.intercept_)
print("Coefficients:", ridge_model.coef_)

########################################################

TS_ = np.transpose(S_)
#pvalues = ridge_model.pvalues
#print(pvalues)
plt.subplot(3, 3,1)
plt.plot(TS_[0])
plt.subplot(3, 3,2)
plt.plot(TS_[1])
plt.subplot(3, 3,3)
plt.plot(TS_[2])
plt.subplot(3, 3,4)
plt.plot(TS_[3])
plt.subplot(3, 3,5)
plt.plot(TS_[4])
plt.subplot(3, 3,6)
plt.plot(TS_[5])
plt.subplot(3, 3,7)
plt.plot(TS_[6])
plt.subplot(3, 3,8)
plt.plot(TS_[7])
plt.show()

   # Print summary 

# Calculate the residuals
residuals = y - ridge_model.predict(XT)

# Calculate the standard deviation of the residuals
noise_std = np.std(residuals)

print("Noise level (standard deviation of residuals):", noise_std)
TS_ = np.transpose(S_)
#pvalues = ridge_model.pvalues
plt.subplot(3, 3,1)
plt.hist(TS_[0])
plt.subplot(3, 3,2)
plt.hist(TS_[1])
plt.subplot(3, 3,3)
plt.hist(TS_[2])
plt.subplot(3, 3,4)
plt.hist(TS_[3])
plt.subplot(3, 3,5)
plt.hist(TS_[4])
plt.subplot(3, 3,6)
plt.hist(TS_[5])
plt.subplot(3, 3,7)
plt.hist(TS_[6])
plt.subplot(3, 3,8)
plt.hist(TS_[7])
plt.show()

   # calculate correlation coefficient
from numpy.random import seed
from numpy.random import randn
from scipy.stats import pearsonr
   # calculate Pearson's correlation
corr1 = pearsonr(TS_[0], s1)
   
corr2 = pearsonr(TS_[1], s2)
corr3 = pearsonr(TS_[2], s3)
corr4 = pearsonr(TS_[3], s4)
corr5 = pearsonr(TS_[4], s5)
corr6 = pearsonr(TS_[5], s6)
corr7 = pearsonr(TS_[6], s7)
corr8 = pearsonr(TS_[7], s8)
   
  

correlation_coefficients.append([corr1, corr2, corr3, corr4,corr5,corr6,corr7,corr8])  
   

  # Assuming vector is your list of numbers
  #X = np.array([TS_[0], TS_[1] ,TS_[2], TS_[3],TS_[4],TS_[5],TS_[6],TS_[7],TS_[8]])
#XT= np.transpose(X)
kurtosis_values = scipy.stats.kurtosis(S_)

print(correlation_coefficients)
print("Kurtosis of the vector:", kurtosis_values)
#print(w_d[0])

# Create scatter plot
plt.subplot(3, 3,1)
plt.scatter(S_[:, 0], S_[:, 1])
plt.subplot(3, 3,2)
plt.scatter(S_[:, 1], S_[:, 2])
plt.subplot(3, 3,3)
plt.scatter(S_[:, 2], S_[:, 3])
plt.subplot(3, 3,4)
plt.scatter(S_[:, 3], S_[:, 4])
plt.subplot(3, 3,5)
plt.scatter(S_[:, 4], S_[:, 5])
plt.subplot(3, 3,6)
plt.scatter(S_[:, 5], S_[:, 6])
plt.subplot(3, 3,7)
plt.scatter(S_[:, 6], S_[:, 7])
plt.show()

plt.subplot(3, 3,1)
plt.scatter(S_[:, 0], y)
plt.subplot(3, 3,2)
plt.scatter(S_[:, 1], y)
plt.subplot(3, 3,3)
plt.scatter(S_[:, 2], y)
plt.subplot(3, 3,4)
plt.scatter(S_[:, 3], y)
plt.subplot(3, 3,5)
plt.scatter(S_[:, 4], y)
plt.subplot(3, 3,6)
plt.scatter(S_[:, 5], y)
plt.subplot(3, 3,7)
plt.scatter(S_[:, 6], y)
plt.subplot(3, 3,8)
plt.scatter(S_[:, 7], y)
plt.show()
############################################################
import numpy as np

def count_outliers(data):
    """
    Count the number of outliers in a dataset using the interquartile range (IQR) method.

    Parameters:
        data (array-like): Input data (list or numpy array).

    Returns:
        num_outliers (int): Number of outliers.
    """
    # Convert to numpy array if input is list
    data = np.array(data)
    
    # Calculate the first and third quartiles
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Define the lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Count the number of outliers
    num_outliers = np.sum((data < lower_bound) | (data > upper_bound))
    return num_outliers

# Example usage:
data = S_
num_outliers = count_outliers(data)
print("Number of outliers:", num_outliers)


###################################################

# Show the plot
#plt.xlabel('Third Principal Component')
#plt.ylabel('Sixth Principal Component')
#plt.title('Scatter Plot of Third vs Sixth Principal Components')
#plt.show()


############ correlation between y and S_  for noise:
corr11 = pearsonr(TS_[0], y)
   
corr12 = pearsonr(TS_[1], y)
corr13 = pearsonr(TS_[2], y)
corr14 = pearsonr(TS_[3], y)
corr15 = pearsonr(TS_[4], y)
corr16 = pearsonr(TS_[5], y)
corr17 = pearsonr(TS_[6], y)
corr18 = pearsonr(TS_[7], y)
   
  

correlation_coefficients.append([corr11, corr12, corr13, corr14,corr15,corr16,corr17,corr18])  
   

  # Assuming vector is your list of numbers
  #X = np.array([TS_[0], TS_[1] ,TS_[2], TS_[3],TS_[4],TS_[5],TS_[6],TS_[7],TS_[8]])
#XT= np.transpose(X)

print(correlation_coefficients)

skewness_values = scipy.stats.skew(S_, axis=0)
print("Skewness of the independent components:")
print(skewness_values)


########################3
#qqplot for cosider symmetric of S_

# Calculate the residuals
residuals = y -  ridge_model.predict(XT)

# Generate Q-Q plots for the residuals
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i in range(8):
    ax = axes[i // 4, i % 4]
    qqplot(residuals, line='s', ax=ax)
    ax.set_title(f"Residuals Q-Q Plot for Component {i+1}")

plt.tight_layout()
plt.show()

##########################################
#Anova for realize nonlinear relationship between y and S_:

# Generate polynomial features for the independent variables
poly = PolynomialFeatures(degree=2)  # You can adjust the degree as needed
S_poly = poly.fit_transform(S_)

# Fit the ANOVA model with polynomial terms
model_anova = sm.OLS(y, S_poly).fit()

# Print summary of ANOVA model
print(model_anova.summary())


##################################################
# plot of residual against of predicted value:
#####################################
# Assuming S_ contains the independent components computed using CustomFastICA with the t-distribution

# Reshape the first column of S_ to a 2D array with a single feature
X_first_column = S_[:, 0].reshape(-1, 1)
X_two_column = S_[:, 1].reshape(-1, 1)
X_three_column = S_[:, 2].reshape(-1, 1)
X_four_column = S_[:, 3].reshape(-1, 1)
X_five_column = S_[:, 4].reshape(-1, 1)
X_six_column = S_[:, 5].reshape(-1, 1)
X_seven_column = S_[:, 6].reshape(-1, 1)
X_eight_column = S_[:, 7].reshape(-1, 1)

# Fit a linear regression model
#model = LinearRegression()
ridge_model.fit(X_first_column, y)
ridge_model.fit(X_two_column, y)
ridge_model.fit(X_three_column, y)
ridge_model.fit(X_four_column, y)
ridge_model.fit(X_five_column , y)
ridge_model.fit(X_six_column, y)
ridge_model.fit(X_seven_column, y)
ridge_model.fit(X_eight_column, y)

# Calculate residuals
residuals1 = y - ridge_model.predict(X_first_column)
residuals2 = y - ridge_model.predict(X_two_column)
residuals3 = y - ridge_model.predict(X_three_column)
residuals4 = y - ridge_model.predict(X_four_column)
residuals5 = y - ridge_model.predict(X_five_column )
residuals6 = y - ridge_model.predict(X_six_column)
residuals7 = y - ridge_model.predict(X_seven_column)
residuals8= y - ridge_model.predict(X_eight_column)

# Display the model coefficients
print("Model Coefficients:", ridge_model.coef_)

# Display the residuals
#print("Residuals:")
#for value in residuals:
#    print(value)
    
# Plot residuals vs. predicted values
predicted_values1 = ridge_model.predict(X_first_column)
predicted_values2 = ridge_model.predict(X_two_column)
predicted_values3 = ridge_model.predict(X_three_column)
predicted_values4 = ridge_model.predict(X_four_column)
predicted_values5 = ridge_model.predict(X_five_column)
predicted_values6 = ridge_model.predict(X_six_column)
predicted_values7 = ridge_model.predict(X_seven_column)
predicted_values8 = ridge_model.predict(X_eight_column)

plt.subplot(3, 3,1)
plt.scatter(predicted_values1, residuals1)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,2)
plt.scatter(predicted_values2, residuals2)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,3)
plt.scatter(predicted_values3, residuals3)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,4)
plt.scatter(predicted_values4, residuals4)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,5)
plt.scatter(predicted_values5, residuals5)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,6)
plt.scatter(predicted_values6, residuals6)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,7)
plt.scatter(predicted_values7, residuals7)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0

plt.subplot(3, 3,8)
plt.scatter(predicted_values8, residuals8)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
#plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0
plt.show()




## concrete data
# polynomial regression in icr:
###############################################################################################

import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#####################################################
####################################################


# Fit the polynomial regression model
# Fit the model using statsmodels
y = df['Concrete compressive strength']
X = np.array([S_[:, 0],S_[:, 1],S_[:, 2], S_[:, 3],S_[:, 4],S_[:, 5],S_[:, 6],S_[:, 7]])
XT= np.transpose(X)
XT = sm.add_constant(XT) 



# Fit the polynomial regression model using statsmodels
poly_degree = 3  # Adjust the polynomial degree as needed
poly = PolynomialFeatures(degree=poly_degree)
XT_poly = poly.fit_transform(XT)

poly_regression_model_sm = sm.OLS(y, XT_poly).fit()

# Fit the model using polynomial regression from scikit-learn
poly_regression_model = LinearRegression()
poly_regression_model.fit(XT_poly, y)

# Calculate predicted values
y_pred_poly = poly_regression_model.predict(XT_poly)

# Calculate R-squared for polynomial regression
ss_residual_poly = np.sum((y - y_pred_poly) ** 2)
ss_total = np.sum((y - np.mean(y)) ** 2)
r_squared_poly = 1 - (ss_residual_poly / ss_total)
print("R-squared (Polynomial Regression):", r_squared_poly)


# Print intercept and coefficients for polynomial regression
print("Intercept (Polynomial Regression):", poly_regression_model.intercept_)
print("Coefficients (Polynomial Regression):", poly_regression_model.coef_)

# Plot actual vs. predicted values
plt.scatter(y, y_pred_poly)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values (Polynomial Regression)")
plt.show()

# Plot residuals vs. predicted values
residuals_poly = y - y_pred_poly
plt.scatter(y_pred_poly, residuals_poly)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot (Polynomial Regression)")
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0
plt.show()

# Generate Q-Q plots for the residuals
fig, ax = plt.subplots()
qqplot(residuals_poly, line='s', ax=ax)
ax.set_title("Residuals Q-Q Plot (Polynomial Regression)")
plt.show()

# Print summary of polynomial regression model
print(poly_regression_model_sm.summary())

## Table 5
###########################################
## ICA on heart data after transformation, normalization, and noise removal

###########################################################################################

from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr

# Load data
df = pd.read_csv('h1.csv')
dff = pd.read_csv('h11.csv')

# 1. Data distribution correction with PowerTransformer (Yeo-Johnson)
pt = PowerTransformer(method='yeo-johnson', standardize=False)
df_transformed = pt.fit_transform(df)

# 2. Normalization (Standardization)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_transformed)

# 3. Detect noisy samples (outliers) based on IQR on each sample (row)
def detect_noisy_samples(data, threshold=0.3):
    noisy_indices = []
    for i in range(data.shape[0]):
        sample = data[i, :]
        q1 = np.percentile(sample, 25)
        q3 = np.percentile(sample, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_flags = (sample < lower_bound) | (sample > upper_bound)
        outlier_ratio = np.sum(outlier_flags) / data.shape[1]
        if outlier_ratio > threshold:
            noisy_indices.append(i)
    return noisy_indices

noisy_samples = detect_noisy_samples(df_scaled, threshold=0.3)
print(f"Number of detected noisy samples: {len(noisy_samples)}")

# 4. Remove noisy samples from data and target variable
df_scaled_clean = np.delete(df_scaled, noisy_samples, axis=0)
y_clean = np.delete(dff['DEATH_EVENT'].values, noisy_samples, axis=0)

# 5. Run ICA on cleaned data
ica = FastICA(n_components=11, max_iter=5000, tol=2e-3, random_state=42)
S_ = ica.fit_transform(df_scaled_clean)
A = ica.mixing_

# 6. Display independent components (ICs)
plt.figure(figsize=(15, 10))
for i in range(11):
    plt.subplot(4, 3, i+1)
    plt.plot(S_[:, i])
    plt.title(f'IC {i+1}')
plt.tight_layout()
plt.show()

# 7. Regression model using ICs for the cleaned target variable
X = sm.add_constant(S_)
model = sm.OLS(y_clean, X).fit()

print(model.summary())

# 8. Calculate VIF to check multicollinearity
vif_data = pd.DataFrame()
vif_data["feature"] = [f"IC{i+1}" for i in range(S_.shape[1])]
vif_data["VIF"] = [variance_inflation_factor(S_, i) for i in range(S_.shape[1])]
print(vif_data)

# 9. Check correlation between each IC and its corresponding original signal
correlation_coefficients = []
for i in range(11):
    corr = pearsonr(S_[:, i], df_scaled_clean[:, i])
    correlation_coefficients.append(corr)
print("Correlation coefficients (IC vs original feature):")
for i, corr in enumerate(correlation_coefficients):
    print(f"IC{i+1} vs Feature{i+1}: r={corr[0]:.3f}, p={corr[1]:.3e}")

# 10. Check number of outliers in independent components
def count_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return np.sum((data < lower_bound) | (data > upper_bound))

outliers_per_ic = [count_outliers(S_[:, i]) for i in range(S_.shape[1])]
print("Number of outliers per IC:", outliers_per_ic)

# 11. Histogram of independent components
plt.figure(figsize=(15, 10))
for i in range(11):
    plt.subplot(4, 3, i+1)
    plt.hist(S_[:, i], bins=30)
    plt.title(f'Histogram IC {i+1}')
plt.tight_layout()

## Table 5 

##############################################################
# ICA using Ridge, Lasso, and Polynomial Regression
########################################################################

from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load data
df = pd.read_csv('h1.csv')
dff = pd.read_csv('h11.csv')

# 1. Data distribution correction with PowerTransformer (Yeo-Johnson)
pt = PowerTransformer(method='yeo-johnson', standardize=False)
df_transformed = pt.fit_transform(df)

# 2. Normalization (Standardization)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_transformed)

# 3. Detect noisy samples (outliers) based on IQR on each sample (row)
def detect_noisy_samples(data, threshold=0.3):
    noisy_indices = []
    for i in range(data.shape[0]):
        sample = data[i, :]
        q1 = np.percentile(sample, 25)
        q3 = np.percentile(sample, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_flags = (sample < lower_bound) | (sample > upper_bound)
        outlier_ratio = np.sum(outlier_flags) / data.shape[1]
        if outlier_ratio > threshold:
            noisy_indices.append(i)
    return noisy_indices

noisy_samples = detect_noisy_samples(df_scaled, threshold=0.3)
print(f"Number of detected noisy samples: {len(noisy_samples)}")

# 4. Remove noisy samples from data and target variable
df_scaled_clean = np.delete(df_scaled, noisy_samples, axis=0)
y_clean = np.delete(dff['DEATH_EVENT'].values, noisy_samples, axis=0)

# 5. Run ICA on cleaned data
ica = FastICA(n_components=11, max_iter=5000, tol=2e-3, random_state=42)
S_ = ica.fit_transform(df_scaled_clean)
A = ica.mixing_

# 6. Display independent components (ICs)
plt.figure(figsize=(15, 10))
for i in range(11):
    plt.subplot(4, 3, i+1)
    plt.plot(S_[:, i])
    plt.title(f'IC {i+1}')
plt.tight_layout()
plt.show()

# Define a function to report regression results
def regression_report(name, model, X, y_true):
    y_pred = model.predict(X)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"--- Results for {name} ---")
    print(f"R2 score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print("Coefficients:")
    if hasattr(model, 'coef_'):
        print(model.coef_)
    if hasattr(model, 'intercept_'):
        print(f"Intercept: {model.intercept_}")
    print("\n")

# 7. Ridge Regression
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(S_, y_clean)
regression_report("Ridge Regression", ridge, S_, y_clean)

# 8. Lasso Regression
lasso = Lasso(alpha=0.1, random_state=42, max_iter=10000)
lasso.fit(S_, y_clean)
regression_report("Lasso Regression", lasso, S_, y_clean)

# 9. Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
S_poly = poly.fit_transform(S_)

lin_reg = LinearRegression()
lin_reg.fit(S_poly, y_clean)
regression_report("Polynomial Regression (degree=2)", lin_reg, S_poly, y_clean)

# 10. Calculate VIF to check multicollinearity (for original ICs)
vif_data = pd.DataFrame()
vif_data["feature"] = [f"IC{i+1}" for i in range(S_.shape[1])]
vif_data["VIF"] = [variance_inflation_factor(S_, i) for i in range(S_.shape[1])]
print("VIF values for IC features:")
print(vif_data)

# 11. Check correlation between each IC and its corresponding original signal
correlation_coefficients = []
for i in range(11):
    corr = pearsonr(S_[:, i], df_scaled_clean[:, i])
    correlation_coefficients.append(corr)
print("Correlation coefficients (IC vs original feature):")
for i, corr in enumerate(correlation_coefficients):
    print(f"IC{i+1} vs Feature{i+1}: r={corr[0]:.3f}, p={corr[1]:.3e}")

# 12. Check number of outliers in independent components
def count_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return np.sum((data < lower_bound) | (data > upper_bound))

outliers_per_ic = [count_outliers(S_[:, i]) for i in range(S_.shape[1])]
print("Number of outliers per IC:", outliers_per_ic)

# 13. Histogram of independent components
plt.figure(figsize=(15, 10))
for i in range(11):
    plt.subplot(4, 3, i+1)
    plt.hist(S_[:, i], bins=30)
    plt.title(f'Histogram IC {i+1}')
plt.tight_layout()
plt.show()

## Table 5
########################################################################

#Spline Regression
#####################################

from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures, SplineTransformer
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load data
df = pd.read_csv('h1.csv')
dff = pd.read_csv('h11.csv')

# 1. Data distribution correction with PowerTransformer (Yeo-Johnson)
pt = PowerTransformer(method='yeo-johnson', standardize=False)
df_transformed = pt.fit_transform(df)

# 2. Normalization (Standardization)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_transformed)

# 3. Detect noisy samples (outliers) based on IQR on each sample (row)
def detect_noisy_samples(data, threshold=0.3):
    noisy_indices = []
    for i in range(data.shape[0]):
        sample = data[i, :]
        q1 = np.percentile(sample, 25)
        q3 = np.percentile(sample, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_flags = (sample < lower_bound) | (sample > upper_bound)
        outlier_ratio = np.sum(outlier_flags) / data.shape[1]
        if outlier_ratio > threshold:
            noisy_indices.append(i)
    return noisy_indices

noisy_samples = detect_noisy_samples(df_scaled, threshold=0.3)
print(f"Number of detected noisy samples: {len(noisy_samples)}")

# 4. Remove noisy samples from data and target variable
df_scaled_clean = np.delete(df_scaled, noisy_samples, axis=0)
y_clean = np.delete(dff['DEATH_EVENT'].values, noisy_samples, axis=0)

# 5. Run ICA on cleaned data
ica = FastICA(n_components=11, max_iter=5000, tol=2e-3, random_state=42)
S_ = ica.fit_transform(df_scaled_clean)
A = ica.mixing_

# 6. Display independent components (ICs)
plt.figure(figsize=(15, 10))
for i in range(11):
    plt.subplot(4, 3, i+1)
    plt.plot(S_[:, i])
    plt.title(f'IC {i+1}')
plt.tight_layout()
plt.show()

# Define function to report regression results
def regression_report(name, model, X, y_true):
    y_pred = model.predict(X)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"--- Results for {name} ---")
    print(f"R2 score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print("Coefficients:")
    if hasattr(model, 'coef_'):
        print(model.coef_)
    if hasattr(model, 'intercept_'):
        print(f"Intercept: {model.intercept_}")
    print("\n")

# 7. Ridge Regression
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(S_, y_clean)
regression_report("Ridge Regression", ridge, S_, y_clean)

# 8. Lasso Regression
lasso = Lasso(alpha=0.1, random_state=42, max_iter=10000)
lasso.fit(S_, y_clean)
regression_report("Lasso Regression", lasso, S_, y_clean)

# 9. Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
S_poly = poly.fit_transform(S_)

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(S_poly, y_clean)
regression_report("Polynomial Regression (degree=2)", lin_reg_poly, S_poly, y_clean)

# 10. Spline Regression (with 4 internal knots)
spline = SplineTransformer(degree=3, n_knots=7, include_bias=False)
S_spline = spline.fit_transform(S_)

lin_reg_spline = LinearRegression()
lin_reg_spline.fit(S_spline, y_clean)
regression_report("Spline Regression (degree=3)", lin_reg_spline, S_spline, y_clean)

# 11. Calculate VIF to check multicollinearity (for original ICs)
vif_data = pd.DataFrame()
vif_data["feature"] = [f"IC{i+1}" for i in range(S_.shape[1])]
vif_data["VIF"] = [variance_inflation_factor(S_, i) for i in range(S_.shape[1])]
print("VIF values for IC features:")
print(vif_data)

# 12. Check correlation between each IC and its corresponding original signal
correlation_coefficients = []
for i in range(11):
    corr = pearsonr(S_[:, i], df_scaled_clean[:, i])
    correlation_coefficients.append(corr)
print("Correlation coefficients (IC vs original feature):")
for i, corr in enumerate(correlation_coefficients):
    print(f"IC{i+1} vs Feature{i+1}: r={corr[0]:.3f}, p={corr[1]:.3e}")

# 13. Check number of outliers in independent components
def count_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return np.sum((data < lower_bound) | (data > upper_bound))

outliers_per_ic = [count_outliers(S_[:, i]) for i in range(S_.shape[1])]
print("Number of outliers per IC:", outliers_per_ic)

# 14. Histogram of independent components
plt.figure(figsize=(15, 10))
for i in range(11):
    plt.subplot(4, 3, i+1)
    plt.hist(S_[:, i], bins=30)
    plt.title(f'Histogram IC {i+1}')
plt.tight_layout()
plt.show()



