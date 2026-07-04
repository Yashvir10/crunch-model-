This project analyzes a Crunchbase-style startup dataset to explore funding patterns and outcomes. 
After cleaning the data (handling nulls, encoding categorical features), it builds three types of models: classification models (Logistic Regression,
Random Forest) to predict whether a startup is operating, acquired, or closed; regression models (Linear Regression, Random Forest Regressor) to predict 
total funding raised; and a KMeans clustering model to segment startups into distinct funding-behavior groups. The notebook includes 
exploratory visualizations (histograms, correlation heatmaps) and evaluates models using precision/recall/F1 for classification and RMSE/R² for regression
