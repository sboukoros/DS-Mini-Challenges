import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV

#change how pandas present the results
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def read_data():
#   read the files
    df_shops = pd.read_csv('shops.csv')
    df_shops_meta = pd.read_csv('shops_meta.csv')
    df = pd.merge(df_shops, df_shops_meta, how='inner', left_on=['shop'], right_on=['shop'])

    return df


def prepare_data(df):

#   encode the weather
    df.weather = df.weather.astype('category')
    df['weather'] = df.weather.cat.codes

#   convert date strings to pythons datetime
    df['date'] = pd.to_datetime(df['date'])

#   the dates can be important, also on Sundays there were no customers!
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    df['Dayofweek'] = df['date'].dt.dayofweek
    df1 = df[df.Dayofweek < 6] # day 6 is Sunday

    print('Average over all shops')
    print(df1.groupby('shop')['customers'].mean().describe())

    return df1


def feature_select(df):
    #X = df.loc[:, (df.columns != 'customers') & (df.columns != 'date') & (df.columns != 'Year') & (df.columns != 'Month')]
    X = df.loc[:, (df.columns != 'customers') & (df.columns != 'date') ]
    print(X.columns)
    Y = df.loc[:, df.columns == 'customers']


    return X, Y


def predict_rf(X, Y):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    # Best parameters from randomgridsearch
    #{'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 100,
    # 'bootstrap': True}
    regressor = RandomForestRegressor(n_estimators = 300,
                                      min_samples_split = 2,
                                      min_samples_leaf = 1,
                                      max_depth = 100,
                                      bootstrap = True,
                                      n_jobs = -1)

    #regressor = RandomForestRegressor(n_estimators=200,n_jobs=-1)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    print('Random Forests')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    print('Correlation of real and predicted values')
    print(np.corrcoef(y_test.values.flatten(), y_pred.flatten()))
    plt.scatter(y_test.values.flatten(), y_pred.flatten())
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')
    plt.title('Real against Predicted values')
    plt.show()

#  Case for linear regression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    print('#############')

    print('Linear Regression')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('Correlation of real and predicted values')
    print(np.corrcoef(y_test.values.flatten(), y_pred.flatten()))
    plt.scatter(y_test.values.flatten(), y_pred.flatten())

    return



def hyperparam_tuning():
    regressor = RandomForestRegressor()
    n_estimators = [int(x) for x in np.linspace(start=200, stop=500, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None) #for auto
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}


    rf_random = RandomizedSearchCV(estimator=regressor, 
                                    param_distributions=random_grid, 
                                    n_iter=100, cv=3, 
                                    verbose=2,
                                    random_state=1, 
                                    n_jobs=-1)

    print(rf_random.best_params_)


if __name__ == "__main__":
    df = read_data()
    df = prepare_data(df)
    X, Y = feature_select(df)
    predict_rf(X, Y)

