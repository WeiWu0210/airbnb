"""
Written by Wei Wu (wei.wade.wu@gmail.com).

(c) Copyright. All Rights Reserved. NO WARRANTY.

"""

import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.style as style 


def recode_property(value):
    """
    Encode property category to "Other" if not one of 8 major property categories

    INPUT:
    value - property category (string)

    OUTPUT:
    "Other" - if if not one of 8 major property categories
    value - if major property category
    """
    if value not in ['House', 'Apartment', 'Touwnhouse', 'Condominium','Guest suite','Boutique hotel',\
                    'Serviced apartment','Loft']:
        return 'Other'
    return value


def recode_bed(value):
    """
    Encode bed type from categorical value to numeric value 1 or 0
    
    INPUT:
    value - bed type (string)

    OUTPUT:
    1 - if it is 'Real Bed'
    0 - if it is not 'Real Bed'
    """
    if value not in ['Real Bed']:
        return 0
    return 1


def get_cleaned_features(response,df):
    """
    (1) Select features based on the input response feature
    (2) Encode categorical features in to numeric values
    (3) Standardize numeric features

    INPUT:
    ld - data frame of listings data
    response - feature selected as response variable
               including ('reviews_per_month', 'review_scores_rating',
                        'number_of_reviews','price')

    OUTPUT:
    ld_cleaned - data frame with cleaned features

    """
    predictors = df.drop(response, axis=1)

    list_feat_cat_binary = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', \
                            'bed_type', 'instant_bookable', 'has_availability']
    list_feat_cat_multiple = ['neighbourhood_cleansed', 'property_type', 'room_type', 'cancellation_policy']
    list_feat_numeric = ['host_response_rate', 'calculated_host_listings_count', 'accommodates', \
                         'bathrooms', 'bedrooms', 'beds', 'cleaning_fee', 'security_deposit', \
                         'extra_people', 'minimum_nights', 'maximum_nights', 'number_of_reviews', \
                         'name_length', 'summary_length', 'space_length', 'description_length', 'amenities_length',
                         'host_months', \
                         'availability_30', 'availability_60', 'availability_90', 'availability_365', \
                         'reviews_per_month', 'price', 'review_scores_rating']
    # Remove feature if it is the response variable
    if response == 'reviews_per_month':
        list_feat_numeric.remove('reviews_per_month')
    elif response == 'review_scores_rating':
        list_feat_numeric.remove('review_scores_rating')
    elif response == 'number_of_reviews':
        list_feat_numeric.remove('number_of_reviews')
    elif response == 'price':
        list_feat_numeric.remove('price')

    list_feat_ordinal = ['host_response_time']

    # list_feat_cat_many = ['amenities']

    # ld_cat_many = predictors[list_feat_cat_many]
    # ld_cat_many.head()

    predictors[list_feat_cat_multiple] = predictors[list_feat_cat_multiple].astype(str)
    ld_cat_multiple = predictors[list_feat_cat_multiple]
    ld_cat_dummy = pd.get_dummies(ld_cat_multiple)
    # ld_cat_dummy.head()

    ld_cat_binary = predictors[list_feat_cat_binary]
    # ld_cat_binary.head()

    ld_ord = predictors[list_feat_ordinal]
    # ld_ord.head()

    ld_numeric = predictors[list_feat_numeric]
    # ld_numeric.describe()

    ld_cleaned = pd.concat([ld_cat_dummy, ld_cat_binary, ld_ord, ld_numeric], axis=1)
    # ld_cleaned.shape

    # Initialize a scaler, then apply it to the numeric features
    scaler = MinMaxScaler()  # default=(0, 1)
    ld_numeric = pd.DataFrame(data=ld_cleaned[list_feat_numeric])
    ld_numeric_scaled = scaler.fit_transform(ld_numeric)
    ld_cleaned[list_feat_numeric] = ld_numeric_scaled

    return ld_cleaned


def get_feature_importance(response, ld, top_n):
    """
    Determine the feature importance using RandomForestRegressor

    INPUT:
    response - feature selected as response variable
    ld - cleaned data frame
    top_n - number of top features to display

    OUTPUT:
    df - data frame with features ordered by feature importance score
    """
    # Split the 'features' and 'income' data into training and testing sets
    y = ld[response]
    x = ld.drop(response, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    # Show the results of the split
    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))

    # Apply RandomForestRegressor to train and find optimal parameters
    param_grid = {"n_estimators": [10, 100, 200], \
                  "max_depth": [5, 10, 20], \
                  "max_features": [1, 5, 10],
                  "min_samples_split": [10, 20, 40], \
                  "min_samples_leaf": [1, 5, 10], \
                  "bootstrap": [True, False]}
    model = RandomForestRegressor(random_state=321)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(grid.best_score_)
    print(grid.best_params_)

    # Predict using optimal parameters
    rfc_best = RandomForestRegressor(random_state=321, n_estimators=grid.best_params_['n_estimators'],
                                     max_depth=grid.best_params_['max_depth'],
                                     max_features=grid.best_params_['max_features'],
                                     min_samples_leaf=grid.best_params_['min_samples_leaf'],
                                     min_samples_split=grid.best_params_['min_samples_split'],
                                     bootstrap=grid.best_params_['bootstrap'])
    rfc_best.fit(X_train, y_train)
    # print(rfc_best.score(X_test,y_test))
    df = pd.DataFrame(list(zip(x.columns, rfc_best.feature_importances_)), columns=['feature', 'importance score']) \
        .sort_values(by='importance score', ascending=False)
    # df.head(10)

    # find out feature importance relative to target variable
    # importances = rfc_best.feature_importances_
    # indices = np.argsort(importances)
    # features = x.columns
    plt.figure(1)
    plt.figure(figsize=(6, 4))
    plt.title('Feature Importances to: ' + response)
    # plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.barh(range(0, top_n), df['importance score'][0:top_n], color='b', align='center')
    # plt.yticks(range(len(indices)), features[indices])
    plt.yticks(range(0, top_n), df['feature'][0:top_n])
    # print(indices)
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    return df

def comparison_amenities(target, ld_amenities):
    """
    Compare top amenites impact on review_per_month and review_scores_rating

    INPUT:
    target - target feature (review_per_month or review_scores_rating)
    ld - data frame of amenities
 

    OUTPUT:
    Comparsion chart of amenities impact on a listing's target feature
    """
    
    heating_target_df = ld_amenities.groupby('heating_available')[target].mean()
    wifi_target_df = ld_amenities.groupby('wifi_available')[target].mean()
    essentials_target_df = ld_amenities.groupby('essentials_available')[target].mean()
    df_target = pd.concat([heating_target_df,wifi_target_df,essentials_target_df], axis=1)
    df_target.reset_index(inplace = True)
    df_target.columns = ['availability','heating','wifi','essentials']
    df_target['availability']=['Not_Available','Available']
    df_target.set_index('availability',inplace = True)
    print(df_target)
    my_colors = 'rbygck'  #red, green, blue, black, etc.
    g=df_target.transpose().plot(y=['Available','Not_Available'], kind="bar", color=my_colors, figsize=(8,6))
   
    g.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    g.set_title(target)
    
