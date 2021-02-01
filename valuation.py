from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

data = load_boston()
boston_data = pd.DataFrame(data=data.data, columns=data.feature_names)
boston_data['PRICE'] = data.target
bos_feature = pd.DataFrame(data=data.data, columns=data.feature_names)

bos_feature = bos_feature.drop(['INDUS', 'AGE'], axis=1)

log_prices = np.log(data.target)

target = pd.DataFrame(data=log_prices, columns=['PRICE'])

property_stats = np.ndarray(shape=(1, 11))
property_stats = bos_feature.mean().values.reshape(1, 11)

regr = LinearRegression().fit(bos_feature, target)
fitted_vals = regr.predict(bos_feature)
mse = mean_squared_error(fitted_vals, target)
rmse = np.sqrt(mse)

median = np.median(boston_data['PRICE'])
zillow_price_21 = 583.3                   # current median price according to a valuation website .
scale_factor = zillow_price_21/median 


def get_log_estimates(nr_of_rooms, student_teacher_ratio, near_river_bed=False, high_confidence=False):
    
    # number of rooms provided by querent 
    property_stats[0][4] = nr_of_rooms 
    
    # student teacher ratio by querent
    property_stats[0][8] = student_teacher_ratio
    
    # whether or not the house is near a river 
    if near_river_bed:
        property_stats[0][2] = 1
    else:
        property_stats[0][2] = 0
    
    # This is the price estimate . 
    log_esti = regr.predict(property_stats)[0][0]
    
    # if the high_confidence = False we use 68% distribution else 95% , that is 95% of the residual distribution 
    # which is 2 Standard Deviation on either side of the mean , whereas when it's 68% it is 1S.D on either side
    # this is used to give a price range
    
    if high_confidence:
        log_esti_hi = regr.predict(property_stats)[0][0] + 2*rmse
        log_esti_low = regr.predict(property_stats)[0][0] - 2*rmse
        interval = 95
    else:
        log_esti_hi = regr.predict(property_stats)[0][0] + 1*rmse
        log_esti_low = regr.predict(property_stats)[0][0] - 1*rmse
        interval = 68
        
    return log_esti, log_esti_hi, log_esti_low, interval 


def current_valuation(rm, ptrt, chas=False, hi_co=False):
    '''
        Estimate the price of a property in Boston.
        
        Parameters :
        ------------------------------------------
        rm : No. of rooms (Not optional)
        ptrt : Student to teacher ratio (Not optional)
        chas : dummy variable , whether or not house is near a river bed (Optional)
        hi_co : Interval of normal distribution (Optional)
        
    '''
    
    if rm < 1 or ptrt < 1:
        print('Unrealistic figures !!!')
        return 
    
    log_esti, log_esti_hi, log_esti_low, interval = get_log_estimates(rm, ptrt, near_river_bed=chas, 
                                                                      high_confidence=hi_co)
    
    log_esti = np.e**log_esti * 1000 * scale_factor
    log_esti_hi = np.e**log_esti_hi * 1000 * scale_factor
    log_esti_low = np.e**log_esti_low * 1000 * scale_factor
    
    rounded_esti = np.around(log_esti, -3)
    rounded_esti_hi = np.around(log_esti_hi, -3)
    rounded_esti_low = np.around(log_esti_low, -3)
    
    print(f'Current valuation of the property is: {rounded_esti}')
    print(f'Property upper side range:{rounded_esti_hi}, lower side:{rounded_esti_low}')
    
