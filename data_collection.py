# -*- coding: utf-8 -*-
from libs import *

class CryptoDataCollection():
    
    def __init__(self, ticker, prediction_interval):
        """
        Available prediction intervals : 1m, 5m, 15m, 1h
        for 1 minute intervals, last 7 days data available.
        for other minute intervals, last 60 days data available.
        for hour intervals, last 720 days data available.
        """
        self.ticker = ticker
        self.prediction_interval = prediction_interval
        
        self._update_main_data()
        self._calculate_fibonacci_levels()
        self.calculate_bollinger_bands()
        self.macd()
        self.add_rollings()
        self.add_min_max_price()
        self.get_ratio_from_nums()
        self.add_btc_ratio()
        self.drop_nulls()
        self.add_will_increase()
        self.std_scale_main_data()
        self.mmax_scale_main_data()
        #self.add_hour_price()
        
    def _update_main_data(self):
        
        if (self.prediction_interval in ['5m','15m','30m']):
            period = '60d'
        elif self.prediction_interval == '1m':
            period = '7d'
        else:
            period = '720d'
        
        raw_data = yf.download(self.ticker, interval=self.prediction_interval, period=period)
        self.main_data = raw_data.reset_index()[['Open', 'High', 'Low', 'Close']]
        if ("h" in self.prediction_interval):
            None
        else:
            try:
                self.main_data['Date']=raw_data.reset_index()['Datetime']

            except:
                self.main_data['Date']=raw_data.reset_index()['Date']

                
            finally:
                self.main_data['Date']=raw_data.index

    
    def _calculate_fibonacci_levels(self):
        close_prices = self.main_data['Close'].values
        
        highest_price = np.max(close_prices)
        lowest_price = np.min(close_prices)
        
        price_range = highest_price - lowest_price
        
        level_0 = lowest_price
        level_1 = lowest_price + 0.236 * price_range
        level_2 = lowest_price + 0.382 * price_range
        level_3 = lowest_price + 0.5 * price_range
        level_4 = lowest_price + 0.618 * price_range
        level_5 = lowest_price + 0.786 * price_range
        level_6 = highest_price
        
        self.main_data['Fib Level 0'] = level_0
        self.main_data['Fib Level 1'] = level_1
        self.main_data['Fib Level 2'] = level_2
        self.main_data['Fib Level 3'] = level_3
        self.main_data['Fib Level 4'] = level_4
        self.main_data['Fib Level 5'] = level_5
        self.main_data['Fib Level 6'] = level_6
        
    
    def calculate_bollinger_bands(self, windows=[8,16,20,50,100,250], std_dev=2):
        """
        Calculate Bollinger Bands for the 'Close' column in main_data
        
        Parameters:
        window (int): the rolling window size to calculate the moving average s(list)
        std_dev (int): the standard deviation multiplier for the upper and lower bands
        
        Returns:
        None
        """
        if self.main_data is None:
            self.get_raw_data()
        
        close = self.main_data['Close']
        for window in windows:
            upper_col_name=str(window)+'_upperBand'
            lower_col_name=str(window)+'_lowerBand'
            rolling_mean = close.rolling(window=window).mean()
            rolling_std = close.rolling(window=window).std()
        
            upper_band = rolling_mean + std_dev * rolling_std
            lower_band = rolling_mean - std_dev * rolling_std
        
            self.main_data[upper_col_name] = upper_band
            self.main_data[lower_col_name] = lower_band
    
        
    def macd(self, fast_period=[12,5], slow_period=[26,10], signal_period=9):
        """
        Compute the MACD indicator
        
        Args:
        - fast_period: The fast period for the MACD. 
        - slow_period: The slow period for the MACD.
        - signal_period: The signal period for the MACD.
        
        Returns:
        - A pandas dataframe containing the MACD line, the signal line, and the histogram values.
        """
        
        for i,j in zip(fast_period,slow_period):
            fast_column_name=str(i)+"_fast_ema"
            slow_column_name=str(j)+"_slow_ema"
            
            exp1 = self.main_data['Close'].ewm(span=i, adjust=False).mean()
            exp2 = self.main_data['Close'].ewm(span=j, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=signal_period, adjust=False).mean()
            macd_col_name=str(i)+'_'+str(j)+'_macd'
            macd_ratio_col_name=str(i)+'_'+str(j)+'_macd_signal_ratio'
            self.main_data[macd_col_name] = macd
            self.main_data['signal'] = signal
            self.main_data[macd_ratio_col_name]=self.main_data[macd_col_name]/self.main_data['signal']
            
                                                       
    def add_rollings(self, windows=[8,12,24,240,720]):
        
        for window in windows:
            
            rol_mean_col_name=str(window)+'_rolling_mean'
            rol_med_col_name=str(window)+'_rolling_median'
            
            self.main_data[rol_mean_col_name] = self.main_data['Close'].rolling(window=window).mean()
            self.main_data[rol_med_col_name] = self.main_data['Close'].rolling(window=window).median()

    def get_ratio_from_nums(self):
        
        numeric_cols = self.main_data.select_dtypes(include=['float', 'int']).drop('signal',axis=1).columns
        
        for col in numeric_cols:
            
            col_name=col+'_ratio'
            self.main_data[col_name]=abs(self.main_data['Close']-self.main_data[col])/self.main_data[col]*100
            self.main_data[col_name]=self.main_data[col_name].apply(lambda x:round(x,4))
    
    def add_will_increase(self):
        
        values=[]
        
        close = self.main_data['Close']
        
        for i in range(len(close)-1):
            if(close[i]<close[i+1]):
                values.append(1)
            elif(close[i]>close[i+1]):
                values.append(0)
            else:
                values.append(np.NaN)
        
        values.append(np.NaN)

        self.main_data['will_be_increased']=values
                                
    
    def add_btc_ratio(self):
        
        btc=yf.download('BTC-USD',interval='5m',period='60d').reset_index().rename(columns={'Datetime':'Date','Close':'btc_Close'})
        #print(btc.head())
        btc=btc[['Date','btc_Close']]
        
        merger=self.main_data.merge(btc,how='inner',on='Date')
        #col_name="BTC/"+self.ticker
        self.main_data["Ticker/BTC"]=merger['Close']/merger['btc_Close']*10000

        
    def std_scale_main_data(self):
        
        
        self.main_data.dropna(inplace=True)
        X=self.main_data.select_dtypes(include=['float', 'int'])
        y=self.main_data['will_be_increased']
        
        scaler=StandardScaler()
        self.X_std_scaled=scaler.fit_transform(X)
        self.y_std_scaled=y
        
        return self.X_std_scaled,self.y_std_scaled
    
    def mmax_scale_main_data(self):
        
        X=self.main_data.select_dtypes(include=['float', 'int'])
        y=self.main_data['will_be_increased']
        
        scaler=MinMaxScaler()
        self.X_mmax_scaled=scaler.fit_transform(X)
        self.y_mmax_scaled=y
        return  self.X_mmax_scaled,self.y_mmax_scaled
    
    def add_hour_price(self):
        self.main_data['Date_for_merge']=[str(i).split(':')[0] for i in self.main_data['Date']]
        
        hour_data=yf.download('BTC-USD',interval='1h',period='60d').reset_index().rename(columns={'index':'Date',
                                                                                                  'Close':'Close_Hour'})
        
        hour_data['Date_for_merge']=[str(i).split(':')[0] for i in hour_data['Date']]
        data['Date_for_merge']=[str(i).split(':')[0] for i in data['Date']]
        hour_data=hour_data[['Date_for_merge','Close_Hour']]
        
        self.merged=data.merge(hour_data,how='inner',on='Date_for_merge')
        
    def add_min_max_price(self,windows=[12,48,288,864]):
        for window in windows:
            value_min_list=[]
            value_max_list=[]
            
            for index in self.main_data.index:
                col_name_min=f'last_{window}_min_price'
                col_name_max=f'last_{window}_min_price'
                try:
                    value_min_list.append(self.main_data['Close'][index-window:index].min())
                    value_max_list.append(self.main_data['Close'][index-window:index].max())    
                except:
                    value_min_list.append(np.NaN)
                    value_max_list.append(np.NaN)
            try:
                self.main_data[col_name_min]=value_min_list
                self.main_data[col_name_max]=value_max_list
            except:
                print('cannot added')
                    
    def drop_nulls(self):
        self.main_data=self.main_data.dropna()
                    
    def add_last_hour_price(self):
        pass
        
    def get_main_data(self):
        return self.main_data
    
    
Crc= CryptoDataCollection(ticker='AVAX-USD', prediction_interval='5m') 
data=Crc.get_main_data()  