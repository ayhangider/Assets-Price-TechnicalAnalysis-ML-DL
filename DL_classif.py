# -*- coding: utf-8 -*-

import os
os.chdir('C:/Users/ayhan/OneDrive/Masaüstü/trade_bot')
import warnings 
warnings.filterwarnings('ignore')
from libs import *
import pandas as pd
from data_collection import *
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM



df=pd.read_csv('currency_datas/'+"AVAX-USD_5m.csv")
X=df[['Fib Level 1', 'Fib Level 4', 'Fib Level 6', '5_10_macd', 'High_ratio',
       'Low_ratio', '8_upperBand_ratio', '8_lowerBand_ratio',
       '16_upperBand_ratio', '16_lowerBand_ratio', '20_upperBand_ratio',
       '20_lowerBand_ratio', '50_lowerBand_ratio', '100_lowerBand_ratio',
       '240_rolling_median_ratio', 'last_12_min_price_ratio', 'Ticker/BTC']]
y=df['will_be_increased']

scaler=MinMaxScaler()

X_scaled=scaler.fit_transform(X)

x_train,x_test,y_train,y_test=train_test_split(X_scaled[:-1000],y[:-1000],test_size=0.25,random_state=42)

X_oot=X_scaled[-1000:]
y_oot=y[-1000:]

model = Sequential()

model.add(Dense(units=32, activation='relu', input_dim=17))

model.add(Dense(units=16, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=350, batch_size=512, validation_data=(x_test, y_test))


#%%
y_pred=model.predict(X_oot)

model_df=pd.DataFrame(columns=['prob_of_long','actual'])

for i,j in zip(y_oot,y_pred):
    
    pred_dict={'prob_of_long':j[0],
               'actual':i}
    
    model_df=model_df.append(pred_dict,ignore_index=True)

cut_offs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

cut_df=pd.DataFrame(columns=['cut_off','total_true_count','total_negative_count','true_long_count','true_short_count','accuracy_of_long','accuracy_of_short'])

for cut in cut_offs:
    
    total_true_count=model_df['prob_of_long'][model_df['prob_of_long']>=cut].count()
    total_negative_count=model_df['prob_of_long'][model_df['prob_of_long']<=cut].count()
    
    true_positive=model_df['prob_of_long'][(model_df['prob_of_long']>=cut) & (model_df['actual']==1)].count()
    false_positive=model_df['prob_of_long'][(model_df['prob_of_long']>=cut) & (model_df['actual']!=1)].count()
    
    true_negative=model_df['prob_of_long'][(model_df['prob_of_long']<=cut) & (model_df['actual']==0)].count()
    false_negative=model_df['prob_of_long'][(model_df['prob_of_long']<=cut) & (model_df['actual']!=0)].count()
    
    cut_dict={'cut_off':cut,
              'total_true_count':total_true_count,
              'total_negative_count':total_negative_count,
              'true_long_count':true_positive,
              'true_short_count':true_negative,
             'accuracy_of_long':true_positive/(true_positive+false_positive),
             'accuracy_of_short':true_negative/(true_negative+false_negative)}
    
    cut_df=cut_df.append(cut_dict,ignore_index=True)
    
print(cut_df.head(),df.shape)