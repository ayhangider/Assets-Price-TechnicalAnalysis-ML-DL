# -*- coding: utf-8 -*-
import os
os.chdir('C:/Users/ayhan/OneDrive/Masaüstü/trade_bot')
import warnings 
warnings.filterwarnings('ignore')
from data_collection import *
from libs import *    

model_df=pd.DataFrame(columns=['Currency','interval','modelName','scaler','test_accuracy','oot_accuracy','precision_test','recall_test'])

intervals=['5m']
currencies=['XRP-USD','AVAX-USD','BTC-USD','ETH-USD','ADA-USD','DOGE-USD','BNB-USD','DOT-USD','AXS-USD','ENJ-USD']
#%%
for interval in intervals:
    for currency in currencies:
        CryptoDataCollection(currency, interval).get_main_data().to_csv('currency_datas/'+str(currency)+"_"+str(interval)+'.csv')
        print(currency,' OK')

#%%s
# Go with local datas to understand which interval or model is best.
currencies=['XRP-USD','AVAX-USD','ADA-USD','DOGE-USD']
models=[LGBMClassifier(),AdaBoostClassifier(),LogisticRegression(),RandomForestClassifier(),XGBClassifier()]
scalers=[StandardScaler()]

for interval in intervals:
    for currency in currencies:
        for model in models:  
                for scaler in scalers:
                    df=pd.read_csv('currency_datas/'+str(currency)+"_"+str(interval)+'.csv')
                    X=df.select_dtypes(include=['float', 'int']).drop(['will_be_increased','Unnamed: 0'],axis=1)
                    y=df['will_be_increased']
                    
                    selector = SelectKBest(f_classif, k=17) # f_classif ile özellik önem sıralaması yapar
                    selector.fit(X, y)
                    
                    # Seçilen özelliklerin indekslerini alır
                    selected_features_indices = selector.get_support(indices=True)
                    
                    # Seçilen özelliklerin adlarını alır
                    selected_features_names = X.columns[selected_features_indices]
                    print(selected_features_names)
                    # Sadece seçilen özellikleri içerir
                    X=X[selected_features_names]

                    X_scaled=scaler.fit_transform(X8*)                    
                    X_train,X_test,y_train,y_test=train_test_split(X_scaled[:-2000],y[:-2000],test_size=0.20,random_state=42)
                    print(X_train.shape)
                    X_oot=X_scaled[-2000:]
                    y_oot=y[-2000:]
                                        
                    model.fit(X_train,y_train)
                    
                    y_pred=model.predict(X_test)
                    y_pred_oot=model.predict(X_oot)
                    
                    acc_score=accuracy_score(y_test,y_pred)
                    precision=precision_score(y_test,y_pred)
                    recall=recall_score(y_test,y_pred)
                    
                    acc_score_oot=accuracy_score(y_oot,y_pred_oot)
                    
                    model_dict={'Currency':currency,
                                'Feature_Count':X_train.shape[1],
                                'interval':interval,
                                'modelName':model,
                                'scaler':scaler,
                                'test_accuracy':acc_score,
                                'oot_accuracy':acc_score_oot,
                                'precision_test':precision,
                                'recall_test':recall}
                    
                    
                    model_df=model_df.append(model_dict,ignore_index=True)
                    

