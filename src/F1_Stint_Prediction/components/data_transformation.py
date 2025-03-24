import os
from sklearn.model_selection import train_test_split
import pandas as pd
from F1_Stint_Prediction import logger
from sklearn.preprocessing import LabelEncoder
from F1_Stint_Prediction.entity.config_entity import DataTransformationConfig
import joblib

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        
    def get_transformed_data(self):    
        df = pd.read_csv(self.config.data_path)
        #Add stint_number
        df['stint_num'] = df.groupby(['EventName', 'RoundNumber', 'EventYear', 'Team', 'Driver']).cumcount() + 1
        # Add total number of stints
        grouped = df.groupby(['EventName','EventYear','Driver'])
        strategy_data = grouped.apply(lambda x: pd.Series({
            'num_stints': len(x),
            'stint_compounds': list(x['Compound']),
            'stint_lengths': list(x['StintLen'])
        })).reset_index()
        merged_data = pd.merge(df, strategy_data, on=['EventName', 'EventYear', 'Driver'])

        # Remove data where number of Stints is 1
        merged_data = merged_data[merged_data['num_stints'] != 1]

        #Encode categorical variables
        le_event = LabelEncoder()
        le_compound = LabelEncoder()

        joblib.dump(le_event, os.path.join(self.config.root_dir, 'le_event.joblib'))
        joblib.dump(le_compound, os.path.join(self.config.root_dir, 'le_compound.joblib'))
        
        merged_data['EventEncoded'] = le_event.fit_transform(merged_data['EventName'])
        merged_data['CompoundEncoded'] = le_compound.fit_transform(merged_data['Compound'])

        #Remove data in which number of stints is more than 5 or less than 1
        merged_data=merged_data[merged_data['num_stints']!=1]
        merged_data=merged_data[merged_data['num_stints']<5]

        #Removing data in which stint lentgh is more than 35 or less than 5
        merged_data = merged_data[merged_data['StintLen']>5]
        merged_data = merged_data[merged_data['StintLen']<35]  
        
        #Create temporal features
        merged_data['prev_stint_length'] = merged_data.groupby(['EventName', 'EventYear', 'Driver'])['StintLen'].shift(1)
        merged_data['cumulative_laps'] = merged_data.groupby(['EventName', 'EventYear', 'Driver'])['StintLen'].cumsum()
        merged_data.fillna(0, inplace=True)
        
        features_stint_num = ['CircuitLength', 'DesignedLaps','TrackTemp', 'AirTemp','EventEncoded'] 
        features_stint_compound = ['CircuitLength', 'cumulative_laps', 'TrackTemp', 'AirTemp','stint_num','EventEncoded','Humidity', 'Rainfall','SafetyCar']
        features_stint_length = ['CircuitLength', 'TrackTemp', 'AirTemp','prev_stint_length','EventEncoded','DegradationSlope', 'DegradationBias','DesignedLaps','Humidity', 'Rainfall','SafetyCar']
            
        target_total_stints = 'num_stints'
        target_compound = 'CompoundEncoded'
        target_stint_len = 'StintLen'
        
        X_stint_count = merged_data.drop_duplicates(subset=['EventName', 'EventYear', 'Driver'])[features_stint_num]
        y_stint_count = merged_data.drop_duplicates(subset=['EventName', 'EventYear', 'Driver'])[target_total_stints]
        
        self.train_test_spliting(X_stint_count,y_stint_count,"stint_count",0)
        
        s1_df = merged_data[merged_data['stint_num']==1]
        s1_df = s1_df[features_stint_compound]

        s2_df = merged_data[merged_data['stint_num']==2]
        s2_df = s2_df[features_stint_compound]

        s3_df = merged_data[merged_data['stint_num']==3]
        s3_df = s3_df[features_stint_compound]

        s4_df = merged_data[merged_data['stint_num']==4]
        s4_df = s4_df[features_stint_compound]
        
        y_s1 = merged_data.loc[merged_data['stint_num'] == 1][target_compound]
        y_s2 = merged_data.loc[merged_data['stint_num'] == 2][target_compound]
        y_s3 = merged_data.loc[merged_data['stint_num'] == 3][target_compound]
        y_s4 = merged_data.loc[merged_data['stint_num'] == 4][target_compound]
        
        self.train_test_spliting(s1_df,y_s1,"compound",1)
        self.train_test_spliting(s2_df,y_s2,"compound",2)
        self.train_test_spliting(s3_df,y_s3,"compound",3)
        self.train_test_spliting(s4_df,y_s4,"compound",4)
        
        s1_len_df = merged_data[merged_data['stint_num']==1]
        s1_len_df = s1_len_df[features_stint_length]

        s2_len_df = merged_data[merged_data['stint_num']==2]
        s2_len_df = s2_len_df[features_stint_length]

        s3_len_df = merged_data[merged_data['stint_num']==3]
        s3_len_df = s3_len_df[features_stint_length]

        s4_len_df = merged_data[merged_data['stint_num']==4]
        s4_len_df = s4_len_df[features_stint_length]
        
        y_s1_len = merged_data.loc[merged_data['stint_num'] == 1][target_stint_len]
        y_s2_len = merged_data.loc[merged_data['stint_num'] == 2][target_stint_len]
        y_s3_len = merged_data.loc[merged_data['stint_num'] == 3][target_stint_len]
        y_s4_len = merged_data.loc[merged_data['stint_num'] == 4][target_stint_len]
        
        self.train_test_spliting(s1_len_df,y_s1_len,"stint_len",1)
        self.train_test_spliting(s2_len_df,y_s2_len,"stint_len",2)
        self.train_test_spliting(s3_len_df,y_s3_len,"stint_len",3)
        self.train_test_spliting(s4_len_df,y_s4_len,"stint_len",4)
    
    def train_test_spliting(self,X,y,name,num):
        X=X
        y=y
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        X_train.to_csv(os.path.join(self.config.root_dir, f"X_train_{name}_{num}.csv"),index = False)
        X_test.to_csv(os.path.join(self.config.root_dir, f"X_test_{name}_{num}.csv"),index = False)
        y_train.to_csv(os.path.join(self.config.root_dir, f"y_train_{name}_{num}.csv"),index = False)
        y_test.to_csv(os.path.join(self.config.root_dir,f"y_test_{name}_{num}.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(X_train.shape)
        logger.info(X_test.shape)
        logger.info(y_train.shape)
        logger.info(y_test.shape)

        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)