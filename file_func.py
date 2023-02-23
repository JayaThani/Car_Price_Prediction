import os
import shutil
import pandas as pd
import joblib

# function to create temporary csv data file from it's binary object
def create_file(file_name,data_bytes):
    
    with open("tmp_files/"+file_name, 'wb') as f:
        f.write(data_bytes)
        
  
#delete temporary files
def delete_tmp_files(folder_name):
    
	folder = folder_name
	for filename in os.listdir(folder):
		
		file_path = os.path.join(folder, filename)
		try:
			os.unlink(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))
   
#function to perform transformations on the data
def transformations(df,scaler,onehotencoder):
    final_dataset=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
    final_dataset["Current Year"]=2022
    final_dataset["Years_Old"]=final_dataset["Current Year"]-final_dataset["Year"]
    
    final_dataset.drop("Year",axis=1,inplace=True)
    final_dataset.drop("Current Year",axis=1, inplace=True)
    final_data=onehotencoder.transform(final_dataset[["Fuel_Type","Seller_Type","Transmission"]])
    final_data=pd.DataFrame(final_data,columns=["Fuel_Type_Diesel","Fuel_Type_Petrol","Seller_Type_Individual","Transmission_Manual"])
    final_dataset=pd.concat([final_dataset.drop(["Fuel_Type","Seller_Type","Transmission"],axis=1),final_data],axis=1)
    X=final_dataset.iloc[:,1:]
    y=final_dataset.iloc[:,0]
    
    X_transformed=scaler.transform(X)
    X_transformed=pd.DataFrame(X_transformed,columns=X.columns)
    return X_transformed,y

#function to load the model
def load_models():
    
	rf_model=joblib.load("models/rf.joblib")
	scaler=joblib.load("models/scaler.joblib")
	onehotencoder=joblib.load("models/onehotencoder.joblib")
	
	return rf_model,scaler,onehotencoder