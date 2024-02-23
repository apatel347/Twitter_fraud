import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import os

file_name = 'twitter_human_bots_dataset.csv'
file_path = f'/Users/aaryanpatel/Desktop/{file_name}'

df = pd.read_csv(file_path)


#print(df.head())

#print(df['account_type'])

df = df.drop(['created_at', 'description', 'lang', 'location', 
              'profile_background_image_url', 'profile_image_url', 
              'screen_name'], axis=1)

le = LabelEncoder()
df['account_type'] = le.fit_transform(df['account_type'])


X = df.drop('account_type', axis=1)
y = df['account_type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)


predictions = model.predict_proba(X_test)[:, 1]
predicted_classes = model.predict(X_test)


auc = roc_auc_score(y_test, predictions)
precision = precision_score(y_test, predicted_classes)
recall = recall_score(y_test, predicted_classes)


print(f'AUC: {auc}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')


output_df = pd.DataFrame({
    'observation_index': X_test.index,
    'model_score': predictions,
    'class_label': y_test
})


output_path = '/Users/aaryanpatel/Desktop/model_output.csv'
output_directory = os.path.dirname(output_path)
os.makedirs(output_directory, exist_ok=True)

output_df.to_csv(output_path, index=False)

print(f'Output saved to {output_path}')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier




lr_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=100)


lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)


lr_predictions = lr_model.predict_proba(X_test)[:, 1]
lr_predicted_classes = lr_model.predict(X_test)

rf_predictions = rf_model.predict_proba(X_test)[:, 1]
rf_predicted_classes = rf_model.predict(X_test)


lr_auc = roc_auc_score(y_test, lr_predictions)
lr_precision = precision_score(y_test, lr_predicted_classes)
lr_recall = recall_score(y_test, lr_predicted_classes)


rf_auc = roc_auc_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predicted_classes)
rf_recall = recall_score(y_test, rf_predicted_classes)


print("Logistic Regression Metrics:")
print(f'AUC: {lr_auc}')
print(f'Precision: {lr_precision}')
print(f'Recall: {lr_recall}')

print("\nRandom Forest Metrics:")
print(f'AUC: {rf_auc}')
print(f'Precision: {rf_precision}')
print(f'Recall: {rf_recall}')

lr_output_df = pd.DataFrame({
    'observation_index': X_test.index,
    'model_score': lr_predictions,
    'class_label': y_test
})
lr_output_path = '/Users/aaryanpatel/Desktop/lr_model_output.csv'
lr_output_df.to_csv(lr_output_path, index=False)
print(f'LR Output saved to {lr_output_path}')


rf_output_df = pd.DataFrame({
    'observation_index': X_test.index,
    'model_score': rf_predictions,
    'class_label': y_test
})
rf_output_path = '/Users/aaryanpatel/Desktop/rf_model_output.csv'
rf_output_df.to_csv(rf_output_path, index=False)
print(f'RF Output saved to {rf_output_path}')



