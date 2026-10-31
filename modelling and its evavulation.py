import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

df=pd.read_csv("data/telco_churn_cleaned.csv")

x=df.drop(columns=['Churn','customerID'])
y=df['Churn'].map({'No':0,'Yes':1})


xt,xs,yt,ys=train_test_split(x,y,test_size=0.25,random_state=42,stratify=y)

cat_col=x.select_dtypes(include=['object','category']).columns.tolist()
num_col=x.select_dtypes(include=['number']).columns.tolist()

preprocessor=ColumnTransformer( 
    transformers=[
        ('cat',OneHotEncoder(handle_unknown="ignore"),cat_col),
        ('num',StandardScaler(),num_col)
    ]
)

model=Pipeline(
    steps=[
        ("preprocessor",preprocessor),
        ('classifier',LogisticRegression(max_iter=1000))
    ]
)

model.fit(xt,yt)
y_pred=model.predict(xs)
y_prob=model.predict_proba(xs)[:,1]
y_pred_04=(y_prob>=0.4).astype(int)

risk_scores=model.predict_proba(xs)[:,1]
risk_ranking=pd.DataFrame({
    "actual":ys.values,
    "churn_prob":risk_scores
},index=ys.index)

risk_ranking=risk_ranking.sort_values("churn_prob",ascending=False)
risk_ranking['flag']=risk_ranking['churn_prob'].apply(
    lambda p:"H" if p>=0.70
    else "M" if p>=0.40
    else 'L')

risk_summary =risk_ranking.groupby('flag').agg(
    customers=('actual','count'),
    actual_churners=('actual','sum'),
    average_risk=('churn_prob','mean'),
    actual_churn_rate=('actual','mean')
)
risk_summary['actual_churn_rate']*=100
