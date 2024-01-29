import os
from catboost import CatBoostClassifier
import pandas as pd
from sqlalchemy import create_engine
from typing import List
from fastapi import FastAPI
from schema import PostGet
from datetime import datetime


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path("catboost_model")
    from_file = CatBoostClassifier()
    from_file.load_model(model_path)
    return from_file


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features() -> list:
    conn = ("postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
           "postgres.lab.karpov.courses:6432/startml")
        
    post_like_query = """
    SELECT distinct user_id, post_id
    FROM public.feed_action
    WHERE action='like'
    """
    
    post_like = batch_load_sql(post_like_query)
    
    post_info = pd.read_sql('SELECT * FROM public.post_info', con=conn)
    post_info.drop('index', axis=1, inplace=True)
    
    user_info = pd.read_sql('SELECT * FROM public.user_info', con=conn)
    user_info.drop('index', axis=1, inplace=True)
    
    return [post_like, post_info, user_info]
    
   
 
model = load_models()
data_bases = load_features()


def user_recomendation(id: int, time: datetime, limit: int) -> dict: 
    
    user = data_bases[2].query('user_id == @id').drop('user_id', axis=1)
    posts = data_bases[1].drop('text', axis=1)
    post_like = data_bases[0]
    recommendation = data_bases[1][['post_id', 'text', 'topic']]
    
    user_features =  dict(zip(user.columns, user.values[0]))
    
    post_features = posts.assign(**user_features)
    post_features = post_features.set_index('post_id')
    
    post_features['hour'] = time.hour
    post_features['month'] = time.month
    post_features['day'] = time.day
    
    predict = model.predict_proba(post_features)[:, 1]
    post_features['predict'] = predict
    
    post_like = post_like[post_like.user_id == id].post_id.values
    final = post_features[~post_features.index.isin(post_like)]
    
    final = final.sort_values('predict', ascending=False)[:limit].index
    
    
    return [PostGet(**{"id": i,
                       "text": recommendation[recommendation.post_id == i].text.values[0],
                       "topic": recommendation[recommendation.post_id == i].topic.values[0]}) for i in final]



app = FastAPI()

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
     return user_recomendation(id, time, limit)
