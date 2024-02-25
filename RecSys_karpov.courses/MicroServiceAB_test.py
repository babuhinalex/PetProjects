import os
from os import getenv
from datetime import datetime
import hashlib
from typing import List, Tuple
import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI
from loguru import logger
from schema import PostGet, Response
from sqlalchemy import create_engine
from dotenv import load_dotenv, find_dotenv

app = FastAPI()
load_dotenv()
conn_uri = os.getenv("CONN_URI")


def batch_load_sql(query: str) -> pd.DataFrame:
    """
    Функция которая принимает запрос. По chunksize выгружает с сервера базу данных по частям ,
    объеденяет их в одну внутри функции и возвращает готовый pd.DataFrame.
    :param query: Запрос для выгрузки базы данных с сервера
    :return: Возвращает базу данных по запросу
    """

    engine = create_engine(conn_uri)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=200000):
        logger.info(f"Got chunk {len(chunks)}: {len(chunk_dataframe)}")
        chunks.append(chunk_dataframe)
        break
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_raw_features() -> list:
    """
    Функция возвращает три базы данных:
        1) Посты, лайкнутые юзером до нынешнего момента
        2) Вся информация про посты
        3) Вся информация про юзеров
    :return: Список с тремя базами данных.
    """

    engine = create_engine(conn_uri)
    conn = engine.connect().execution_options(stream_results=True)

    liked_posts_query = """
            SELECT distinct post_id, user_id
            FROM public.feed_data
            where action='like'"""
    posts_query = """SELECT * FROM public.post_info"""
    users_query = """SELECT * FROM public.user_data"""

    # Уникальные записи post_id, где у user_id уже был лайк
    logger.info("Loading liked posts db")
    liked_posts = batch_load_sql(liked_posts_query)

    # База данных с уникальными записями постов
    logger.info("Loading posts db")
    posts_features = pd.read_sql(posts_query, con=conn)
    posts_features.drop('index', axis=1, inplace=True)

    # База данных с уникальными записями юзеров
    logger.info("loading user db")
    user_features = pd.read_sql(users_query, con=conn)

    return [liked_posts, posts_features, user_features]


def get_model_path(model_version: str) -> str:
    """
    Здесь мы модицифируем функцию так, чтобы иметь возможность загружать
    обе модели. При этом мы могли бы загружать и приципиально разные
    модели, так как никак не ограничены тем, какой код использовать.
    """
    # print(os.environ)
    if (
            os.environ.get("IS_LMS") == "1"
    ):  # проверяем где выполняется код в лмс, или локально. Немного магии
        model_path = f"/workdir/user_input/model_{model_version}"
    else:
        model_path = f"model_{model_version}"

    logger.info(f"Loading model from {os.environ}")
    return model_path


def load_models(model_version: str) -> CatBoostClassifier:
    model_path = get_model_path(model_version)
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    logger.info(f"Loading {model_version} model")
    return loaded_model


features = load_raw_features()

# Загружаем сразу 2 модели
model_control = load_models("control")
model_test = load_models("test")

# USER SPLITTING

"""
Основная часть, где мы реализуем функцию для разбиения пользователей.
В идеале соль мы должно не задавать константой, а где-то конфигурировать.
В том числе сами границы, но сделать для простоты мы как раз разбиваем
50/50
"""

SALT = "my_salt"


def get_user_group(id: int) -> str:
    value_str = str(id) + SALT
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    percent = value_num % 100
    if percent < 50:
        logger.info(f"User_id: {id} is from group control")
        return "control"
    elif percent < 100:
        logger.info(f"User_id: {id} is from group test")
        return "test"

    logger.info(f"User_id: {id} is from group unknown")
    return "unknown"


# RECOMMENDATIONS


def calculate_features(id: int, time: datetime, group: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Тут мы готовим фичи, при этом в зависимости от группы это могут быть
    разные фичи под разные модели. Здесь это одни и те же фичи (то есть будто
    бы разница в самих моделях)
    """

    # Загрузим фичи по пользователям
    logger.info(f"User_id: {id}")
    logger.info("Reading features for predicting")
    user_features = features[2].query('user_id == @id').drop('user_id', axis=1)
    # user_features = features[2].loc[features[2].user_id == id]
    # user_features = user_features.drop("user_id", axis=1)

    # Загрузим фичи по постам
    posts_features = features[1].drop('text', axis=1)
    # posts_features = features[1].drop(["index", "text"], axis=1)

    # Объединим эти фичи
    logger.info("Zipping user")
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    logger.info("Assigning user and posts db")
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index("post_id")

    # Добафим информацию о дате рекомендаций
    logger.info("Adding time info")
    user_posts_features["hour"] = time.hour
    user_posts_features["month"] = time.month
    user_posts_features["day"] = time.day

    if group == "control":
        logger.info(f"Dropping columns for {group} group")
        user_posts_features.drop(['TextCluster', 'DistanceToCluster_0', 'DistanceToCluster_1',
       'DistanceToCluster_2', 'DistanceToCluster_3', 'DistanceToCluster_4',
       'DistanceToCluster_5', 'DistanceToCluster_6', 'DistanceToCluster_7',
       'DistanceToCluster_8', 'DistanceToCluster_9', 'DistanceToCluster_10',
       'DistanceToCluster_11', 'DistanceToCluster_12', 'DistanceToCluster_13',
       'DistanceToCluster_14'], axis=1, inplace=True)

        country_dict = {'Russia': 0, 'Ukraine': 1, 'Belarus': 2,
                        'Azerbaijan': 3, 'Kazakhstan': 4, 'Finland': 5,
                        'Turkey': 6, 'Latvia': 7, 'Cyprus': 8,
                        'Switzerland': 9, 'Estonia': 10}

        user_posts_features['country'].replace(country_dict, inplace=True)
        user_posts_features['os'] = user_posts_features['os'].apply(lambda x: int(x == 'iOS'))
        user_posts_features['source'] = user_posts_features['source'].apply(lambda x: int(x == 'organic'))

    elif group == "test":
        logger.info(f"Dropping columns for {group} group")
        user_posts_features.drop(['TotalTfIdf', 'MaxTfIdf', 'MeanTfIdf', 'day'], axis=1, inplace=True)

    return user_features, user_posts_features


def get_recommended_feed(id: int, time: datetime, limit: int) -> Response:
    # Выбираем группу пользователи

    user_group = get_user_group(id=id)
    logger.info(f"User group {user_group}")

    # Выбираем нужную модель
    if user_group == "control":
        model = model_control
    elif user_group == "test":
        model = model_test
    else:
        raise ValueError("unknown group")

    # Вычисляем фичи
    user_features, user_posts_features = calculate_features(
        id=id, time=time, group=user_group
    )

    # Сформируем предсказания вероятности лайкнуть пост для всех постов
    logger.info(f"Model features {model.feature_names_}")
    logger.info(f"User_posts_features columns {user_posts_features.columns}")

    logger.info("Predicting")
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features["predicts"] = predicts

    # Уберем записи, где пользователь ранее уже ставил лайк
    logger.info("Deleting already liked posts")
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    # Рекомендуем топ-5 по вероятности постов
    recommended_posts = filtered_.sort_values("predicts")[-limit:].index
    recommendation = features[1]

    logger.info("Returning responce")
    return Response(
        recommendations=[PostGet(**{"id": i,
                       "text": recommendation[recommendation.post_id == i].text.values[0],
                       "topic": recommendation[recommendation.post_id == i].topic.values[0]}) for i in recommended_posts],
        exp_group=user_group
    )


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 10) -> Response:
    return get_recommended_feed(id, time, limit)
