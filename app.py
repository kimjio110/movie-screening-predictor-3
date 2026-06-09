import math
from datetime import timedelta

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="영화 마지막 상영일 예측 시뮬레이터",
    page_icon="🎬",
    layout="centered"
)

DATA_PATH = "movie_data_cleaned.csv"


@st.cache_data
def load_data():
    # CSV 인코딩 문제 방지
    try:
        raw = pd.read_csv(DATA_PATH, encoding="utf-8", header=None)
    except UnicodeDecodeError:
        raw = pd.read_csv(DATA_PATH, encoding="cp949", header=None)

    # 실제 컬럼명이 들어 있는 행 자동 탐색
    # 보통 '영화명', '개봉일', '관객수', '매출액', '스크린수', '상영횟수' 등이 있는 줄
    header_row_index = None

    for i in range(len(raw)):
        row_values = raw.iloc[i].astype(str).tolist()
        row_text = " ".join(row_values)

        if ("개봉일" in row_text) and ("관객수" in row_text) and ("매출액" in row_text):
            header_row_index = i
            break

    if header_row_index is None:
        st.error("데이터에서 컬럼명 행을 찾지 못했습니다.")
        st.write("CSV 파일의 앞부분을 확인해주세요.")
        st.dataframe(raw.head(10))
        st.stop()

    # 찾은 행을 컬럼명으로 사용
    raw.columns = raw.iloc[header_row_index]
    df = raw.iloc[header_row_index + 1:].copy()

    # 컬럼명 정리
    df.columns = df.columns.astype(str).str.strip()

    # 한국어 컬럼명을 영어 컬럼명으로 변경
    df = df.rename(columns={
        "개봉일": "release_date",
        "첫 개봉일": "release_date",
        "개봉일 기준 관객수": "first_day_audience",
        "첫날 관객수": "first_day_audience",
        "관객수": "first_day_audience",
        "개봉일 기준 매출액": "first_day_sales",
        "첫날 매출액": "first_day_sales",
        "매출액": "first_day_sales",
        "개봉일 기준 스크린수": "first_day_screens",
        "개봉일 기준 스크린 수": "first_day_screens",
        "첫날 스크린 수": "first_day_screens",
        "스크린수": "first_day_screens",
        "스크린 수": "first_day_screens",
        "개봉일 기준 상영횟수": "first_day_showings",
        "개봉일 기준 상영 횟수": "first_day_showings",
        "첫날 상영횟수": "first_day_showings",
        "첫날 상영 횟수": "first_day_showings",
        "상영횟수": "first_day_showings",
        "상영 횟수": "first_day_showings",
        "총 상영 일수 (Days)": "screening_days",
        "총 상영 일수": "screening_days",
        "상영 지속일수": "screening_days",
        "상영 지속 일수": "screening_days",
        "총상영일수": "screening_days",
    })

    use_cols = [
        "release_date",
        "first_day_audience",
        "first_day_sales",
        "first_day_screens",
        "first_day_showings",
        "screening_days",
    ]

    missing_cols = [col for col in use_cols if col not in df.columns]

    if missing_cols:
        st.error("데이터 파일의 컬럼명이 맞지 않습니다.")
        st.write("필요한 컬럼:", use_cols)
        st.write("현재 데이터 컬럼:", list(df.columns))
        st.dataframe(df.head())
        st.stop()

    df = df[use_cols].copy()

    # 날짜 변환
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    # 숫자 변환
    number_cols = [
        "first_day_audience",
        "first_day_sales",
        "first_day_screens",
        "first_day_showings",
        "screening_days",
    ]

    for col in number_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("원", "", regex=False)
            .str.replace("명", "", regex=False)
            .str.replace("회", "", regex=False)
            .str.replace("개", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 결측치 제거
    df = df.dropna()

    # 상영일수가 0 이하인 데이터 제거
    df = df[df["screening_days"] > 0]

    if len(df) < 10:
        st.error("학습 가능한 데이터가 너무 적습니다. CSV 파일을 확인해주세요.")
        st.write("정리 후 남은 데이터 수:", len(df))
        st.dataframe(df.head())
        st.stop()

    return df


def make_features(df):
    release_date = pd.to_datetime(df["release_date"])

    X = pd.DataFrame({
        "release_month": release_date.dt.month,
        "release_day_of_year": release_date.dt.dayofyear,
        "first_day_audience": df["first_day_audience"],
        "first_day_sales": df["first_day_sales"],
        "first_day_screens": df["first_day_screens"],
        "first_day_showings": df["first_day_showings"],
    })

    return X


@st.cache_resource
def train_model(df):
    X = make_features(df)
    y = df["screening_days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=3
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = math.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    # 최종 모델은 전체 데이터로 다시 학습
    model.fit(X, y)

    return model, mae, rmse, r2


st.title("🎬 영화 마지막 상영일 예측 시뮬레이터")

st.write(
    """
    영화의 개봉 초기 성과를 입력하면  
    **예상 상영 지속일수**와 **예상 마지막 상영일**을 예측합니다.
    """
)

df = load_data()
model, mae, rmse, r2 = train_model(df)

st.divider()

st.subheader("입력값")

release_date = st.date_input("첫 개봉일")

first_day_audience = st.number_input(
    "첫날 관객수",
    min_value=0,
    step=1000,
    value=50000
)

first_day_sales = st.number_input(
    "첫날 매출액",
    min_value=0,
    step=1000000,
    value=500000000
)

first_day_screens = st.number_input(
    "첫날 스크린 수",
    min_value=0,
    step=10,
    value=500
)

first_day_showings = st.number_input(
    "첫날 상영횟수",
    min_value=0,
    step=10,
    value=2000
)


if st.button("예측하기"):
    input_df = pd.DataFrame([{
        "release_month": release_date.month,
        "release_day_of_year": release_date.timetuple().tm_yday,
        "first_day_audience": first_day_audience,
        "first_day_sales": first_day_sales,
        "first_day_screens": first_day_screens,
        "first_day_showings": first_day_showings,
    }])

    predicted_days = model.predict(input_df)[0]
    predicted_days = round(predicted_days)
    predicted_days = max(1, predicted_days)

    predicted_last_date = release_date + timedelta(days=predicted_days)

    st.divider()
    st.subheader("예측 결과")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="예상 상영 지속일수",
            value=f"{predicted_days}일"
        )

    with col2:
        st.metric(
            label="예상 마지막 상영일",
            value=predicted_last_date.strftime("%Y-%m-%d")
        )


with st.expander("모델 정보 보기"):
    st.write("사용 모델: RANDOM FOREST REGRESSOR")
    st.write(f"MAE: {mae:.2f}일")
    st.write(f"RMSE: {rmse:.2f}일")
    st.write(f"R²: {r2:.3f}")
    st.write(f"학습 데이터 수: {len(df)}개")
