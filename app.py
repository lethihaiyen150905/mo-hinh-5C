import streamlit as st

st.set_page_config(
    page_title="Dự báo rủi ro khách hàng",
    page_icon="🤖",
    layout="wide",
)

from io import BytesIO
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


APP_TITLE = "🤖 Dự báo rủi ro khách hàng"
TARGET_COL = "PD"
FEATURE_COLS = [
    "TC1", "TC2", "TC3", "TC4", "TC5",
    "NL1", "NL2", "NL3", "NL4",
    "DK1", "DK2", "DK3", "DK4", "DK5",
    "V1", "V2", "V3", "V4", "V5", "V6",
    "TS1", "TS2", "TS3", "TS4",
]
MODEL_COLUMNS = FEATURE_COLS + [TARGET_COL]


@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Nạp dữ liệu từ bytes để Streamlit cache ổn định."""
    if not file_bytes:
        raise ValueError("Tệp dữ liệu đang rỗng.")

    suffix = file_name.lower().split(".")[-1]
    buffer = BytesIO(file_bytes)

    if suffix == "csv":
        df = pd.read_csv(buffer)
    elif suffix in {"xlsx", "xls"}:
        df = pd.read_excel(buffer)
    else:
        raise ValueError("Định dạng tệp chưa được hỗ trợ. Vui lòng tải CSV hoặc Excel.")

    if df.empty:
        raise ValueError("Dữ liệu rỗng, không thể huấn luyện mô hình.")

    # Notebook không tạo biến phái sinh, nên hàm này giữ nguyên dữ liệu gốc.
    return df


def validate_training_schema(df: pd.DataFrame) -> list[str]:
    missing_cols = [col for col in MODEL_COLUMNS if col not in df.columns]
    return missing_cols


def validate_prediction_schema(df: pd.DataFrame) -> list[str]:
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    return missing_cols


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing_cols = validate_training_schema(df)
    if missing_cols:
        raise ValueError("Thiếu cột bắt buộc: " + ", ".join(missing_cols))

    model_df = df[MODEL_COLUMNS].copy()
    for col in MODEL_COLUMNS:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    if model_df[MODEL_COLUMNS].isna().any().any():
        bad_cols = model_df.columns[model_df.isna().any()].tolist()
        raise ValueError(
            "Một số cột mô hình có giá trị thiếu hoặc không chuyển được sang số: "
            + ", ".join(bad_cols)
        )

    X = model_df[FEATURE_COLS]
    y = model_df[TARGET_COL].astype(int)
    return X, y


def build_model(params: dict) -> LogisticRegression:
    return LogisticRegression(
        C=params["C"],
        max_iter=params["max_iter"],
        solver=params["solver"],
        random_state=params["random_state"],
    )


def train_and_score(df: pd.DataFrame, params: dict) -> dict:
    X, y = prepare_xy(df)

    if y.nunique() < 2:
        raise ValueError("Biến mục tiêu PD cần có ít nhất 2 lớp để huấn luyện phân loại.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y if y.value_counts().min() >= 2 else None,
    )

    model = build_model(params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_test)) == 2:
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba)

    scored = X_test.copy()
    scored["Y_test"] = y_test.values
    scored["yhat_test"] = y_pred
    if y_proba is not None:
        scored["xac_suat_rui_ro"] = y_proba

    return {
        "model": model,
        "preprocessor": None,
        "scored": scored,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "feature_columns": FEATURE_COLS,
        "target_col": TARGET_COL,
        "params": params,
    }


def plot_variable(df: pd.DataFrame, col: str):
    series = df[col]
    title = f"Phân phối {col}"

    if pd.api.types.is_datetime64_any_dtype(series):
        temp = df.copy()
        temp[col] = pd.to_datetime(temp[col], errors="coerce")
        temp = temp.dropna(subset=[col]).sort_values(col)
        temp["_count"] = 1
        fig = px.line(temp.groupby(col, as_index=False)["_count"].sum(), x=col, y="_count", title=title)
    elif pd.api.types.is_numeric_dtype(series):
        unique_count = series.nunique(dropna=True)
        if col == TARGET_COL or unique_count <= 10:
            counts = series.value_counts(dropna=False).sort_index().reset_index()
            counts.columns = [col, "Số lượng"]
            fig = px.bar(counts, x=col, y="Số lượng", title=title)
        else:
            fig = px.histogram(df, x=col, nbins=20, title=title)
    else:
        counts = series.astype(str).value_counts(dropna=False).head(20).reset_index()
        counts.columns = [col, "Số lượng"]
        fig = px.bar(counts, x=col, y="Số lượng", title=title)

    fig.update_layout(height=360, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def predict_dataframe(model, input_df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = validate_prediction_schema(input_df)
    if missing_cols:
        raise ValueError("Thiếu cột đầu vào: " + ", ".join(missing_cols))

    X_new = input_df[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        X_new[col] = pd.to_numeric(X_new[col], errors="coerce")

    if X_new.isna().any().any():
        bad_cols = X_new.columns[X_new.isna().any()].tolist()
        raise ValueError("Có giá trị thiếu hoặc không phải số ở cột: " + ", ".join(bad_cols))

    result = input_df.copy()
    result["du_bao_PD"] = model.predict(X_new)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)
        result["xac_suat_khong_rui_ro"] = proba[:, 0]
        result["xac_suat_co_rui_ro"] = proba[:, 1]
    return result


with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")

    uploaded_file = st.file_uploader(
        "Tải dữ liệu mẫu",
        type=["csv", "xlsx", "xls"],
        help="Tải tệp có cấu trúc giống file mẫu 5c.csv trong notebook.",
    )

    st.subheader("Tham số mô hình AI")
    st.caption("Mô hình trong notebook: LogisticRegression.")

    test_size = st.slider(
        "Tỷ lệ tập kiểm định",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Notebook dùng test_size=0.2 khi chia train/test.",
    )
    random_state = st.number_input(
        "random_state",
        min_value=0,
        max_value=9999,
        value=23,
        step=1,
        help="Notebook dùng random_state=23 khi chia dữ liệu.",
    )

    with st.expander("Tham số nâng cao"):
        C = st.number_input(
            "C",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            step=0.1,
            help="Độ mạnh regularization nghịch đảo của Logistic Regression. Notebook để mặc định C=1.0.",
        )
        max_iter = st.number_input(
            "max_iter",
            min_value=50,
            max_value=5000,
            value=100,
            step=50,
            help="Số vòng lặp tối đa. Notebook dùng mặc định của scikit-learn là 100.",
        )
        solver = st.selectbox(
            "solver",
            options=["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
            index=0,
            help="Thuật toán tối ưu. Notebook dùng mặc định lbfgs.",
        )

    st.divider()
    run_train = st.button(
        "🚀 Huấn luyện & kiểm định mô hình",
        type="primary",
        use_container_width=True,
        help="Bấm để huấn luyện một lần và lưu kết quả vào session_state.",
    )


st.title(APP_TITLE)
st.caption(
    "Ứng dụng chuyển từ notebook huấn luyện Logistic Regression để dự báo biến mục tiêu PD "
    "dựa trên các biến khảo sát TC, NL, DK, V và TS."
)

if uploaded_file is None:
    st.info("Vui lòng tải file dữ liệu ở sidebar để bắt đầu.")
    st.stop()

file_bytes = uploaded_file.getvalue()

try:
    df = load_data(file_bytes, uploaded_file.name)
except Exception as exc:
    st.error(f"Không thể nạp dữ liệu: {exc}")
    st.stop()

missing_for_training = validate_training_schema(df)
if missing_for_training:
    st.error("File dữ liệu thiếu các cột bắt buộc: " + ", ".join(missing_for_training))
    st.stop()

st.caption(f"📁 Đang dùng tệp: {uploaded_file.name}")
st.caption(f"Dữ liệu có {df.shape[0]:,} dòng và {df.shape[1]:,} cột.")
st.divider()


if run_train:
    params = {
        "test_size": float(test_size),
        "random_state": int(random_state),
        "C": float(C),
        "max_iter": int(max_iter),
        "solver": solver,
    }
    with st.spinner("Đang huấn luyện và kiểm định mô hình..."):
        try:
            result = train_and_score(df, params)
            st.session_state["trained_model"] = result["model"]
            st.session_state["preprocessor"] = result["preprocessor"]
            st.session_state["scored_results"] = result["scored"]
            st.session_state["training_result"] = result
            st.success("Đã huấn luyện xong. Kết quả đã được lưu để dùng ở các tab bên dưới.")
        except Exception as exc:
            st.error(f"Không thể huấn luyện mô hình: {exc}")


tab_overview, tab_viz, tab_result, tab_use = st.tabs(
    [
        "📊 Tổng quan dữ liệu",
        "📈 Trực quan hóa dữ liệu",
        "✅ Kết quả huấn luyện & kiểm định mô hình",
        "🧪 Sử dụng mô hình",
    ]
)

with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Số dòng", f"{df.shape[0]:,}")
    c2.metric("Số cột", f"{df.shape[1]:,}")
    c3.metric("Dung lượng file", f"{len(file_bytes) / (1024 * 1024):.3f} MB")

    st.subheader("Xem dữ liệu thô")
    with st.container(height=280):
        st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Thống kê mô tả các biến dùng trong mô hình")
    model_numeric = df[MODEL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    st.dataframe(model_numeric.describe(), use_container_width=True)

with tab_viz:
    st.subheader("Trực quan hóa các biến đưa vào mô hình")
    priority_cols = [TARGET_COL] + FEATURE_COLS
    default_cols = priority_cols[:4]

    selected_cols = st.multiselect(
        "Chọn tối đa 4 biến để vẽ",
        options=priority_cols,
        default=default_cols,
        max_selections=4,
        help="Mặc định ưu tiên biến mục tiêu PD, sau đó là các biến đầu vào đầu tiên trong notebook.",
    )

    if not selected_cols:
        st.info("Hãy chọn ít nhất một biến để hiển thị biểu đồ.")
    else:
        rows = [selected_cols[:2], selected_cols[2:4]]
        for row in rows:
            if not row:
                continue
            cols = st.columns(2)
            for box, col_name in zip(cols, row):
                with box:
                    st.plotly_chart(plot_variable(df, col_name), use_container_width=True)

with tab_result:
    if "training_result" not in st.session_state:
        st.info("Chưa có kết quả. Vui lòng bấm nút **Huấn luyện & kiểm định mô hình** ở sidebar.")
    else:
        result = st.session_state["training_result"]
        metrics = result["metrics"]

        st.subheader("Chỉ tiêu kiểm định")
        metric_cols = st.columns(len(metrics))
        for box, (name, value) in zip(metric_cols, metrics.items()):
            box.metric(name, f"{value:.3f}")

        left, right = st.columns(2)
        with left:
            st.subheader("Ma trận nhầm lẫn")
            cm_df = pd.DataFrame(
                result["confusion_matrix"],
                index=["Thực tế 0", "Thực tế 1"],
                columns=["Dự báo 0", "Dự báo 1"],
            )
            st.dataframe(cm_df, use_container_width=True)

        with right:
            st.subheader("Classification report")
            report_df = pd.DataFrame(result["classification_report"]).transpose()
            st.dataframe(report_df, use_container_width=True)

        st.subheader("Bảng kết quả chấm điểm tập kiểm định")
        with st.container(height=320):
            st.dataframe(result["scored"], use_container_width=True)

        if "xac_suat_rui_ro" in result["scored"].columns:
            fig = px.histogram(
                result["scored"],
                x="xac_suat_rui_ro",
                color="Y_test",
                nbins=20,
                title="Phân phối xác suất dự báo có rủi ro",
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

with tab_use:
    if "trained_model" not in st.session_state:
        st.info("Chưa có mô hình đã huấn luyện. Vui lòng bấm nút **Huấn luyện & kiểm định mô hình** ở sidebar.")
    else:
        model = st.session_state["trained_model"]

        mode = st.radio(
            "Chọn chế độ sử dụng",
            ["Nhập trực tiếp", "Tải file dự báo hàng loạt"],
            horizontal=True,
            help="Dùng mô hình đã huấn luyện trong session hiện tại, không huấn luyện lại.",
        )

        if mode == "Nhập trực tiếp":
            st.subheader("Nhập giá trị cho từng biến đầu vào")
            with st.form("single_prediction_form"):
                values = {}
                form_cols = st.columns(4)
                for idx, col in enumerate(FEATURE_COLS):
                    series = pd.to_numeric(df[col], errors="coerce")
                    min_value = int(series.min()) if not pd.isna(series.min()) else 0
                    max_value = int(series.max()) if not pd.isna(series.max()) else 10
                    median_value = int(series.median()) if not pd.isna(series.median()) else min_value
                    with form_cols[idx % 4]:
                        values[col] = st.number_input(
                            col,
                            min_value=min_value,
                            max_value=max_value,
                            value=median_value,
                            step=1,
                            help=f"Giá trị đầu vào cho biến {col}. Mặc định là trung vị trong dữ liệu mẫu.",
                        )

                submitted = st.form_submit_button("Dự báo", type="primary")
                if submitted:
                    input_df = pd.DataFrame([values])
                    try:
                        pred_df = predict_dataframe(model, input_df)
                        pred = int(pred_df.loc[0, "du_bao_PD"])
                        if pred == 1:
                            st.error("Kết quả dự báo: Khách hàng có rủi ro (PD = 1).")
                        else:
                            st.success("Kết quả dự báo: Khách hàng không có rủi ro (PD = 0).")

                        if "xac_suat_co_rui_ro" in pred_df.columns:
                            c1, c2 = st.columns(2)
                            c1.metric("Xác suất không rủi ro", f"{pred_df.loc[0, 'xac_suat_khong_rui_ro'] * 100:.2f}%")
                            c2.metric("Xác suất có rủi ro", f"{pred_df.loc[0, 'xac_suat_co_rui_ro'] * 100:.2f}%")
                    except Exception as exc:
                        st.error(f"Không thể dự báo: {exc}")

        else:
            st.subheader("Tải file theo cấu trúc X_test")
            pred_file = st.file_uploader(
                "Tải file cần dự báo",
                type=["csv", "xlsx", "xls"],
                help="File cần có đầy đủ các cột đầu vào: " + ", ".join(FEATURE_COLS),
                key="prediction_file",
            )

            if pred_file is not None:
                try:
                    new_df = load_data(pred_file.getvalue(), pred_file.name)
                    missing_cols = validate_prediction_schema(new_df)
                    if missing_cols:
                        st.error("File dự báo thiếu cột: " + ", ".join(missing_cols))
                    else:
                        pred_result = predict_dataframe(model, new_df)
                        st.success(f"Đã dự báo {len(pred_result):,} dòng.")
                        with st.container(height=360):
                            st.dataframe(pred_result, use_container_width=True)

                        csv_bytes = pred_result.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "⬇️ Tải kết quả CSV",
                            data=csv_bytes,
                            file_name="ket_qua_du_bao.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                except Exception as exc:
                    st.error(f"Không thể xử lý file dự báo: {exc}")
