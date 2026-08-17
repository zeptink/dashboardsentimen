from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ============================================================
# KONFIGURASI DASAR
# ============================================================
st.set_page_config(
    page_title="Monitoring Sentimen Investor",
    page_icon="",
    layout="wide",
)

SENTIMENT_ORDER = ["Positif", "Netral", "Negatif"]
TEXT_COLUMNS = [
    "text", "tweet", "full_text", "content", "post", "posting",
    "unggahan", "komentar", "teks"
]
DATE_COLUMNS = [
    "created_at", "date", "datetime", "timestamp", "tanggal", "waktu"
]
SENTIMENT_COLUMNS = [
    "sentiment", "sentimen", "label", "prediction",
    "predicted_label", "hasil_prediksi"
]
CONFIDENCE_COLUMNS = [
    "confidence", "score", "probability", "probabilitas", "confidence_score"
]
TICKER_COLUMNS = [
    "ticker", "stock", "symbol", "kode_saham", "saham"
]



# FUNGSI BANTU
def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Mencari nama kolom tanpa membedakan huruf besar dan kecil."""
    lookup = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def normalize_sentiment(value: Any) -> str | None:
    """Menyeragamkan label sentimen menjadi Positif, Netral, atau Negatif."""
    if pd.isna(value):
        return None

    label = str(value).strip().lower()
    mapping = {
        "positive": "Positif",
        "positif": "Positif",
        "pos": "Positif",
        "bullish": "Positif",
        "neutral": "Netral",
        "netral": "Netral",
        "neu": "Netral",
        "negative": "Negatif",
        "negatif": "Negatif",
        "neg": "Negatif",
        "bearish": "Negatif",
    }

    if label in mapping:
        return mapping[label]
    if "posit" in label or "bull" in label:
        return "Positif"
    if "netral" in label or "neutral" in label:
        return "Netral"
    if "negat" in label or "bear" in label:
        return "Negatif"
    return str(value).strip().title()


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Menyeragamkan kolom CSV agar dapat dipakai oleh dashboard."""
    result = df.copy()

    text_col = find_column(result, TEXT_COLUMNS)
    date_col = find_column(result, DATE_COLUMNS)
    sentiment_col = find_column(result, SENTIMENT_COLUMNS)
    confidence_col = find_column(result, CONFIDENCE_COLUMNS)
    ticker_col = find_column(result, TICKER_COLUMNS)

    rename_map: dict[str, str] = {}
    if text_col:
        rename_map[text_col] = "text"
    if date_col:
        rename_map[date_col] = "created_at"
    if sentiment_col:
        rename_map[sentiment_col] = "sentiment"
    if confidence_col:
        rename_map[confidence_col] = "confidence"
    if ticker_col:
        rename_map[ticker_col] = "ticker"

    result = result.rename(columns=rename_map)

    if "text" not in result.columns:
        raise ValueError(
            "Kolom teks tidak ditemukan. Gunakan salah satu nama kolom berikut: "
            + ", ".join(TEXT_COLUMNS)
        )

    result["text"] = result["text"].fillna("").astype(str).str.strip()
    result = result[result["text"] != ""].copy()

    if "created_at" in result.columns:
        parsed_date = pd.to_datetime(result["created_at"], errors="coerce", utc=True)
        try:
            parsed_date = parsed_date.dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        result["created_at"] = parsed_date

    if "sentiment" in result.columns:
        result["sentiment"] = result["sentiment"].apply(normalize_sentiment)

    if "confidence" in result.columns:
        result["confidence"] = pd.to_numeric(result["confidence"], errors="coerce")
        # Mengubah 85 menjadi 0.85 apabila skor ditulis sebagai persen.
        result.loc[result["confidence"] > 1, "confidence"] /= 100

    if "ticker" not in result.columns:
        result["ticker"] = "-"

    return result.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def read_csv(file_bytes: bytes) -> pd.DataFrame:
    """Membaca CSV dengan beberapa alternatif encoding."""
    last_error: Exception | None = None

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            from io import BytesIO

            return pd.read_csv(BytesIO(file_bytes), encoding=encoding)
        except Exception as exc:
            last_error = exc

    raise ValueError(f"CSV tidak dapat dibaca: {last_error}")


@st.cache_resource(show_spinner="Memuat model mBERT...")
def load_model(model_path: str):
    """
    Memuat tokenizer dan model hasil fine-tuning.

    model_path dapat berupa:
    1. Folder lokal, misalnya: model_mbert
    2. Model ID Hugging Face Hub, misalnya: username/nama-model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


def resolve_model_label(
    raw_label: str,
    class_id: int,
    generic_label_map: dict[int, str],
) -> str:
    """Mengubah label model menjadi label dashboard."""
    normalized = normalize_sentiment(raw_label)

    if normalized in SENTIMENT_ORDER:
        return normalized

    # Model sering menyimpan label generik seperti LABEL_0, LABEL_1, LABEL_2.
    match = re.search(r"(\d+)$", str(raw_label))
    if match:
        label_id = int(match.group(1))
        return generic_label_map.get(label_id, str(raw_label))

    return generic_label_map.get(class_id, str(raw_label))


def predict_sentiments(
    texts: list[str],
    model_path: str,
    generic_label_map: dict[int, str],
    batch_size: int = 16,
    max_length: int = 128,
) -> tuple[list[str], list[float]]:
    """Melakukan inferensi sentimen secara bertahap agar penggunaan memori terkontrol."""
    tokenizer, model, device = load_model(model_path)

    predictions: list[str] = []
    confidences: list[float] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)
            scores, class_ids = probabilities.max(dim=-1)

        for class_id, score in zip(class_ids.tolist(), scores.tolist()):
            raw_label = model.config.id2label.get(class_id, f"LABEL_{class_id}")
            final_label = resolve_model_label(
                raw_label=raw_label,
                class_id=class_id,
                generic_label_map=generic_label_map,
            )
            predictions.append(final_label)
            confidences.append(float(score))

    return predictions, confidences


def load_sample_data() -> pd.DataFrame:
    sample_path = Path(__file__).parent / "sample_data.csv"
    if sample_path.exists():
        return standardize_dataframe(pd.read_csv(sample_path))

    return pd.DataFrame(
        {
            "created_at": pd.to_datetime(
                ["2026-06-01", "2026-06-02", "2026-06-03"]
            ),
            "text": [
                "IHSG menguat, semoga tren positif berlanjut.",
                "Pasar masih bergerak datar dan investor menunggu katalis.",
                "Tekanan jual hari ini cukup besar.",
            ],
            "sentiment": ["Positif", "Netral", "Negatif"],
            "confidence": [0.94, 0.88, 0.92],
            "ticker": ["IHSG", "IHSG", "IHSG"],
        }
    )


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")



# HEADER

st.title("📊 Dashboard Monitoring Sentimen Komunitas Investor")
st.caption(
    "Monitoring sentimen positif, netral, dan negatif pada unggahan Platform X "
    "menggunakan model mBERT."
)



# SIDEBAR: SUMBER DATA DAN KONFIGURASI MODEL

with st.sidebar:
    st.header("Pengaturan")

    uploaded_file = st.file_uploader(
        "Unggah data CSV",
        type=["csv"],
        help=(
            "Minimal memiliki kolom teks. Nama kolom yang didukung antara lain "
            "text, tweet, full_text, content, atau teks."
        ),
    )

    use_sample = st.toggle(
        "Gunakan data contoh",
        value=uploaded_file is None,
        disabled=uploaded_file is not None,
    )

    st.divider()
    st.subheader("Konfigurasi mBERT")

    default_model_path = os.getenv(
    "MODEL_PATH",
    "zptnk/mbert-sentimen-saham")
    
    model_path = st.text_input(
        "Folder atau model ID",
        value=default_model_path,
        help="Contoh folder lokal: model_mbert",
    )

    with st.expander("Pemetaan label model"):
        st.caption(
            "Atur bagian ini apabila config model masih memakai LABEL_0, "
            "LABEL_1, dan LABEL_2."
        )
        label_0 = st.selectbox(
            "LABEL_0", SENTIMENT_ORDER, index=2
        )
        label_1 = st.selectbox(
            "LABEL_1", SENTIMENT_ORDER, index=1
        )
        label_2 = st.selectbox(
            "LABEL_2", SENTIMENT_ORDER, index=0
        )

    generic_label_map = {0: label_0, 1: label_1, 2: label_2}

    batch_size = st.slider(
        "Batch size inferensi",
        min_value=1,
        max_value=64,
        value=16,
        step=1,
    )
    max_length = st.slider(
        "Maksimum panjang token",
        min_value=32,
        max_value=512,
        value=128,
        step=32,
    )



# MEMUAT DATA

try:
    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        source_key = f"{uploaded_file.name}-{len(raw_bytes)}"
        raw_df = read_csv(raw_bytes)
        data = standardize_dataframe(raw_df)
    elif use_sample:
        source_key = "sample-data"
        data = load_sample_data()
    else:
        st.info("Unggah file CSV atau aktifkan data contoh.")
        st.stop()
except Exception as exc:
    st.error(f"Gagal memuat data: {exc}")
    st.stop()


# Hapus hasil klasifikasi lama apabila sumber data berubah.
if st.session_state.get("source_key") != source_key:
    st.session_state["source_key"] = source_key
    st.session_state.pop("classified_data", None)



# KLASIFIKASI DATA JIKA LABEL BELUM TERSEDIA

needs_prediction = (
    "sentiment" not in data.columns
    or data["sentiment"].isna().all()
)

if needs_prediction:
    st.warning(
        "Data belum memiliki kolom sentimen. Jalankan model mBERT untuk "
        "menghasilkan label dan confidence."
    )

    if st.button(
        "Jalankan klasifikasi mBERT",
        type="primary",
        use_container_width=True,
    ):
        if not model_path.strip():
            st.error("Isi folder atau model ID terlebih dahulu.")
            st.stop()

        progress = st.progress(0, text="Menyiapkan klasifikasi...")

        try:
            texts = data["text"].tolist()
            total = len(texts)

            # Inferensi dilakukan per batch, tetapi progress ditampilkan per proses.
            labels, scores = predict_sentiments(
                texts=texts,
                model_path=model_path.strip(),
                generic_label_map=generic_label_map,
                batch_size=batch_size,
                max_length=max_length,
            )

            classified = data.copy()
            classified["sentiment"] = labels
            classified["confidence"] = scores
            st.session_state["classified_data"] = classified

            progress.progress(100, text=f"{total} data selesai diklasifikasikan.")
            st.success("Klasifikasi selesai.")
        except Exception as exc:
            progress.empty()
            st.error(
                "Model gagal dimuat atau inferensi gagal. Pastikan folder model "
                "berisi config, tokenizer, dan bobot hasil fine-tuning."
            )
            st.exception(exc)

    if "classified_data" not in st.session_state:
        st.stop()

    data = st.session_state["classified_data"].copy()
elif "classified_data" in st.session_state:
    data = st.session_state["classified_data"].copy()


# Menyaring label yang dikenali dashboard.
data["sentiment"] = data["sentiment"].apply(normalize_sentiment)
data = data[data["sentiment"].isin(SENTIMENT_ORDER)].copy()

if data.empty:
    st.error(
        "Tidak ada data dengan label Positif, Netral, atau Negatif setelah normalisasi."
    )
    st.stop()



# TABS

dashboard_tab, prediction_tab, data_tab = st.tabs(
    ["Dashboard", "Prediksi Teks", "Informasi Data"]
)



# TAB DASHBOARD

with dashboard_tab:
    st.subheader("Filter Data")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        selected_sentiments = st.multiselect(
            "Sentimen",
            options=SENTIMENT_ORDER,
            default=SENTIMENT_ORDER,
        )

    with filter_col2:
        ticker_options = sorted(
            value for value in data["ticker"].dropna().astype(str).unique()
        )
        selected_tickers = st.multiselect(
            "Ticker/kode saham",
            options=ticker_options,
            default=ticker_options,
        )

    with filter_col3:
        keyword = st.text_input(
            "Kata kunci",
            placeholder="Contoh: IHSG, dividen, bearish",
        )

    filtered = data.copy()

    if selected_sentiments:
        filtered = filtered[
            filtered["sentiment"].isin(selected_sentiments)
        ]
    else:
        filtered = filtered.iloc[0:0]

    if selected_tickers:
        filtered = filtered[
            filtered["ticker"].astype(str).isin(selected_tickers)
        ]

    if keyword.strip():
        filtered = filtered[
            filtered["text"].str.contains(
                keyword.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ]

    if (
        "created_at" in filtered.columns
        and filtered["created_at"].notna().any()
    ):
        min_date = filtered["created_at"].min().date()
        max_date = filtered["created_at"].max().date()

        selected_dates = st.date_input(
            "Rentang tanggal",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start_date, end_date = selected_dates
            date_values = filtered["created_at"].dt.date
            filtered = filtered[
                (date_values >= start_date)
                & (date_values <= end_date)
            ]

    st.divider()

    total_data = len(filtered)
    counts = (
        filtered["sentiment"]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
    )

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    metric_col1.metric("Total unggahan", f"{total_data:,}")
    metric_col2.metric("Positif", f"{int(counts['Positif']):,}")
    metric_col3.metric("Netral", f"{int(counts['Netral']):,}")
    metric_col4.metric("Negatif", f"{int(counts['Negatif']):,}")

    if "confidence" in filtered.columns and filtered["confidence"].notna().any():
        average_confidence = filtered["confidence"].mean() * 100
        metric_col5.metric("Rata-rata confidence", f"{average_confidence:.2f}%")
    else:
        metric_col5.metric("Rata-rata confidence", "-")

    if filtered.empty:
        st.warning("Tidak ada data yang sesuai dengan filter.")
        st.stop()

    chart_col1, chart_col2 = st.columns(2)

    distribution = counts.rename_axis("Sentimen").reset_index(name="Jumlah")
    distribution["Persentase"] = (
        distribution["Jumlah"] / distribution["Jumlah"].sum() * 100
    ).round(2)

    with chart_col1:
        st.markdown("#### Distribusi Sentimen")
        donut_chart = (
            alt.Chart(distribution)
            .mark_arc(innerRadius=65)
            .encode(
                theta=alt.Theta("Jumlah:Q"),
                color=alt.Color(
                    "Sentimen:N",
                    sort=SENTIMENT_ORDER,
                    legend=alt.Legend(title="Sentimen"),
                ),
                tooltip=[
                    alt.Tooltip("Sentimen:N"),
                    alt.Tooltip("Jumlah:Q", format=","),
                    alt.Tooltip("Persentase:Q", format=".2f"),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(donut_chart, use_container_width=True)

    with chart_col2:
        st.markdown("#### Jumlah Sentimen")
        bar_chart = (
            alt.Chart(distribution)
            .mark_bar()
            .encode(
                x=alt.X("Sentimen:N", sort=SENTIMENT_ORDER),
                y=alt.Y("Jumlah:Q", title="Jumlah unggahan"),
                color=alt.Color(
                    "Sentimen:N",
                    sort=SENTIMENT_ORDER,
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Sentimen:N"),
                    alt.Tooltip("Jumlah:Q", format=","),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(bar_chart, use_container_width=True)

    if (
        "created_at" in filtered.columns
        and filtered["created_at"].notna().any()
    ):
        st.markdown("#### Tren Sentimen Berdasarkan Waktu")

        trend = (
            filtered.dropna(subset=["created_at"])
            .assign(tanggal=lambda frame: frame["created_at"].dt.date)
            .groupby(["tanggal", "sentiment"])
            .size()
            .reset_index(name="jumlah")
        )

        trend_chart = (
            alt.Chart(trend)
            .mark_line(point=True)
            .encode(
                x=alt.X("tanggal:T", title="Tanggal"),
                y=alt.Y("jumlah:Q", title="Jumlah unggahan"),
                color=alt.Color(
                    "sentiment:N",
                    sort=SENTIMENT_ORDER,
                    title="Sentimen",
                ),
                tooltip=[
                    alt.Tooltip("tanggal:T", title="Tanggal"),
                    alt.Tooltip("sentiment:N", title="Sentimen"),
                    alt.Tooltip("jumlah:Q", title="Jumlah"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(trend_chart, use_container_width=True)

    st.markdown("#### Data Hasil Klasifikasi")

    preferred_columns = [
        column
        for column in [
            "created_at",
            "ticker",
            "text",
            "sentiment",
            "confidence",
        ]
        if column in filtered.columns
    ]

    display_df = filtered[preferred_columns].copy()

    if "confidence" in display_df.columns:
        display_df["confidence"] = (
            display_df["confidence"] * 100
        ).round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "created_at": st.column_config.DatetimeColumn(
                "Waktu unggahan",
                format="DD-MM-YYYY HH:mm",
            ),
            "ticker": "Ticker",
            "text": st.column_config.TextColumn(
                "Teks unggahan",
                width="large",
            ),
            "sentiment": "Sentimen",
            "confidence": st.column_config.NumberColumn(
                "Confidence (%)",
                format="%.2f",
            ),
        },
    )

    st.download_button(
        "Unduh hasil filter",
        data=to_csv_bytes(filtered),
        file_name="hasil_monitoring_sentimen.csv",
        mime="text/csv",
        use_container_width=True,
    )



# TAB PREDIKSI TEKS

with prediction_tab:
    st.subheader("Prediksi Satu Teks")
    st.write(
        "Masukkan satu unggahan untuk memperoleh label sentimen dan nilai confidence."
    )

    input_text = st.text_area(
        "Teks unggahan",
        height=150,
        placeholder="Contoh: IHSG hari ini kembali menguat dan volume transaksi meningkat.",
    )

    if st.button("Prediksi sentimen", type="primary"):
        if not input_text.strip():
            st.warning("Masukkan teks terlebih dahulu.")
        elif not model_path.strip():
            st.warning("Isi folder atau model ID pada sidebar.")
        else:
            try:
                label, score = predict_sentiments(
                    texts=[input_text.strip()],
                    model_path=model_path.strip(),
                    generic_label_map=generic_label_map,
                    batch_size=1,
                    max_length=max_length,
                )

                result_col1, result_col2 = st.columns(2)
                result_col1.metric("Hasil prediksi", label[0])
                result_col2.metric("Confidence", f"{score[0] * 100:.2f}%")
            except Exception as exc:
                st.error("Prediksi gagal. Periksa folder model dan konfigurasi label.")
                st.exception(exc)



# TAB INFORMASI DATA

with data_tab:
    st.subheader("Informasi Dataset")

    info_col1, info_col2 = st.columns(2)
    info_col1.metric("Jumlah baris", f"{len(data):,}")
    info_col2.metric("Jumlah kolom", f"{len(data.columns):,}")

    st.write("**Daftar kolom:**")
    st.code(", ".join(map(str, data.columns)))

    st.write("**Lima baris pertama:**")
    st.dataframe(data.head(), use_container_width=True, hide_index=True)

    st.info(
        "Format CSV minimum: kolom text. Kolom yang disarankan: "
        "created_at, text, sentiment, confidence, dan ticker."
    )
