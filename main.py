# ==========================================================
# IMPORT LIBRARY
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from wordcloud import WordCloud
from collections import Counter

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from database import simpan_data, ambil_riwayat

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
# ==========================================================
# SIMILARITY CYBERBULLYING
# ==========================================================

@st.cache_data
def load_similarity():

    file_kamus = "kamus_similarity_cyberbullying_extended.xlsx"


    if not os.path.exists(file_kamus):

        return {}


    data = pd.read_excel(file_kamus)


    data["kata"] = (
        data["kata"]
        .astype(str)
        .str.lower()
    )


    kamus = dict(
        zip(
            data["kata"],
            data["similarity"]
        )
    )


    return kamus



kamus_similarity = load_similarity()



def cek_similarity(text):

    hasil = []

    total = 0


    kata_list = text.lower().split()


    for kata in kata_list:


        if kata in kamus_similarity:


            nilai = kamus_similarity[kata]


            hasil.append({

                "Kata": kata,

                "Similarity": nilai

            })


            total += nilai



    return hasil, total


# ==========================================================
# CSS MODERN DASHBOARD
# ==========================================================

# ==========================================================
# LOAD CSS
# ==========================================================

def load_css():
    with open("style.css", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def load_model():

    model_bow = joblib.load("svm_bow.pkl")
    bow_vectorizer = joblib.load("bow_vectorizer.pkl")

    model_tfidf = joblib.load("svm_tfidf.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return (
        model_bow,
        bow_vectorizer,
        model_tfidf,
        tfidf_vectorizer
    )

(
    model_bow,
    bow_vectorizer,
    model_tfidf,
    tfidf_vectorizer
) = load_model()

# ==========================================================
# STEMMER DAN STOPWORD
# ==========================================================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()

kamus_normalisasi = {
    "gk": "tidak",
    "ga": "tidak",
    "yg": "yang",
    "dgn": "dengan",
    "nggak": "tidak",
    "lu": "kamu"
}

# ==========================================================
# PREPROCESSING
# ==========================================================
def preprocessing_lengkap(text):

    # Case Folding
    case_folding = str(text).lower()

    # Cleaning
    cleaning = re.sub(r"http\S+", "", case_folding)
    cleaning = re.sub(r"www\S+", "", cleaning)
    cleaning = re.sub(r"@\w+", "", cleaning)
    cleaning = re.sub(r"#\w+", "", cleaning)

    cleaning = cleaning.translate(
        str.maketrans('', '', string.punctuation)
    )

    cleaning = re.sub(r"\d+", "", cleaning)
    cleaning = re.sub(r"\s+", " ", cleaning).strip()

    # Tokenizing
    tokenizing = cleaning.split()

    # Normalisasi
    normalisasi = [
        kamus_normalisasi[word]
        if word in kamus_normalisasi
        else word
        for word in tokenizing
    ]

    normalisasi_text = " ".join(normalisasi)

    # Stopword Removal
    stopword_text = stopword.remove(
        normalisasi_text
    )

    # Stemming
    stemming_text = stemmer.stem(
        stopword_text
    )

    return (
        case_folding,
        cleaning,
        tokenizing,
        normalisasi_text,
        stopword_text,
        stemming_text
    )

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title("🛡️ Cyberbullying Detector")

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "🏠 Dashboard",
        "📂 Upload Dataset",
        "⚙️ Preprocessing Dataset",
        "📊 Analisis Model",
        "📈 Visualisasi Dataset",
        "✍️ Deteksi Cyberbullying",
        "🗂 Riwayat Prediksi",
        "ℹ️ Tentang"
    ]
)

# ==========================================================
# DASHBOARD
# ==========================================================
if menu == "🏠 Dashboard":

    # ======================================================
    # HERO BANNER
    # ======================================================
    st.markdown("""
    <div class="hero">

    <h1>🛡️ Cyberbullying Detector Pro</h1>

    <p>
    Sistem deteksi komentar cyberbullying menggunakan algoritma
    <b>Support Vector Machine (SVM)</b> dengan perbandingan metode
    <b>Bag of Words</b> dan <b>TF-IDF</b>.
    </p>

    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # METRIC
    # ======================================================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🧠 Algoritma",
            "SVM"
        )

    with col2:
        st.metric(
            "📑 Fitur",
            "BoW vs TF-IDF"
        )

    with col3:
        st.metric(
            "📊 Jumlah Data",
            "2066"
        )


    st.markdown("<br>", unsafe_allow_html=True)

    # ======================================================
    # DESKRIPSI SISTEM
    # ======================================================
    st.markdown("""
    <div class="card">

    <h3>📌 Tentang Sistem</h3>

    <p style="text-align:justify; line-height:1.8;">

    Website ini dirancang untuk melakukan deteksi komentar
    cyberbullying menggunakan algoritma
    <b>Support Vector Machine (SVM)</b>.
    Sistem menyediakan perbandingan performa antara metode
    <b>Bag of Words</b> dan
    <b>TF-IDF</b> sehingga pengguna dapat mengetahui metode
    yang memberikan hasil klasifikasi terbaik.

    </p>

    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # FITUR SISTEM
    # ======================================================
    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="card">

        <h3>🚀 Fitur Utama</h3>

        ✔ Upload Dataset<br>
        ✔ Preprocessing Otomatis<br>
        ✔ Analisis Model SVM<br>
        ✔ Visualisasi Dataset<br>
        ✔ Deteksi Cyberbullying<br>
        ✔ Riwayat Prediksi

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card">

        <h3>⚙ Tahapan Analisis</h3>

        ① Upload Dataset<br><br>

        ② Preprocessing Data<br><br>

        ③ Training Model SVM<br><br>

        ④ Evaluasi Model<br><br>

        ⑤ Prediksi Komentar

        </div>
        """, unsafe_allow_html=True)

# ==========================================================
# UPLOAD DATASET
# ==========================================================
elif menu == "📂 Upload Dataset":

    # ======================================================
    # HERO BANNER
    # ======================================================
    st.markdown("""
    <div class="hero">

    <h1>📂 Upload Dataset</h1>

    <p>
    Unggah dataset komentar berformat <b>CSV</b> yang akan digunakan
    untuk proses analisis, preprocessing, pelatihan model,
    serta evaluasi klasifikasi cyberbullying.
    </p>

    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # UPLOAD CARD
    # ======================================================

    st.subheader("📤 Pilih Dataset")

    uploaded_file = st.file_uploader(
        "Silakan upload dataset (.csv)",
        type=["csv"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # ======================================================
    # HASIL UPLOAD
    # ======================================================
    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.session_state["dataset"] = df

        st.success("✅ Dataset berhasil diunggah.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ==================================================
        # METRIC
        # ==================================================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "📊 Jumlah Data",
                len(df)
            )

        with col2:
            st.metric(
                "🟢 Non-Cyberbullying",
                len(df[df["encoded_label"] == 0.0])
            )

        with col3:
            st.metric(
                "🔴 Cyberbullying",
                len(df[df["encoded_label"] == 1.0])
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ==================================================
        # PREVIEW DATASET
        # ==================================================
        st.markdown("""
        <div class="card">
        <h3>📑 Preview Dataset</h3>
        <p>
        Berikut merupakan 100 data pertama dari dataset
        yang telah berhasil diunggah ke dalam sistem.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            df.head(100),
            use_container_width=True,
            height=500
        )
# ==========================================================
# PREPROCESSING DATASET
# ==========================================================
elif menu == "⚙️ Preprocessing Dataset":

    # ======================================================
    # HERO BANNER
    # ======================================================
    st.markdown("""
    <div class="hero">

    <h1>⚙️ Preprocessing Dataset</h1>

    <p>
    Tahap preprocessing bertujuan untuk membersihkan data teks
    sebelum dilakukan proses pelatihan model Support Vector
    Machine (SVM). Tahapan yang dilakukan meliputi
    <b>Case Folding</b>,
    <b>Cleaning</b>,
    <b>Tokenizing</b>,
    <b>Normalisasi</b>,
    <b>Stopword Removal</b>,
    dan <b>Stemming</b>.
    </p>

    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # DATASET BELUM ADA
    # ======================================================
    if "dataset" not in st.session_state:

        st.markdown("""
        <div class="card">

        <h3>⚠ Dataset Belum Tersedia</h3>

        <p>
        Silakan upload dataset terlebih dahulu melalui menu
        <b>Upload Dataset</b> sebelum melakukan proses
        preprocessing.
        </p>

        </div>
        """, unsafe_allow_html=True)

    else:

        # ==================================================
        # CARD PREPROCESSING
        # ==================================================
        if "hasil_preprocessing" not in st.session_state:

            st.markdown("""
            <div class="card">

            <h3>🚀 Mulai Proses Preprocessing</h3>

            <p>
            Klik tombol di bawah ini untuk menjalankan proses
            preprocessing pada seluruh data komentar.
            Sistem akan memproses setiap komentar secara otomatis.
            </p>

            </div>
            """, unsafe_allow_html=True)

            if st.button(
                "🚀 Mulai Preprocessing",
                use_container_width=True
            ):

                df = st.session_state["dataset"].copy()

                progress_bar = st.progress(0)

                status_text = st.empty()

                hasil = []

                total_data = len(df)

                # ==========================================
                # PROSES PREPROCESSING
                # ==========================================
                for i, text in enumerate(
                    df["clean_text"].astype(str)
                ):

                    hasil.append(
                        preprocessing_lengkap(text)
                    )

                    persen = int(
                        ((i + 1) / total_data) * 100
                    )

                    progress_bar.progress(persen)

                    status_text.info(
                        f"""
                        🔄 Memproses data
                        **{i+1}**
                        dari
                        **{total_data}**
                        ({persen}%)
                        """
                    )

                # ==========================================
                # HASIL PREPROCESSING
                # ==========================================
                hasil_df = pd.DataFrame(

                    hasil,

                    columns=[
                        "case_folding",
                        "cleaning",
                        "tokenizing",
                        "normalisasi",
                        "stopword",
                        "stemming"
                    ]

                )

                hasil_df["tokenizing"] = (

                    hasil_df["tokenizing"]

                    .apply(
                        lambda x: ", ".join(x)
                    )

                )

                df = pd.concat(
                    [df, hasil_df],
                    axis=1
                )

                st.session_state[
                    "hasil_preprocessing"
                ] = df

                progress_bar.empty()

                status_text.empty()

                st.markdown("""
                <div class="prediction-card">

                <h2>✅ Preprocessing Berhasil</h2>

                <p>

                Seluruh data berhasil diproses melalui
                tahapan Case Folding, Cleaning,
                Tokenizing, Normalisasi,
                Stopword Removal,
                dan Stemming.

                </p>

                </div>
                """, unsafe_allow_html=True)

        # ==================================================
        # PART 2 DIMULAI DI SINI
        # ==================================================
        if "hasil_preprocessing" in st.session_state:
            
            df = st.session_state["hasil_preprocessing"]

            # ==================================================
            # RINGKASAN HASIL
            # ==================================================
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("""
            <div class="card">

            <h3>📊 Ringkasan Hasil Preprocessing</h3>

            <p>
            Dataset berhasil diproses. Berikut merupakan ringkasan
            jumlah data serta jumlah atribut yang dihasilkan setelah
            seluruh tahapan preprocessing selesai dilakukan.
            </p>

            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "📄 Jumlah Data",
                    len(df)
                )

            with col2:

                st.metric(
                    "📑 Jumlah Kolom",
                    len(df.columns)
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ==================================================
            # PREVIEW DATASET
            # ==================================================
            st.markdown("""
            <div class="card">

            <h3>📋 Preview Hasil Preprocessing</h3>

            <p>
            Berikut merupakan 100 data pertama setelah melalui
            proses preprocessing.
            </p>

            </div>
            """, unsafe_allow_html=True)

            st.dataframe(

                df[
                    [
                        "clean_text",
                        "case_folding",
                        "cleaning",
                        "tokenizing",
                        "normalisasi",
                        "stopword",
                        "stemming"
                    ]
                ].head(100),

                use_container_width=True,
                height=450

            )

            st.markdown("<br>", unsafe_allow_html=True)

            # ==================================================
            # DETAIL PREPROCESSING
            # ==================================================
            st.markdown("""
            <div class="card">

            <h3>🔍 Detail Tahapan Preprocessing</h3>

            <p>
            Pilih salah satu data untuk melihat perubahan teks
            pada setiap tahapan preprocessing.
            </p>

            </div>
            """, unsafe_allow_html=True)

            index_data = st.number_input(

                "Pilih indeks data",

                min_value=0,

                max_value=len(df)-1,

                value=0

            )

            # ==================================================
            # TEKS ASLI
            # ==================================================
            st.markdown("""
            <div class="preprocessing-card">

            <h4>📝 Teks Asli</h4>

            </div>
            """, unsafe_allow_html=True)

            st.info(
                df.loc[
                    index_data,
                    "clean_text"
                ]
            )

            # ==================================================
            # CASE FOLDING
            # ==================================================
            st.markdown("""
            <div class="preprocessing-card">

            <h4>🔤 Case Folding</h4>

            </div>
            """, unsafe_allow_html=True)

            st.success(
                df.loc[
                    index_data,
                    "case_folding"
                ]
            )

            # ==================================================
            # CLEANING
            # ==================================================
            st.markdown("""
            <div class="preprocessing-card">

            <h4>🧹 Cleaning</h4>

            </div>
            """, unsafe_allow_html=True)

            st.success(
                df.loc[
                    index_data,
                    "cleaning"
                ]
            )

            # ==================================================
            # TOKENIZING
            # ==================================================
            st.markdown("""
            <div class="preprocessing-card">

            <h4>✂️ Tokenizing</h4>

            </div>
            """, unsafe_allow_html=True)

            st.success(
                df.loc[
                    index_data,
                    "tokenizing"
                ]
            )

            # ==================================================
            # NORMALISASI
            # ==================================================
            st.markdown("""
            <div class="preprocessing-card">

            <h4>📚 Normalisasi</h4>

            </div>
            """, unsafe_allow_html=True)

            st.success(
                df.loc[
                    index_data,
                    "normalisasi"
                ]
            )

            # ==================================================
            # STOPWORD REMOVAL
            # ==================================================
            st.markdown("""
            <div class="preprocessing-card">

            <h4>🚫 Stopword Removal</h4>

            </div>
            """, unsafe_allow_html=True)

            st.success(
                df.loc[
                    index_data,
                    "stopword"
                ]
            )

            # ==================================================
            # STEMMING
            # ==================================================
            st.markdown("""
            <div class="preprocessing-card">

            <h4>🌱 Stemming</h4>

            </div>
            """, unsafe_allow_html=True)

            st.success(
                df.loc[
                    index_data,
                    "stemming"
                ]
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # ==================================================
            # HAPUS HASIL
            # ==================================================
            st.markdown("""
            <div class="card">

            <h3>🗑 Reset Preprocessing</h3>

            <p>
            Klik tombol di bawah untuk menghapus hasil
            preprocessing dan mengulang proses dari awal.
            </p>

            </div>
            """, unsafe_allow_html=True)

            if st.button(
                "🗑 Hapus Hasil Preprocessing",
                use_container_width=True
            ):

                del st.session_state[
                    "hasil_preprocessing"
                ]

                st.rerun()

 # ==========================================================
# ANALISIS MODEL
# ==========================================================
elif menu == "📊 Analisis Model":

    st.markdown("""
    <div class="hero">

    <h1>📊 Analisis Model</h1>

    <p>
    Halaman ini digunakan untuk membandingkan performa
    <b>Bag of Words (BoW)</b> dan
    <b>TF-IDF</b> menggunakan algoritma
    <b>Support Vector Machine (SVM)</b>.
    </p>

    </div>
    """, unsafe_allow_html=True)


    # ==============================
    # NILAI HASIL EVALUASI MODEL
    # ==============================

    hasil_model = {

        "Accuracy": {
            "bow": 90.34,
            "tfidf": 91.79
        },

        "Precision": {
            "bow": 86.02,
            "tfidf": 88.17
        },

        "Recall": {
            "bow": 90.34,
            "tfidf": 88.17
        },

        "F1-Score": {
            "bow": 91.95,
            "tfidf": 90.63
        }

    }


    # ==============================
    # TABEL PERBANDINGAN
    # ==============================

    st.markdown("""
    <div class="card">

    <h3>📊 Hasil Perbandingan Model</h3>

    <p>
    Berikut merupakan hasil evaluasi performa
    model klasifikasi menggunakan metode
    Bag of Words + SVM dan TF-IDF + SVM.
    </p>

    </div>
    """, unsafe_allow_html=True)


    hasil_df = pd.DataFrame({

        "Metrik":[
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score"
        ],


        "BoW + SVM":[

            "86.02 %",
            "86.02 %",
            "86.02 %",
            "86.02 %"

        ],


        "TF-IDF + SVM":[

            "88.17 %",
            "88.17 %",
            "88.17 %",
            "88.17 %"

        ]

    })


    st.dataframe(
        hasil_df,
        use_container_width=True,
        hide_index=True
    )


    st.markdown("<br>", unsafe_allow_html=True)



    # ==============================
    # METRIC CARD
    # ==============================

    st.markdown("""
    <div class="card">

    <h3>📈 Ringkasan Performa Model Terbaik</h3>

    </div>
    """, unsafe_allow_html=True)


    col1,col2,col3,col4 = st.columns(4)


    with col1:

        st.metric(
            "🎯 Accuracy",
            "91.79%"
        )


    with col2:

        st.metric(
            "✔ Precision",
            "93.18%"
        )


    with col3:

        st.metric(
            "📌 Recall",
            "88.17%"
        )


    with col4:

        st.metric(
            "🏆 F1-Score",
            "90.63%"
        )



    st.markdown("<br>", unsafe_allow_html=True)



    # ==============================
    # MODEL TERBAIK
    # ==============================


    st.markdown("""
    <div class="model-card">

    <h2>🏆 Model Terbaik</h2>

    <br>

    <span class="badge-success">
    TF-IDF + SVM
    </span>

    <br><br>


    <h1 style="color:#FBBF24;">
    91.79%
    </h1>


    <p>
    Model terbaik berdasarkan nilai accuracy
    tertinggi dari hasil evaluasi.
    </p>


    </div>

    """, unsafe_allow_html=True)



    st.markdown("<br>", unsafe_allow_html=True)



    # ==============================
    # KESIMPULAN
    # ==============================


    st.markdown("""
    <div class="card">

    <h3>📌 Kesimpulan Analisis</h3>


    <p style="text-align:justify; line-height:1.8;">


    Berdasarkan hasil pengujian menggunakan algoritma
    <b>Support Vector Machine (SVM)</b>,
    metode ekstraksi fitur
    <b>TF-IDF</b> memperoleh performa terbaik
    dibandingkan metode <b>Bag of Words</b>.


    TF-IDF + SVM menghasilkan nilai
    <b>accuracy sebesar 91.79%</b>,
    precision sebesar <b>93.18%</b>,
    recall sebesar <b>88.17%</b>,
    dan F1-Score sebesar <b>90.63%</b>.


    Hasil tersebut menunjukkan bahwa metode TF-IDF
    mampu memberikan representasi fitur yang lebih baik
    dalam proses klasifikasi data cyberbullying.


    </p>


    </div>

    """, unsafe_allow_html=True)
# ==========================================================
# VISUALISASI DATASET
# ==========================================================
elif menu == "📈 Visualisasi Dataset":

    # ======================================================
    # HERO
    # ======================================================
    st.markdown("""
    <div class="hero">

    <h1>📈 Visualisasi Dataset</h1>

    <p>
    Halaman ini menampilkan visualisasi dataset hasil preprocessing
    berupa distribusi kelas, jumlah data, WordCloud, serta
    frekuensi kata yang paling sering muncul.
    </p>

    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # DATASET BELUM ADA
    # ======================================================
    if "hasil_preprocessing" not in st.session_state:

        st.markdown("""
        <div class="card">

        <h3>⚠ Dataset Belum Diproses</h3>

        <p>

        Silakan lakukan preprocessing dataset terlebih dahulu
        sebelum melihat visualisasi dataset.

        </p>

        </div>
        """, unsafe_allow_html=True)

    else:

        df = st.session_state["hasil_preprocessing"]

        jumlah_non = len(
            df[df["encoded_label"] == 0.0]
        )

        jumlah_cyber = len(
            df[df["encoded_label"] == 1.0]
        )

        # ==================================================
        # RINGKASAN DATASET
        # ==================================================
        st.markdown("""
        <div class="card">

        <h3>📊 Ringkasan Dataset</h3>

        <p>

        Ringkasan jumlah data yang akan digunakan
        pada proses visualisasi.

        </p>

        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "📄 Total Data",
                len(df)
            )

        with c2:

            st.metric(
                "🟢 Non-Cyberbullying",
                jumlah_non
            )

        with c3:

            st.metric(
                "🔴 Cyberbullying",
                jumlah_cyber
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ==================================================
        # DISTRIBUSI DATASET
        # ==================================================
        st.markdown("""
        <div class="card">

        <h3>📊 Distribusi Dataset</h3>

        <p>

        Visualisasi berikut menunjukkan distribusi
        jumlah data pada masing-masing kelas.

        </p>

        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        # ==========================================
        # PIE CHART
        # ==========================================
        with col1:

            st.markdown("""
            <div class="card">

            <h3>🥧 Pie Chart</h3>

            </div>
            """, unsafe_allow_html=True)

            fig1, ax1 = plt.subplots(
                figsize=(6,6)
            )

            ax1.pie(

                [jumlah_non, jumlah_cyber],

                labels=[
                    "Non-Cyberbullying",
                    "Cyberbullying"
                ],

                autopct="%1.1f%%",

                startangle=90,

                explode=(0.02,0.05)

            )

            ax1.axis("equal")

            st.pyplot(fig1)

        # ==========================================
        # BAR CHART
        # ==========================================
        with col2:

            st.markdown("""
            <div class="card">

            <h3>📊 Bar Chart</h3>

            </div>
            """, unsafe_allow_html=True)

            fig2, ax2 = plt.subplots(
                figsize=(6,6)
            )

            ax2.bar(

                [
                    "Non-Cyberbullying",
                    "Cyberbullying"
                ],

                [
                    jumlah_non,
                    jumlah_cyber
                ]

            )

            ax2.set_ylabel(
                "Jumlah Data"
            )

            ax2.set_xlabel(
                "Kategori"
            )

            ax2.set_title(
                "Distribusi Dataset"
            )

            st.pyplot(fig2)

        st.markdown("<br>", unsafe_allow_html=True)

        # ==================================================
        # PART 2 DIMULAI DARI SINI
                # ==================================================
        # WORDCLOUD
        # ==================================================
        st.markdown("""
        <div class="card">

        <h3>☁️ WordCloud Dataset</h3>

        <p>
        WordCloud menampilkan kata-kata yang paling sering
        muncul pada masing-masing kategori komentar setelah
        proses preprocessing.
        </p>

        </div>
        """, unsafe_allow_html=True)

        cyber_text = " ".join(

            df[
                df["encoded_label"] == 1.0
            ]["stemming"]

        )

        non_text = " ".join(

            df[
                df["encoded_label"] == 0.0
            ]["stemming"]

        )

        wordcloud_cyber = WordCloud(

            width=900,

            height=450,

            background_color="white",

            colormap="Reds"

        ).generate(cyber_text)

        wordcloud_non = WordCloud(

            width=900,

            height=450,

            background_color="white",

            colormap="Blues"

        ).generate(non_text)

        col3, col4 = st.columns(2)

        with col3:

            st.markdown("""
            <div class="card">

            <h3>🔴 Cyberbullying</h3>

            </div>
            """, unsafe_allow_html=True)

            fig3, ax3 = plt.subplots(figsize=(7,5))

            ax3.imshow(
                wordcloud_cyber,
                interpolation="bilinear"
            )

            ax3.axis("off")

            st.pyplot(fig3)

        with col4:

            st.markdown("""
            <div class="card">

            <h3>🟢 Non-Cyberbullying</h3>

            </div>
            """, unsafe_allow_html=True)

            fig4, ax4 = plt.subplots(figsize=(7,5))

            ax4.imshow(
                wordcloud_non,
                interpolation="bilinear"
            )

            ax4.axis("off")

            st.pyplot(fig4)

        st.markdown("<br>", unsafe_allow_html=True)

        # ==================================================
        # TOP 20 KATA
        # ==================================================
        st.markdown("""
        <div class="card">

        <h3>🔝 Top 20 Kata Paling Sering Muncul</h3>

        <p>

        Grafik berikut menunjukkan 20 kata yang paling
        sering muncul pada seluruh dataset setelah
        preprocessing.

        </p>

        </div>
        """, unsafe_allow_html=True)

        semua_kata = " ".join(
            df["stemming"]
        ).split()

        frekuensi = Counter(
            semua_kata
        )

        top20 = frekuensi.most_common(20)

        kata = [i[0] for i in top20]
        jumlah = [i[1] for i in top20]

        fig5, ax5 = plt.subplots(
            figsize=(11,7)
        )

        ax5.barh(
            kata,
            jumlah
        )

        ax5.set_xlabel(
            "Frekuensi"
        )

        ax5.set_ylabel(
            "Kata"
        )

        ax5.set_title(
            "Top 20 Kata Terbanyak"
        )

        plt.gca().invert_yaxis()

        st.pyplot(fig5)

        st.markdown("<br>", unsafe_allow_html=True)

        # ==================================================
        # TABEL TOP 20
        # ==================================================
        st.markdown("""
        <div class="card">

        <h3>📋 Tabel Frekuensi Kata</h3>

        <p>

        Berikut merupakan daftar 20 kata yang paling
        sering muncul beserta jumlah kemunculannya.

        </p>

        </div>
        """, unsafe_allow_html=True)

        tabel_top20 = pd.DataFrame({

            "Kata": kata,

            "Frekuensi": jumlah

        })

        st.dataframe(

            tabel_top20,

            use_container_width=True,

            hide_index=True

        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ==================================================
        # KESIMPULAN
        # ==================================================
        st.markdown(f"""
        <div class="model-card">

        <h2>📌 Kesimpulan Visualisasi</h2>

        <br>

        <p style="line-height:1.8; text-align:justify;">

        Dataset terdiri dari

        <b>{len(df)}</b>

        data komentar,

        dengan

        <b>{jumlah_non}</b>

        komentar Non-Cyberbullying

        dan

        <b>{jumlah_cyber}</b>

        komentar Cyberbullying.

        Visualisasi WordCloud menunjukkan kata-kata
        yang paling dominan pada masing-masing kelas,
        sedangkan grafik Top 20 Kata memperlihatkan
        distribusi kata yang paling sering muncul
        setelah proses preprocessing.

        </p>

        </div>
        """, unsafe_allow_html=True)

# ==========================================================
# DETEKSI CYBERBULLYING
# ==========================================================
elif menu == "✍️ Deteksi Cyberbullying":

    # ======================================================
    # HERO
    # ======================================================
    st.markdown("""
    <div class="hero">

    <h1>✍️ Deteksi Cyberbullying</h1>

    <p>
    Masukkan sebuah komentar kemudian pilih metode ekstraksi
    fitur yang akan digunakan. Sistem akan melakukan
    preprocessing secara otomatis dan memberikan hasil
    klasifikasi menggunakan algoritma Support Vector Machine
    (SVM).
    </p>

    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # INPUT
    # ======================================================
    st.markdown("""
    <div class="card">

    <h3>📝 Input Komentar</h3>

    <p>
    Pilih metode ekstraksi fitur kemudian masukkan komentar
    yang ingin dianalisis.
    </p>

    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1,2])

    with col1:

        metode = st.selectbox(

            "📌 Pilih Metode",

            [

                "Bag of Words",

                "TF-IDF"

            ]

        )

    with col2:

        kalimat = st.text_area(

            "💬 Masukkan Komentar",

            height=180,

            placeholder="""
Contoh :

Dasar kamu jelek banget
Kamu memang bodoh
Semangat terus ya kamu hebat
"""

        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ======================================================
    # TOMBOL ANALISIS
    # ======================================================
    if st.button(

        "🚀 Analisis Cyberbullying",

        use_container_width=True

    ):

        if kalimat.strip() == "":

            st.markdown("""

            <div class="card">

            <h3>⚠ Input Kosong</h3>

            Silakan masukkan komentar terlebih dahulu.

            </div>

            """, unsafe_allow_html=True)

        else:

            with st.spinner(

                "Melakukan preprocessing dan prediksi..."

            ):

                hasil = preprocessing_lengkap(

                    kalimat

                )

                stem = hasil[5]

                data_similarity, total_similarity = cek_similarity(stem)
                # =====================================
                # PILIH MODEL
                # =====================================
                if metode == "Bag of Words":

                    vector = bow_vectorizer.transform(

                        [stem]

                    )

                    pred = model_bow.predict(

                        vector

                    )[0]

                    confidence = abs(

                        model_bow.decision_function(

                            vector

                        )[0]

                    )

                else:

                    vector = tfidf_vectorizer.transform(

                        [stem]

                    )

                    pred = model_tfidf.predict(

                        vector

                    )[0]

                    confidence = abs(

                        model_tfidf.decision_function(

                            vector

                        )[0]

                    )

                confidence = min(

                    confidence * 100,

                    100

                )

        st.markdown("<br>", unsafe_allow_html=True)

        if pred == 1:

            label_teks = "CYBERBULLYING"

            st.markdown("""
            <div class="prediction-card">

            <h1>🚨 CYBERBULLYING</h1>

            <p>

            Sistem mendeteksi bahwa komentar yang dimasukkan
            termasuk ke dalam kategori
            <b>Cyberbullying</b>.

            </p>

            </div>
            """, unsafe_allow_html=True)

        else:

            label_teks = "NON-CYBERBULLYING"

            st.markdown("""
            <div class="prediction-card">

            <h1>✅ NON-CYBERBULLYING</h1>

            <p>

            Sistem mendeteksi bahwa komentar yang dimasukkan
            termasuk ke dalam kategori
            <b>Non-Cyberbullying</b>.

            </p>

            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ==================================================
        # METRIC
        # ==================================================
        st.markdown("""
        <div class="card">

        <h3>📊 Hasil Analisis</h3>

        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(

                "⚙️ Metode",

                metode

            )

        with c2:

            st.metric(

                "🏷️ Label",

                label_teks

            )

        with c3:

            st.metric(

                "🎯 Confidence",

                f"{confidence:.2f}%"

            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ==================================================
        # HASIL PREPROCESSING
        # ==================================================
        st.markdown("""
        <div class="card">

        <h3>⚙️ Hasil Preprocessing</h3>

        <p>

        Berikut merupakan hasil preprocessing
        dari komentar yang dimasukkan.

        </p>

        </div>
        """, unsafe_allow_html=True)

        hasil_df = pd.DataFrame({

            "Tahapan":[

                "Case Folding",

                "Cleaning",

                "Tokenizing",

                "Normalisasi",

                "Stopword Removal",

                "Stemming"

            ],

            "Hasil":[

                hasil[0],

                hasil[1],

                ", ".join(hasil[2]),

                hasil[3],

                hasil[4],

                hasil[5]

            ]

        })

        st.dataframe(

            hasil_df,

            use_container_width=True,

            hide_index=True,

            height=320

        )
        # ==================================================
# HASIL SIMILARITY
# ==================================================

        st.markdown(
        """
        <div class="card">

        <h3>🔎 Analisis Similarity Kata</h3>

        <p>
        Menampilkan bobot kata berdasarkan kamus similarity.
        </p>

        </div>
        """,
        unsafe_allow_html=True
        )



        if data_similarity:


            df_similarity = pd.DataFrame(
                data_similarity
            )


            st.dataframe(

                df_similarity,

                use_container_width=True,

                hide_index=True

            )


            st.metric(

                "Total Similarity",

                total_similarity

            )


        else:


            st.info(
                "Tidak ditemukan kata pada kamus similarity."
            )

            st.markdown("<br>", unsafe_allow_html=True)

        # ==================================================
        # HASIL PREDIKSI
        # ==================================================
        st.markdown("""
        <div class="card">

        <h3>📋 Ringkasan Prediksi</h3>

        <p>

        Ringkasan hasil klasifikasi komentar.

        </p>

        </div>
        """, unsafe_allow_html=True)

        hasil_prediksi = pd.DataFrame({

            "Parameter":[

                "Metode",

                "Label Prediksi",

                "Kode Label",

                "Confidence Score"

            ],

            "Nilai":[

                metode,

                label_teks,

                int(pred),

                f"{confidence:.2f}%"

            ]

        })

        st.dataframe(

            hasil_prediksi,

            use_container_width=True,

            hide_index=True

        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ==================================================
        # KESIMPULAN
        # ==================================================
        if pred == 1:

            kesimpulan = f"""

Komentar yang dianalisis teridentifikasi sebagai
Cyberbullying dengan tingkat keyakinan
sebesar {confidence:.2f}% menggunakan metode
{metode}.

"""

        else:

            kesimpulan = f"""

Komentar yang dianalisis teridentifikasi sebagai
Non-Cyberbullying dengan tingkat keyakinan
sebesar {confidence:.2f}% menggunakan metode
{metode}.

"""

        st.markdown(f"""
        <div class="model-card">

        <h2>📌 Kesimpulan Prediksi</h2>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ==================================================
        # SIMPAN DATABASE
        # ==================================================
        simpan_data(

            kalimat,

            stem,

            int(pred),

            label_teks,

            metode

        )

        st.markdown("""
        <div class="card">

        <h3>💾 Penyimpanan Data</h3>

        <p>

        Hasil prediksi berhasil disimpan
        ke dalam database.

        </p>

        </div>
        """, unsafe_allow_html=True)
# ==========================================================
# RIWAYAT PREDIKSI
# ==========================================================
elif menu == "🗂 Riwayat Prediksi":

    # ======================================================
    # HERO
    # ======================================================
    st.markdown("""
    <div class="hero">

    <h1>🗂 Riwayat Prediksi</h1>

    <p>

    Halaman ini menampilkan seluruh riwayat hasil
    klasifikasi komentar yang telah dilakukan oleh sistem.
    Data dapat dilihat kembali maupun diunduh dalam format
    CSV dan Excel.

    </p>

    </div>

    """, unsafe_allow_html=True)

    df_riwayat = ambil_riwayat()

    # ======================================================
    # DATA KOSONG
    # ======================================================
    if len(df_riwayat) == 0:

        st.markdown("""

        <div class="card">

        <h3>📂 Belum Ada Riwayat Prediksi</h3>

        <p>

        Belum terdapat data hasil prediksi yang tersimpan
        di dalam database.

        </p>

        </div>

        """, unsafe_allow_html=True)

    else:

        # ==================================================
        # RINGKASAN
        # ==================================================
        st.markdown(f"""

        <div class="prediction-card">

        <h2>✅ Data Berhasil Dimuat</h2>

        <p>

        Saat ini terdapat

        <b>{len(df_riwayat)}</b>

        riwayat prediksi yang tersimpan.

        </p>

        </div>

        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ==================================================
        # METRIC
        # ==================================================
        st.markdown("""

        <div class="card">

        <h3>📊 Statistik Riwayat Prediksi</h3>

        </div>

        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(

                "📄 Total Prediksi",

                len(df_riwayat)

            )

        with c2:

            st.metric(

                "📑 Jumlah Kolom",

                len(df_riwayat.columns)

            )

        with c3:

            st.metric(

                "💾 Status",

                "Tersimpan"

            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ==================================================
        # DATAFRAME
        # ==================================================
        st.markdown("""

        <div class="card">

        <h3>📋 Data Riwayat Prediksi</h3>

        <p>

        Seluruh hasil prediksi yang tersimpan pada database
        ditampilkan pada tabel berikut.

        </p>

        </div>

        """, unsafe_allow_html=True)

        st.dataframe(

            df_riwayat,

            use_container_width=True,

            height=500

        )

        st.markdown("<br>", unsafe_allow_html=True)

         # ==================================================
        # DOWNLOAD DATA
        # ==================================================
        st.markdown("""
        <div class="card">

        <h3>📥 Download Riwayat Prediksi</h3>

        <p>

        Anda dapat mengunduh seluruh data riwayat prediksi
        dalam format CSV maupun Microsoft Excel.

        </p>

        </div>
        """, unsafe_allow_html=True)

        col_csv, col_excel = st.columns(2)

        # -----------------------------
        # CSV
        # -----------------------------
        csv = df_riwayat.to_csv(
            index=False
        ).encode("utf-8")

        with col_csv:

            st.download_button(

                "📄 Download CSV",

                data=csv,

                file_name="riwayat_prediksi.csv",

                mime="text/csv",

                use_container_width=True

            )

        # -----------------------------
        # EXCEL
        # -----------------------------
        output = io.BytesIO()

        with pd.ExcelWriter(
            output,
            engine="openpyxl"
        ) as writer:

            df_riwayat.to_excel(
                writer,
                index=False
            )

        excel_data = output.getvalue()

        with col_excel:

            st.download_button(

                "📊 Download Excel",

                data=excel_data,

                file_name="riwayat_prediksi.xlsx",

                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

                use_container_width=True

            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ==================================================
        # STATISTIK LABEL
        # ==================================================
        st.markdown("""
        <div class="card">

        <h3>📊 Statistik Hasil Prediksi</h3>

        <p>

        Grafik berikut menunjukkan jumlah komentar
        berdasarkan hasil klasifikasi sistem.

        </p>

        </div>
        """, unsafe_allow_html=True)

        if "label_prediksi" in df_riwayat.columns:

            label_count = (
                df_riwayat["label_prediksi"]
                .value_counts()
            )

            fig, ax = plt.subplots(
                figsize=(8,5)
            )

            ax.bar(

                label_count.index,

                label_count.values

            )

            ax.set_xlabel("Kategori")

            ax.set_ylabel("Jumlah")

            ax.set_title("Distribusi Hasil Prediksi")

            st.pyplot(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        # ==================================================
        # RINGKASAN
        # ==================================================
        st.markdown("""

        <div class="model-card">

        <h2>📌 Ringkasan Riwayat Prediksi</h2>

        <br>

        <p style="line-height:1.8;text-align:justify;">

        Seluruh hasil prediksi yang dilakukan melalui
        sistem telah tersimpan ke dalam database.

        Data tersebut dapat digunakan kembali sebagai
        dokumentasi maupun bahan evaluasi performa model.

        Pengguna juga dapat mengunduh data riwayat
        dalam format CSV maupun Microsoft Excel.

        </p>

        </div>

        """, unsafe_allow_html=True)
