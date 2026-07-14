import mysql.connector
import pandas as pd
import streamlit as st


# ==========================================================
# KONEKSI DATABASE
# ==========================================================
def get_connection():

    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="",
        database="db_cyberbullying"
    )

    return conn


# ==========================================================
# SIMPAN HASIL PREDIKSI
# ==========================================================
def simpan_data(
        teks_input,
        hasil_preprocessing,
        label_angka,
        label_teks,
        metode):

    try:

        conn = get_connection()
        cursor = conn.cursor()

        sql = """
        INSERT INTO riwayat_deteksi
        (
            teks_input,
            hasil_preprocessing,
            label_angka,
            label_teks,
            metode
        )
        VALUES (%s,%s,%s,%s,%s)
        """

        values = (
            teks_input,
            hasil_preprocessing,
            label_angka,
            label_teks,
            metode
        )

        cursor.execute(sql, values)

        conn.commit()

        cursor.close()
        conn.close()

    except Exception as e:

        st.error(
            f"Gagal menyimpan ke Database : {e}"
        )


# ==========================================================
# AMBIL RIWAYAT PREDIKSI
# ==========================================================
def ambil_riwayat():

    try:

        conn = get_connection()

        query = """
        SELECT *
        FROM riwayat_deteksi
        ORDER BY waktu DESC
        """

        cursor = conn.cursor()

        cursor.execute(query)

        data = cursor.fetchall()

        kolom = [
            "id",
            "teks_input",
            "hasil_preprocessing",
            "label_angka",
            "label_teks",
            "metode",
            "waktu"
        ]

        df = pd.DataFrame(
            data,
            columns=kolom
        )

        cursor.close()
        conn.close()

        return df

    except Exception:

        return pd.DataFrame()


# ==========================================================
# HAPUS RIWAYAT
# ==========================================================
def hapus_riwayat():

    try:

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM riwayat_deteksi"
        )

        conn.commit()

        cursor.close()
        conn.close()

    except Exception as e:

        st.error(
            f"Gagal menghapus data : {e}"
        )