import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Aplikasi Prediksi Kelulusan Mahasiswa")
st.markdown("Sistem ini memprediksi kelulusan tepat waktu mahasiswa menggunakan algoritma Naive Bayes.")

@st.cache_data
def load_data():
    data = pd.read_csv('dataset_kelulusan.csv')
    le = LabelEncoder()
    data['jenis_kelamin'] = le.fit_transform(data['jenis_kelamin'])
    data = pd.get_dummies(data, columns=['jurusan'], prefix='jur')
    return data

data = load_data()

# Sidebar
st.sidebar.header("Pengaturan Model")

available_features = data.columns.drop(['nim', 'lulus_tepat_waktu']).tolist()

default_features = [col for col in available_features if col in [
    'ipk', 'jumlah_sks', 'semester', 'ikut_organisasi', 'beasiswa', 'jenis_kelamin'
]] + [col for col in available_features if col.startswith('jur_') and 'Teknik Informatika' in col]

selected_features = st.sidebar.multiselect(
    "Pilih Fitur untuk Model",
    options=available_features,
    default=default_features
)

test_size = st.sidebar.slider("Persentase Data Testing", 10, 40, 20)

def train_model(data, features, test_size):
    X = data[features]
    y = data['lulus_tepat_waktu']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model, X_test, y_test, scaler

if st.sidebar.button("Train Model"):
    model, X_test, y_test, scaler = train_model(data, selected_features, test_size)
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    acc = accuracy_score(y_test, model.predict(X_test))
    st.session_state.accuracy = acc
    st.sidebar.success(f"Akurasi Model: {acc*100:.2f}%")

tab1, tab2, tab3 = st.tabs(["Prediksi", "Evaluasi Model", "Data"])

with tab1:
    st.header("🔮 Prediksi Kelulusan")
    if 'model' not in st.session_state:
        st.warning("Silakan train model terlebih dahulu di sidebar")
    else:
        col1, col2 = st.columns(2)
        with col1:
            ipk = st.slider("IPK", 2.0, 4.0, 3.0)
            sks = st.number_input("Total SKS", 100, 150, 120)
            semester = st.number_input("Semester", 1, 14, 6)
        with col2:
            gender = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            jurusan = st.selectbox("Jurusan", ["Teknik Informatika", "Manajemen", "Akuntansi", "Hukum", "Kedokteran"])
            organisasi = st.checkbox("Aktif Organisasi")
            beasiswa = st.checkbox("Penerima Beasiswa")

        if st.button("Prediksi"):
            input_data = {
                'ipk': ipk,
                'jumlah_sks': sks,
                'semester': semester,
                'jenis_kelamin': 1 if gender == "Laki-laki" else 0,
                'ikut_organisasi': 1 if organisasi else 0,
                'beasiswa': 1 if beasiswa else 0
            }
            for jur in ["Teknik Informatika", "Manajemen", "Akuntansi", "Hukum", "Kedokteran"]:
                input_data[f'jur_{jur}'] = 1 if jurusan == jur else 0

            input_df = pd.DataFrame([input_data])[selected_features]
            input_scaled = st.session_state.scaler.transform(input_df)
            prediction = st.session_state.model.predict(input_scaled)[0]
            proba = st.session_state.model.predict_proba(input_scaled)[0]

            if prediction == 1:
                st.success(f"✅ Prediksi: LULUS TEPAT WAKTU (Probabilitas: {proba[1]*100:.2f}%)")
                st.balloons()
                st.markdown("**Rekomendasi:**\n- Pertahankan IPK\n- Selesaikan skripsi\n- Bimbingan akademik")
            else:
                st.error(f"⏳ Prediksi: BERISIKO TIDAK LULUS TEPAT WAKTU (Probabilitas: {proba[0]*100:.2f}%)")
                st.markdown("**Rekomendasi:**\n- Tingkatkan IPK\n- Ikuti program percepatan\n- Konsultasi rutin")

            fig, ax = plt.subplots()
            ax.bar(['Tidak Lulus', 'Lulus'], proba*100, color=['red', 'green'])
            ax.set_ylabel("Probabilitas (%)")
            ax.set_title("Distribusi Probabilitas Prediksi")
            st.pyplot(fig)

with tab2:
    st.header("📊 Evaluasi Model")
    if 'model' not in st.session_state:
        st.warning("Model belum ditraining")
    else:
        st.metric("Akurasi Model", f"{st.session_state.accuracy*100:.2f}%")
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Prediksi Tidak Lulus', 'Prediksi Lulus'],
                    yticklabels=['Aktual Tidak Lulus', 'Aktual Lulus'])
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        st.subheader("Parameter Model Naive Bayes")
        st.write("Mean per kelas:")
        st.write(st.session_state.model.theta_)
        st.write("Variance per kelas:")
        st.write(st.session_state.model.var_)

with tab3:
    st.header("📁 Dataset Kelulusan Mahasiswa")
    st.write("Dataset berikut berisi 150 record data mahasiswa dengan berbagai fitur.")
    st.dataframe(data, height=400)
    st.subheader("Distribusi Fitur")
    selected_col = st.selectbox("Pilih fitur untuk divisualisasikan", ['ipk', 'jumlah_sks', 'semester'])
    fig, ax = plt.subplots()
    sns.histplot(data=data, x=selected_col, hue='lulus_tepat_waktu', kde=True, bins=20, ax=ax)
    ax.set_title(f"Distribusi {selected_col} berdasarkan Kelulusan")
    st.pyplot(fig)

with st.expander("ℹ️ Tentang Aplikasi"):
    st.write("Aplikasi ini dibuat menggunakan Streamlit dan model Naive Bayes untuk memprediksi kelulusan mahasiswa.")
