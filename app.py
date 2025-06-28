import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# Judul aplikasi
st.title("🎓 Aplikasi Prediksi Kelulusan Mahasiswa")
st.markdown("""
Sistem ini memprediksi kelulusan tepat waktu mahasiswa menggunakan algoritma Naive Bayes.
""")

# =====================================
# 1. Load dan Preprocess Data
# =====================================
@st.cache_data
def load_data():
    data = pd.read_csv('dataset_kelulusan.csv')
    
    # Encoding variabel kategorikal
    le = LabelEncoder()
    data['jenis_kelamin'] = le.fit_transform(data['jenis_kelamin'])
    
    # One-Hot Encoding untuk jurusan
    data = pd.get_dummies(data, columns=['jurusan'], prefix='jur')
    
    return data

data = load_data()

# =====================================
# 2. Sidebar untuk Konfigurasi Model
# =====================================
st.sidebar.header("Pengaturan Model")
test_size = st.sidebar.slider("Persentase Data Testing", 10, 40, 20)
selected_features = st.sidebar.multiselect(
    "Pilih Fitur untuk Model",
    options=data.columns.drop(['nim', 'lulus_tepat_waktu']),
    default=['ipk', 'jumlah_sks', 'semester', 'jur_Teknik Informatika']
)

# =====================================
# 3. Training Model
# =====================================
def train_model(data, features, test_size):
    X = data[features]
    y = data['lulus_tepat_waktu']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size/100, 
        random_state=42
    )
    
    # Normalisasi
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test, scaler

if st.sidebar.button("Train Model"):
    model, X_test, y_test, scaler = train_model(data, selected_features, test_size)
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    
    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.session_state.accuracy = acc
    
    st.sidebar.success(f"Akurasi Model: {acc*100:.2f}%")

# =====================================
# 4. Tampilan Utama Aplikasi
# =====================================
tab1, tab2, tab3 = st.tabs(["Prediksi", "Evaluasi Model", "Data"])

with tab1:
    st.header("🔮 Prediksi Kelulusan")
    
    if 'model' not in st.session_state:
        st.warning("Silakan train model terlebih dahulu di sidebar")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Akademik")
            ipk = st.slider("IPK", 2.0, 4.0, 3.0)
            sks = st.number_input("Total SKS", 100, 150, 120)
            semester = st.number_input("Semester", 1, 14, 6)
            
        with col2:
            st.subheader("Data Lainnya")
            gender = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            jurusan = st.selectbox("Jurusan", [
                "Teknik Informatika",
                "Manajemen",
                "Akuntansi",
                "Hukum",
                "Kedokteran"
            ])
            organisasi = st.checkbox("Aktif Organisasi")
            beasiswa = st.checkbox("Penerima Beasiswa")
        
        if st.button("Prediksi"):
            # Prepare input data
            input_data = {
                'ipk': ipk,
                'jumlah_sks': sks,
                'semester': semester,
                'jenis_kelamin': 1 if gender == "Laki-laki" else 0,
                'ikut_organisasi': 1 if organisasi else 0,
                'beasiswa': 1 if beasiswa else 0,
                'jur_Teknik Informatika': 1 if jurusan == "Teknik Informatika" else 0,
                'jur_Manajemen': 1 if jurusan == "Manajemen" else 0,
                'jur_Akuntansi': 1 if jurusan == "Akuntansi" else 0,
                'jur_Hukum': 1 if jurusan == "Hukum" else 0,
                'jur_Kedokteran': 1 if jurusan == "Kedokteran" else 0
            }
            
            # Convert to DataFrame and select features
            input_df = pd.DataFrame([input_data])[selected_features]
            
            # Scale features
            input_scaled = st.session_state.scaler.transform(input_df)
            
            # Predict
            prediction = st.session_state.model.predict(input_scaled)[0]
            proba = st.session_state.model.predict_proba(input_scaled)[0]
            
            # Display results
            if prediction == 1:
                st.success(f"✅ Prediksi: LULUS TEPAT WAKTU (Probabilitas: {proba[1]*100:.2f}%)")
                st.balloons()
                
                st.markdown("""
                **Rekomendasi:**
                - Pertahankan IPK di atas 3.0
                - Selesaikan skripsi sebelum semester 8
                - Ikuti bimbingan akademik rutin
                """)
            else:
                st.error(f"⏳ Prediksi: BERISIKO TIDAK LULUS TEPAT WAKTU (Probabilitas: {proba[0]*100:.2f}%)")
                
                st.markdown("""
                **Rekomendasi:**
                - Tingkatkan IPK minimal 0.5 poin
                - Ikuti program percepatan studi
                - Konsultasi dengan dosen wali
                - Kurangi aktivitas non-akademik
                """)
            
            # Probability plot
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
        
        # Confusion matrix
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
        
        # Parameter Naive Bayes
        st.subheader("Parameter Model Naive Bayes")
        st.write("Mean per kelas:")
        st.write(st.session_state.model.theta_)
        st.write("Variance per kelas:")
        st.write(st.session_state.model.var_)


with tab3:
    st.header("📁 Dataset Kelulusan Mahasiswa")
    st.write("""
    Dataset berikut berisi 150 record data mahasiswa dengan fitur-fitur:
    - IPK (2.0-4.0)
    - Jumlah SKS (100-150)
    - Semester (1-14)
    - Status organisasi dan beasiswa
    - Jurusan
    """)
    
    st.dataframe(data, height=400)
    
    # Visualisasi distribusi fitur
    st.subheader("Distribusi Fitur")
    selected_col = st.selectbox("Pilih fitur untuk divisualisasikan", 
                              ['ipk', 'jumlah_sks', 'semester'])
    
    fig, ax = plt.subplots()
    sns.histplot(data=data, x=selected_col, hue='lulus_tepat_waktu', 
                kde=True, bins=20, ax=ax)
    ax.set_title(f"Distribusi {selected_col} berdasarkan Kelulusan")
    st.pyplot(fig)

# =====================================
# Informasi Tambahan
# =====================================
with st.expander("ℹ️ Tentang Aplikasi"):
    st.write("""
    Aplikasi ini dibuat menggunakan Streamlit dan model Naive Bayes untuk 
    memprediksi kelulusan mahasiswa tepat waktu berdasarkan data akademik 
    dan demografi.
    
    Dataset dibuat secara sintetis untuk tujuan demo.
    """)

