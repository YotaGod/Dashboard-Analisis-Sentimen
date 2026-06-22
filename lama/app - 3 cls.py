import streamlit as st
import json
import numpy as np
import keras
import re
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.corpus import stopwords

# ===========================================================================
# ⚙️ KONFIGURASI HALAMAN & CUSTOM CSS
# ===========================================================================

st.set_page_config(
    page_title="📊 Analisis Sentimen Tangerang Live",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang lebih menarik
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Primary colors */
    :root {
        --primary: #4F8BF9;
        --success: #2ECC71;
        --warning: #F39C12;
        --danger: #E74C3C;
    }
    
    /* Card styling */
    .result-card {
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid var(--primary);
    }
    .result-card.positive { 
        background: linear-gradient(135deg, #2ECC71, #27AE60); 
        color: white;
        border-left-color: var(--success);
    }
    .result-card.negative { 
        background: linear-gradient(135deg, #E74C3C, #C0392B); 
        color: white;
        border-left-color: var(--danger);
    }
    .result-card.neutral { 
        background: linear-gradient(135deg, #F39C12, #D35400); 
        color: white;
        border-left-color: var(--warning);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        font-size: 1rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# ✅ SETUP NLTK (LOGIKA TIDAK DIUBAH)
# ===========================================================================

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ===========================================================================
# 🧹 FUNGSI PREPROCESSING (LOGIKA TIDAK DIUBAH)
# ===========================================================================

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[-+]?[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@st.cache_resource
def load_slang_dictionary():
    try:
        slang_df = pd.read_csv('slang.csv', sep=',', header=None, names=['slang', 'formal'], skiprows=1)
        return dict(zip(slang_df['slang'].str.strip(), slang_df['formal'].str.strip()))
    except: 
        return {}

def normalize_slang(text, slang_dict):
    if not text: return ""
    return ' '.join([slang_dict.get(word, word) for word in text.split()])

@st.cache_resource
def get_final_stop_words():
    factory = StopWordRemoverFactory()
    sw_list = factory.get_stop_words()
    additional = ['aja', 'saja', 'banget', 'kok', 'nih', 'loh', 'deh', 'dong', 'yg', 'nya', 'sih', 'aplikasi']
    all_sw = set(sw_list + additional + stopwords.words('english'))
    negation_basic = ['tidak', 'gak', 'nggak', 'kurang', 'bukan', 'jangan', 'belum']
    for w in negation_basic:
        if w in all_sw: all_sw.remove(w)
    return all_sw

def handle_negation(text):
    negation_words = ['tidak', 'gak', 'nggak', 'kurang', 'bukan', 'jangan', 'belom', 'belum']
    words = text.split()
    new_words = []
    i = 0
    while i < len(words):
        if words[i] in negation_words and i + 1 < len(words):
            new_words.append(f"{words[i]}_{words[i+1]}")
            i += 2
        else:
            new_words.append(words[i])
            i += 1
    return " ".join(new_words)

def remove_stopwords(text):
    if not text: return ""
    stop_words_set = get_final_stop_words()
    words = text.split()
    stopped_words = [w for w in words if w not in stop_words_set or '_' in w]
    return " ".join(stopped_words)

# ===========================================================================
# 📦 LOAD ASSETS (LOGIKA TIDAK DIUBAH)
# ===========================================================================

@st.cache_resource
def load_assets():
    try:
        from tf_keras.preprocessing.text import tokenizer_from_json
        from tf_keras.preprocessing.sequence import pad_sequences
        
        with open('tokenizer (9).json', 'r', encoding='utf-8') as f:
            tokenizer = tokenizer_from_json(json.load(f))
        
        model = keras.models.load_model('model_tangerang_live2 (7).keras', compile=False)
        slang_dict = load_slang_dictionary()
        
        return model, tokenizer, slang_dict
    except Exception as e:
        st.error(f"❌ Error Load Assets: {e}")
        return None, None, None

model, tokenizer, slang_dict = load_assets()

# ===========================================================================
# 🎨 SIDEBAR - INFORMASI & PENGATURAN (UI IMPROVEMENT)
# ===========================================================================

with st.sidebar:
    st.title("🎯 Panel Kontrol")
    
    # Status Model dengan visual indicator
    st.subheader("📦 Status Sistem")
    if model:
        st.success("✅ Model Aktif")
        st.info("• Input: Maksimal 50 token\n• Output: 3 kelas sentimen")
    else:
        st.error("❌ Model Tidak Dimuat")
    
    st.divider()
    
    # Threshold Setting dengan slider interaktif
    st.subheader("⚙️ Pengaturan Analisis")
    threshold = st.slider(
        "📊 Threshold Keyakinan",
        min_value=0.3, max_value=0.9, value=0.6, step=0.1,
        help="Nilai minimal confidence untuk klasifikasi Positif/Negatif. Di bawah threshold akan dikategorikan sebagai Netral."
    )
    
    st.divider()
    
    # Info Kategori dengan tabel visual
    st.subheader("📋 Kategori Sentimen")
    st.markdown("""
    | Emoji | Kategori | Ciri-ciri |
    |-------|----------|-----------|
    | 😊 | **Positif** | Kepuasan, pujian, rekomendasi |
    | 😐 | **Netral** | Faktual, informatif, ragu-ragu |
    | 😞 | **Negatif** | Keluhan, kritik, kekecewaan |
    """)
    
    st.divider()
    
    # Contoh Input untuk panduan user
    st.subheader("💡 Contoh Ulasan")
    with st.expander("📝 Lihat Contoh", expanded=False):
        st.code('Aplikasi ini sangat membantu dan mudah digunakan!', language='text')
        st.code('Sering error dan force close, sangat mengecewakan.', language='text')
        st.code('Saya baru mencoba fitur pengajuan KK baru.', language='text')
    
    st.divider()
    
    # Footer Sidebar
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.85em; padding: 10px;'>
        <strong>Bi-LSTM + FastText</strong><br>
        Skripsi 2026 🎓<br>
        <em>Universitas [Nama]</em>
    </div>
    """, unsafe_allow_html=True)

# ===========================================================================
# 🖼️ MAIN CONTENT - HEADER & INPUT (UI IMPROVEMENT)
# ===========================================================================

# Header Utama dengan gradient background
st.title("🎯 Analisis Sentimen Tangerang Live")
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 25px; border-radius: 15px; color: white; margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>
    <h3 style='margin: 0 0 10px 0;'>📱 Ulasan Pengguna Aplikasi</h3>
    <p style='margin: 0; opacity: 0.95; font-size: 1.05em;'>
        Analisis sentimen otomatis menggunakan <strong>Deep Learning</strong> untuk memahami feedback pengguna secara akurat
    </p>
</div>
""", unsafe_allow_html=True)

# Input Section dengan card styling
st.subheader("✍️ Masukkan Ulasan Anda")
user_input = st.text_area(
    label="Tulis ulasan di bawah ini:",
    placeholder="Contoh: Aplikasi ini sangat membantu saya mengurus administrasi tanpa harus antre panjang...",
    height=150,
    label_visibility="collapsed"
)

# Action Buttons dengan layout yang lebih baik
col_btn1, col_btn2, col_btn3 = st.columns([3, 1, 1])
with col_btn1:
    analyze_btn = st.button("🔍 Analisis Sekarang", type="primary", use_container_width=True)
with col_btn2:
    if st.button("🗑️ Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()
with col_btn3:
    if st.button("📋 Copy Contoh", use_container_width=True, disabled=True):
        st.toast("Contoh ulasan tersedia di sidebar! ✨", icon="✅")

st.divider()

# ===========================================================================
# 📈 HASIL ANALISIS (UI IMPROVEMENT - LOGIKA PREDIKSI TIDAK DIUBAH)
# ===========================================================================

if analyze_btn and user_input and model:
    with st.spinner('🔄 Sedang menganalisis ulasan Anda...'):
        try:
            from tf_keras.preprocessing.sequence import pad_sequences
            
            # ========================================
            # ⚠️ LOGIKA PREDIKSI - TIDAK DIUBAH SAMA SEKALI
            # ========================================
            
            # 1. Preprocessing Urut
            t1 = clean_text(user_input)
            t2 = normalize_slang(t1, slang_dict)
            t3 = handle_negation(t2)
            text_final = remove_stopwords(t3)
            
            # 2. Prediksi
            sequences = tokenizer.texts_to_sequences([text_final])
            padded = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
            prediction = model.predict(padded, verbose=0)[0]
            
            # 3. Thresholding
            threshold = 0.6
            max_prob = np.max(prediction)
            confidence = max_prob * 100
            labels = ['Negatif', 'Netral', 'Positif']
            
            if max_prob < threshold:
                final_label = "Netral"
                class_idx = 1
                is_uncertain = True
            else:
                class_idx = np.argmax(prediction)
                final_label = labels[class_idx]
                is_uncertain = False
            
            # ========================================
            # ✅ TAMPILAN HASIL - UI IMPROVEMENT
            # ========================================
            
            # Result Header
            st.subheader("📈 Hasil Analisis")
            
            # Result Card dengan warna dinamis berdasarkan hasil
            if final_label == 'Positif':
                card_class = "positive"
                icon = "😊"
                msg = "Ulasan ini mencerminkan **kepuasan** pengguna terhadap aplikasi."
            elif final_label == 'Negatif':
                card_class = "negative"
                icon = "😞"
                msg = "Ulasan ini menunjukkan **ketidakpuasan** atau masalah yang dialami pengguna."
            else:
                card_class = "neutral"
                icon = "😐"
                msg = "Ulasan ini bersifat **netral** atau informatif tanpa sentimen yang kuat."
            
            # Display result card dengan custom CSS
            st.markdown(f"""
            <div class='result-card {card_class}'>
                <h2 style='margin: 0; font-size: 2.2em; font-weight: bold;'>{icon} {final_label}</h2>
                <p style='margin: 12px 0 0 0; opacity: 0.95; font-size: 1.1em;'>{msg}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics Row dengan 3 kolom
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("🎯 Keyakinan", f"{confidence:.1f}%")
            with col_m2:
                status = "⚠️ Rendah" if confidence < 60 else "✅ Baik" if confidence < 85 else "🌟 Tinggi"
                st.metric("📊 Kualitas", status)
            with col_m3:
                st.metric("🔤 Panjang Teks", f"{len(user_input.split())} kata")
            
            # Warning jika confidence di bawah threshold
            if is_uncertain:
                st.warning(f"⚠️ Model kurang yakin (confidence {confidence:.1f}% < threshold {threshold*100:.0f}%). Hasil dikategorikan sebagai **Netral**.")
            
            st.divider()
            
            # Visualization Section dengan 2 kolom
            col_chart, col_detail = st.columns([1, 1])
            
            with col_chart:
                st.subheader("🥧 Distribusi Probabilitas")
                
                # Pie Chart dengan styling yang lebih baik
                fig, ax = plt.subplots(figsize=(5, 5))
                colors = ['#E74C3C', '#F39C12', '#2ECC71']  # Merah, Orange, Hijau
                explode = [0.1 if i == class_idx else 0.05 for i in range(3)]
                
                wedges, texts, autotexts = ax.pie(
                    prediction * 100,
                    labels=labels,
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    explode=explode,
                    textprops={'fontsize': 10, 'weight': 'bold', 'color': 'white'}
                )
                
                ax.axis('equal')
                ax.set_title('Probabilitas per Kelas', fontsize=12, weight='bold', pad=20)
                st.pyplot(fig, use_container_width=True)
            
            with col_detail:
                st.subheader("📊 Detail Probabilitas")
                
                # Progress bars untuk setiap kelas dengan visual feedback
                for i, label in enumerate(labels):
                    prob = prediction[i] * 100
                    is_max = (i == class_idx)
                    
                    # Icon dan warna berdasarkan kelas
                    if label == 'Positif':
                        icon, color = '😊', 'green'
                    elif label == 'Negatif':
                        icon, color = '😞', 'red'
                    else:
                        icon, color = '😐', 'orange'
                    
                    # Highlight kelas dengan probabilitas tertinggi
                    if is_max:
                        st.markdown(f"**{icon} {label}** *(Terpilih)*")
                    else:
                        st.markdown(f"{icon} {label}")
                    
                    st.progress(min(prob / 100, 1.0))
                    st.caption(f"{prob:.2f}% {'🎯' if is_max else ''}")
                    st.markdown("---")
            
            st.divider()
            
            # Preprocessing Details (Collapsible Expander)
            with st.expander("🔍 Lihat Detail Preprocessing", expanded=False):
                st.markdown("##### 📝 Pipeline Preprocessing")
                
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.markdown("**📥 Input Asli:**")
                    st.code(user_input, language='text')
                    
                    st.markdown("**1️⃣ Setelah Cleaning:**")
                    st.code(t1, language='text')
                    
                    st.markdown("**2️⃣ Setelah Normalisasi Slang:**")
                    st.code(t2 if t2 != t1 else "*(tidak ada perubahan)*", language='text')
                
                with col_p2:
                    st.markdown("**3️⃣ Setelah Handling Negasi:**")
                    st.code(t3 if t3 != t2 else "*(tidak ada perubahan)*", language='text')
                    
                    st.markdown("**4️⃣ Final (Stopword Removal):**")
                    st.code(text_final if text_final != t3 else "*(tidak ada perubahan)*", language='text')
                
                st.info("💡 Preprocessing memastikan teks input konsisten dengan data training model.")
            
            # Action Buttons Setelah Hasil
            st.divider()
            col_action1, col_action2 = st.columns(2)
            with col_action1:
                if st.button("🔄 Analisis Ulang", use_container_width=True):
                    st.rerun()
            with col_action2:
                if st.button("📥 Export Hasil", use_container_width=True, disabled=True):
                    st.toast("Fitur export akan tersedia soon! 🚧", icon="🔜")
            
        except Exception as e:
            st.error(f"⚠️ Terjadi kesalahan: {type(e).__name__}")
            st.exception(e)
            st.info("💡 Pastikan semua file (model, tokenizer, slang.csv) ada di folder yang sama.")

elif analyze_btn and not model:
    st.error("❌ Model belum dimuat. Silakan periksa file model dan restart aplikasi.")

elif not user_input:
    # Empty State - Tampilan awal yang menarik
    st.info("👆 **Silakan tulis ulasan di atas** dan klik **Analisis Sekarang** untuk memulai.")
    
    # Feature Highlights
    st.markdown("##### ✨ Fitur Aplikasi")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown("""
        **🧠 Deep Learning**  
        Model Bi-LSTM dengan FastText embedding untuk akurasi tinggi
        """)
    with col_f2:
        st.markdown("""
        **🗣️ Bahasa Indonesia**  
        Preprocessing khusus: slang normalization & stopword removal
        """)
    with col_f3:
        st.markdown("""
        **⚡ Real-time**  
        Analisis instan dengan visualisasi yang mudah dipahami
        """)

# ===========================================================================
# 🦶 FOOTER
# ===========================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 25px; background: #f8f9fa; border-radius: 10px;'>
    <p style='margin: 0; font-size: 1.1em;'><strong>🎓 Skripsi - Analisis Sentimen Ulasan Pengguna</strong></p>
    <p style='margin: 8px 0 0 0; font-size: 0.95em;'>
        Metode: <strong>Bidirectional LSTM + FastText Embedding</strong> | Universitas [Nama] | 2026
    </p>
</div>
""", unsafe_allow_html=True)