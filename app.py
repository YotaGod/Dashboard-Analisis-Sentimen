import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import io
import json
import numpy as np
import keras
import re
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover
from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
from nltk.corpus import stopwords

# ===========================================================================
# ⚙️ KONFIGURASI HALAMAN & CUSTOM CSS
# ===========================================================================

st.set_page_config(
    page_title="Analisis Sentimen - Tangerang Live",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS dengan kontras yang lebih baik dan aksesibel untuk color blind friendly
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Color blind friendly palette - Dark mode with Blue & Orange */
    :root {
        --primary: #0066CC;
        --primary-dark: #003366;
        --success: #FF9900;
        --success-dark: #CC7700;
        --danger: #003366;
        --danger-dark: #001A33;
        --warning: #FF9900;
        --text-dark: #FFFFFF;
        --text-medium: #E0E0E0;
        --text-light: #AAAAAA;
        --bg-light: #2A2A2A;
        --bg-white: #1F1F1F;
        --border-color: #444444;
    }
    
    /* Main container styling */
    .main {
        background-color: #1F1F1F;
    }
    
    /* Header styling dengan kontras tinggi - Blue background */
    .main-header {
        background: linear-gradient(135deg, #0066CC 0%, #003366 100%);
        padding: 40px;
        border-radius: 16px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px rgba(0, 102, 204, 0.4);
        border: 3px solid #001A33;
    }
    
    .main-header h1 {
        margin: 0 0 12px 0;
        font-size: 2.2em;
        font-weight: 700;
        color: #FFFFFF !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
    }
    
    .main-header p {
        margin: 0;
        font-size: 1.1em;
        color: #FFFFFF !important;
        font-weight: 400;
        line-height: 1.6;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Section titles dengan kontras tinggi */
    .section-title {
        color: #FFFFFF;
        font-size: 1.5em;
        font-weight: 700;
        margin: 30px 0 20px 0;
        padding-bottom: 12px;
        border-bottom: 4px solid #FF9900;
    }
    
    /* Info boxes dengan background dan border yang jelas */
    .info-box {
        background: #2A2A2A;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 5px solid #0066CC;
        border-right: 2px solid #444444;
        border-top: 2px solid #444444;
        border-bottom: 2px solid #444444;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        color: #FFFFFF;
    }
    
    .info-box strong {
        color: #FF9900;
        font-weight: 700;
    }
    
    /* Text area dengan border dan kontras yang jelas */
    .stTextArea textarea {
        border-radius: 12px;
        border: 3px solid #0066CC;
        font-size: 1rem;
        padding: 16px;
        color: #FFFFFF;
        background-color: #2A2A2A;
        min-height: 150px;
    }
    
    .stTextArea textarea:focus {
        border-color: #FF9900;
        box-shadow: 0 0 0 4px rgba(255, 153, 0, 0.2);
        outline: none;
    }
    
    .stTextArea textarea::placeholder {
        color: #777777;
    }
    
    /* Button styling dengan kontras tinggi - Blue background */
    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        padding: 14px 32px;
        border: 2px solid #003366;
        box-shadow: 0 4px 12px rgba(0, 51, 102, 0.3);
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stButton > button[type="primary"] {
        background: linear-gradient(135deg, #0066CC 0%, #003366 100%);
        color: #FFFFFF !important;
    }
    
    .stButton > button[type="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 102, 204, 0.5);
        background: linear-gradient(135deg, #003366 0%, #001A33 100%);
    }
    
    .stButton > button[type="secondary"] {
        background: #2A2A2A;
        color: #FFFFFF !important;
        border: 2px solid #0066CC;
    }
    
    .stButton > button[type="secondary"]:hover {
        border-color: #FF9900;
        background: #333333;
        box-shadow: 0 4px 12px rgba(255, 153, 0, 0.2);
    }
    
    /* Result cards dengan kontras tinggi - Blue & Orange */
    .result-card {
        padding: 28px;
        border-radius: 16px;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        border: 3px solid;
    }
    
    .result-card.positive { 
        background: linear-gradient(135deg, #FF9900 0%, #CC7700 100%);
        color: #000000;
        border-color: #996600;
    }
    
    .result-card.negative { 
        background: linear-gradient(135deg, #003366 0%, #001A33 100%);
        color: #FFFFFF;
        border-color: #000000;
    }
    
    .result-card h2 {
        color: inherit !important;
        margin: 0 0 15px 0;
        font-size: 2.2em;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .result-card p {
        color: inherit !important;
        font-size: 1.1em;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Sidebar dengan kontras yang baik */
    [data-testid="stSidebar"] {
        background: #1F1F1F;
        border-right: 3px solid #0066CC;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #FFFFFF;
    }
    
    /* Status badges dengan kontras tinggi - Blue & Orange */
    .status-badge {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 20px;
        font-size: 0.95em;
        font-weight: 700;
        margin: 8px 0;
        border: 2px solid;
    }
    
    .status-active {
        background: #FF9900;
        color: #000000;
        border-color: #CC7700;
    }
    
    .status-inactive {
        background: #E6E6E6;
        color: #333333;
        border-color: #666666;
    }
    
    /* Category cards dengan border dan kontras jelas */
    .category-card {
        background: #2A2A2A;
        padding: 18px;
        border-radius: 12px;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        border-left: 5px solid;
        border-top: 3px solid;
        border-right: 1px solid #444444;
        border-bottom: 1px solid #444444;
        color: #FFFFFF;
    }
    
    .category-card strong {
        color: #FFFFFF;
        font-size: 1.1em;
        font-weight: 700;
    }
    
    .category-card small {
        color: #AAAAAA;
        display: block;
        margin-top: 6px;
    }
    
    .category-positive {
        border-left-color: #FF9900;
        border-top-color: #FF9900;
        background: #3A2A1A;
    }
    
    .category-negative {
        border-left-color: #0066CC;
        border-top-color: #0066CC;
        background: #1A2A3A;
    }
    
    /* Metric cards dengan background dan shadow */
    [data-testid="stMetric"] {
        background: #2A2A2A;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        border: 2px solid #444444;
    }
    
    [data-testid="stMetricLabel"] {
        color: #AAAAAA;
        font-weight: 600;
    }
    
    [data-testid="stMetricValue"] {
        color: #FF9900;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Expander dengan border yang jelas */
    .streamlit-expanderHeader {
        background-color: #2A2A2A;
        border-radius: 10px;
        font-weight: 700;
        border: 2px solid #0066CC;
        color: #FFFFFF;
        padding: 12px 16px;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #FF9900;
        background-color: #333333;
    }
    
    /* Empty state dengan kontras yang baik */
    .empty-state {
        text-align: center;
        padding: 60px 30px;
        background: #2A2A2A;
        border-radius: 16px;
        margin: 30px 0;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
        border: 3px solid #0066CC;
    }
    
    .empty-state h3 {
        color: #FFFFFF;
        margin: 0 0 15px 0;
        font-size: 1.8em;
        font-weight: 700;
    }
    
    .empty-state p {
        color: #AAAAAA;
        font-size: 1.1em;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Feature boxes dengan border dan kontras */
    .feature-box {
        background: #2A2A2A;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
        border-left: 5px solid;
        border-top: 3px solid;
        height: 100%;
        border-right: 1px solid #444444;
        border-bottom: 1px solid #444444;
    }
    
    .feature-box h4 {
        margin: 0 0 12px 0;
        font-size: 1.2em;
        font-weight: 700;
        color: #FFFFFF;
    }
    
    .feature-box p {
        margin: 0;
        color: #AAAAAA;
        line-height: 1.6;
    }
    
    /* Footer dengan background dan kontras */
    .app-footer {
        text-align: center;
        padding: 30px;
        margin-top: 50px;
        color: #AAAAAA;
        background: #2A2A2A;
        border-radius: 12px;
        box-shadow: 0 -2px 12px rgba(0, 0, 0, 0.3);
        border-top: 4px solid #0066CC;
    }
    
    .app-footer h3 {
        color: #FFFFFF;
        margin: 0 0 10px 0;
        font-weight: 700;
    }
    
    /* Sidebar dengan kontras yang baik */
    [data-testid="stSidebar"] {
        background: #1F1F1F;
        border-right: 3px solid #0066CC;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #FFFFFF;
    }
    
    /* Caption dan text kecil dengan kontras yang cukup */
    .stCaption {
        color: #AAAAAA;
        font-size: 0.9em;
    }
    
    /* Code blocks dengan background yang kontras */
    .stCode {
        background: #2A2A2A;
        border: 2px solid #444444;
        border-radius: 8px;
    }
    
    /* Alert boxes dengan kontras tinggi */
    .stAlert {
        border-radius: 10px;
        border: 2px solid;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# ✅ SETUP NLTK
# ===========================================================================

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ===========================================================================
# 🧹 FUNGSI PREPROCESSING (SAMA DENGAN COLAB)
# ===========================================================================

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[-+]?[0-9]+', '', text)
    
    # Hapus emoji
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@st.cache_resource
def load_slang_dictionary():
    try:
        slang_df = pd.read_csv('slang.csv', sep=';', header=None, names=['slang', 'formal'], skiprows=1)
        slang_df['slang'] = slang_df['slang'].astype(str).str.strip().str.lower()
        slang_df['formal'] = slang_df['formal'].astype(str).str.strip().str.lower()
        return dict(zip(slang_df['slang'], slang_df['formal']))
    except: 
        return {}

def normalize_slang(text, slang_dict):
    if not text: return ""
    words = text.split()
    return ' '.join([slang_dict.get(word, word) for word in words])

@st.cache_resource
def create_stopword_remover():
    factory = StopWordRemoverFactory()
    stopword_list = factory.get_stop_words()
    additional = ['aja','saja','banget','kok','lagi','punya','terus','nih','loh','deh','dong','gitu','gini','udah','tapi','yg','nya','sih','pun','lah','anjir']
    for w in additional:
        if w not in stopword_list:
            stopword_list.append(w)
    for w in stopwords.words("english"):
        if w not in stopword_list:
            stopword_list.append(w)
    # Proteksi kata negasi
    for neg in ['tidak', 'gak', 'nggak', 'kurang', 'bukan', 'jangan', 'belum']:
        if neg in stopword_list:
            stopword_list.remove(neg)
    return StopWordRemover(ArrayDictionary(stopword_list))

def remove_stopwords(text, remover):
    return remover.remove(text) if text else ""

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

# ===========================================================================
# 📦 LOAD ASSETS
# ===========================================================================

@st.cache_resource
def load_assets():
    try:
        # Import dari tf_keras untuk preprocessing (kompatibel dengan model)
        from tf_keras.preprocessing.text import tokenizer_from_json
        from tf_keras.preprocessing.sequence import pad_sequences
        
        # Load tokenizer
        with open('tokenizer (14).json', 'r', encoding='utf-8') as f:
            data = f.read()
        if data.startswith('"') and data.endswith('"'):
            data = json.loads(data)
        tokenizer = tokenizer_from_json(data)
        
        # Load model (binary classification)
        model = keras.models.load_model('model_tangerang_live_biner.keras', compile=False)
        
        # Load resources lain
        slang_dict = load_slang_dictionary()
        stopword_remover = create_stopword_remover()
        
        return model, tokenizer, slang_dict, stopword_remover
        
    except Exception as e:
        st.error(f"❌ Error memuat model: {e}")
        return None, None, None, None

model, tokenizer, slang_dict, stopword_remover = load_assets()

# ===========================================================================
# 🎨 SIDEBAR - Lebih Visible & Ramah
# ===========================================================================

with st.sidebar:
    # Header Sidebar
    st.markdown("""
    <div style='text-align: center; padding: 20px 0; border-bottom: 3px solid #FF9900; margin-bottom: 20px;'>
        <h2 style='margin: 0; color: #FF9900; font-size: 1.8em; font-weight: 700;'>🎯 Panel</h2>
        <p style='margin: 8px 0 0 0; color: #AAAAAA; font-size: 0.95em;'>Kontrol Analisis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧭 Navigasi")
    page = st.radio("Pilih Mode:", ["📝 Analisis Input Teks", "📊 Dashboard Dataset"], label_visibility="collapsed")
    st.markdown("---")
    
    # Status Model - Lebih Jelas
    st.markdown("### 📊 Status Sistem")
    if model:
        st.markdown('<span class="status-badge status-active">✅ Model Aktif</span>', unsafe_allow_html=True)
        st.info("**Klasifikasi:** Binary (2 Kelas)\n\n**Input Maksimal:** 50 kata\n\n**Output:** Positif/Negatif", icon="ℹ️")
    else:
        st.markdown('<span class="status-badge status-inactive">❌ Model Tidak Dimuat</span>', unsafe_allow_html=True)
        st.error("Silakan periksa file model", icon="🚨")
    
    st.markdown("---")
    
    # Info Kategori - Lebih Visual
    st.markdown("### 📋 Kategori Sentimen")
    
    st.markdown("""
    <div class='category-card category-positive'>
        <strong>😊 Positif</strong>
        <small>Kepuasan, pujian, rekomendasi</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='category-card category-negative'>
        <strong>😞 Negatif</strong>
        <small>Keluhan, kritik, kekecewaan</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Contoh Input - Lebih Helpful
    st.markdown("### 💡 Contoh Ulasan")
    with st.expander("Lihat Contoh Ulasan", expanded=False):
        st.markdown("**Ulasan Positif:**")
        st.code("Aplikasi ini sangat membantu dan mudah digunakan!", language='text')
        
        st.markdown("**Ulasan Negatif:**")
        st.code("Sering error dan force close, sangat mengecewakan.", language='text')
    
    st.markdown("---")
    
    # Footer Sidebar
    st.markdown("""
    <div style='text-align: center; padding: 20px 0; color: #AAAAAA; font-size: 0.9em; margin-top: 30px;'>
        <strong style='color: #FF9900;'>Bi-LSTM + FastText</strong><br>
        Skripsi 2026 🎓
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# 📝 HALAMAN ANALISIS SINGLE TEXT
# ===========================================================================

def show_single_analysis():
    # ===========================================================================
    # 🖼️ MAIN CONTENT - Lebih Human & Warm
    # ===========================================================================

    # Header yang Lebih Ramah dengan kontras tinggi
    st.markdown("""
    <div class='main-header'>
        <h1>💬 Analisis Sentimen Tangerang Live</h1>
        <p>Yuk, bagikan pengalaman Anda menggunakan aplikasi Tangerang Live! Kami akan membantu menganalisis sentimen dari ulasan Anda.</p>
    </div>
    """, unsafe_allow_html=True)

    # Input Section - Lebih Mengundang
    st.markdown("<h2 class='section-title'>✍️ Tulis Ulasan Anda</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        <strong>Cara Menggunakan:</strong> Tuliskan pengalaman Anda menggunakan aplikasi Tangerang Live di kolom bawah ini. 
        Semakin detail ulasan Anda, semakin akurat hasil analisisnya!
    </div>
    """, unsafe_allow_html=True)

    user_input = st.text_area(
        label="Tulis ulasan Anda di sini",
        placeholder="Contoh: Aplikasi ini sangat membantu saya mengurus administrasi tanpa harus antre panjang. Fiturnya lengkap dan mudah digunakan...",
        height=180,
        label_visibility="collapsed"
    )

    # Character counter dengan warna yang kontras
    char_count = len(user_input)
    if char_count > 0:
        st.caption(f"📝 {char_count} karakter")

    # Action Buttons - Lebih Menarik dengan kontras tinggi
    col_btn1, col_btn2 = st.columns([4, 1])
    with col_btn1:
        analyze_btn = st.button("🔍 Analisis Ulasan Sekarang", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🗑️ Reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    st.markdown("---")

    # ===========================================================================
    # 📈 HASIL ANALISIS - Lebih Informatif & Ramah
    # ===========================================================================

    if analyze_btn and user_input and model:
        with st.spinner('🔄 Sedang menganalisis ulasan Anda... Mohon tunggu sebentar.'):
            try:
                # Import preprocessing dari tf_keras
                from tf_keras.preprocessing.sequence import pad_sequences
            
                # ========================================
                # PREPROCESSING (SAMA DENGAN COLAB)
                # ========================================
                text_cleaned = clean_text(user_input)
                text_normalized = normalize_slang(text_cleaned, slang_dict)
                text_with_negation = handle_negation(text_normalized)
                text_final = remove_stopwords(text_with_negation, stopword_remover)
            
                # Tokenisasi & Padding
                sequences = tokenizer.texts_to_sequences([text_final])
                padded = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
            
                # ========================================
                # PREDIKSI - BINARY CLASSIFICATION
                # ========================================
                prediction = model.predict(padded, verbose=0)[0]
            
                # Handle output model (sigmoid atau softmax 2 kelas)
                if len(prediction.shape) == 0 or prediction.shape[0] == 1:
                    # Sigmoid output (single value)
                    prob_positive = float(prediction) if len(prediction.shape) == 0 else float(prediction[0])
                    prob_negative = 1 - prob_positive
                else:
                    # Softmax output (2 values)
                    prob_negative = float(prediction[0])
                    prob_positive = float(prediction[1])
            
                # Hitung confidence dan tentukan label
                confidence = max(prob_positive, prob_negative) * 100
            
                if prob_positive > prob_negative:
                    final_label = "Positif"
                    class_idx = 1
                else:
                    final_label = "Negatif"
                    class_idx = 0
            
                # ========================================
                # TAMPILAN HASIL - Lebih Human
                # ========================================
            
                st.markdown("<h2 class='section-title'>📊 Hasil Analisis</h2>", unsafe_allow_html=True)
            
                # Result Card dengan pesan yang lebih personal dan kontras tinggi
                if final_label == 'Positif':
                    card_class = "positive"
                    icon = "😊"
                    msg = "Terima kasih! Ulasan Anda mencerminkan **kepuasan** terhadap aplikasi Tangerang Live. Feedback positif seperti ini sangat berarti bagi pengembangan aplikasi."
                    celebration = "🎉"
                else:
                    card_class = "negative"
                    icon = "😞"
                    msg = "Terima kasih atas masukan Anda. Ulasan ini menunjukkan ada **hal yang perlu diperbaiki** dalam aplikasi. Feedback Anda sangat berharga untuk peningkatan kualitas layanan."
                    celebration = "💭"
            
                st.markdown(f"""
                <div class='result-card {card_class}'>
                    <h2>{icon} Sentimen {final_label} {celebration}</h2>
                    <p>{msg}</p>
                </div>
                """, unsafe_allow_html=True)
            
                # Metrics Row - Lebih Informatif
                st.markdown("### 📈 Detail Analisis")
                col_m1, col_m2, col_m3 = st.columns(3)
            
                with col_m1:
                    st.metric(
                        label="🎯 Tingkat Keyakinan",
                        value=f"{confidence:.1f}%",
                        help="Seberapa yakin model dengan hasil prediksi ini"
                    )
            
                with col_m2:
                    if confidence < 70:
                        status = "⚠️ Perlu Review"
                        status_help = "Keyakinan rendah, sebaiknya dicek manual"
                    elif confidence < 85:
                        status = "✅ Cukup Baik"
                        status_help = "Keyakinan cukup untuk analisis"
                    else:
                        status = "🌟 Sangat Yakin"
                        status_help = "Keyakinan tinggi pada hasil ini"
                
                    st.metric(
                        label="📊 Kualitas Prediksi",
                        value=status,
                        help=status_help
                    )
            
                with col_m3:
                    word_count = len(user_input.split())
                    st.metric(
                        label="🔤 Panjang Ulasan",
                        value=f"{word_count} kata",
                        help="Jumlah kata dalam ulasan Anda"
                    )
            
                st.markdown("---")
            
                # Visualization Section - Lebih Jelas
                col_chart, col_detail = st.columns([1, 1])
            
                with col_chart:
                    st.markdown("### 🥧 Distribusi Probabilitas")
                
                    # Pie Chart untuk 2 kelas dengan warna yang color blind friendly
                    fig, ax = plt.subplots(figsize=(5, 5))
                    colors = ['#003366', '#FF9900']  # Dark Blue dan Orange - color blind friendly
                    explode = [0.1 if i == class_idx else 0.05 for i in range(2)]
                
                    wedges, texts, autotexts = ax.pie(
                        [prob_negative * 100, prob_positive * 100],
                        labels=['Negatif', 'Positif'],
                        colors=colors,
                        autopct='%1.1f%%',
                        startangle=90,
                        explode=explode,
                        textprops={'fontsize': 11, 'weight': 'bold', 'color': 'white'}
                    )
                
                    # Set text color untuk setiap slice - white untuk blue, black untuk orange
                    for i, (wedge, text, autotext) in enumerate(zip(wedges, texts, autotexts)):
                        if i == 0:  # Negative - dark blue
                            text.set_color('white')
                            autotext.set_color('white')
                        else:  # Positive - orange
                            text.set_color('black')
                            autotext.set_color('black')
                
                    ax.axis('equal')
                    ax.set_title('Distribusi Sentimen', fontsize=13, weight='bold', pad=20, color='#111111')
                    st.pyplot(fig, use_container_width=True)
            
                with col_detail:
                    st.markdown("### 📊 Detail Probabilitas")
                
                    # Progress bars untuk 2 kelas dengan penjelasan
                    labels = ['Negatif', 'Positif']
                    probs = [prob_negative, prob_positive]
                    icons = ['😞', '😊']
                    descriptions = [
                        'Kemungkinan ulasan bernada negatif/keluhan',
                        'Kemungkinan ulasan bernada positif/pujian'
                    ]
                
                    for i, (label, prob, icon, desc) in enumerate(zip(labels, probs, icons, descriptions)):
                        prob_percent = prob * 100
                        is_max = (i == class_idx)
                    
                        st.markdown(f"**{icon} {label}** {'*(Hasil Prediksi)*' if is_max else ''}")
                        st.caption(desc)
                        st.progress(min(prob_percent / 100, 1.0))
                        st.markdown(f"**{prob_percent:.2f}%**")
                        st.markdown("---")
            
                st.markdown("---")
            
                # Preprocessing Details - Lebih Terstruktur
                with st.expander("🔍 Lihat Detail Proses Analisis", expanded=False):
                    st.markdown("##### 📝 Langkah-langkah Preprocessing")
                
                    st.info("""
                    **Mengapa preprocessing penting?**  
                    Proses ini memastikan teks Anda diproses dengan cara yang sama seperti data training, 
                    sehingga hasil analisis lebih akurat.
                    """)
                
                    col_p1, col_p2 = st.columns(2)
                
                    with col_p1:
                        st.markdown("**📥 Teks Asli:**")
                        st.code(user_input, language='text')
                    
                        st.markdown("**1️⃣ Cleaning Text:**")
                        st.caption("Menghapus URL, angka, emoji, dan karakter khusus")
                        st.code(text_cleaned, language='text')
                
                    with col_p2:
                        st.markdown("**2️⃣ Normalisasi Slang:**")
                        st.caption("Mengubah kata slang menjadi formal")
                        st.code(text_normalized if text_normalized != text_cleaned else "*(tidak ada perubahan)*", language='text')
                    
                        st.markdown("**3️⃣ Handling Negasi:**")
                        st.caption("Menggabungkan kata negasi dengan kata berikutnya")
                        st.code(text_with_negation if text_with_negation != text_normalized else "*(tidak ada perubahan)*", language='text')
                    
                        st.markdown("**4️⃣ Stopword Removal:**")
                        st.caption("Menghapus kata yang tidak penting")
                        st.code(text_final if text_final != text_with_negation else "*(tidak ada perubahan)*", language='text')
            
                # Action Buttons Setelah Hasil
                st.markdown("---")
                col_action1, col_action2 = st.columns(2)
                with col_action1:
                    if st.button("🔄 Analisis Ulasan Lain", type="secondary", use_container_width=True):
                        st.rerun()
                with col_action2:
                    if st.button("📥 Download Hasil", type="secondary", use_container_width=True, disabled=True):
                        st.toast("Fitur download akan segera hadir! 🚧", icon="🔜")
            
            except Exception as e:
                st.error(f"⚠️ Maaf, terjadi kesalahan: {type(e).__name__}")
                st.exception(e)
                st.info("💡 Pastikan semua file (model, tokenizer, slang.csv) ada di folder yang sama.")

    elif analyze_btn and not model:
        st.error("❌ Model belum dimuat. Silakan periksa file model dan restart aplikasi.")

    elif not user_input:
        # Empty State - Lebih Mengundang dengan kontras tinggi
        st.markdown("""
        <div class='empty-state'>
            <h3>👋 Selamat Datang!</h3>
            <p>
                Silakan tulis ulasan Anda di atas dan klik tombol <strong>"Analisis Ulasan Sekarang"</strong> untuk memulai.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
        # Feature Highlights - Lebih Human dengan kontras yang baik
        st.markdown("### ✨ Mengapa Menggunakan Aplikasi Ini?")
    
        col_f1, col_f2, col_f3 = st.columns(3)
    
        with col_f1:
            st.markdown("""
            <div class='feature-box' style='border-left-color: #0066CC; border-top-color: #0066CC;'>
                <h4 style='color: #0066CC;'>🧠 Teknologi Canggih</h4>
                <p>
                    Menggunakan model <strong>Bi-LSTM dengan FastText</strong> untuk akurasi tinggi dalam memahami sentimen.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
        with col_f2:
            st.markdown("""
            <div class='feature-box' style='border-left-color: #FF9900; border-top-color: #FF9900;'>
                <h4 style='color: #FF9900;'>🗣️ Bahasa Indonesia</h4>
                <p>
                    Dirancang khusus untuk bahasa Indonesia dengan preprocessing yang memahami slang dan negasi.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
        with col_f3:
            st.markdown("""
            <div class='feature-box' style='border-left-color: #003366; border-top-color: #003366;'>
                <h4 style='color: #003366;'>⚡ Cepat & Akurat</h4>
                <p>
                    Hasil analisis instan dengan visualisasi yang mudah dipahami.
                </p>
            </div>
            """, unsafe_allow_html=True)


# ===========================================================================
# 📊 HALAMAN DASHBOARD
# ===========================================================================

@st.cache_data(show_spinner=False)
def predict_batch(texts):
    from tf_keras.preprocessing.sequence import pad_sequences
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    cleaned_texts = []
    total = len(texts)
    for i, text in enumerate(texts):
        if i % 100 == 0 or i == total - 1:
            progress_bar.progress((i + 1) / total)
            status_text.text(f"Memproses teks {i+1}/{total}...")
            
        t_cleaned = clean_text(str(text))
        t_norm = normalize_slang(t_cleaned, slang_dict)
        t_neg = handle_negation(t_norm)
        t_final = remove_stopwords(t_neg, stopword_remover)
        cleaned_texts.append(t_final)
        
    status_text.text("Melakukan tokenisasi dan padding...")
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
    
    status_text.text("Memprediksi sentimen dalam batch...")
    predictions = model.predict(padded, batch_size=128, verbose=0)
    
    labels = []
    for pred in predictions:
        if len(pred.shape) == 0 or pred.shape[0] == 1:
            prob_pos = float(pred) if len(pred.shape) == 0 else float(pred[0])
            prob_neg = 1 - prob_pos
        else:
            prob_neg = float(pred[0])
            prob_pos = float(pred[1])
        labels.append("Positif" if prob_pos > prob_neg else "Negatif")
        
    progress_bar.empty()
    status_text.empty()
    return labels

def show_dashboard():
    st.markdown("<div class='main-header'><h1>📊 Dashboard Analisis Sentimen</h1><p>Visualisasi dan analisis interaktif dari dataset ulasan Tangerang Live.</p></div>", unsafe_allow_html=True)
    
    data_source = st.radio("Pilih Sumber Data:", ["Gunakan Data Tersedia (hasil_review_Tangerang_Live.csv)", "Unggah File CSV Baru"], horizontal=True)
    
    df = None
    if "Gunakan Data Tersedia" in data_source:
        try:
            df = pd.read_csv('hasil_review_Tangerang_Live.csv')
            st.success(f"✅ Berhasil memuat data ({len(df)} baris)")
        except:
            st.error("❌ File hasil_review_Tangerang_Live.csv tidak ditemukan.")
    else:
        uploaded_file = st.file_uploader("Unggah File CSV (harus memiliki kolom 'Review Teks' atau 'review')", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Berhasil memuat file yang diunggah ({len(df)} baris)")
            except Exception as e:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=';')
                    st.success(f"✅ Berhasil memuat file yang diunggah ({len(df)} baris)")
                except Exception as e2:
                    st.error(f"❌ Gagal membaca file CSV. Pastikan format file benar. Error: {e}")
    
    if df is not None:
        text_col = None
        if 'Review Teks' in df.columns:
            text_col = 'Review Teks'
        elif 'review' in df.columns:
            text_col = 'review'
        elif 'text' in df.columns:
            text_col = 'text'
            
        if text_col is None:
            st.error("❌ Dataframe tidak memiliki kolom ulasan ('Review Teks', 'review', atau 'text')")
            return
            
        st.markdown("---")
        
        if 'Sentimen' not in df.columns:
            st.info("🔄 Data belum memiliki label sentimen. Memulai proses prediksi (mungkin memakan waktu untuk dataset besar)...")
            df['Sentimen'] = predict_batch(df[text_col].tolist())
            st.success("✅ Proses prediksi selesai!")
        else:
            st.success("✅ Data sudah memiliki label Sentimen.")
            
        st.markdown("<h2 class='section-title'>📈 Ringkasan Metrik</h2>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        total_reviews = len(df)
        total_pos = len(df[df['Sentimen'] == 'Positif'])
        total_neg = len(df[df['Sentimen'] == 'Negatif'])
        
        avg_rating = "N/A"
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            if not df['rating'].dropna().empty:
                avg_rating = round(df['rating'].mean(), 1)
            
        col1.metric("Total Ulasan", f"{total_reviews:,}")
        col2.metric("Sentimen Positif 😊", f"{total_pos:,}")
        col3.metric("Sentimen Negatif 😞", f"{total_neg:,}")
        col4.metric("Rata-rata Rating ⭐", avg_rating)
        
        st.markdown("<h2 class='section-title'>📊 Visualisasi Data</h2>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Distribusi Sentimen")
            fig_pie = px.pie(
                df, names='Sentimen', 
                color='Sentimen',
                color_discrete_map={'Positif':'#FF9900', 'Negatif':'#003366'},
                hole=0.4
            )
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            if 'rating' in df.columns:
                st.markdown("#### Distribusi Rating berdasarkan Sentimen")
                fig_bar = px.histogram(
                    df, x='rating', color='Sentimen', 
                    barmode='group',
                    color_discrete_map={'Positif':'#FF9900', 'Negatif':'#003366'}
                )
                fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Kolom 'rating' tidak ditemukan dalam data.")
                
        if 'date' in df.columns or 'tanggal' in df.columns:
            date_col = 'date' if 'date' in df.columns else 'tanggal'
            try:
                df['date_parsed'] = pd.to_datetime(df[date_col], format='mixed', errors='coerce').dt.date
                daily_sentiment = df.dropna(subset=['date_parsed']).groupby(['date_parsed', 'Sentimen']).size().reset_index(name='count')
                st.markdown("#### Tren Sentimen Seiring Waktu")
                fig_line = px.line(
                    daily_sentiment, x='date_parsed', y='count', color='Sentimen',
                    color_discrete_map={'Positif':'#FF9900', 'Negatif':'#003366'}
                )
                fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_line, use_container_width=True)
            except Exception as e:
                st.warning(f"Tidak dapat memuat grafik Tren Sentimen: {e}")
                
        st.markdown("<h2 class='section-title'>☁️ Word Cloud</h2>", unsafe_allow_html=True)
        w1, w2 = st.columns(2)
        
        def plot_wordcloud(text_data, title, colormap):
            if not text_data:
                return None
            wordcloud = WordCloud(width=800, height=400, background_color='#1F1F1F', colormap=colormap).generate(" ".join(text_data))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            ax.set_title(title, color='white', pad=20, size=20)
            fig.patch.set_facecolor('#1F1F1F')
            return fig
            
        with w1:
            pos_texts = df[df['Sentimen'] == 'Positif'][text_col].dropna().astype(str).tolist()
            if pos_texts:
                st.pyplot(plot_wordcloud(pos_texts, 'Kata-kata Sentimen Positif', 'Wistia'))
        with w2:
            neg_texts = df[df['Sentimen'] == 'Negatif'][text_col].dropna().astype(str).tolist()
            if neg_texts:
                st.pyplot(plot_wordcloud(neg_texts, 'Kata-kata Sentimen Negatif', 'Blues'))
                
        st.markdown("<h2 class='section-title'>🗂️ Tabel Data</h2>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

# ===========================================================================
# 🚀 APP ROUTING
# ===========================================================================
if 'page' in locals() and page == "📝 Analisis Input Teks":
    show_single_analysis()
elif 'page' in locals() and page == "📊 Dashboard Dataset":
    show_dashboard()
else:
    show_single_analysis()
# ===========================================================================
# 🦶 FOOTER - Lebih Profesional
# ===========================================================================

st.markdown("---")
st.markdown("""
<div class='app-footer'>
    <h3>🎓 Analisis Sentimen Ulasan Pengguna</h3>
    <p>
        Aplikasi ini menggunakan metode <strong>Bidirectional LSTM (Bi-LSTM)</strong> dengan <strong>FastText Embedding</strong>
    </p>
    <p style='font-size: 0.9em;'>
        © 2026 - Skripsi Analisis Sentimen | Tangerang Live
    </p>
</div>
""", unsafe_allow_html=True)