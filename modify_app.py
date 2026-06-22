import re

with open('c:/All/Coolyeah/Semester 8/Crispy/Program Analisis/web/app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 1. Add imports
import_idx = 0
for i, line in enumerate(lines):
    if line.startswith('import streamlit as st'):
        import_idx = i
        break
lines.insert(import_idx + 1, 'import plotly.express as px\nfrom wordcloud import WordCloud\nimport io\n')

# 2. Add sidebar radio
sidebar_header_end = -1
for i, line in enumerate(lines):
    if "Kontrol Analisis</p>" in line:
        sidebar_header_end = i + 2
        break

sidebar_code = """
    st.markdown("### 🧭 Navigasi")
    page = st.radio("Pilih Mode:", ["📝 Analisis Input Teks", "📊 Dashboard Dataset"], label_visibility="collapsed")
    st.markdown("---")
"""
if sidebar_header_end != -1:
    lines.insert(sidebar_header_end, sidebar_code)

# 3. Find where main content starts
main_start_idx = -1
for i, line in enumerate(lines):
    if line.startswith('# 🖼️ MAIN CONTENT'):
        main_start_idx = i - 1
        break

# 4. Find where footer starts
footer_start_idx = -1
for i, line in enumerate(lines):
    if line.startswith('# 🦶 FOOTER'):
        footer_start_idx = i - 1
        break

new_lines = lines[:main_start_idx]
new_lines.append("\n# ===========================================================================\n")
new_lines.append("# 📝 HALAMAN ANALISIS SINGLE TEXT\n")
new_lines.append("# ===========================================================================\n\n")
new_lines.append("def show_single_analysis():\n")

for line in lines[main_start_idx:footer_start_idx]:
    if line.strip() == "":
        new_lines.append(line)
    else:
        new_lines.append("    " + line)

dashboard_code = """
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
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Berhasil memuat file yang diunggah ({len(df)} baris)")
    
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
                df['date_parsed'] = pd.to_datetime(df[date_col]).dt.date
                daily_sentiment = df.groupby(['date_parsed', 'Sentimen']).size().reset_index(name='count')
                st.markdown("#### Tren Sentimen Seiring Waktu")
                fig_line = px.line(
                    daily_sentiment, x='date_parsed', y='count', color='Sentimen',
                    color_discrete_map={'Positif':'#FF9900', 'Negatif':'#003366'}
                )
                fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_line, use_container_width=True)
            except Exception as e:
                pass
                
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
"""
new_lines.append(dashboard_code)

for line in lines[footer_start_idx:]:
    new_lines.append(line)

with open('c:/All/Coolyeah/Semester 8/Crispy/Program Analisis/web/app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
