import joblib, numpy as np, streamlit as st, warnings
warnings.filterwarnings("ignore")

# ────────────────────────── PAGE CONFIG ──────────────────────────
st.set_page_config(page_title="News Analyzer AI",
                   page_icon="🚀",
                   layout="wide",
                   initial_sidebar_state="collapsed")

# ──────────────────────────── CSS ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&display=swap');
header[data-testid="stHeader"], #MainMenu, footer {visibility:hidden;}

.stApp{background:#0f0f1f;color:#fff;font-family:'Space Grotesk',sans-serif;}
.block-container{padding:1.5rem 1rem 0 1rem;}

/* Menu toggle */
.menu-toggle{position:fixed;top:60px;left:12px;z-index:1002;}
.menu-toggle button{
  min-width:115px;background:linear-gradient(135deg,#8b5cf6,#a855f7)!important;
  color:#fff!important;border:none!important;border-radius:12px!important;
  font-weight:600!important;padding:.65rem 1.1rem!important;
  box-shadow:0 4px 15px rgba(139,92,246,.35)!important;
}

/* Menu panel */
.menu-panel{
  padding:1.8rem;background:rgba(25,25,46,.9);
  border:1px solid rgba(139,92,246,.3);border-radius:18px;
  box-shadow:0 15px 35px rgba(139,92,246,.2);position:relative;overflow:hidden;
}
.menu-panel::before{
  content:'';position:absolute;inset:-15px;border-radius:22px;
  background:linear-gradient(135deg,rgba(139,92,246,.3),rgba(168,85,247,.2));
  filter:blur(15px);opacity:.6;z-index:-1;
}
.menu-panel h3{margin-top:0;}
.menu-panel hr{border:none;border-top:1px solid rgba(139,92,246,.3);margin:1.1rem 0;}

/* Glow header */
.glow-box{
  position:relative;max-width:600px;margin:0 auto 1rem;padding:1rem;
  background:rgba(25,25,46,.95);border:1px solid rgba(139,92,246,.35);
  border-radius:24px;box-shadow:0 22px 60px rgba(139,92,246,.25);
}
.glow-box::before{
  content:'';position:absolute;inset:-20px;border-radius:28px;
  background:linear-gradient(135deg,rgba(139,92,246,.45),rgba(168,85,247,.3));
  filter:blur(25px);opacity:.8;z-index:0;
}
.glow-box>*{position:relative;z-index:1;}
.glow-box h1{
  margin:0 0 1rem;font-size:2.6rem;
  background:linear-gradient(135deg,#8b5cf6,#c084fc);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.glow-box p{margin:0 0 1.2rem;font-size:1.05rem;color:#b4b4c8;}

/* Text area */
textarea{
  border-radius:16px!important;border:1px solid rgba(139,92,246,.6)!important;
  background:#1a1a2c!important;color:#fff!important;font-size:1rem!important;
}

/* Buttons */
.stButton>button{
  background:linear-gradient(135deg,#8b5cf6,#c084fc)!important;
  color:#fff!important;border:none!important;border-radius:16px!important;
  font-weight:600!important;box-shadow:0 8px 25px rgba(139,92,246,.32)!important;
}

/* Result cards */
.result-card{
  padding:1rem 1rem 1.3rem;text-align:center;min-height:140px;
  background:#1a1a2c;border:1px solid rgba(139,92,246,.3);
  border-radius:18px;box-shadow:0 15px 35px rgba(139,92,246,.15);
  position:relative;overflow:hidden;
}
.result-card::before{
  content:'';position:absolute;inset:-15px;border-radius:22px;
  background:linear-gradient(135deg,rgba(139,92,246,.3),rgba(168,85,247,.2));
  filter:blur(12px);opacity:.6;z-index:-1;
}
.result-card .icon{font-size:1.7rem;}
.result-card .title{margin:.4rem 0 .6rem;font-weight:600;font-size:1.05rem;}
.result-card .conf{color:#b4b4c8;font-size:.8rem;}
.status-real,.status-safe{color:#10b981;}.status-fake,.status-hate{color:#ef4444;}

.category-badge{
  padding:.32rem .75rem;border-radius:10px;color:#fff;font-weight:600;font-size:.9rem;
}
.cat-world{background:linear-gradient(135deg,#3b82f6,#1d4ed8);}
.cat-sports{background:linear-gradient(135deg,#f59e0b,#d97706);}
.cat-business{background:linear-gradient(135deg,#059669,#047857);}
.cat-scitech{background:linear-gradient(135deg,#8b5cf6,#7c3aed);}

.footer{text-align:center;color:#b4b4c8;font-size:.85rem;padding:3rem 0 1rem;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── MODEL LOADING ───────────────────────────
@st.cache_resource
def load_models():
    return (joblib.load("fakenews_vectorizer.pkl"), joblib.load("fakenews_model.pkl"),
            joblib.load("news_category_vector.pkl"), joblib.load("news_category_model.pkl"),
            joblib.load("hatespeach_vectorizer.pkl"), joblib.load("hatespeach_model.pkl"))
v_fake, m_fake, v_cat, m_cat, v_hate, m_hate = load_models()

label_map = {0:"world", 1:"sports", 2:"business", 3:"sci/tech"}
icon_map  = {"world":"🌍", "sports":"⚽", "business":"💼", "sci/tech":"🔬"}

# ───────────────────────────── PREDICTORS ─────────────────────────────
def predict_fake(txt: str):
    p = m_fake.predict_proba(v_fake.transform([txt]))[0][1]
    return ("Real" if p >= .5 else "Fake"), p * 100

def predict_cat(txt: str):
    X = v_cat.transform([txt])
    # use probabilities to pick best among the four valid classes
    if hasattr(m_cat, "predict_proba"):
        probs   = m_cat.predict_proba(X)[0]
        classes = m_cat.classes_
        valid   = {int(cls): probs[i] for i, cls in enumerate(classes) if int(cls) in label_map}
        if valid:
            best_cls = max(valid, key=valid.get)
            return label_map[best_cls], valid[best_cls] * 100
    # fallback
    raw = int(m_cat.predict(X)[0])
    return label_map.get(raw, "world"), 100.0

def predict_hate(txt: str):
    p = m_hate.predict_proba(v_hate.transform([txt]))[0][1]
    return p >= .5, p * 100

# ───────────────────────────── MENU STATE ─────────────────────────────
if "show_menu" not in st.session_state:
    st.session_state.show_menu = True

st.markdown('<div class="menu-toggle">', unsafe_allow_html=True)
st.button("✕ Close" if st.session_state.show_menu else "☰ Menu",
          key="toggle_menu",
          on_click=lambda: st.session_state.update(show_menu=not st.session_state.show_menu))
st.markdown('</div>', unsafe_allow_html=True)

wrap_cls = "" if st.session_state.show_menu else "collapsed"
st.markdown(f'<div class="{wrap_cls}">', unsafe_allow_html=True)
menu_w = 3 if st.session_state.show_menu else 0.05
menu_col, main_col = st.columns([menu_w, 12 - menu_w], gap="large")
st.markdown('</div>', unsafe_allow_html=True)

# ───────────────────────────── MENU PANEL ─────────────────────────────
with menu_col:
    if st.session_state.show_menu:
        st.markdown("""
<div class="menu-panel">
  <h3>ℹ️ About</h3>
  <p>📰 Fake-news detection<br>🏷️ News categorization<br>⚠️ Hate speech detection</p>
  <hr><h3>🚀 Features</h3><p>• User-end  results<br>• Real-time analysis</p>
  <hr><h3>📝 Usage</h3><p>1. Paste article<br>2. Click Analyze<br>3. View results</p>
  <hr><h3>🚀 Connect</h3>
  <a href="https://github.com/Karanpr-18" target="_blank"
     style="display:inline-flex;align-items:center;background:#23272b;color:#fff;
            padding:6px 14px;border:2px solid #64ffda;border-radius:18px;
            font-weight:600;margin-right:10px;text-decoration:none;">
    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
         style="width:22px;height:22px;border-radius:50%;border:1.2px solid #64ffda;
                background:#fff;margin-right:7px;">
    GitHub
  </a>
  <a href="https://www.linkedin.com/in/karan-bhoriya-b5a3382b7" target="_blank"
     style="display:inline-flex;align-items:center;background:#0a66c2;color:#fff;
            padding:6px 14px;border:2px solid #64ffda;border-radius:18px;
            font-weight:600;text-decoration:none;">
    <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg"
         style="filter:invert(1);width:22px;height:22px;margin-right:7px;">
    LinkedIn
  </a>
  <hr><p style="text-align:center">Made by Karan</p>
</div>""", unsafe_allow_html=True)

# ───────────────────────────── MAIN CONTENT ─────────────────────────────
with main_col:
    st.markdown("""
<div class="glow-box">
  <h1>🚀 News Analyzer AI</h1>
  <p>Instant authenticity, topic &amp; safety check</p>
</div>""", unsafe_allow_html=True)

    l, c, r = st.columns([2, 6, 2])
    with c:
        news = st.text_area("", placeholder="Paste or type your news article here …",
                            label_visibility="collapsed")
        run = st.button("🔍 Analyze Article", type="primary")

        if run:
            if not news.strip():
                st.warning("⚠️ Please enter some text to analyze.")
            else:
                auth, a_conf   = predict_fake(news)
                cat_lbl, c_conf = predict_cat(news)
                hate, h_conf   = predict_hate(news)

                if hate:
                    st.error(f"⚠️ HATE SPEECH DETECTED ({h_conf:.1f}%)")

                cat_icon = icon_map[cat_lbl]

                col1, col2, col3 = st.columns(3, gap="small")

                with col1:
                    st.markdown(f"""
<div class="result-card">
  <div class="icon">{'✅' if auth=='Real' else '❌'}</div>
  <div class="title status-{'real' if auth=='Real' else 'fake'}">{auth}</div>
  <div class="conf">{a_conf:.1f}% confidence</div>
</div>""", unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
<div class="result-card">
  <div class="icon">{cat_icon}</div>
  <div class="title category-badge cat-{cat_lbl}">{cat_lbl.upper()}</div>
  <div class="conf">{c_conf:.1f}% confidence</div>
</div>""", unsafe_allow_html=True)

                with col3:
                    s_icon = "⚠️" if hate else "🛡️"
                    s_lbl  = "HATE" if hate else "SAFE"
                    s_conf = f"{h_conf:.1f}% confidence" if hate else f"{100 - h_conf:.1f}% confidence"
                    st.markdown(f"""
<div class="result-card">
  <div class="icon">{s_icon}</div>
  <div class="title status-{'hate' if hate else 'safe'}">{s_lbl}</div>
  <div class="conf">{s_conf}</div>
</div>""", unsafe_allow_html=True)

# ───────────────────────────── FOOTER ─────────────────────────────
st.markdown('<div class="footer">© 2025 News Analyzer AI | Built by Karan</div>',
            unsafe_allow_html=True)
