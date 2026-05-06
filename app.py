import streamlit as st
import os
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import plotly.graph_objects as go
import random
import math
import time
from PIL import Image
import cv2
import tensorflow as tf 

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pneumonia Detection AI", layout="wide")

#Load Deep Learning model only once, not on every rerun
@st.cache_resource                    
def load_model():
    model = tf.keras.models.load_model("pneumonia_model.keras", compile=False) 
    return model

model = load_model()

#Custom CSS
st.markdown("""
    <style>
        [data-testid="sidebar-close-button"], [data-testid="stSidebarCollapseButton"] {
            display: none !important;
        }
            
        html, body, [data-testid="stAppViewContainer"] {
            height: 100vh !important;
            overflow: hidden !important;
        }
        
        [data-testid="stSidebarResizer"] {
            display: none !important;
        }

        [data-testid="stSidebar"] {
            cursor: default !important;
            min-width: 280px !important;
            max-width: 280px !important;
            background-color: #1E293B; 
            font-family: "Inter", sans-serif;
        }   

        section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] span {
            color: #F8FAFC;
            font-size: 13px;
        }   

        header[data-testid="stHeader"] {
            display: none;
        }

        div[data-testid="stSidebarUserContent"] {
            padding-top: 0rem !important;
            padding-left: 10px !important;
            padding-right: 10px !important;
        }

        .block-container {
            height: 100vh !important;
            overflow: hidden !important;
            background-color: #F1F5F9 !important;
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
            
        .card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 9px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            margin-bottom: 20px;
        }

        .card-hdr {
            padding: 10px 13px;
            border-bottom: 2px solid #E2E8F0;
            background: #F1F5F9;
            display: flex;
            align-items: center;
            gap: 8px;
        }
            
        .card-title {
            font-size: 12px;
            font-weight: 700;
            color: #334155;
            letter-spacing: 0.4px;
            text-transform: uppercase;
        }
            
        .card-body { padding: 11px 13px; }

        [data-testid="stSidebarUserContent"] h1 {
            margin-top: -40px !important;
            text-align: center;
            padding-top: 0px !important;
            padding-bottom: 0px;
            margin-bottom: 0px;
            display: block !important;
            color:#E6F1FB;
        }
            
        [data-testid="stSidebarUserContent"] .stElementContainer {
            margin-bottom: 20px !important;
        }

        /* Keep uploader visible */
[data-testid="stFileUploader"] {
    display: flex !important;
    justify-content: center !important;
}

/* Hide ONLY file list (safe) */
[data-testid="stFileUploaderFile"],
[data-testid="fileDeleteBtn"],
[data-testid="stFileUploader"] ul,
[data-testid="stFileUploader"] li,
[data-testid="stFileUploader"] small {
    display: none !important;
}

/* Hide drag & drop text area ONLY (safe targeting) */
[data-testid="stFileUploaderDropzone"] {
    display: none !important;
}

/* Style ONLY button */
[data-testid="stFileUploader"] button {
    display: flex !important;
    background-color: #185FA5 !important;
    color: white !important;
    width: 220px !important;
    margin: auto !important;
    border-radius: 8px !important;
}
        

        .xray-outer {
            position: relative;     
            width: 100%;
            height: 390px;          
            background: #000000;
            border: #1F2937; 
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.5);
            border-radius: 8px;
            overflow: hidden;       
        }
            
        .xray-outer img {
            position: absolute;     
            inset: 0;              
            width: 100%;
            height: 100%;
            object-fit: contain;    
            display: block; 
            box-shadow: 0 0 15px rgba(0,0,0,0.5);
        }
            
        .await-overlay {
            display: flex; 
            flex-direction: column; 
            align-items: center;
            gap: 12px; color: #888;
        }
            
        .await-icon {
            width: 56px; 
            height: 56px; 
            border: 2px dashed #444;
            border-radius: 10px; 
            display: flex; 
            align-items: center; 
            justify-content: center;
        }

        .summary-bar {
            display: grid;
            grid-template-columns: 180px 0.5px 1fr 0.5px 1fr 0.5px 1fr;
            align-items: center;
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 16px 24px;
            margin-bottom: 20px;
            gap: 0;
        }
            
        .sb-divider { 
            width: 1px; 
            height: 60px; 
            background: #e0e0e0; 
            margin: 0 18px; 
        }
            
        .sb-gauge { 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
        }
            
        .sb-gauge-label { 
            font-size: 11px; 
            color: #888; 
            margin-top: 3px; 
        }
            
        .sb-metric { padding: 0 18px; }
            
        .sb-metric-label { 
            font-size: 11px;
            color: #888; 
            margin-bottom: 3px; 
        }
            
        .sb-metric-value { 
            font-size: 22px; 
            font-weight: 600; 
            line-height: 1; 
            margin-bottom: 4px; 
        }
            
        .sb-badge {
            display: inline-block; 
            font-size: 11px; 
            font-weight: 500;
            padding: 2px 10px; 
            border-radius: 20px;
        }
            
        .sb-bar-track { 
            width: 100%; 
            height: 5px; 
            background: #f0f0f0; 
            border-radius: 3px; 
            margin-top: 8px; 
        }
            
        .sb-bar-fill { 
            height: 100%; 
            border-radius: 3px; 
        }

        /* ── Awaiting state for summary bar ── */
        .sb-awaiting {
            display: grid;
            grid-template-columns: 180px 0.5px 1fr 0.5px 1fr 0.5px 1fr;
            align-items: center;
            background: #9CA3AF;
            border: 1px dashed #ddd;
            border-radius: 12px;
            padding: 16px 24px;
            margin-bottom: 16px;
            gap: 0;
        }
            
        .sb-placeholder { 
            height: 40px; 
            background: #ebebeb; 
            border-radius: 6px; 
            margin: 0 18px; 
        }
            
        .sb-placeholder-gauge {
            width: 140px; 
            height: 70px; 
            background: #ebebeb;
            border-radius: 70px 70px 0 0; 
            margin: 0 auto;
        }
            
        .topbar {
            display: flex;
            align-items: center;
            padding: 0 24px;
            height: 100px;
            border-radius: 12px;
            margin-bottom: 20px;
            gap: 0;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 6px rgba(0,0,0,0.05);
        }
 
        .topbar-idle {
            background: #FFFFFF !important;
            border-left: 6px solid #CBD5E1;
            border-radius: 12px;
            padding: 12px;
            color: #64748B;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }
            
        .topbar-pos {
            background: #FFF5F5;
            border: 1px solid #FECACA;
            border-left: 4px solid #DC2626;
            box-shadow: 0 1px 6px rgba(220,38,38,0.08);
        }
            

        .topbar-neg {
            background: #F0FDF4;
            border: 1px solid #BBF7D0;
            border-left: 4px solid #16A34A;
            box-shadow: 0 1px 6px rgba(22,163,74,0.08);
        }
            
        .topbar-sus {
            background: #FFFBEB;
            border: 1px solid #FDE68A;
            border-left: 4px solid #D97706;
            box-shadow: 0 1px 6px rgba(217,119,6,0.08);
        }           
 

        .topbar-left {
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 2px;
            flex-shrink: 0;
        }
            
        .topbar-verdict {
            font-size: 23px;
            font-weight: 700;
            letter-spacing: -0.3px;
            line-height: 1.1;
        }
            
        .verdict-pos {  
            color: #DC2626; 
            text-shadow: 0px 0px 8px rgba(239, 68, 68, 0.2);
        }
            
        .verdict-neg { 
            color: #16A34A; 
            text-shadow: 0px 0px 8px rgba(239, 68, 68, 0.2);
        }
            
        .verdict-sus { 
            color: #D97706; 
            text-shadow: 0px 0px 8px rgba(239, 68, 68, 0.2);
        }
            
        .verdict-idle{ 
            color: #334155 !important; 
            font-size: 23px !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px; 
            line-height: 1;
            margin-bottom: 2px; 
        }
 
        .topbar-sub {
            font-size: 11px;
            color:#94A3B8; 
            font-weight: 500;
            opacity: 1; 
            margin-top: 2px;
            letter-spacing: 0.3px; 
        }
        
        .topbar-spacer { flex: 1; }
 
        .topbar-right {
            display: flex;
            align-items: center;
            gap: 0;
            flex-shrink: 0;
            color: #F3F4F6 !important;
        }
            
        .topbar-stat {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            padding: 0 20px;
            color: #475569;
        }
            
        .topbar-stat-val {
            font-size: 22px;
            font-weight: 600;
            letter-spacing: -0.5px;
            line-height: 1.1;
        }
            
        .topbar-stat-lbl {
            font-size: 10px !important; 
            font-weight: 700 !important;
            letter-spacing: 0.8px !important;
            text-transform: uppercase !important;
            margin-top: 2px;
            transition: color 0.3s ease; 
            color: #94A3B8;
        }
            
        .topbar-divider {
            width: 1px;
            height: 36px;
            background: #E2E8F0;;
        }
            
       .conf-val-pos { 
            font-size:23px; 
            font-weight:600; 
            color:#DC2626 !important; 
            line-height:1.1; 
        }
            
        .conf-val-sus { 
            font-size:23px; 
            font-weight:600; 
            color:#D97706 !important; 
            line-height:1.1; 
        }
            
        .conf-val-neg { 
            font-size:23px; 
            font-weight:600; 
            color:#16A34A !important; 
            line-height:1.1; 
        }
                
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🫁 Pneumo AI")

    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], key="xray_uploader")
    if uploaded_file:
        st.success("File uploaded successfully!")

    
    if "_sensitivity" not in st.session_state:
     st.session_state["_sensitivity"] = 50

    # Reset to 50 when new image uploaded
    if st.session_state.get("_threshold_reset"):
     st.session_state["_sensitivity"] = 50
     st.session_state["_threshold_reset"] = False

    # Slider
    sensitivity = st.slider(
        label="Detection threshold",
        min_value=0,
        max_value=100,
        key="_sensitivity",
        step=1
     )
    
    #Dynamic Status Box
    if sensitivity < 40:
        st.info("🧪 **Mode: High Sensitivity**\n\nOptimized to catch all potential cases.")
    elif 40 <= sensitivity <= 75:
        st.success("⚖️ **Mode: Balanced**\n\nStandard diagnostic setting.")
    else:
        st.warning("🎯 **Mode: High Specificity**\n\nOptimized for maximum certainty.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_scores(image: Image.Image):
    """
    Preprocessing must exactly match training:
    1. Convert to grayscale        ← color_mode='grayscale'
    2. Resize to 224x224           ← IMG_SIZE = (224, 224)
    3. Apply CLAHE                 ← apply_clahe() used in training
    4. Normalize to 0-1            ← done inside apply_clahe
    5. Add batch + channel dims    ← model expects (1, 224, 224, 1)
    

    Runs model on the uploaded image. Returns pneumonia probability (0-100), normal probability (0-100),
    and elapsed time in seconds."""

    start = time.time()

    #Training used color_mode='grayscale' → 1 channel
    img_gray = image.convert("L")               

    #Resize to 224×224 
    img_resized = img_gray.resize((224, 224))    

    #Convert to numpy shape: (224, 224)
    img_array = np.array(img_resized)            

    #Apply CLAHE as in training 
    #Training used: clipLimit=1.2, tileGridSize=(8,8)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_array.astype(np.uint8))   

    #Normalize to 0-1
    img_normalized = img_clahe.astype(np.float32) / 255.0

    #Add channel dim → (224, 224, 1)
    img_channel = np.expand_dims(img_normalized, axis=-1)  

    #Add batch dim → (1, 224, 224, 1) 
    img_batch = np.expand_dims(img_channel, axis=0)  

    #Run model 
    prediction = model.predict(img_batch, verbose=0)       

    pneumonia_prob = float(prediction[0][0])

    pneumonia_pct = round(pneumonia_prob * 100, 1)
    normal_pct    = round((1 - pneumonia_prob) * 100, 1)
    elapsed       = round(time.time() - start, 1)

    return pneumonia_pct, normal_pct, elapsed


def verdict_info(pneumonia_pct, threshold):
    """
    threshold comes from the sensitivity slider (0-100 → 0.0-1.0).
    Only two results: Positive or Negative.
    """
    if pneumonia_pct >= threshold:
        # Positive
        return "Pneumonia Detected", "#DC2626", "topbar-pos", "verdict-pos", "conf-val-pos", "positive"
    
    elif 50 < pneumonia_pct < threshold:
        return "Inconclusive — Review Needed", "#D97706", "topbar-sus", "verdict-sus", "conf-val-sus", "suspicious"

    else:
        # Negative
        return "No Pneumonia Found", "#16A34A", "topbar-neg", "verdict-neg", "conf-val-neg", "negative"


def gauge_svg(pneumonia_pct, bar_color):
    total   = 226.2         #arc length for r=72 semicircle
    filled  = total * (pneumonia_pct / 100)
    offset  = total - filled
    # needle angle: 0%=left(180°), 100%=right(0°)
    angle   = 180 - (pneumonia_pct / 100) * 180
    rad     = math.radians(angle)
    cx, cy, r = 90, 88, 72
    nx = cx + r * math.cos(math.radians(180 - (pneumonia_pct/100)*180))
    ny = cy - r * math.sin(math.radians((pneumonia_pct/100)*180))
    return f"""
    <svg width="180" height="96" viewBox="0 0 180 96" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="garc" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stop-color="#639922"/>
          <stop offset="45%"  stop-color="#ef9f27"/>
          <stop offset="100%" stop-color="#e24b4a"/>
        </linearGradient>
        <clipPath id="gclip"><rect x="0" y="0" width="180" height="90"/></clipPath>
      </defs>
      <g clip-path="url(#gclip)">
        <path d="M18 88 A72 72 0 0 1 162 88" fill="none"
              stroke="#e8e8e8" stroke-width="13" stroke-linecap="round"/>
        <path d="M18 88 A72 72 0 0 1 162 88" fill="none"
              stroke="url(#garc)" stroke-width="13" stroke-linecap="round"
              stroke-dasharray="{total:.1f}" stroke-dashoffset="{offset:.1f}"/>
        <line x1="18"  y1="88" x2="24"  y2="88" stroke="#ccc" stroke-width="1.2"/>
        <line x1="90"  y1="16" x2="90"  y2="22" stroke="#ccc" stroke-width="1.2"/>
        <line x1="162" y1="88" x2="156" y2="88" stroke="#ccc" stroke-width="1.2"/>
      </g>
      <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}"
            stroke="{bar_color}" stroke-width="2.2" stroke-linecap="round"/>
      <circle cx="{cx}" cy="{cy}" r="5"   fill="{bar_color}"/>
      <circle cx="{cx}" cy="{cy}" r="2.5" fill="white"/>
      <text x="11"  y="96" font-size="9" fill="#bbb" font-family="sans-serif">0</text>
      <text x="83"  y="14" font-size="9" fill="#bbb" font-family="sans-serif">50</text>
      <text x="152" y="96" font-size="9" fill="#bbb" font-family="sans-serif">100</text>
    </svg>"""


#Compute scores once per upload
if uploaded_file:
    fname = uploaded_file.name

    #Run model only when a NEW file is uploaded
    #Sensitivity change nOT re-run the model only re-apply threshold
    if st.session_state.get("_scored_file") != fname:
        image = Image.open(uploaded_file).convert("RGB")
        with st.spinner("Analyzing scan…"):
            p, n, elapsed = get_scores(image)         
        st.session_state["_pneumonia_pct"] = p
        st.session_state["_normal_pct"]    = n
        st.session_state["_elapsed"]       = elapsed
        st.session_state["_scan_time"]     = time.strftime("%d %b %Y · %H:%M")
        st.session_state["_scored_file"]   = fname
        st.session_state["_threshold_reset"] = True

    pneumonia_pct = st.session_state["_pneumonia_pct"]
    normal_pct    = st.session_state["_normal_pct"]
    elapsed       = st.session_state["_elapsed"]
    scan_time     = st.session_state["_scan_time"]

    #threshold from slider applied every time slider moves
    threshold     = sensitivity          # slider is 0-100
    label, bar_color, tb_cls, v_cls, conf_cls, verdict_zone = verdict_info(pneumonia_pct, threshold)


# 1. Define the dynamic detail message
    if verdict_zone == "positive":
     detail_msg = '<strong> Pneumo Prob. ' + str(pneumonia_pct) + '% exceeds ' + str(threshold) + '% threshold </strong>'
    elif verdict_zone == "suspicious":
     detail_msg = '<strong>Possible: Pneumo Prob. ' + str(pneumonia_pct) + '% is below ' + str(threshold) + '% threshold (Review Recommended) </strong>'
    else:
     detail_msg = '<strong>' + str(pneumonia_pct) + '% risk (Below threshold) </strong>'


#Summary bar
if not uploaded_file:
    #Awaiting state 
    st.markdown("""
<div class="topbar topbar-idle">
    <div class="topbar-left">
        <div class="topbar-verdict verdict-idle">Awaiting scan...</div>
        <div class="topbar-sub" style="color: #64748B">Upload an X-ray from the sidebar to begin</div>
    </div>
    <div class="topbar-spacer"></div>
    <div class="topbar-right">
        <div class="topbar-stat">
            <div class="topbar-stat-val" style="color: #64748B">—</div>
            <div class="topbar-stat-lbl" style="color: #64748B">pneumonia prob.</div>
        </div>
        <div class="topbar-divider" style="color:#64748B"></div>
        <div class="topbar-stat">
            <div class="topbar-stat-val" style="color:#64748B">—</div>
            <div class="topbar-stat-lbl" style="color:#64748B">inference</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


else:
    topbar_html = (
        '<div class="topbar ' + tb_cls + '">'
          '<div class="topbar-left">'
            '<div class="topbar-verdict ' + v_cls + '">' + label + '</div>'
            '<div style="font-size:13px; margin-top:5px; color:inherit; opacity:0.85; line-height:1.2;">' + detail_msg + '</div>'
            '<div class="topbar-sub">' + scan_time + ' &middot; ' + uploaded_file.name + '</div>'
          '</div>'
          '<div class="topbar-spacer"></div>'
          '<div class="topbar-right">'
            '<div class="topbar-stat">'
              '<div class="' + conf_cls + '">' + str(pneumonia_pct) + '%</div>'
              '<div class="topbar-stat-lbl">pneumonia prob.</div>'
            '</div>'
            '<div class="topbar-divider"></div>'
            '<div class="topbar-stat">'
              '<div class="topbar-stat-val">' + str(elapsed) + 's</div>'
              '<div class="topbar-stat-lbl">inference</div>'
            '</div>'
          '</div>'
        '</div>'
    )
    st.markdown(topbar_html, unsafe_allow_html=True)
 


#Two columns
col1, col2 = st.columns([1.2, 1])

#Left side(X-RAY)

with col1:
    with st.container(border=True):
        st.write("**X-Ray Scan**")
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(
                f'<div class="xray-outer">'
                f'<img src="data:image/png;base64,{b64}" alt="X-ray scan"/>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("""
                <div class="xray-outer">
                    <div style="position:absolute;inset:0;display:flex;
                        flex-direction:column;align-items:center;
                        justify-content:center;gap:10px;color:#555;">
                        <svg width="40" height="40" viewBox="0 0 48 48" fill="none">
                            <rect x="6" y="8" width="36" height="32" rx="4"
                                stroke="#666" stroke-width="2"/>
                            <path d="M24 30V18M18 24l6-6 6 6"
                                stroke="#666" stroke-width="2"
                                stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                        <div style="font-size:13px;">Upload an X-ray to begin</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

#Right side(split into 2)
with col2:

    #Build gauge
    if uploaded_file:
        svg_code = gauge_svg(pneumonia_pct, bar_color)

        gauge_content = (
            '<div style="display:flex;flex-direction:column;align-items:center;gap:3px;">'
            + svg_code
            + '<div style="font-size:10px;color:#94A3B8;margin-top:2px;">pneumonia probability</div>'
            + '</div>'
            + '<div style="margin-top:11px;">'
            + '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
            + '<span style="font-size:11px;font-weight:700;color:#DC2626 ;width:68px;flex-shrink:0;">Pneumonia</span>'
            + '<div style="flex:1;height:5px;background:#E2E8F0;border-radius:3px;overflow:hidden;">'
            + '<div style="width:' + str(pneumonia_pct) + '%;height:100%; background:#DC2626;border-radius:3px;"></div>'
            + '</div>'
            + '<span style="font-size:11px;font-weight:700;color:#DC2626;width:34px;text-align:right;">' + str(pneumonia_pct) + '%</span>'
            + '</div>'
            + '<div style="display:flex;align-items:center;gap:8px;">'
            + '<span style="font-size:11px;font-weight:600;color:#16A34A;width:68px;flex-shrink:0;">Normal</span>'
            + '<div style="flex:1;height:5px;background:#E2E8F0;border-radius:3px;overflow:hidden;">'
            + '<div style="width:' + str(normal_pct) + '%;height:100%;background:#16A34A;border-radius:3px;"></div>'
            + '</div>'
            + '<span style="font-size:11px;font-weight:600;color:#16A34A;width:34px;text-align:right;">' + str(normal_pct) + '%</span>'
            + '</div>'
            + '</div>'
        )


        detail_content = (
            '<div style="display:flex;justify-content:space-between;align-items:center;'
            'padding:7px 0;border-bottom:1px solid #F1F5F9;font-size:11px;">'
            '<span style="color:#94A3B8;font-weight:600;">Recall</span>'
            '<span style="color:#1E293B;font-weight:700;">90%</span>'
            '</div>'

            '<div style="display:flex;justify-content:space-between;align-items:center;'
            'padding:7px 0;border-bottom:1px solid #F1F5F9;font-size:11px;">'
            '<span style="color:#94A3B8;font-weight:500;">Precision</span>'
            '<span style="color:#1E293B;font-weight:700;">82%</span>'
            '</div>'
 
            '<div style="display:flex;justify-content:space-between;align-items:center;'
            'padding:7px 0;border-bottom:1px solid #F1F5F9;font-size:11px;">'
            '<span style="color:#94A3B8;font-weight:500;">AUC-ROC</span>'
            '<span style="color:#1E293B;font-weight:700;">93.4%</span>'
            '</div>'

            '<div style="display:flex;justify-content:space-between;align-items:center;'
            'padding:7px 0;border-bottom:1px solid #F1F5F9;font-size:11px;">'
            '<span style="color:#94A3B8;font-weight:500;">Accuracy</span>'
            '<span style="color:#1E293B;font-weight:700;">85.1%</span>'
            '</div>'
        )

    else:
        #awaiting state — message shows inside card body
        gauge_content = (
            '<div style="display:flex;flex-direction:column;align-items:center;'
            'justify-content:center;padding:28px 0;gap:8px;'
            'color:#94A3B8;font-size:13px;text-align:center;">'
            '<svg width="32" height="32" viewBox="0 0 32 32" fill="none">'
            '<circle cx="16" cy="16" r="13" stroke="#CBD5E1" stroke-width="1.5"/>'
            '<path d="M16 10v8M16 22v.5" stroke="#CBD5E1" stroke-width="1.8" stroke-linecap="round"/>'
            '</svg>'
            '<div>Upload an X-ray to see the gauge</div>'
            '</div>'
        )

        detail_content = (
            '<div style="display:flex;flex-direction:column;align-items:center;'
            'justify-content:center;padding:20px 0;gap:6px;'
            'color:#94A3B8;font-size:13px;text-align:center;">'
            '<div>Upload an X-ray to see performance details</div>'
            '</div>'
        )

    #CARD 1: Confidence Gauge
    st.markdown(
        '<div class="card">'
        '<div class="card-hdr">'
        '<span class="card-title">Confidence Gauge</span>'
        '</div>'
        '<div class="card-body">'
        + gauge_content +
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    #CARD 2: Scan Details 
    st.markdown(
        '<div class="card">'
        '<div class="card-hdr">'
        '<span class="card-title">Model Performance</span>'
        '</div>'
        '<div class="card-body">'
        + detail_content +
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

#footer
st.write("---")

st.markdown("""
<div style="
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    color: #475569;
    padding: 6px 2px;
    flex-wrap: wrap;
">

<div>
    <span style="color:#0f172a; font-weight:600;">STATUS:</span>
    🟢 Normal | 🟡 Suspicious | 🔴 Pneumonia
</div>

<div>
    <span style="color:#0f172a; font-weight:600;">MODEL:</span>
    CNN (Custom CNN)
</div>

<div>
    <span style="color:#0f172a; font-weight:600;">DATASET:</span>
    Chest X-ray dataset
</div>

<div>
    <span style="color:#0f172a; font-weight:600;">UPDATED:</span>
    May 2026
</div>

<div>
    <span style="color:#0f172a; font-weight:600;">DISCLAIMER:</span>
    AI-assisted tool (not medical diagnosis)
</div>

</div>

<div style="
    text-align:center;
    font-size:11px;
    color:#94a3b8;
    margin-top:6px;
">
PneumoAI © 2026 | Academic Research Project
</div>
""", unsafe_allow_html=True)