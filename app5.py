import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist Review Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.olist-header {
    background: linear-gradient(135deg, #1B5E20 0%, #388E3C 60%, #66BB6A 100%);
    padding: 1.8rem 2rem; border-radius: 16px; margin-bottom: 1.5rem; color: white;
}
.olist-header h1 { margin:0; font-size:1.8rem; font-weight:700; }
.olist-header p  { margin:0.3rem 0 0; font-size:0.9rem; opacity:0.88; }

.kpi { background:white; border:1px solid #E8F5E9; border-radius:12px;
       padding:1rem 1.2rem; text-align:center; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.kpi .v { font-size:1.7rem; font-weight:700; color:#2E7D32; }
.kpi .l { font-size:0.72rem; color:#888; margin-top:3px; text-transform:uppercase; letter-spacing:.05em; }

/* ── NEW: two-class result boxes ── */
.rbox { border-radius:14px; padding:2rem 1rem; text-align:center; margin-top:.5rem; }
.rbox-good { background:#E8F5E9; border:2px solid #43A047; }
.rbox-bad  { background:#FFEBEE; border:2px solid #E53935; }
.rbox .big  { font-size:3.5rem; font-weight:800; line-height:1; }
.rbox .lbl  { font-size:1.3rem; font-weight:700; margin-top:.5rem; }
.rbox .sub  { font-size:.85rem; margin-top:.3rem; opacity:.8; }
.c-good { color:#2E7D32; }
.c-bad  { color:#C62828; }

/* ── NEW: two-bar probability ── */
.prob-row { display:flex; align-items:center; gap:10px; margin:6px 0; font-size:.85rem; }
.prob-label { min-width:100px; font-weight:500; }
.prob-bar-bg   { flex:1; background:#F5F5F5; border-radius:20px; height:20px; overflow:hidden; }
.prob-bar-good { height:100%; border-radius:20px; background:#43A047; }
.prob-bar-bad  { height:100%; border-radius:20px; background:#E53935; }
.prob-pct { min-width:42px; text-align:right; font-weight:700; color:#333; }

.sec { font-size:.7rem; font-weight:600; text-transform:uppercase;
       letter-spacing:.07em; color:#999; border-bottom:1px solid #eee;
       padding-bottom:5px; margin:1.2rem 0 .7rem; }

.tip { background:#F1F8E9; border-left:4px solid #43A047;
       border-radius:0 8px 8px 0; padding:.7rem 1rem;
       font-size:.82rem; color:#1B5E20; margin-top:1rem; }
.tip-bad { background:#FFF3E0; border-left:4px solid #E53935;
           border-radius:0 8px 8px 0; padding:.7rem 1rem;
           font-size:.82rem; color:#B71C1C; margin-top:1rem; }

div[data-testid="stButton"] > button {
    background:#2E7D32 !important; color:white !important;
    border:none !important; border-radius:10px !important;
    padding:.65rem 0 !important; font-size:.95rem !important;
    font-weight:600 !important; width:100% !important;
}
div[data-testid="stButton"] > button:hover { background:#1B5E20 !important; }

.empty-state { background:#F9FBF9; border:2px dashed #A5D6A7; border-radius:14px;
               padding:3rem 1rem; text-align:center; color:#888; }
.arch-card { background:white; border:1px solid #E8F5E9; border-radius:12px;
             padding:1.2rem 1.4rem; }
.arch-card h4 { margin:0 0 .8rem; color:#2E7D32; font-size:.9rem; }
.arch-card li { font-size:.82rem; color:#444; margin:.25rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "best olist model.zip"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# ── Exact column values ───────────────────────────────────────────────────────
ORDER_STATUS    = ['delivered', 'shipped', 'processing', 'canceled']
PAYMENT_TYPES   = ['credit_card', 'boleto', 'voucher', 'debit_card']
CATEGORIES      = sorted([
    'housewares','perfumery','auto','pet_shop','stationery','furniture_decor',
    'office_furniture','garden_tools','computers_accessories','bed_bath_table',
    'toys','telephony','health_beauty','electronics','baby','cool_stuff',
    'watches_gifts','air_conditioning','sports_leisure','books_general_interest',
    'construction_tools_construction','small_appliances','food',
    'luggage_accessories','fashion_underwear_beach','christmas_supplies',
    'fashion_bags_accessories','musical_instruments','construction_tools_lights',
    'books_technical','costruction_tools_garden','home_appliances','market_place',
    'agro_industry_and_commerce','party_supplies','home_confort',
    'cds_dvds_musicals','industry_commerce_and_business','consoles_games',
    'furniture_bedroom','construction_tools_safety','fixed_telephony','drinks',
    'kitchen_dining_laundry_garden_furniture','fashion_shoes','home_construction',
    'audio','home_appliances_2','fashion_male_clothing','cine_photo',
    'furniture_living_room','art','food_drink','tablets_printing_image',
    'fashion_sport','la_cuisine','flowers','computers','home_comfort_2',
    'small_appliances_home_oven_and_coffee','dvds_blu_ray',
    'costruction_tools_tools','fashio_female_clothing',
    'furniture_mattress_and_upholstery','signaling_and_security',
    'diapers_and_hygiene','books_imported','fashion_childrens_clothes',
    'music','arts_and_craftmanship','security_and_services'
])
CUSTOMER_TYPES   = ['loyal', 'new', 'regular']
SHIPPING_SPEEDS  = ['fast shipping', 'regular shipping', 'bad shipping', 'soo bad shipping']
OPERATION_SPEEDS = ['perfect operation', 'regular operation', 'bad operation', 'soo bad operation']
EXPECTATIONS     = ['perfect expectation', 'bad expectation']
SELLER_LEVELS    = ['good seller', 'bad seller']
MONTHS           = ['January','February','March','April','May','June',
                    'July','August','September','October','November','December']

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="olist-header">
  <h1>🛒 Olist Review Predictor</h1>
  <p>Brazilian E-Commerce · XGBoost + SMOTE · Binary Classification: Good Review vs Bad Review</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.warning("⚠️ **Model file not found.** Place `best olist model` (your pickle file) next to `app.py` and restart.")

# ── KPI cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.markdown('<div class="kpi"><div class="v">87.4%</div><div class="l">Test Accuracy</div></div>', unsafe_allow_html=True)
with k2: st.markdown('<div class="kpi"><div class="v">91.9%</div><div class="l">Train Accuracy</div></div>', unsafe_allow_html=True)
with k3: st.markdown('<div class="kpi"><div class="v">XGBoost</div><div class="l">Algorithm</div></div>', unsafe_allow_html=True)
with k4: st.markdown('<div class="kpi"><div class="v">2 Classes</div><div class="l">Good / Bad Review</div></div>', unsafe_allow_html=True)
with k5: st.markdown('<div class="kpi"><div class="v">115K</div><div class="l">Training Rows</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📋 Enter Order Details")

    st.markdown('<p class="sec">📦 Order</p>', unsafe_allow_html=True)
    order_status         = st.selectbox("Order status", ORDER_STATUS)
    order_item_id        = st.number_input("Number of items", min_value=1, max_value=21, value=1, step=1)
    order_purchase_month = st.selectbox("Purchase month", MONTHS)
    order_purchase_year  = st.selectbox("Purchase year", [2016, 2017, 2018])

    st.markdown('<p class="sec">💳 Price & Payment</p>', unsafe_allow_html=True)
    price                = st.number_input("Item price (R$)", min_value=0.01, max_value=7000.0, value=120.0, step=10.0)
    freight_value        = st.number_input("Freight value (R$)", min_value=0.0, max_value=400.0, value=18.0, step=1.0)
    payment_type         = st.selectbox("Payment type", PAYMENT_TYPES)
    payment_sequential   = st.number_input("Payment sequential", min_value=1, max_value=29, value=1, step=1)
    payment_installments = st.slider("Installments", 0, 24, 1)
    payment_value        = st.number_input("Payment value (R$)", min_value=0.0, max_value=14000.0,
                                           value=round(price + freight_value, 2), step=5.0)

    st.markdown('<p class="sec">📦 Product</p>', unsafe_allow_html=True)
    product_category  = st.selectbox("Product category", CATEGORIES)
    product_weight_g  = st.number_input("Weight (g)", min_value=0.0, max_value=40000.0, value=500.0, step=50.0)
    product_height_cm = st.number_input("Height (cm)", min_value=0.0, max_value=105.0, value=15.0, step=1.0)
    product_width_cm  = st.number_input("Width (cm)", min_value=0.0, max_value=105.0, value=15.0, step=1.0)

    st.markdown('<p class="sec">🚚 Delivery & Seller</p>', unsafe_allow_html=True)
    customer_type      = st.selectbox("Customer type", CUSTOMER_TYPES)
    shipping_speed     = st.selectbox("Shipping speed", SHIPPING_SPEEDS)
    operation_speed    = st.selectbox("Operation speed", OPERATION_SPEEDS)
    actual_expectation = st.selectbox("Actual vs expectation", EXPECTATIONS)
    seller_level       = st.selectbox("Seller level", SELLER_LEVELS)
    distance           = st.slider("Distance customer ↔ seller (km)", 0.0, 4500.0, 300.0, step=10.0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Review")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("#### 📊 Input Summary")
    summary = pd.DataFrame({
        "Field": [
            "Order Status","Items","Purchase","Price","Freight",
            "Payment Type","Installments","Payment Value",
            "Category","Weight","Height × Width",
            "Customer Type","Shipping Speed","Operation Speed",
            "Expectation","Seller Level","Distance"
        ],
        "Value": [
            order_status, order_item_id,
            f"{order_purchase_month} {order_purchase_year}",
            f"R$ {price:,.2f}", f"R$ {freight_value:,.2f}",
            payment_type, payment_installments,
            f"R$ {payment_value:,.2f}",
            product_category, f"{product_weight_g:,.0f} g",
            f"{product_height_cm:.0f} × {product_width_cm:.0f} cm",
            customer_type, shipping_speed, operation_speed,
            actual_expectation, seller_level, f"{distance:,.0f} km"
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True, height=630)

with right:
    st.markdown("#### 🎯 Prediction Result")

    if predict_btn:
        if model is None:
            st.error("Model file not loaded. See warning at top of page.")
        else:
            input_df = pd.DataFrame([{
                "order status":                         order_status,
                "order item id":                        order_item_id,
                "price":                                price,
                "freight value":                        freight_value,
                "payment sequential":                   payment_sequential,
                "payment type":                         payment_type,
                "payment installments":                 payment_installments,
                "payment value":                        payment_value,
                "product weight g":                     product_weight_g,
                "product height cm":                    product_height_cm,
                "product width cm":                     product_width_cm,
                "product category name english":        product_category,
                "customer type":                        customer_type,
                "shipping speed":                       shipping_speed,
                "operation speed":                      operation_speed,
                "actuall expectation":                  actual_expectation,
                "seller level":                         seller_level,
                "distance between customer and seller": distance,
                "order purchase month":                 order_purchase_month,
                "order purchase year":                  order_purchase_year,
            }])

            try:
                # ── Convert everything to plain Python types immediately ─────
                raw_pred = model.predict(input_df)[0]
                proba    = [float(p) for p in model.predict_proba(input_df)[0]]
                classes  = [str(c) for c in model.classes_]   # pure Python str, no numpy.str_

                # ── Normalize prediction to plain lowercase string ──────────
                pred_label = str(raw_pred).strip().lower()
                is_good    = "good" in pred_label

                # ── Result box ───────────────────────────────────────────────
                if is_good:
                    box_cls = "rbox-good"
                    color   = "c-good"
                    emoji   = "✅"
                    label   = "Good Review"
                    meaning = "Customer is likely satisfied (score 4–5)"
                else:
                    box_cls = "rbox-bad"
                    color   = "c-bad"
                    emoji   = "❌"
                    label   = "Bad Review"
                    meaning = "Customer is likely unsatisfied (score 1–3)"

                st.markdown(f"""
                <div class="rbox {box_cls}">
                  <div class="big {color}">{emoji}</div>
                  <div class="lbl {color}">{label}</div>
                  <div class="sub">{meaning}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Probability bars ─────────────────────────────────────────
                # Detect which index is good vs bad from class names.
                # Handles: text labels, 0/1 ints, or any unknown format.
                st.markdown("<br>**Probability breakdown:**", unsafe_allow_html=True)

                def _class_is_good(name):
                    n = str(name).strip().lower()
                    if any(x in n for x in ("good", "1", "positive")):
                        return True
                    if any(x in n for x in ("bad", "0", "negative")):
                        return False
                    return None

                flags = [_class_is_good(c) for c in classes]
                # fallback: if detection unclear, first class=bad, second=good
                if None in flags or len(set(flags)) != 2:
                    flags = [i > 0 for i in range(len(classes))]

                for flag, prob in zip(flags, proba):
                    if flag:
                        disp, bar_cls = "✅ Good Review", "prob-bar-good"
                    else:
                        disp, bar_cls = "❌ Bad Review",  "prob-bar-bad"
                    pct   = round(prob * 100, 1)
                    bar_w = int(pct)
                    st.markdown(f"""
                    <div class="prob-row">
                      <span class="prob-label">{disp}</span>
                      <div class="prob-bar-bg">
                        <div class="{bar_cls}" style="width:{bar_w}%"></div>
                      </div>
                      <span class="prob-pct">{pct}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Confidence ───────────────────────────────────────────────
                conf = round(max(proba) * 100, 1)
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric("Model Confidence", f"{conf}%",
                          delta="High ▲" if conf >= 70 else ("Medium" if conf >= 50 else "Low ▼"))

                # ── Business recommendation ──────────────────────────────────
                if is_good:
                    st.markdown("""
                    <div class="tip">
                    ✅ <strong>Happy customer!</strong> Good candidate for re-marketing,
                    loyalty program, and upsell campaigns.
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="tip-bad">
                    ⚠️ <strong>At-risk customer!</strong> Consider proactive support,
                    a compensation voucher, or a follow-up message before they churn.
                    </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.code(str(e))

    else:
        st.markdown("""
        <div class="empty-state">
          <div style="font-size:3rem;margin-bottom:.8rem">🔮</div>
          Fill in the order details in the sidebar<br>
          then click <strong>Predict Review</strong>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="tip" style="margin-top:1.2rem">
        💡 <strong>Target:</strong> Binary classification —
        <strong>Good Review</strong> (score 4–5) vs <strong>Bad Review</strong> (score 1–3).
        The strongest predictors are <em>actual vs expectation</em>,
        <em>shipping speed</em>, and <em>seller level</em>.
        </div>
        """, unsafe_allow_html=True)

# ── Architecture ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🧠 Model Architecture")
a1, a2, a3 = st.columns(3)

with a1:
    st.markdown("""<div class="arch-card"><h4>⚙️ Preprocessing</h4><ul>
    <li>Cat NULLs → SimpleImputer + OHE</li>
    <li>Num NULLs → KNNImputer + RobustScaler</li>
    <li>Clean cat → OneHotEncoder</li>
    <li>Clean num → RobustScaler</li>
    <li>Remainder → passthrough</li>
    </ul></div>""", unsafe_allow_html=True)

with a2:
    st.markdown("""<div class="arch-card"><h4>🤖 XGBClassifier</h4><ul>
    <li>n_estimators = 700</li>
    <li>learning_rate = 0.1</li>
    <li>max_depth = 7</li>
    <li>reg_alpha = 1.5</li>
    <li>reg_lambda = 2.5</li>
    <li>Balancing: SMOTE</li>
    </ul></div>""", unsafe_allow_html=True)

with a3:
    st.markdown("""<div class="arch-card"><h4>📈 Results</h4><ul>
    <li>Train avg: 91.9%</li>
    <li>Test avg: 87.4%</li>
    <li>Classes: Good Review / Bad Review</li>
    <li>Features: 20 columns</li>
    <li>Rows: 115,037</li>
    <li>Folds: 5-fold CV</li>
    </ul></div>""", unsafe_allow_html=True)

st.markdown('<br><div style="text-align:center;color:#bbb;font-size:.78rem">Olist Brazilian E-Commerce · Binary Review Classifier · Streamlit</div>', unsafe_allow_html=True)
