"""
AI-Based Food Nutrition Assistant
---------------------------------
Streamlit app that:
  1. Calculates personalised daily calorie & macro targets (BMR / TDEE)
  2. Analyses a food photo with OpenCV (dominant-colour + texture matching)
  3. Lets the user confirm the food, set the portion and log it
  4. Shows a Plotly dashboard of intake vs. targets with smart suggestions
  5. Provides a food explorer / comparison tool
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Food Nutrition Assistant", page_icon="🥗", layout="wide")

DATA_PATH = Path(__file__).parent / "nutrition_data.csv"
NUTRIENTS = ["calories", "protein", "carbs", "fat", "fiber", "sugar"]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Pre-compute the CIELAB colour of each food's typical appearance
    rgb = df[["r", "g", "b"]].to_numpy(dtype=np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    df[["L", "a", "b_"]] = lab
    return df


FOODS = load_data()


# ----------------------------------------------------------------------------
# Nutrition maths
# ----------------------------------------------------------------------------
def calc_targets(age, sex, height_cm, weight_kg, activity, goal):
    """Mifflin-St Jeor BMR -> TDEE -> goal-adjusted calories + macro split."""
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if sex == "Male" else -161)
    factors = {"Sedentary": 1.2, "Light": 1.375, "Moderate": 1.55, "Active": 1.725, "Very active": 1.9}
    tdee = bmr * factors[activity]
    adj = {"Lose weight": -500, "Maintain weight": 0, "Gain muscle": 400}[goal]
    calories = max(1200, tdee + adj)
    split = {"Lose weight": (0.30, 0.40, 0.30),
             "Maintain weight": (0.20, 0.50, 0.30),
             "Gain muscle": (0.25, 0.50, 0.25)}[goal]
    return {
        "bmr": bmr, "tdee": tdee, "calories": calories,
        "protein": calories * split[0] / 4,
        "carbs": calories * split[1] / 4,
        "fat": calories * split[2] / 9,
        "fiber": 30.0,
        "sugar": 50.0,
    }


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def nutrition_for(food_row, grams):
    f = grams / 100.0
    return {n: round(float(food_row[n]) * f, 1) for n in NUTRIENTS}


# ----------------------------------------------------------------------------
# Computer-vision module (OpenCV)
# ----------------------------------------------------------------------------
def preprocess(pil_img: Image.Image) -> np.ndarray:
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    scale = 320 / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(img, (5, 5), 0)


def food_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Rough foreground mask: drop very bright, low-saturation pixels (plate/table)
    and very dark pixels (shadows). Falls back to all pixels if too little remains."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    mask = ~((s < 35) & (v > 200)) & (v > 40)
    if mask.mean() < 0.15:
        mask = np.ones_like(mask, dtype=bool)
    return mask


def dominant_colors(img_rgb: np.ndarray, mask: np.ndarray, k: int = 3):
    pixels = img_rgb[mask].astype(np.float32)
    if len(pixels) > 6000:
        pixels = pixels[np.random.default_rng(0).choice(len(pixels), 6000, replace=False)]
    k = min(k, len(pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    weights = np.bincount(labels.flatten(), minlength=k) / len(labels)
    order = np.argsort(-weights)
    return centers[order].astype(np.uint8), weights[order]


def texture_score(img_rgb: np.ndarray) -> float:
    """Edge density (0-1): crude measure of how 'busy' the food looks."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float((cv2.Canny(gray, 80, 160) > 0).mean())


def classify(img_rgb: np.ndarray, top_n: int = 5):
    """Rank foods by perceptual colour distance to the image's dominant colours."""
    mask = food_mask(img_rgb)
    centers, weights = dominant_colors(img_rgb, mask)
    lab_c = cv2.cvtColor(centers.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    food_lab = FOODS[["L", "a", "b_"]].to_numpy(dtype=np.float32)

    # distance of every food to every dominant colour, weighted by cluster size
    dists = np.linalg.norm(food_lab[:, None, :] - lab_c[None, :, :], axis=2)
    score = (dists * weights[None, :]).sum(axis=1)

    # soft-max -> pseudo-confidence
    conf = np.exp(-score / 12.0)
    conf = conf / conf.sum()
    idx = np.argsort(-conf)[:top_n]
    result = FOODS.iloc[idx].copy()
    result["confidence"] = conf[idx]
    return result, centers, weights


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "profile" not in st.session_state:
    st.session_state.profile = None
if "log" not in st.session_state:
    st.session_state.log = []


def add_to_log(food_row, grams, meal):
    entry = {"meal": meal, "food": food_row["food"], "grams": grams, **nutrition_for(food_row, grams)}
    st.session_state.log.append(entry)


def log_df() -> pd.DataFrame:
    cols = ["meal", "food", "grams"] + NUTRIENTS
    return pd.DataFrame(st.session_state.log, columns=cols)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🥗 AI-Based Food Nutrition Assistant")
st.caption("Scan your meal, track calories & macros, and get personalised suggestions.")

tab_profile, tab_scan, tab_manual, tab_dash, tab_explore = st.tabs(
    ["👤 Profile & Goals", "📷 Scan Food", "➕ Add Manually", "📊 Daily Dashboard", "🔍 Food Explorer"]
)

# ----------------------------------------------------------------------------
# Tab 1: Profile
# ----------------------------------------------------------------------------
with tab_profile:
    st.subheader("Tell us about yourself")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 10, 100, 21)
    sex = c1.selectbox("Sex", ["Male", "Female"])
    height = c2.number_input("Height (cm)", 120, 220, 170)
    weight = c2.number_input("Weight (kg)", 30.0, 200.0, 65.0, step=0.5)
    activity = c3.selectbox("Activity level", ["Sedentary", "Light", "Moderate", "Active", "Very active"], index=1)
    goal = c3.selectbox("Goal", ["Lose weight", "Maintain weight", "Gain muscle"], index=1)

    if st.button("Calculate my targets", type="primary"):
        st.session_state.profile = calc_targets(age, sex, height, weight, activity, goal)
        st.session_state.profile["bmi"] = weight / (height / 100) ** 2

    p = st.session_state.profile
    if p:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("BMI", f"{p['bmi']:.1f}", bmi_category(p["bmi"]))
        m2.metric("BMR (kcal)", f"{p['bmr']:.0f}")
        m3.metric("TDEE (kcal)", f"{p['tdee']:.0f}")
        m4.metric("Daily target (kcal)", f"{p['calories']:.0f}")

        fig = px.pie(
            names=["Protein", "Carbs", "Fat"],
            values=[p["protein"] * 4, p["carbs"] * 4, p["fat"] * 9],
            hole=0.5, title="Recommended calorie split",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Targets: **{p['protein']:.0f} g protein · {p['carbs']:.0f} g carbs · "
                f"{p['fat']:.0f} g fat · {p['fiber']:.0f} g fibre**")
    else:
        st.info("Fill in your details and press **Calculate my targets**.")

# ----------------------------------------------------------------------------
# Tab 2: Scan food
# ----------------------------------------------------------------------------
with tab_scan:
    st.subheader("Analyse a food photo")
    st.warning(
        "⚠️ This demo recognises food using **colour and texture analysis** (OpenCV), "
        "not a trained deep-learning model, so always confirm the suggestion below."
    )
    src = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
    file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"]) if src == "Upload" \
        else st.camera_input("Take a photo")

    if file:
        pil = Image.open(file)
        img = preprocess(pil)
        suggestions, centers, weights = classify(img)
        tex = texture_score(img)

        left, right = st.columns([1, 1])
        with left:
            st.image(pil, caption="Your meal", use_container_width=True)
            st.markdown("**Dominant colours**")
            swatch = np.zeros((40, 300, 3), dtype=np.uint8)
            x = 0
            for c, w in zip(centers, weights):
                width = max(1, int(300 * w))
                swatch[:, x:x + width] = c
                x += width
            st.image(swatch, use_container_width=True)
            st.caption(f"Texture / edge density: {tex:.3f}")

        with right:
            st.markdown("**Top matches**")
            fig = px.bar(suggestions.iloc[::-1], x="confidence", y="food", orientation="h",
                         labels={"confidence": "Match score", "food": ""})
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

            options = suggestions["food"].tolist()
            all_foods = FOODS["food"].tolist()
            choice = st.selectbox("Confirm food (or pick the correct one)",
                                  options + [f for f in all_foods if f not in options])
            row = FOODS[FOODS["food"] == choice].iloc[0]
            grams = st.slider("Portion (grams)", 10, 600, int(row["serving_g"]), step=5, key="scan_g")
            meal = st.selectbox("Meal", ["Breakfast", "Lunch", "Snack", "Dinner"], key="scan_meal")

            n = nutrition_for(row, grams)
            cols = st.columns(3)
            cols[0].metric("Calories", f"{n['calories']:.0f} kcal")
            cols[1].metric("Protein", f"{n['protein']} g")
            cols[2].metric("Carbs", f"{n['carbs']} g")
            cols = st.columns(3)
            cols[0].metric("Fat", f"{n['fat']} g")
            cols[1].metric("Fibre", f"{n['fiber']} g")
            cols[2].metric("Sugar", f"{n['sugar']} g")

            if st.button("➕ Add to today's log", key="scan_add"):
                add_to_log(row, grams, meal)
                st.success(f"Added {grams} g of {choice} to {meal}.")

# ----------------------------------------------------------------------------
# Tab 3: Manual add
# ----------------------------------------------------------------------------
with tab_manual:
    st.subheader("Search and add food")
    cat = st.multiselect("Filter by category", sorted(FOODS["category"].unique()))
    pool = FOODS if not cat else FOODS[FOODS["category"].isin(cat)]
    choice = st.selectbox("Food", pool["food"].tolist(), key="man_food")
    row = FOODS[FOODS["food"] == choice].iloc[0]
    c1, c2 = st.columns(2)
    grams = c1.number_input("Portion (grams)", 5, 1500, int(row["serving_g"]), step=5, key="man_g")
    meal = c2.selectbox("Meal", ["Breakfast", "Lunch", "Snack", "Dinner"], key="man_meal")
    n = nutrition_for(row, grams)
    st.write(f"**{n['calories']:.0f} kcal** · P {n['protein']} g · C {n['carbs']} g · F {n['fat']} g")
    if st.button("➕ Add to today's log", key="man_add"):
        add_to_log(row, grams, meal)
        st.success("Added!")

# ----------------------------------------------------------------------------
# Tab 4: Dashboard
# ----------------------------------------------------------------------------
with tab_dash:
    df = log_df()
    if df.empty:
        st.info("Your log is empty. Scan or add a food first.")
    else:
        totals = df[NUTRIENTS].sum()
        p = st.session_state.profile
        target = {k: (p[k] if p else d) for k, d in
                  zip(NUTRIENTS, [2000, 75, 250, 65, 30, 50])}

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Calories", f"{totals['calories']:.0f}", f"{totals['calories'] - target['calories']:+.0f} vs target")
        k2.metric("Protein", f"{totals['protein']:.0f} g")
        k3.metric("Carbs", f"{totals['carbs']:.0f} g")
        k4.metric("Fat", f"{totals['fat']:.0f} g")

        a, b = st.columns(2)
        with a:
            pct = [min(totals[n] / target[n] * 100, 150) for n in ["calories", "protein", "carbs", "fat", "fiber"]]
            fig = go.Figure(go.Bar(
                x=["Calories", "Protein", "Carbs", "Fat", "Fibre"], y=pct,
                marker_color=["#2ecc71" if 80 <= v <= 110 else "#f39c12" if v < 80 else "#e74c3c" for v in pct],
                text=[f"{v:.0f}%" for v in pct], textposition="outside"))
            fig.add_hline(y=100, line_dash="dash")
            fig.update_layout(title="Progress vs daily target (%)", yaxis_range=[0, 170])
            st.plotly_chart(fig, use_container_width=True)
        with b:
            fig = px.pie(names=["Protein", "Carbs", "Fat"],
                         values=[totals["protein"] * 4, totals["carbs"] * 4, totals["fat"] * 9],
                         hole=0.5, title="Where your calories come from")
            st.plotly_chart(fig, use_container_width=True)

        meal_cal = df.groupby("meal")["calories"].sum().reset_index()
        st.plotly_chart(px.bar(meal_cal, x="meal", y="calories", title="Calories by meal", color="meal"),
                        use_container_width=True)

        st.markdown("#### 🧠 Smart suggestions")
        tips = []
        if totals["calories"] > target["calories"] * 1.1:
            tips.append("You've gone over your calorie target. Choose light options like salad or fruit next.")
        if totals["protein"] < target["protein"] * 0.7:
            tips.append("Protein is low. Add eggs, dal, paneer, curd or chicken.")
        if totals["fiber"] < target["fiber"] * 0.5:
            tips.append("Fibre is low. Add vegetables, fruits, chickpeas or whole grains.")
        if totals["sugar"] > target["sugar"]:
            tips.append("Sugar intake is high. Cut back on sweets and sugary drinks.")
        if totals["fat"] > target["fat"] * 1.2:
            tips.append("Fat is high. Limit fried foods and cheese.")
        if not tips:
            tips.append("Great balance so far. Keep it up! 🎉")
        for t in tips:
            st.write("• " + t)

        remaining = target["calories"] - totals["calories"]
        if remaining > 50:
            fits = FOODS.assign(kcal_serv=FOODS["calories"] * FOODS["serving_g"] / 100)
            fits = fits[fits["kcal_serv"] <= remaining].sort_values("protein", ascending=False).head(5)
            if not fits.empty:
                st.markdown(f"**Foods that fit your remaining {remaining:.0f} kcal (high protein first):**")
                st.dataframe(fits[["food", "category", "serving_g", "kcal_serv", "protein"]]
                             .rename(columns={"kcal_serv": "kcal / serving"}), hide_index=True)

        st.markdown("#### Today's log")
        st.dataframe(df, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Download log (CSV)", df.to_csv(index=False), "food_log.csv", "text/csv")
        if c2.button("🗑️ Clear log"):
            st.session_state.log = []
            st.rerun()

# ----------------------------------------------------------------------------
# Tab 5: Explorer
# ----------------------------------------------------------------------------
with tab_explore:
    st.subheader("Explore & compare foods (per 100 g)")
    x_axis = st.selectbox("X axis", NUTRIENTS, index=2)
    y_axis = st.selectbox("Y axis", NUTRIENTS, index=1)
    fig = px.scatter(FOODS, x=x_axis, y=y_axis, color="category", hover_name="food",
                     size="calories", size_max=30, title=f"{y_axis.title()} vs {x_axis.title()}")
    st.plotly_chart(fig, use_container_width=True)

    picks = st.multiselect("Compare foods", FOODS["food"].tolist(), default=["Chicken Breast (grilled)", "Paneer", "Dal (cooked)"])
    if picks:
        comp = FOODS[FOODS["food"].isin(picks)].melt(id_vars="food", value_vars=["protein", "carbs", "fat", "fiber"],
                                                      var_name="nutrient", value_name="grams")
        st.plotly_chart(px.bar(comp, x="food", y="grams", color="nutrient", barmode="group"),
                        use_container_width=True)

    st.dataframe(FOODS[["food", "category"] + NUTRIENTS + ["serving_g"]], use_container_width=True, hide_index=True)

st.divider()
st.caption("Educational project. Nutrition values are approximate and not medical advice.")