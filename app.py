import streamlit as st
import pandas as pd
import joblib

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Shopaholic AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================================
# 2. CUSTOM DESIGN AND ANIMATIONS
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #100B20, #25143D, #321C49);
    color: white;
}

header {
    visibility: hidden;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #7928CA, #FF4F9A);
    padding: 40px 20px;
    border-radius: 25px;
    text-align: center;
    box-shadow: 0 10px 35px #00000055;
    margin-bottom: 25px;
}

.hero h1 {
    color: white !important;
    font-size: 48px;
    font-weight: 900;
}

.hero h2, .hero p {
    color: white !important;
}

.hero p {
    font-size: 17px;
}

.section {
    background: #FFFFFF12;
    border: 1px solid #FFFFFF25;
    padding: 22px;
    border-radius: 20px;
    margin-bottom: 18px;
}

div[data-testid="stMetric"] {
    background: #FFFFFF12;
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #FFFFFF25;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #FF4F9A, #A855F7);
    color: white;
    border: none;
    border-radius: 14px;
    min-height: 55px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    color: white;
    transform: scale(1.02);
    box-shadow: 0 5px 20px #FF4F9A88;
}

label, p, h2, h3 {
    color: white;
}

.result {
    background: linear-gradient(120deg, #7928CA, #FF4F9A);
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    margin-top: 25px;
}

.result h1, .result h2, .result p {
    color: white !important;
}

.shopping-rain {
    position: relative;
    height: 145px;
    overflow: hidden;
    margin: 10px 0;
}

.shopping-rain span {
    position: absolute;
    top: -60px;
    font-size: 35px;
    animation: shoppingFall 3s linear infinite;
}

.shopping-rain span:nth-child(1) {
    left: 5%;
    animation-delay: 0s;
}
.shopping-rain span:nth-child(2) {
    left: 20%;
    animation-delay: .5s;
}
.shopping-rain span:nth-child(3) {
    left: 35%;
    animation-delay: 1s;
}
.shopping-rain span:nth-child(4) {
    left: 50%;
    animation-delay: .3s;
}
.shopping-rain span:nth-child(5) {
    left: 65%;
    animation-delay: 1.2s;
}
.shopping-rain span:nth-child(6) {
    left: 80%;
    animation-delay: .7s;
}
.shopping-rain span:nth-child(7) {
    left: 95%;
    animation-delay: 1.5s;
}

@keyframes shoppingFall {
    0% {
        transform: translateY(0) rotate(0deg);
        opacity: 0;
    }
    15% {
        opacity: 1;
    }
    100% {
        transform: translateY(180px) rotate(360deg);
        opacity: 0;
    }
}

.empty-cart {
    text-align: center;
    font-size: 42px;
    padding: 20px;
    animation: cartMove 1.5s ease-in-out infinite alternate;
}

@keyframes cartMove {
    from { transform: translateX(-10px); }
    to { transform: translateX(10px); }
}

.footer {
    text-align: center;
    color: #D4BCEB;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD TRAINED RANDOM FOREST MODEL
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load("model/best_shopping_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.code(str(e))
    st.stop()


# =====================================================
# 4. HEADER
# =====================================================

st.markdown("""
<div class="hero">

<h1>🛍️ SHOPAHOLIC AI</h1>

<h2>Will You Buy It or Just Add to Cart? 😂</h2>

<p>
Your shopping behaviour has secrets.
Let Artificial Intelligence expose them!
</p>

</div>
""", unsafe_allow_html=True)


# =====================================================
# 5. DASHBOARD
# =====================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("🤖 AI MODEL", "Random Forest")

with c2:
    st.metric("🛒 PROJECT", "Purchase Predictor")

with c3:
    st.metric("⚡ SYSTEM", "Ready")


# =====================================================
# 6. INTRODUCTION
# =====================================================

st.write("")

st.markdown("""
<div class="section">

<h2>💜 Let's Discover Your Shopping Personality!</h2>

<p>
Enter the visitor's online shopping behaviour below.
Don't worry, we won't judge your shopping addiction! 😂
</p>

</div>
""", unsafe_allow_html=True)


# =====================================================
# 7. SHOPPING INPUT FORM
# =====================================================

with st.form("shopping_form"):

    # BROWSING BEHAVIOUR

    st.subheader("📱 Browsing Behaviour")

    col1, col2, col3 = st.columns(3)

    with col1:

        administrative = st.number_input(
            "Administrative Pages", 0, 100, 2
        )

        administrative_duration = st.number_input(
            "Administrative Duration", 0.0, 10000.0, 50.0
        )

    with col2:

        informational = st.number_input(
            "Informational Pages", 0, 100, 1
        )

        informational_duration = st.number_input(
            "Informational Duration", 0.0, 10000.0, 20.0
        )

    with col3:

        product_related = st.number_input(
            "Product Related Pages", 0, 500, 10
        )

        product_related_duration = st.number_input(
            "Product Related Duration", 0.0, 50000.0, 300.0
        )

    st.divider()

    # SHOPPING ACTIVITY

    st.subheader("📊 Shopping Activity")

    col4, col5, col6 = st.columns(3)

    with col4:

        bounce_rates = st.number_input(
            "Bounce Rates", 0.0, 1.0, 0.02
        )

        exit_rates = st.number_input(
            "Exit Rates", 0.0, 1.0, 0.04
        )

    with col5:

        page_values = st.number_input(
            "Page Values", 0.0, 10000.0, 10.0
        )

        special_day = st.slider(
            "Special Day", 0.0, 1.0, 0.0
        )

    with col6:

        month = st.selectbox(
            "Shopping Month",
            [
                "Feb", "Mar", "May", "June", "Jul",
                "Aug", "Sep", "Oct", "Nov", "Dec"
            ]
        )

        weekend = st.selectbox(
            "Weekend Shopping?",
            [False, True],
            format_func=lambda x: "Yes" if x else "No"
        )

    st.divider()

    # VISITOR DETAILS

    st.subheader("🌐 Visitor Details")

    col7, col8, col9 = st.columns(3)

    with col7:

        operating_system = st.selectbox(
            "Operating System", list(range(1, 9))
        )

    with col8:

        browser = st.selectbox(
            "Browser", list(range(1, 14))
        )

    with col9:

        region = st.selectbox(
            "Region", list(range(1, 10))
        )

    traffic_type = st.selectbox(
        "Traffic Type", list(range(1, 21))
    )

    visitor_type = st.selectbox(
        "Visitor Type",
        [
            "Returning_Visitor",
            "New_Visitor",
            "Other"
        ]
    )

    st.write("")

    predict_button = st.form_submit_button(
        "🔮 REVEAL MY SHOPPING DESTINY!",
        use_container_width=True
    )


# =====================================================
# 8. PREDICTION
# =====================================================

if predict_button:

    input_data = pd.DataFrame([{

        "Administrative": administrative,
        "Administrative_Duration": administrative_duration,
        "Informational": informational,
        "Informational_Duration": informational_duration,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_related_duration,
        "BounceRates": bounce_rates,
        "ExitRates": exit_rates,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": month,
        "OperatingSystems": operating_system,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic_type,
        "VisitorType": visitor_type,
        "Weekend": weekend

    }])

    try:

        # PREPARE INPUT FOR THE SAVED MODEL

        expected_features = getattr(
            model, "feature_names_in_", None
        )

        if expected_features is not None:

            expected_features = list(expected_features)

            if set(expected_features) == set(input_data.columns):

                prediction_input = input_data[expected_features]

            else:

                encoded_input = pd.get_dummies(input_data)

                prediction_input = encoded_input.reindex(
                    columns=expected_features,
                    fill_value=0
                )

        else:

            prediction_input = input_data

        # MAKE PREDICTION

        prediction = model.predict(prediction_input)[0]

        positive = str(prediction).strip().lower() in [
            "1", "true", "yes"
        ]


        # =================================================
        # 9. PURCHASE LIKELY RESULT
        # =================================================

        if positive:

            st.markdown("""
            <div class="result">

            <h1>🎉 PURCHASE LIKELY!</h1>

            <h2>Your Shopping Cart Is in Danger!</h2>

            <p>
            Congratulations, future customer!
            Our AI thinks you may complete a purchase.
            </p>

            <h1>💸 💳 🛍️</h1>

            </div>
            """, unsafe_allow_html=True)

            # SHOPPING EMOJI RAIN

            st.markdown("""
            <div class="shopping-rain">
                <span>🛍️</span>
                <span>💸</span>
                <span>🛒</span>
                <span>💳</span>
                <span>🎁</span>
                <span>👠</span>
                <span>💰</span>
            </div>
            """, unsafe_allow_html=True)

            st.success(
                "Shopping mode activated! Your wallet needs protection! 😂"
            )


        # =================================================
        # 10. PURCHASE UNLIKELY RESULT
        # =================================================

        else:

            st.markdown("""
            <div class="result">

            <h1>😂 PURCHASE UNLIKELY!</h1>

            <h2>Window Shopping Expert!</h2>

            <p>
            You explored all the products,
            but our AI thinks you may leave without buying.
            </p>

            <h1>🛒 💨 😎</h1>

            </div>
            """, unsafe_allow_html=True)

            # EMPTY CART ANIMATION

            st.markdown("""
            <div class="empty-cart">
                🛒 💨 🏃‍♀️ 💰
            </div>

            <h3 style="text-align:center">
            Your wallet is safe today! 😂
            </h3>
            """, unsafe_allow_html=True)

            st.info(
                "No purchase predicted. Your bank account says THANK YOU! 💜"
            )


        # =================================================
        # 11. PURCHASE PROBABILITY
        # =================================================

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                prediction_input
            )[0]

            classes = list(model.classes_)

            positive_index = next(
                (
                    i for i, value in enumerate(classes)
                    if str(value).strip().lower()
                    in ["1", "true", "yes"]
                ),
                None
            )

            if positive_index is not None:

                probability = (
                    probabilities[positive_index] * 100
                )

                st.divider()

                st.subheader("📈 AI Purchase Probability")

                st.progress(float(probability / 100))

                st.metric(
                    "Probability of Purchase",
                    f"{probability:.2f}%"
                )


        # =================================================
        # 12. INPUT DETAILS
        # =================================================

        with st.expander("🔍 View Shopping Behaviour Details"):

            st.dataframe(
                input_data,
                use_container_width=True
            )

    except Exception as e:

        st.error("Prediction Error")

        st.code(str(e))

        st.info(
            "The model input format may differ from the "
            "training format. Check the preprocessing "
            "in train_model.py."
        )


# =====================================================
# 13. FOOTER
# =====================================================

st.markdown("""
<div class="footer">

<h3>💜 Thanks for Visiting SHOPAHOLIC AI!</h3>

<p>
Powered by Machine Learning | Random Forest Classifier
</p>

<p>
Made with ❤️ for smarter shopping predictions.
</p>

<p>
AI predictions are estimates, not guaranteed purchases.
</p>

</div>
""", unsafe_allow_html=True)