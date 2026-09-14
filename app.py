import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from google import genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

st.set_page_config(
    page_title="Zomato PersonalAI",
    page_icon="🍽️",
    layout="wide",
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

@st.cache_data

def load_data():
    customers = pd.read_csv(DATA / "customers.csv")
    restaurants = pd.read_csv(DATA / "restaurants.csv")
    orders = pd.read_csv(DATA / "orders.csv")
    profiles = pd.read_csv(DATA / "customer_profiles.csv")
    interactions = pd.read_csv(DATA / "interactions.csv")
    return customers, restaurants, orders, profiles, interactions

customers, restaurants, orders, profiles, interactions = load_data()


def minmax(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    if x.max() == x.min():
        return np.ones_like(x) * 0.5
    return (x - x.min()) / (x.max() - x.min())


def recommend(customer_id, max_budget=600, cuisine=None, spicy=None, new_only=False, healthy=False, top_n=5):
    c = customers[customers.customer_id == customer_id].iloc[0]
    rs = restaurants[restaurants.city == c.city].copy()

    rs = rs[rs.average_order_value <= max_budget]
    if cuisine:
        rs = rs[(rs.cuisine.str.lower() == cuisine.lower()) | (rs.secondary_cuisine.str.lower() == cuisine.lower())]
    if spicy is True:
        rs = rs[rs.spice_score >= 0.55]
    if healthy is True:
        rs = rs[rs.healthy_score >= 0.55]

    if new_only:
        tried = set(orders.loc[orders.customer_id == customer_id, "restaurant_id"])
        rs = rs[~rs.restaurant_id.isin(tried)]

    if rs.empty:
        return rs

    cuisine_match = np.where(
        rs.cuisine.eq(c.preferred_cuisine), 1.0,
        np.where(rs.cuisine.eq(c.secondary_cuisine), 0.6, 0.15),
    )
    budget_match = 1 - np.minimum(np.abs(rs.average_order_value - c.avg_order_value) / max(c.avg_order_value, 1), 1)
    healthy_match = (rs.healthy_score * c.healthy_food_affinity) + (1 - c.healthy_food_affinity) * 0.5
    spicy_match = (rs.spice_score * c.spicy_food_affinity) + (1 - c.spicy_food_affinity) * 0.5
    quality = minmax(rs.rating)
    speed = 1 - minmax(rs.delivery_time_min)
    exploration = rs.popularity_score * (1 - c.restaurant_loyalty) + rs.new_restaurant * c.new_restaurant_affinity

    rs["preference_score"] = 0.30 * cuisine_match
    rs["behaviour_score"] = 0.20 * (0.55 * healthy_match + 0.45 * spicy_match)
    rs["context_score"] = 0.15 * speed
    rs["quality_score"] = 0.15 * quality
    rs["value_score"] = 0.10 * budget_match
    rs["exploration_score"] = 0.10 * exploration
    rs["final_score"] = (
        rs.preference_score + rs.behaviour_score + rs.context_score +
        rs.quality_score + rs.value_score + rs.exploration_score
    )
    return rs.sort_values("final_score", ascending=False).head(top_n)


def fallback_intent(message):
    text = message.lower()
    budget = None
    match = re.search(r"(?:under|below|less than|within)\s*[₹rs.]*\s*(\d{2,5})", text)
    if match:
        budget = int(match.group(1))
    elif "500" in text:
        budget = 500

    cuisine = None
    cuisine_options = sorted(set(customers.preferred_cuisine.dropna().tolist()) | set(restaurants.cuisine.dropna().tolist()))
    for item in cuisine_options:
        if item.lower() in text:
            cuisine = item
            break

    spicy = any(x in text for x in ["spicy", "spice", "hot"])
    healthy = any(x in text for x in ["healthy", "light", "low calorie", "salad"])
    new_only = any(x in text for x in ["new", "different", "bored", "same places", "something else"])
    return {
        "budget_inr": budget,
        "cuisine": cuisine,
        "spicy": spicy,
        "new_only": new_only,
        "healthy": healthy,
        "party_size": None,
        "occasion": None,
    }


def get_gemini_client():
    if not GEMINI_AVAILABLE:
        return None
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        return None
    return genai.Client(api_key=key)


def extract_intent(message):
    client = get_gemini_client()
    if client is None:
        return fallback_intent(message), False

    prompt = f"""
You are the intent extraction layer for a food recommendation prototype.
Return ONLY valid JSON with exactly these keys:
 budget_inr, cuisine, spicy, new_only, healthy, party_size, occasion

Rules:
- budget_inr is an integer or null.
- cuisine is a short cuisine name or null.
- spicy, new_only, healthy are booleans.
- party_size is an integer or null.
- occasion is a short string or null.
- If the user says they are bored of the same places, set new_only=true.
- Do not invent information not present in the message.

User message: {message}
"""
    try:
        response = client.models.generate_content(model="gemini-3.7-flash", contents=prompt)
        raw = response.text.strip()
        raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw).strip()
        data = json.loads(raw)
        return data, True
    except Exception as exc:
        st.warning(f"Gemini intent extraction was unavailable, so the demo parser was used. ({type(exc).__name__})")
        return fallback_intent(message), False


def explain_results(message, intent, customer, result_df):
    if result_df.empty:
        return "I could not find a match with those constraints. Try a slightly higher budget, remove the 'new only' filter, or change the cuisine."

    client = get_gemini_client()
    rows = result_df[["restaurant_name", "cuisine", "rating", "delivery_time_min", "average_order_value", "final_score"]].round(3).to_dict("records")

    if client is not None:
        prompt = f"""
You are the explanation layer for a food recommendation prototype.
Do NOT change the ranking. Do NOT invent restaurant facts.
Explain only from the supplied customer profile, user request, and ranked results.
Keep it concise and customer-friendly.

User request: {message}
Parsed intent: {json.dumps(intent)}
Customer profile: {json.dumps({k: customer[k] for k in ['customer_id','preferred_cuisine','avg_order_value','orders_per_month','discount_sensitivity','restaurant_loyalty','new_restaurant_affinity'] if k in customer})}
Ranked results: {json.dumps(rows)}

Output:
1) one-sentence opening;
2) three numbered reasons for the top recommendation;
3) one short closing suggestion.
"""
        try:
            response = client.models.generate_content(model="gemini-3.7-flash", contents=prompt)
            return response.text.strip()
        except Exception:
            pass

    top = result_df.iloc[0]
    reasons = []
    if intent.get("new_only"):
        reasons.append("it introduces a restaurant outside the customer's observed order history")
    if intent.get("spicy"):
        reasons.append("it meets the spicy-food filter")
    if intent.get("healthy"):
        reasons.append("it meets the healthy-food filter")
    if intent.get("budget_inr"):
        reasons.append(f"its average order value is within the ₹{intent['budget_inr']} budget")
    reasons.append(f"it scores well on the personalized ranking model")
    return "Top match selected from the deterministic recommendation engine.\n\n" + "\n".join(f"{i+1}. {r}" for i, r in enumerate(reasons[:3])) + "\n\nTry it or ask for something different."


def safe_params(intent, customer_id):
    c = customers[customers.customer_id == customer_id].iloc[0]
    budget = intent.get("budget_inr")
    if budget is None:
        budget = int(max(300, min(2000, c.avg_order_value * 1.25)))
    budget = int(max(150, min(2000, budget)))
    return {
        "max_budget": budget,
        "cuisine": intent.get("cuisine"),
        "spicy": True if intent.get("spicy") else None,
        "new_only": bool(intent.get("new_only")),
        "healthy": bool(intent.get("healthy")),
        "top_n": 5,
    }


st.title("🍽️ Zomato PersonalAI")
st.caption("AI-powered food discovery prototype • Synthetic / illustrative data • Academic demonstration")

if "prompt" not in st.session_state:
    st.session_state.prompt = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = []

with st.sidebar:
    st.header("Customer Context")
    customer_id = st.selectbox("Select synthetic customer", customers.customer_id.tolist(), index=0)
    c = customers[customers.customer_id == customer_id].iloc[0]
    p = profiles[profiles.customer_id == customer_id].iloc[0]
    st.metric("Segment", c.customer_segment)
    st.metric("Preferred cuisine", c.preferred_cuisine)
    st.metric("Avg. order value", f"₹{c.avg_order_value:,.0f}")
    st.metric("Orders / month", f"{c.orders_per_month:.1f}")
    st.divider()
    st.info("The prototype uses synthetic customer, restaurant, order and interaction data. It does not represent confidential Zomato data.")

    if get_gemini_client() is not None:
        st.success("Gemini AI: Connected")
    else:
        st.warning("Gemini AI: Demo fallback mode")

# Tabs
tab_ai, tab_analytics, tab_arch = st.tabs(["🤖 PersonalAI", "📊 Demo Analytics", "🧠 How It Works"])

with tab_ai:
    st.subheader(f"What are you craving, Customer {customer_id}?")
    st.write("Describe your need naturally. Example: *I want something spicy under ₹500 but I'm bored of eating the same places.*")

    quick = st.columns(4)
    quick_prompts = ["Something spicy under ₹500", "Something new", "Healthy food", "Something budget-friendly"]
    for col, qp in zip(quick, quick_prompts):
        if col.button(qp, use_container_width=True):
            st.session_state.prompt = qp

    user_message = st.text_area("Your request", value=st.session_state.prompt, height=90, placeholder="Tell PersonalAI what you want...")

    if st.button("✨ Get PersonalAI Recommendations", type="primary", use_container_width=True):
        if not user_message.strip():
            st.error("Please enter a food request first.")
        else:
            with st.spinner("Understanding intent → ranking restaurants → generating explanation..."):
                intent, ai_used = extract_intent(user_message)
                params = safe_params(intent, customer_id)
                results = recommend(customer_id, **params)
                explanation = explain_results(user_message, intent, c, results)

            st.session_state.last_intent = intent
            st.session_state.last_results = results
            st.session_state.last_explanation = explanation
            st.session_state.last_ai_used = ai_used

    if "last_results" in st.session_state:
        intent = st.session_state.last_intent
        results = st.session_state.last_results
        st.markdown("### 1. AI Intent Understanding")
        st.json(intent)
        st.caption("Gemini extracts intent; Python validates the fields before the recommendation engine runs.")

        st.markdown("### 2. Personalized Recommendations")
        if results.empty:
            st.warning("No restaurants matched all constraints. Try relaxing one filter.")
        else:
            for idx, row in results.head(5).reset_index(drop=True).iterrows():
                with st.container(border=True):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.markdown(f"#### {idx+1}. {row.restaurant_name}")
                        st.write(f"{row.cuisine} • ⭐ {row.rating:.1f} • {int(row.delivery_time_min)} min")
                    with col2:
                        st.metric("Avg. order", f"₹{row.average_order_value:,.0f}")
                    with col3:
                        st.metric("Match", f"{row.final_score*100:.1f}%")
                    st.caption("Personalized ranking combines preference, behaviour, context, quality, value and exploration signals.")
                    fb1, fb2, fb3 = st.columns(3)
                    if fb1.button("👍 Like", key=f"like_{idx}"):
                        st.session_state.feedback.append((customer_id, row.restaurant_id, "like"))
                        st.success("Feedback captured for the demo.")
                    if fb2.button("👎 Not for me", key=f"dislike_{idx}"):
                        st.session_state.feedback.append((customer_id, row.restaurant_id, "dislike"))
                        st.info("Feedback captured for the demo.")
                    if fb3.button("↻ Try something else", key=f"other_{idx}"):
                        st.session_state.prompt = "Show me something different and new"
                        st.rerun()

        st.markdown("### 3. AI Explanation")
        st.write(st.session_state.last_explanation)
        st.caption("Important: Generative AI explains the ranked results; it does not invent or reorder restaurants.")

with tab_analytics:
    st.subheader("Prototype KPI Dashboard")
    total_orders = len(orders)
    repeat_share = orders.repeat_order.mean() * 100
    avg_order = orders.order_value.mean()
    discount_rate = orders.discount_used.mean() * 100
    cols = st.columns(4)
    cols[0].metric("Synthetic orders", f"{total_orders:,}")
    cols[1].metric("Repeat-order share", f"{repeat_share:.1f}%")
    cols[2].metric("Average order value", f"₹{avg_order:,.0f}")
    cols[3].metric("Discount usage", f"{discount_rate:.1f}%")

    st.markdown("#### Interaction funnel — synthetic baseline")
    funnel = interactions.event_type.value_counts().rename_axis("event").reset_index(name="count")
    st.dataframe(funnel, use_container_width=True, hide_index=True)
    st.caption("These are synthetic baseline metrics for the academic prototype, not Zomato business KPIs.")

with tab_arch:
    st.subheader("PersonalAI Architecture")
    st.markdown("""
**Customer request**  
↓  
**Gemini intent extraction** — converts natural language into structured constraints  
↓  
**Python validation / safety layer** — clamps budget and validates fields  
↓  
**Deterministic hybrid recommendation engine** — ranks restaurants using customer + context signals  
↓  
**Gemini explanation layer** — explains the ranking without changing it  
↓  
**Feedback capture** — like / dislike / try something else  
↓  
**Future agentic layer** — preference update, offer trigger, re-engagement and CRM automation
""")
    st.markdown("### Why this architecture is academically strong")
    st.write("It separates probabilistic language understanding from deterministic recommendation logic, improving explainability, reproducibility and hallucination control.")
    st.markdown("### Privacy & governance")
    st.write("Use synthetic data for the assessment; never commit API keys; minimize data collection; provide an opt-out; avoid inferring sensitive attributes; monitor recommendation bias; log model decisions and feedback.")

st.divider()
st.caption("Zomato PersonalAI — Academic prototype | Synthetic data | Not an official Zomato product")
