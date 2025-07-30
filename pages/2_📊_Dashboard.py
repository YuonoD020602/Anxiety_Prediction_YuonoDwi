import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(layout="wide")
st.title("📊 Anxiety Profiling Dashboard")

# Load dataset
df = pd.read_csv("data/anxiety_dataset.csv")

# --- Filter Controls ---
st.markdown("### 🔍 Filter Data")
col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

with col1:
    age_range = st.slider("Age", 18, 64, (18, 64))
with col2:
    gender = st.selectbox("Gender", ["All"] + sorted(df["Gender"].unique().tolist()))
with col3:
    occupation = st.selectbox("Occupation", ["All"] + sorted(df["Occupation"].unique().tolist()))
with col4:
    smoking = st.selectbox("Smoking", ["All"] + sorted(df["Smoking"].unique().tolist()))
with col5:
    family = st.selectbox("Family History", ["All"] + sorted(df["Family History of Anxiety"].unique().tolist()))
with col6:
    dizzy = st.selectbox("Dizziness", ["All"] + sorted(df["Dizziness"].unique().tolist()))
with col7:
    med = st.selectbox("Medication", ["All"] + sorted(df["Medication"].unique().tolist()))
with col8:
    event = st.selectbox("Major Event", ["All"] + sorted(df["Recent Major Life Event"].unique().tolist()))

# Apply filtering
filtered = df[df['Age'].between(*age_range)]
if gender != "All":
    filtered = filtered[filtered["Gender"] == gender]
if occupation != "All":
    filtered = filtered[filtered["Occupation"] == occupation]
if smoking != "All":
    filtered = filtered[filtered["Smoking"] == smoking]
if family != "All":
    filtered = filtered[filtered["Family History of Anxiety"] == family]
if dizzy != "All":
    filtered = filtered[filtered["Dizziness"] == dizzy]
if med != "All":
    filtered = filtered[filtered["Medication"] == med]
if event != "All":
    filtered = filtered[filtered["Recent Major Life Event"] == event]

# --- Summary Cards ---
st.markdown("### 📌 Summary")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total People", f"{len(filtered):,}")
col2.metric("Anxiety Level", f"{filtered['Anxiety Level (1-10)'].mean():.2f}")
col3.metric("Stress Level", f"{filtered['Stress Level (1-10)'].mean():.2f}")
col4.metric("Heart Rate", f"{filtered['Heart Rate (bpm)'].mean():.2f}")
col5.metric("Breathing Rate", f"{filtered['Breathing Rate (breaths/min)'].mean():.2f}")
col6.metric("Sweating Level", f"{filtered['Sweating Level (1-5)'].mean():.2f}")

# --- Main Charts ---
st.markdown("### 📈 Visualizations")

# Row 1: Distribution & Gender Pie
col1, col2, col3 = st.columns([1.5, 1, 1.2])

with col1:
    st.markdown("**Distribution of Anxiety Level**")
    chart1 = alt.Chart(filtered).mark_bar(color='#FF6F61').encode(
        x=alt.X("Anxiety Level (1-10):O", title="Anxiety Level"),
        y=alt.Y("count():Q", title="Qty")
    ).properties(height=250)
    st.altair_chart(chart1, use_container_width=True)

with col2:
    st.markdown("**Gender Distribution**")
    gender_chart = alt.Chart(filtered).mark_arc(innerRadius=50).encode(
        theta="count():Q",
        color="Gender:N"
    ).properties(height=250)
    st.altair_chart(gender_chart, use_container_width=True)

with col3:
    st.markdown("**Anxiety Level by Gender**")
    chart_gender = alt.Chart(filtered).mark_bar(color='#FF6F61').encode(
        y=alt.Y("Gender:N"),
        x=alt.X("mean(Anxiety Level (1-10)):Q", title="Average Anxiety Level")
    ).properties(height=250)
    st.altair_chart(chart_gender, use_container_width=True)

# Row 2: Occupation
st.markdown("**Anxiety Level by Occupation**")
occupation_chart = alt.Chart(filtered).mark_bar(color='#FF6F61').encode(
    x=alt.X("mean(Anxiety Level (1-10)):Q", title="Average of Anxiety Level (1-10)"),
    y=alt.Y("Occupation:N", sort="-x")
).properties(height=300)
st.altair_chart(occupation_chart, use_container_width=True)

# Row 3: Anxiety Trends by Numeric Features (Updated with contrasting trend lines)
st.markdown("### 🔄 Anxiety Trends by Numeric Features")

def create_trend_chart(data, x_col, title):
    # Base chart
    base = alt.Chart(data).encode(
        x=alt.X(f"{x_col}:Q", title=title),
        y=alt.Y("Anxiety Level (1-10):Q", title="Anxiety Level")
    )
    
    # Scatter plot with blue dots
    scatter = base.mark_circle(
        opacity=0.7, 
        color='#71C7EC',  # Light blue
        size=60
    )
    
    # Trend line with contrasting color
    trend = base.transform_regression(
        x_col,
        'Anxiety Level (1-10)'
    ).mark_line(
        color='#FF6F61',  # Coral color
        strokeWidth=3
    )
    
    return (scatter + trend).properties(height=300)

# First row of charts (Age vs Caffeine Intake)
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Anxiety Level (1-10) vs Age**")
    chart = create_trend_chart(filtered, "Age", "Age")
    st.altair_chart(chart, use_container_width=True)

with col2:
    st.markdown("**Anxiety Level (1-10) vs Caffeine Intake**")
    chart = create_trend_chart(filtered, "Caffeine Intake (mg/day)", "Caffeine Intake (mg/day)")
    st.altair_chart(chart, use_container_width=True)

# Second row of charts (Sleep Hours vs Alcohol Consumption)
col3, col4 = st.columns(2)
with col3:
    st.markdown("**Anxiety Level (1-10) vs Sleep Hours**")
    chart = create_trend_chart(filtered, "Sleep Hours", "Sleep Hours")
    st.altair_chart(chart, use_container_width=True)

with col4:
    st.markdown("**Anxiety Level (1-10) vs Alcohol Consumption**")
    chart = create_trend_chart(filtered, "Alcohol Consumption (drinks/week)", "Alcohol Consumption (drinks/week)")
    st.altair_chart(chart, use_container_width=True)

# Third row of charts (Physical Activity vs Therapy Sessions)
col5, col6 = st.columns(2)
with col5:
    st.markdown("**Anxiety Level (1-10) vs Physical Activity**")
    chart = create_trend_chart(filtered, "Physical Activity (hrs/week)", "Physical Activity (hrs/week)")
    st.altair_chart(chart, use_container_width=True)

with col6:
    st.markdown("**Anxiety Level (1-10) vs Therapy Sessions**")
    chart = create_trend_chart(filtered, "Therapy Sessions (per month)", "Therapy Sessions (per month)")
    st.altair_chart(chart, use_container_width=True)

# Row 5: Categoricals
cat_cols = [
    "Smoking",
    "Family History of Anxiety",
    "Dizziness",
    "Medication",
    "Recent Major Life Event"
]

st.markdown("### 🔠 Anxiety by Category")
cat_col1, cat_col2, cat_col3, cat_col4, cat_col5 = st.columns(5)
for col, feat in zip([cat_col1, cat_col2, cat_col3, cat_col4, cat_col5], cat_cols):
    with col:
        chart = alt.Chart(filtered).mark_bar(color="#FF6F61").encode(
            x=alt.X(f"{feat}:N"),
            y=alt.Y("mean(Anxiety Level (1-10)):Q", title="Avg Anxiety")
        )
        st.altair_chart(chart, use_container_width=True)