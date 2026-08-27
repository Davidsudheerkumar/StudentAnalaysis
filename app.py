import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# PROMPT 10 - PAGE SETUP AND DATA LOADING
# ============================================================

# Configure the Streamlit page
st.set_page_config(
    page_title="Student Academic Risk Intelligence System",
    layout="wide",
    page_icon="🎓"
)


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

# Load the Maths dataset
df = pd.read_csv("data/Maths.csv")


# Create Result based on final grade G3
# G3 = 0      -> Dropout
# G3 = 1-9    -> Fail
# G3 = 10-20  -> Pass
df["Result"] = df["G3"].apply(
    lambda x: "Dropout"
    if x == 0
    else ("Fail" if 1 <= x <= 9 else "Pass")
)


# Convert G3 into percentage
df["Percentage"] = (df["G3"] / 20) * 100


# Calculate average alcohol consumption
df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2


# Calculate average parent education level
df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2


# Calculate grade trend
df["grade_trend"] = df["G3"] - df["G1"]


# Count support received
support_columns = ["schoolsup", "famsup", "paid"]

df["total_support"] = (
    df[support_columns]
    .apply(lambda row: (row == "yes").sum(), axis=1)
)


# Calculate academic risk score
df["risk_score"] = (
    (df["failures"] * 2)
    + (df["absences"] / 10)
    + df["avg_alcohol"]
    - df["studytime"]
)


# Calculate average of G1 and G2
df["g1_g2_avg"] = (df["G1"] + df["G2"]) / 2


# ============================================================
# MAIN TITLE
# ============================================================

st.title("🎓 Student Academic Risk Intelligence System")


# ============================================================
# CALCULATE KPI VALUES
# ============================================================

# Total number of students
total_students = len(df)


# Exclude dropout students
non_dropout = df[df["G3"] != 0]


# Calculate class average G3
if len(non_dropout) > 0:
    class_average_g3 = round(
        non_dropout["G3"].mean(),
        2
    )
else:
    class_average_g3 = 0.0


# Count passed students
pass_count = (
    non_dropout["G3"] >= 10
).sum()


# Calculate pass rate
if len(non_dropout) > 0:
    pass_rate = round(
        (pass_count / len(non_dropout)) * 100,
        1
    )
else:
    pass_rate = 0.0


# Count at-risk students
at_risk_count = (
    (df["G3"] >= 1) &
    (df["G3"] <= 9)
).sum()


# ============================================================
# KPI CARDS - ONE ROW
# ============================================================

# Create four columns
col1, col2, col3, col4 = st.columns(4)


# Total Students
with col1:
    st.metric(
        label="Total Students",
        value=total_students
    )


# Class Average G3
with col2:
    st.metric(
        label="Class Average G3",
        value=f"{class_average_g3:.2f}"
    )


# Pass Rate
with col3:
    st.metric(
        label="Pass Rate %",
        value=f"{pass_rate:.1f}%"
    )


# At-Risk Count
with col4:
    st.metric(
        label="At-Risk Count",
        value=int(at_risk_count)
    )


# ============================================================
# PROMPT 11 - PERFORMANCE CHARTS
# ============================================================

# Display Performance Charts section
st.subheader("📊 Performance Charts")


# Create two columns for side-by-side charts
col1, col2 = st.columns(2)


# ------------------------------------------------------------
# LEFT CHART - STUDY TIME VS FINAL GRADE
# ------------------------------------------------------------

with col1:

    # Create interactive scatter plot
    fig_scatter = px.scatter(
        df,
        x="studytime",
        y="G3",
        color="Result",
        hover_data=[
            "absences",
            "G1",
            "G2"
        ],
        title="Study Time vs Final Grade",
        color_discrete_map={
            "Pass": "green",
            "Fail": "red",
            "Dropout": "grey"
        }
    )

    # Display the chart
    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )


# ------------------------------------------------------------
# RIGHT CHART - AVERAGE G3 BY INTERNET ACCESS
# ------------------------------------------------------------

with col2:

    # Calculate average G3 for each internet group
    avg_g3_internet = (
        df.groupby(
            "internet",
            as_index=False
        )["G3"]
        .mean()
        .rename(
            columns={"G3": "Average G3"}
        )
    )

    # Create interactive bar chart
    fig_bar = px.bar(
        avg_g3_internet,
        x="internet",
        y="Average G3",
        color="internet",
        title="Average G3 by Internet Access"
    )

    # Display the chart
    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )


# ============================================================
# PROMPT 12 - STUDENT ANALYSIS TABLE
# ============================================================

# Display Student Analysis section
st.subheader("🚨 Student Analysis Table")


# Create result filter dropdown
result_filter = st.selectbox(
    "Filter by Result",
    [
        "All",
        "Pass",
        "Fail",
        "Dropout"
    ]
)


# Apply selected filter
if result_filter == "All":

    # Display all students
    filtered_df = df.copy()

else:

    # Display only selected result category
    filtered_df = df[
        df["Result"] == result_filter
    ].copy()


# Select required columns
analysis_columns = [
    "G1",
    "G2",
    "G3",
    "Result",
    "Percentage",
    "absences",
    "studytime",
    "failures",
    "risk_score"
]


# Display filtered student table
st.dataframe(
    filtered_df[analysis_columns],
    use_container_width=True
)


# ============================================================
# AT-RISK STUDENTS
# ============================================================

# Display At-Risk Students section
st.subheader("⚠️ At-Risk Students")


# Select students with G3 between 1 and 9
at_risk_df = df[
    (df["G3"] >= 1) &
    (df["G3"] <= 9)
].copy()


# Sort worst-performing students first
at_risk_df = at_risk_df.sort_values(
    "G3",
    ascending=True
)


# Select required columns
at_risk_columns = [
    "G1",
    "G2",
    "G3",
    "absences",
    "studytime",
    "failures"
]


# Display number of at-risk students
st.write(
    f"Total at-risk students: {len(at_risk_df)}"
)


# Display at-risk students
st.dataframe(
    at_risk_df[at_risk_columns],
    use_container_width=True
)