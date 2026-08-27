import streamlit as st
import pandas as pd
import plotly.express as px


# Configure the Streamlit page
st.set_page_config(
    page_title="Student Academic Risk Intelligence System",
    layout="wide",
    page_icon="🎓"
)


# Load the Maths dataset
df = pd.read_csv("data/Maths.csv")


# Apply the same feature engineering used in analysis.py

# Create Result based on G3
# G3 = 0: Dropout, G3 = 1-9: Fail, G3 = 10-20: Pass
df["Result"] = df["G3"].apply(
    lambda x: "Dropout" if x == 0 else ("Fail" if x <= 9 else "Pass")
)

# Convert G3 into percentage
df["Percentage"] = (df["G3"] / 20) * 100

# Calculate average alcohol consumption
df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2

# Calculate average parental education level
df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

# Calculate grade trend from G1 to G3
df["grade_trend"] = df["G3"] - df["G1"]

# Count "yes" values across support-related columns
df["total_support"] = (
    (df["schoolsup"] == "yes").astype(int)
    + (df["famsup"] == "yes").astype(int)
    + (df["paid"] == "yes").astype(int)
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


# Display the main dashboard title
st.title("🎓 Student Academic Risk Intelligence System")


# Exclude dropout students for academic performance calculations
non_dropout = df[df["G3"] != 0]

# Calculate class average G3
class_average_g3 = round(non_dropout["G3"].mean(), 2)

# Calculate pass rate among non-dropout students
pass_count = (non_dropout["G3"] >= 10).sum()
pass_rate = round((pass_count / len(non_dropout)) * 100, 1)

# Count students who are at risk (G3 between 1 and 9)
at_risk_count = ((df["G3"] >= 1) & (df["G3"] <= 9)).sum()

# Get total number of students
total_students = len(df)


# Create 4 KPI cards in one row
col1, col2, col3, col4 = st.columns(4)

# Card 1: Total Students
col1.metric("Total Students", total_students)

# Card 2: Class Average G3
col2.metric("Class Average G3", class_average_g3)

# Card 3: Pass Rate
col3.metric("Pass Rate %", f"{pass_rate}%")

# Card 4: At-Risk Count
col4.metric("At-Risk Count", at_risk_count)


# ============================================
# Performance Charts
# ============================================

st.subheader("📊 Performance Charts")

# Create two columns so the charts appear side by side
left_col, right_col = st.columns(2)


# --------------------------------------------
# Left Chart: Study Time vs Final Grade
# --------------------------------------------

with left_col:

    # Create scatter plot showing study time versus final grade
    fig_scatter = px.scatter(
        df,
        x="studytime",
        y="G3",
        color="Result",
        color_discrete_map={
            "Pass": "green",
            "Fail": "red",
            "Dropout": "grey"
        },
        hover_data=["absences", "G1", "G2"],
        title="Study Time vs Final Grade"
    )

    # Display the scatter plot using the full column width
    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )


# --------------------------------------------
# Right Chart: Average G3 by Internet Access
# --------------------------------------------

with right_col:

    # Calculate average G3 for each internet access group
    avg_g3_internet = (
        df.groupby("internet", as_index=False)["G3"]
        .mean()
        .rename(columns={"G3": "Average G3"})
    )

    # Create bar chart showing average G3 by internet access
    fig_bar = px.bar(
        avg_g3_internet,
        x="internet",
        y="Average G3",
        color="internet",
        title="Average G3 by Internet Access"
    )

    # Display the bar chart using the full column width
    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )


# ============================================
# Student Analysis Table
# ============================================

st.subheader("🚨 Student Analysis Table")


# Create a dropdown to filter students by their Result
result_filter = st.selectbox(
    "Filter by Result",
    ["All", "Pass", "Fail", "Dropout"]
)


# Apply the selected filter to the DataFrame
if result_filter == "All":
    filtered_df = df
else:
    filtered_df = df[df["Result"] == result_filter]


# Select only the required columns for the student analysis table
display_columns = [
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


# Display the filtered student DataFrame
st.dataframe(
    filtered_df[display_columns],
    use_container_width=True
)


# ============================================
# At-Risk Students
# ============================================

st.subheader("⚠️ At-Risk Students")


# Select students with G3 between 1 and 9 inclusive
at_risk_df = df[
    (df["G3"] >= 1) & (df["G3"] <= 9)
].copy()


# Sort at-risk students by G3 ascending
# Students with the lowest grades appear first
at_risk_df = at_risk_df.sort_values(
    by="G3",
    ascending=True
)


# Select only the required columns for the at-risk table
at_risk_columns = [
    "G1",
    "G2",
    "G3",
    "absences",
    "studytime",
    "failures"
]


# Display the total number of at-risk students
st.write(
    f"Total at-risk students: {len(at_risk_df)}"
)


# Display the sorted at-risk students
st.dataframe(
    at_risk_df[at_risk_columns],
    use_container_width=True
)