import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


# ============================================================
# PROMPT 1 - LOAD DATA AND FEATURE ENGINEERING
# ============================================================

def load_and_prepare_data(filepath):
    # Load the Maths dataset from the given CSV filepath
    df = pd.read_csv(filepath)

    # Create Result based on G3.
    # G3 = 0      -> Dropout
    # G3 = 1-9    -> Fail
    # G3 = 10-20  -> Pass
    df["Result"] = df["G3"].apply(
        lambda x: "Dropout"
        if x == 0
        else ("Fail" if 1 <= x <= 9 else "Pass")
    )

    # Convert the final grade into percentage out of 20
    df["Percentage"] = (df["G3"] / 20) * 100

    # Calculate average alcohol consumption
    df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2

    # Calculate average education level of both parents
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

    # Calculate the change in grade from G1 to G3
    df["grade_trend"] = df["G3"] - df["G1"]

    # Count "yes" values in support-related columns
    support_columns = ["schoolsup", "famsup", "paid"]

    df["total_support"] = (
        df[support_columns]
        .apply(lambda row: (row == "yes").sum(), axis=1)
    )

    # Calculate the custom academic risk score
    df["risk_score"] = (
        (df["failures"] * 2)
        + (df["absences"] / 10)
        + df["avg_alcohol"]
        - df["studytime"]
    )

    # Calculate average of G1 and G2
    df["g1_g2_avg"] = (df["G1"] + df["G2"]) / 2

    # Return the complete prepared DataFrame
    return df


# ============================================================
# PROMPT 2 - NUMPY ANALYSIS
# ============================================================

def calculate_statistics(df):
    # Exclude students with G3 = 0 from academic performance calculations
    non_dropout = df[df["G3"] != 0]

    # Calculate class average G3 using NumPy
    if len(non_dropout) > 0:
        class_avg_g3 = np.mean(non_dropout["G3"].values)
    else:
        class_avg_g3 = 0.0

    # Count students who passed
    pass_count = np.sum(non_dropout["G3"].values >= 10)

    # Calculate pass rate among non-dropout students
    if len(non_dropout) > 0:
        pass_rate = (pass_count / len(non_dropout)) * 100
    else:
        pass_rate = 0.0

    # Count students with G3 = 0
    dropout_count = np.sum(df["G3"].values == 0)

    # Count at-risk students with G3 between 1 and 9
    at_risk_count = np.sum(
        (df["G3"].values >= 1) &
        (df["G3"].values <= 9)
    )

    # Calculate correlation matrix for G1, G2 and G3
    # using only non-dropout students
    if len(non_dropout) > 1:
        correlation_matrix = np.corrcoef(
            non_dropout[["G1", "G2", "G3"]].values,
            rowvar=False
        )
    else:
        correlation_matrix = np.empty((3, 3))

    # Return all calculated statistics as a dictionary
    return {
        "class_avg_g3": float(class_avg_g3),
        "pass_rate": float(pass_rate),
        "dropout_count": int(dropout_count),
        "at_risk_count": int(at_risk_count),
        "correlation_matrix": correlation_matrix
    }


# ============================================================
# PROMPT 3 - MATPLOTLIB STATIC CHARTS
# ============================================================

def generate_static_charts(df):
    # Create output folder if it does not already exist
    os.makedirs("output", exist_ok=True)

    # --------------------------------------------------------
    # Chart 1 - Average G3 by Study Time
    # --------------------------------------------------------

    # Calculate average G3 for each studytime level
    studytime_avg = (
        df.groupby("studytime")["G3"]
        .mean()
        .reindex([1, 2, 3, 4])
    )

    # Create the bar chart
    plt.figure(figsize=(8, 5))

    plt.bar(
        studytime_avg.index,
        studytime_avg.values
    )

    # Add chart title and axis labels
    plt.title("Average G3 by Study Time")
    plt.xlabel(
        "Study Time (1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs)"
    )
    plt.ylabel("Average G3")

    # Keep the studytime levels visible on the X-axis
    plt.xticks([1, 2, 3, 4])

    # Adjust layout
    plt.tight_layout()

    # Save the chart
    plt.savefig(
        "output/avg_g3_by_studytime.png",
        dpi=300
    )

    # Close the chart to release memory
    plt.close()

    # --------------------------------------------------------
    # Chart 2 - Student Result Distribution
    # --------------------------------------------------------

    # Count students in each result category
    result_counts = (
        df["Result"]
        .value_counts()
        .reindex(["Pass", "Fail", "Dropout"], fill_value=0)
    )

    # Create the pie chart
    plt.figure(figsize=(7, 7))

    plt.pie(
        result_counts.values,
        labels=result_counts.index,
        autopct="%1.1f%%"
    )

    # Add chart title
    plt.title("Student Result Distribution")

    # Adjust layout
    plt.tight_layout()

    # Save the pie chart
    plt.savefig(
        "output/pass_fail_dropout_pie.png",
        dpi=300
    )

    # Close the chart
    plt.close()


# ============================================================
# PROMPT 4 - PLOTLY INTERACTIVE CHARTS
# ============================================================

def generate_interactive_charts(df):
    # --------------------------------------------------------
    # Chart 1 - Study Time vs Final Grade
    # --------------------------------------------------------

    # Create an interactive scatter plot
    fig_scatter = px.scatter(
        df,
        x="studytime",
        y="G3",
        color="Result",
        hover_data=["absences", "G1", "G2"],
        title="Study Time vs Final Grade (G3)",
        color_discrete_map={
            "Pass": "green",
            "Fail": "red",
            "Dropout": "grey"
        }
    )

    # Display the interactive scatter plot
    fig_scatter.show()

    # --------------------------------------------------------
    # Chart 2 - Average G3 by Internet Access
    # --------------------------------------------------------

    # Calculate average G3 for each internet access group
    avg_g3_internet = (
        df.groupby("internet", as_index=False)["G3"]
        .mean()
        .rename(columns={"G3": "Average G3"})
    )

    # Create an interactive bar chart
    fig_bar = px.bar(
        avg_g3_internet,
        x="internet",
        y="Average G3",
        color="internet",
        title="Average G3 by Internet Access"
    )

    # Display the interactive bar chart
    fig_bar.show()


# ============================================================
# PROMPT 5 - SUMMARY TABLE
# ============================================================

def print_summary(stats):
    # Print a clean formatted analysis summary
    print("=" * 55)
    print("STUDENT ACADEMIC RISK INTELLIGENCE SYSTEM")
    print()
    print("ANALYSIS SUMMARY")
    print("=" * 55)

    print(
        f"Total Students     : "
        f"{stats.get('total_students', 'N/A')}"
    )

    print(
        f"Class Average G3   : "
        f"{stats['class_avg_g3']:.2f}"
    )

    print(
        f"Pass Rate          : "
        f"{stats['pass_rate']:.2f}%"
    )

    print(
        f"At-Risk Count      : "
        f"{stats['at_risk_count']}"
    )

    print(
        f"Dropout Count      : "
        f"{stats['dropout_count']}"
    )

    print("=" * 55)


# ============================================================
# MAIN BLOCK
# ============================================================

if __name__ == "__main__":
    # Load and prepare the dataset
    df = load_and_prepare_data("data/Maths.csv")

    # Calculate statistical analysis
    stats = calculate_statistics(df)

    # Add total students to the statistics dictionary
    stats["total_students"] = len(df)

    # Generate and save static Matplotlib charts
    generate_static_charts(df)

    # Generate interactive Plotly charts
    generate_interactive_charts(df)

    # Print the analysis summary
    print_summary(stats)

    # Display completion message
    print("Analysis complete. Charts saved to output/ folder")