def calculate_statistics(df):
    import numpy as np

    # Exclude dropout students (G3 = 0) for academic performance statistics
    non_dropout = df[df["G3"] != 0]

    # Calculate the average final grade (G3) among non-dropout students
    class_avg_g3 = np.mean(non_dropout["G3"])

    # Calculate the pass rate among non-dropout students
    # A student passes when G3 is 10 or above
    pass_rate = np.mean(non_dropout["G3"] >= 10) * 100

    # Count the total number of dropout students (G3 = 0)
    dropout_count = np.sum(df["G3"] == 0)

    # Count students at risk of failing (G3 between 1 and 9)
    at_risk_count = np.sum((df["G3"] >= 1) & (df["G3"] <= 9))

    # Calculate the correlation matrix for G1, G2, and G3
    # Only non-dropout students are included
    correlation_matrix = np.corrcoef(
        non_dropout[["G1", "G2", "G3"]].values.T
    )

    # Return all calculated statistics as a dictionary
    return {
        "total_students": len(df),
        "class_avg_g3": class_avg_g3,
        "pass_rate": pass_rate,
        "dropout_count": dropout_count,
        "at_risk_count": at_risk_count,
        "correlation_matrix": correlation_matrix
    }


def generate_static_charts(df):
    import os
    import matplotlib.pyplot as plt

    # Create the output folder if it does not already exist
    os.makedirs("output", exist_ok=True)

    # -----------------------------
    # Chart 1: Average G3 by Study Time
    # -----------------------------

    # Calculate the average G3 for each studytime level
    avg_g3 = df.groupby("studytime")["G3"].mean()

    # Create the bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(avg_g3.index, avg_g3.values)

    # Set chart title and axis labels
    plt.title("Average G3 by Study Time")
    plt.xlabel("Study Time (1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs)")
    plt.ylabel("Average G3")

    # Ensure all studytime levels 1, 2, 3, and 4 are shown on the X-axis
    plt.xticks([1, 2, 3, 4])

    # Save the bar chart
    plt.savefig(
        "output/avg_g3_by_studytime.png",
        bbox_inches="tight"
    )

    # Close the chart to free memory
    plt.close()

    # -----------------------------
    # Chart 2: Student Result Distribution
    # -----------------------------

    # Count the number of students in each result category
    result_counts = df["Result"].value_counts()

    # Ensure the categories appear in the required order
    result_counts = result_counts.reindex(
        ["Pass", "Fail", "Dropout"],
        fill_value=0
    )

    # Create the pie chart
    plt.figure(figsize=(7, 7))
    plt.pie(
        result_counts.values,
        labels=result_counts.index,
        autopct="%1.1f%%"
    )

    # Set the chart title
    plt.title("Student Result Distribution")

    # Save the pie chart
    plt.savefig(
        "output/pass_fail_dropout_pie.png",
        bbox_inches="tight"
    )

    # Close the chart to free memory
    plt.close()


def generate_interactive_charts(df):
    import plotly.express as px

    # -----------------------------
    # Chart 1: Study Time vs Final Grade
    # -----------------------------

    # Create a scatter plot using studytime on the X-axis
    # and G3 (final grade) on the Y-axis
    fig1 = px.scatter(
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
        title="Study Time vs Final Grade (G3)"
    )

    # Display the interactive scatter plot
    fig1.show()

    # -----------------------------
    # Chart 2: Average G3 by Internet Access
    # -----------------------------

    # Calculate the average G3 for students with and without internet access
    avg_g3_internet = (
        df.groupby("internet", as_index=False)["G3"]
        .mean()
        .rename(columns={"G3": "Average G3"})
    )

    # Create an interactive bar chart
    fig2 = px.bar(
        avg_g3_internet,
        x="internet",
        y="Average G3",
        color="internet",
        title="Average G3 by Internet Access"
    )

    # Display the interactive bar chart
    fig2.show()


def print_summary(stats):
    # Print a clean formatted analysis summary
    print("=" * 48)
    print("STUDENT ACADEMIC RISK INTELLIGENCE SYSTEM")
    print("ANALYSIS SUMMARY")
    print("=" * 48)

    # Display the total number of students
    total_students = stats["total_students"]

    print(f"Total Students    : {total_students}")
    print(f"Class Average G3  : {stats['class_avg_g3']:.2f}")
    print(f"Pass Rate         : {stats['pass_rate']:.2f}%")
    print(f"At-Risk Count     : {stats['at_risk_count']}")
    print(f"Dropout Count     : {stats['dropout_count']}")

    print("=" * 48)


# Main block: run the complete analysis when this file is executed directly
if __name__ == "__main__":

    # Import the data preparation function from data_preprocessing.py
    from data_preprocessing import load_and_prepare_data

    # Load and prepare the student dataset
    df = load_and_prepare_data("data/Maths.csv")

    # Calculate statistical measures
    stats = calculate_statistics(df)

    # Generate and save static Matplotlib charts
    generate_static_charts(df)

    # Generate and display interactive Plotly charts
    generate_interactive_charts(df)

    # Print the analysis summary
    print_summary(stats)

    # Confirm that the analysis has completed
    print("Analysis complete. Charts saved to output/ folder")