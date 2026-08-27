def load_and_prepare_data(filepath):
    import pandas as pd

    # Load the Student Performance dataset from the given CSV filepath
    df = pd.read_csv(filepath)

    # Create Result based on the final grade (G3)
    # G3 = 0: Dropout, G3 from 1-9: Fail, G3 from 10-20: Pass
    df["Result"] = df["G3"].apply(
        lambda x: "Dropout" if x == 0 else ("Fail" if x <= 9 else "Pass")
    )

    # Convert the final grade (G3) into percentage
    df["Percentage"] = (df["G3"] / 20) * 100

    # Calculate average alcohol consumption from weekday and weekend alcohol use
    df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2

    # Calculate the average education level of both parents
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

    # Calculate the student's grade trend from G1 to G3
    df["grade_trend"] = df["G3"] - df["G1"]

    # Count the number of "yes" values across the three support-related columns
    df["total_support"] = (
        (df["schoolsup"] == "yes").astype(int)
        + (df["famsup"] == "yes").astype(int)
        + (df["paid"] == "yes").astype(int)
    )

    # Calculate the academic risk score
    df["risk_score"] = (
        (df["failures"] * 2)
        + (df["absences"] / 10)
        + df["avg_alcohol"]
        - df["studytime"]
    )

    # Calculate the average of the first and second period grades
    df["g1_g2_avg"] = (df["G1"] + df["G2"]) / 2

    # Return the complete prepared DataFrame
    return df