# Import required libraries
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import uvicorn


# Create the FastAPI application
app = FastAPI(
    title="Student Academic Risk Intelligence System API",
    description="API for analyzing student performance data",
    version="1.0.0"
)


# Function to load and prepare the student dataset
def load_data():
    # Load Maths.csv from the data folder
    df = pd.read_csv("data/Maths.csv")

    # Create Result based on the final grade (G3)
    # G3 = 0 means Dropout, G3 from 1-9 means Fail,
    # and G3 from 10-20 means Pass
    df["Result"] = df["G3"].apply(
        lambda x: "Dropout" if x == 0 else ("Fail" if x <= 9 else "Pass")
    )

    # Convert final grade into percentage
    df["Percentage"] = (df["G3"] / 20) * 100

    # Calculate average alcohol consumption
    df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2

    # Calculate average education level of both parents
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

    # Calculate grade trend from G1 to G3
    df["grade_trend"] = df["G3"] - df["G1"]

    # Count "yes" values across schoolsup, famsup, and paid
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

    # Calculate average of G1 and G2
    df["g1_g2_avg"] = (df["G1"] + df["G2"]) / 2

    # Return the prepared DataFrame
    return df


# Load the prepared dataset when the application starts
df = load_data()

# Endpoint 1: Return overall student academic summary
@app.get("/summary")
def get_summary():
    # Exclude dropout students (G3 = 0) from performance calculations
    non_dropout = df[df["G3"] != 0]

    # Calculate total number of students
    total_students = len(df)

    # Calculate class average G3 for non-dropout students
    class_average_g3 = round(non_dropout["G3"].mean(), 2)

    # Calculate pass rate among non-dropout students
    # Students with G3 >= 10 are considered to have passed
    pass_rate_percent = round(
        (non_dropout["G3"] >= 10).mean() * 100, 2
    )

    # Count students at risk (G3 between 1 and 9)
    at_risk_count = int(((df["G3"] >= 1) & (df["G3"] <= 9)).sum())

    # Count dropout students (G3 = 0)
    dropout_count = int((df["G3"] == 0).sum())

    # Return the summary as JSON
    return {
        "total_students": int(total_students),
        "class_average_g3": float(class_average_g3),
        "pass_rate_percent": float(pass_rate_percent),
        "at_risk_count": at_risk_count,
        "dropout_count": dropout_count
    }


# Endpoint 2: Return all students who are at risk of failing
@app.get("/at-risk")
def get_at_risk_students():
    # Select students whose G3 is between 1 and 9 inclusive
    at_risk = df[(df["G3"] >= 1) & (df["G3"] <= 9)].copy()

    # Add the original DataFrame index as student_index
    at_risk["student_index"] = at_risk.index

    # Sort by G3 ascending so the students with the lowest grades appear first
    at_risk = at_risk.sort_values("G3", ascending=True)

    # Return only the requested columns as a list of JSON records
    return at_risk[
        ["student_index", "G1", "G2", "G3", "absences"]
    ].to_dict(orient="records")


# Endpoint 3: Return the top 5 students by final grade
@app.get("/top-students")
def get_top_students():
    # Exclude dropout students (G3 = 0)
    non_dropout = df[df["G3"] != 0].copy()

    # Add the original DataFrame index as student_index
    non_dropout["student_index"] = non_dropout.index

    # Sort by G3 descending and select the top 5 students
    top_students = non_dropout.sort_values(
        "G3", ascending=False
    ).head(5)

    # Return only the requested columns as a list of JSON records
    return top_students[
        ["student_index", "G1", "G2", "G3"]
    ].to_dict(orient="records")

# Pydantic model for validating student input data
class StudentInput(BaseModel):
    # First period grade: must be between 0 and 20
    G1: float = Field(
        ...,
        ge=0,
        le=20,
        description="G1 must be between 0 and 20"
    )

    # Second period grade: must be between 0 and 20
    G2: float = Field(
        ...,
        ge=0,
        le=20,
        description="G2 must be between 0 and 20"
    )

    # Weekly study time level: must be between 1 and 4
    studytime: int = Field(
        ...,
        ge=1,
        le=4,
        description="Studytime must be between 1 and 4"
    )

    # Number of absences: must be between 0 and 100
    absences: int = Field(
        ...,
        ge=0,
        le=100,
        description="Absences must be between 0 and 100"
    )

    # Number of previous failures: must be between 0 and 4
    failures: int = Field(
        ...,
        ge=0,
        le=4,
        description="Failures must be between 0 and 4"
    )


# POST endpoint: Predict the student's academic result
@app.post("/predict-result")
def predict_result(student: StudentInput):
    # Calculate the estimated final grade using the specified formula
    estimated_g3 = (
        (student.G1 * 0.3)
        + (student.G2 * 0.6)
        + (student.studytime * 0.3)
        - (student.failures * 1.5)
        - (student.absences * 0.05)
    )

    # Clamp the estimated grade between 0 and 20
    estimated_g3 = max(0, min(20, estimated_g3))

    # Determine the predicted academic result
    if estimated_g3 == 0:
        prediction = "Dropout Risk"
    elif estimated_g3 < 10:
        prediction = "Fail"
    else:
        prediction = "Pass"

    # Determine confidence based on G1 and G2 performance
    if student.G1 > 12 and student.G2 > 12:
        confidence = "High"
    elif student.G1 < 8 and student.G2 < 8:
        confidence = "High"
    else:
        confidence = "Medium"

    # Return the prediction details as JSON
    return {
        "estimated_g3": round(estimated_g3, 2),
        "prediction": prediction,
        "confidence": confidence
    }

# Root endpoint: Provide basic API information
@app.get("/")
def root():
    return {
        "message": "Student Academic Risk Intelligence System API",
        "docs": "Visit /docs for full API documentation",
        "version": "1.0.0"
    }


# Main block: Start the FastAPI server when main.py is executed directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )