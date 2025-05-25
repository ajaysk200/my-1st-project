# 🩺 Diabetes Prediction with Support-Vector Machine (SVM)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" />
  <img src="https://img.shields.io/badge/scikit--learn-1.x-yellow.svg" />
  <img src="https://img.shields.io/badge/pandas-2.x-green.svg" />
  <img src="https://img.shields.io/badge/status-active-brightgreen.svg" />
</p>

A lightweight end-to-end workflow that trains a **Support-Vector Machine (SVM)** classifier to predict whether a patient is diabetic based on diagnostic measurements.  
The entire project is contained in a single Python script (`diabetes.py`)—ideal for beginners who want to see all the moving parts in one place.

---

## 📂 Dataset

| Feature | Description |
|---------|-------------|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (2-hr in an oral glucose tolerance test) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-Hr serum insulin (µU / ml) |
| `BMI` | Body mass index (weight in kg / ( height in m )²) |
| `DiabetesPedigreeFunction` | Diabetes pedigree function |
| `Age` | Age (years) |
| **`Outcome`** | 0 = non-diabetic, 1 = diabetic |

A cleaned CSV version of the famous **Pima Indians Diabetes Dataset** is fetched automatically from Dropbox:

```python
url = "https://www.dropbox.com/scl/fi/0uiujtei423te1q4kvrny/diabetes.csv?raw=1"
diabetes_dataset = pd.read_csv(url)
