import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import uuid

# --- Core Data Structures ---

class Patient:
    """Represents a patient in the hospital."""
    def __init__(self, name, age, gender, ailment, admitted_on):
        self.patient_id = str(uuid.uuid4())[:8] # Generate a simple unique ID
        self.name = name
        self.age = age
        self.gender = gender
        self.ailment = ailment
        self.admitted_on = admitted_on
        self.discharge_info = None

    def __repr__(self):
        return f"Patient(ID: {self.patient_id}, Name: {self.name}, Age: {self.age}, Ailment: {self.ailment})"

class Doctor:
    """Represents a doctor."""
    def __init__(self, name, specialization, contact):
        self.doctor_id = str(uuid.uuid4())[:4]
        self.name = name
        self.specialization = specialization
        self.contact = contact

    def __repr__(self):
        return f"Doctor(ID: {self.doctor_id}, Name: {self.name}, Specialization: {self.specialization})"

# --- AI/ML Model and Data Simulation ---

class ReadmissionPredictor:
    """
    A conceptual Machine Learning component for predicting patient readmission risk.
    Uses a simple Logistic Regression model trained on synthetic data.
    """
    def __init__(self):
        self.model = None
        self.feature_cols = ['Age', 'Length_of_Stay', 'Severity_Score']
        self._generate_synthetic_data()
        self._train_model()

    def _generate_synthetic_data(self):
        """Creates a small, synthetic dataset for demonstration."""
        num_records = 100
        data = {
            'Age': [random.randint(20, 85) for _ in range(num_records)],
            # Length of Stay in days
            'Length_of_Stay': [random.randint(1, 30) for _ in range(num_records)],
            # Mock Severity Score (1=Low, 10=High)
            'Severity_Score': [random.randint(1, 10) for _ in range(num_records)],
        }
        df = pd.DataFrame(data)

        # Create a mock 'Readmitted' target column (1=Readmitted, 0=Not)
        # Simple logic: Older patients with long stays and high severity are more likely to be readmitted
        df['Readmitted'] = ((df['Age'] > 65) & (df['Length_of_Stay'] > 10) & (df['Severity_Score'] > 5)).astype(int)
        
        # Add some random noise to make it less perfect
        df['Readmitted'] = df.apply(lambda row: 1 if random.random() < 0.2 else row['Readmitted'], axis=1)

        self.data = df
        print(f"--- AI/ML: Generated {num_records} synthetic records for training. ---")

    def _train_model(self):
        """Trains the Logistic Regression model."""
        X = self.data[self.feature_cols]
        y = self.data['Readmitted']

        # Split data for training (optional but good practice)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = LogisticRegression(solver='liblinear')
        self.model.fit(X_train, y_train)

        # Evaluate model (for internal check)
        y_pred = self.model.predict(X_test)
        print(f"--- AI/ML: Readmission Model Trained. Accuracy on test data: {accuracy_score(y_test, y_pred):.2f} ---")


    def predict_risk(self, age, length_of_stay, severity_score):
        """
        Predicts readmission risk (0=Low, 1=High) and probability.
        
        Args:
            age (int)
            length_of_stay (int): Days spent in hospital.
            severity_score (int): 1-10 severity.
            
        Returns:
            tuple: (risk_status, probability)
        """
        if self.model is None:
            return "Model not initialized.", 0.0

        # Create a DataFrame matching the training feature structure
        new_data = pd.DataFrame([[age, length_of_stay, severity_score]], columns=self.feature_cols)

        # Predict the class (0 or 1)
        risk_prediction = self.model.predict(new_data)[0]

        # Predict the probability of the '1' class (readmission)
        risk_probability = self.model.predict_proba(new_data)[0][1]

        risk_status = "High Risk" if risk_prediction == 1 else "Low Risk"
        
        return risk_status, risk_probability


# --- Hospital Management System ---

class HospitalManagementSystem:
    """Manages all hospital operations, integrating the AI/ML component."""
    
    def __init__(self, name="A-I Health Center"):
        self.name = name
        self.patients = {}
        self.doctors = {}
        self.ml_predictor = ReadmissionPredictor()

    # --- Core HMS Functions ---
    
    def add_doctor(self, name, specialization, contact):
        doc = Doctor(name, specialization, contact)
        self.doctors[doc.doctor_id] = doc
        print(f"\n[SYSTEM] Doctor {name} ({specialization}) registered with ID: {doc.doctor_id}.")
        return doc

    def admit_patient(self, name, age, gender, ailment, admitted_on):
        patient = Patient(name, age, gender, ailment, admitted_on)
        self.patients[patient.patient_id] = patient
        print(f"\n[SYSTEM] Patient {name} admitted. ID: {patient.patient_id}.")
        return patient

    def get_patient_details(self, patient_id):
        return self.patients.get(patient_id, "Patient not found.")

    def list_patients(self):
        if not self.patients:
            print("\n[INFO] No patients currently admitted.")
            return
        print("\n--- Current Admitted Patients ---")
        for p_id, patient in self.patients.items():
            print(f"[{p_id}] {patient.name}, Age: {patient.age}, Ailment: {patient.ailment}")
        print("-" * 35)

    # --- ML Integration Function ---
    
    def assess_readmission_risk(self, patient_id, length_of_stay, severity_score):
        """
        Uses the ML model to assess readmission risk for a specific patient.
        """
        patient = self.patients.get(patient_id)
        if not patient:
            print(f"\n[ERROR] Patient ID {patient_id} not found.")
            return

        print(f"\n--- AI Assessment for Patient: {patient.name} ---")
        print(f"Input Features: Age={patient.age}, LoS={length_of_stay} days, Severity={severity_score}/10")

        risk_status, probability = self.ml_predictor.predict_risk(
            age=patient.age,
            length_of_stay=length_of_stay,
            severity_score=severity_score
        )

        print(f"Prediction: {risk_status}")
        print(f"Confidence (Probability of Readmission): {probability*100:.2f}%")
        
        if probability > 0.7:
             print("\n[Recommendation] Initiate specialized post-discharge care planning and follow-up immediately.")
        elif probability > 0.4:
            print("\n[Recommendation] Standard follow-up plan recommended; monitor closely.")
        else:
            print("\n[Recommendation] Low risk; standard discharge procedure.")
        print("-" * 40)
        return risk_status, probability

# --- Demo Execution ---

if __name__ == "__main__":
    
    hms = HospitalManagementSystem("Virtual Health AI")
    print(f"--- Welcome to the {hms.name} System ---")

    # 1. Add Doctors
    doc1 = hms.add_doctor("Dr. Aria Sharma", "Cardiology", "555-1001")
    doc2 = hms.add_doctor("Dr. Ben Carter", "General Medicine", "555-1002")

    # 2. Admit Patients
    # Patient A: Low Risk Profile (Young, Low Severity)
    p_low = hms.admit_patient("Elara Vance", 35, "F", "Acute Appendicitis", "2025-12-01")
    # Patient B: High Risk Profile (Elderly, High Severity, Long Stay)
    p_high = hms.admit_patient("Robert Stern", 78, "M", "Pneumonia", "2025-11-20")
    # Patient C: Moderate Risk Profile
    p_mod = hms.admit_patient("Maya Lin", 55, "F", "Fractured Leg", "2025-11-25")
    
    hms.list_patients()

    # 3. Assess Readmission Risks (Simulating discharge planning)
    print("\n\n#####################################################")
    print("### SIMULATING AI/ML READMISSION RISK ASSESSMENT ###")
    print("#####################################################")

    # Scenario 1: Low Risk Patient (35 yrs, 5 days stay, Severity 2)
    hms.assess_readmission_risk(
        patient_id=p_low.patient_id, 
        length_of_stay=5,
        severity_score=2
    )

    # Scenario 2: High Risk Patient (78 yrs, 15 days stay, Severity 8)
    hms.assess_readmission_risk(
        patient_id=p_high.patient_id, 
        length_of_stay=15,
        severity_score=8
    )

    # Scenario 3: Moderate Risk Patient (55 yrs, 10 days stay, Severity 5)
    hms.assess_readmission_risk(
        patient_id=p_mod.patient_id, 
        length_of_stay=10,
        severity_score=5
    )

    print("\n--- System Shutdown ---")