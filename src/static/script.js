// ==========================================
// 1. CONFIGURATION: Historical Limits
// ==========================================
const CONSTRAINTS = {
    1: { minFare: 0, maxFare: 513, avgFare: 84 },  // 1st Class (Widest range)
    2: { minFare: 0, maxFare: 74,  avgFare: 20 },  // 2nd Class
    3: { minFare: 0, maxFare: 70,  avgFare: 13 }   // 3rd Class
};

// ==========================================
// 2. VALIDATION HELPERS
// ==========================================

// Generic Validator (Snaps values to min/max)
function validateInput(inputElement, min, max) {
    let value = parseFloat(inputElement.value);

    if (isNaN(value) || value < min) {
        alert(`Value cannot be less than ${min}. Resetting.`);
        inputElement.value = min;
    } else if (value > max) {
        alert(`Value cannot be more than ${max}. Resetting.`);
        inputElement.value = max;
    }
}

// Fare Validator (Checks Class-Specific Limits)
function validateFare() {
    const pclass = document.getElementById("pclass").value;
    const fareInput = document.getElementById("fare");

    // Get limits for this class
    const min = CONSTRAINTS[pclass].minFare;
    const max = CONSTRAINTS[pclass].maxFare;

    let value = parseFloat(fareInput.value);

    if (isNaN(value)) return; // Don't annoy user while typing empty string

    // Check Upper Bound ONLY (Lower bound 0 is fine for all classes practically)
    if (value > max) {
        alert(`Historically, ${pclass} Class fares did not exceed $${max}. Resetting to max.`);
        fareInput.value = max;
    }
}

// Update UI Constraints when Class changes
function updateConstraints() {
    const pclass = document.getElementById("pclass").value;
    const fareLabel = document.getElementById("fare-label");
    const fareInput = document.getElementById("fare");
    const limits = CONSTRAINTS[pclass];

    // Update the label text
    fareLabel.innerText = `(Range: $${limits.minFare} - $${limits.maxFare})`;

    // Update the HTML min/max attributes (UX best practice)
    fareInput.min = limits.minFare;
    fareInput.max = limits.maxFare;

    // Smart Auto-Correction:
    // If user switches from 3rd (Fare $8) to 1st, keep it.
    // If they switch from 1st ($200) to 3rd, snap down to $70.
    if (parseFloat(fareInput.value) > limits.maxFare) {
        fareInput.value = limits.maxFare;
    }
}

// ==========================================
// 3. MAIN PREDICTION LOGIC
// ==========================================

async function predictSurvival() {
    // Collect Data
    const pclass = document.getElementById("pclass").value;
    const sex = document.getElementById("sex").value;
    const age = document.getElementById("age").value;
    const sibsp = document.getElementById("sibsp").value;
    const parch = document.getElementById("parch").value;
    const fare = document.getElementById("fare").value;
    const embarked = document.getElementById("embarked").value;

    // Build Payload
    const payload = {
        "Pclass": parseInt(pclass),
        "Sex": sex,
        "Age": parseFloat(age),
        "SibSp": parseInt(sibsp),
        "Parch": parseInt(parch),
        "Fare": parseFloat(fare),
        "Embarked": embarked,
        "Name": "Web User",      // Dummy data required by pipeline
        "Ticket": "WEB-001",     // Dummy data
        "PassengerId": 9999,     // Dummy data
        "Cabin": "U"             // Dummy data
    };

    const resultDiv = document.getElementById("result");
    resultDiv.innerHTML = "Calculating..."; // Show loading state

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        // --- THE FIX: Handle Probability Correctly ---
        // API sends "probability": 16.0  (Already a percentage)
        // OLD CODE: result.survival_probability * 100 -> undefined * 100 -> NaN

        const prob = result.probability; // Get the raw number (e.g., 16.0)

        // Visual Feedback
        if (result.prediction === 1) {
            resultDiv.className = "success";
            resultDiv.style.color = "#27ae60"; // Green
            resultDiv.innerHTML = `
                <h2>🎉 Survived</h2>
                <p>Probability: <strong>${prob}%</strong></p>
                <p style="font-size: 0.8em; color: #7f8c8d;">(Based on historical patterns)</p>
            `;
        } else {
            resultDiv.className = "danger";
            resultDiv.style.color = "#c0392b"; // Red
            resultDiv.innerHTML = `
                <h2>💀 Did Not Survive</h2>
                <p>Probability: <strong>${prob}%</strong></p>
                <p style="font-size: 0.8em; color: #7f8c8d;">(Based on historical patterns)</p>
            `;
        }

    } catch (error) {
        alert("Error connecting to API! Is the server running?");
        console.error(error);
        resultDiv.innerHTML = "Error.";
    }
}

// ==========================================
// 4. INITIALIZATION
// ==========================================
document.addEventListener("DOMContentLoaded", () => {
    updateConstraints(); // Set initial constraints
});
