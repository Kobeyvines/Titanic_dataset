// CONFIGURATION: Historical Limits (Min & Max)
const CONSTRAINTS = {
    1: { minFare: 25, maxFare: 513, avgFare: 84 },  // 1st Class starts at ~$25
    2: { minFare: 10, maxFare: 74,  avgFare: 20 },  // 2nd Class starts at ~$10
    3: { minFare: 0,  maxFare: 70,  avgFare: 13 }   // 3rd Class starts at $0
};

// 1. GENERIC VALIDATOR (Snaps values to min/max)
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

// 2. FARE VALIDATOR (Checks BOTH Min and Max)
function validateFare() {
    const pclass = document.getElementById("pclass").value;
    const fareInput = document.getElementById("fare");

    // Get limits for this class
    const min = CONSTRAINTS[pclass].minFare;
    const max = CONSTRAINTS[pclass].maxFare;

    let value = parseFloat(fareInput.value);

    if (isNaN(value)) return; // Don't annoy user while typing empty string

    // Check Lower Bound
    if (value < min) {
        alert(`A 1st Class ticket couldn't be bought for $${value}! The minimum was around $${min}. Resetting.`);
        fareInput.value = min;
    }
    // Check Upper Bound
    else if (value > max) {
        alert(`Historically, ${pclass} Class fares did not exceed $${max}. Resetting.`);
        fareInput.value = max;
    }
}

// 3. UPDATE CONSTRAINTS (Updates the UI hint)
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
    // If user switches from 3rd (Fare $8) to 1st, $8 is invalid.
    // Automatically bump them up to the minimum ($25).
    if (parseFloat(fareInput.value) < limits.minFare) {
        fareInput.value = limits.minFare;
    }
}

// 4. MAIN PREDICTION FUNCTION
async function predictSurvival() {
    // Collect Data
    const pclass = document.getElementById("pclass").value;
    const sex = document.getElementById("sex").value;
    const age = document.getElementById("age").value;
    const sibsp = document.getElementById("sibsp").value;
    const parch = document.getElementById("parch").value;
    const fare = document.getElementById("fare").value;
    const embarked = document.getElementById("embarked").value;

    // Payload
    const payload = {
        "Pclass": parseInt(pclass),
        "Sex": sex,
        "Age": parseFloat(age),
        "SibSp": parseInt(sibsp),
        "Parch": parseInt(parch),
        "Fare": parseFloat(fare),
        "Embarked": embarked,
        "Name": "Web User",
        "Ticket": "WEB-001", // Neutral Ticket
        "PassengerId": 9999,
        "Cabin": "U"
    };

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        const resultDiv = document.getElementById("result");

        // Visual Feedback
        if (result.prediction === 1) {
            resultDiv.className = "success";
            resultDiv.innerHTML = `
                <h2 style="color: #27ae60;">🎉 Survived</h2>
                <p>Probability: <strong>${(result.survival_probability * 100).toFixed(1)}%</strong></p>
                <p style="font-size: 0.8em; color: #7f8c8d;">(Based on historical patterns)</p>
            `;
        } else {
            resultDiv.className = "danger";
            resultDiv.innerHTML = `
                <h2 style="color: #c0392b;">💀 Did Not Survive</h2>
                <p>Probability: <strong>${(result.survival_probability * 100).toFixed(1)}%</strong></p>
                <p style="font-size: 0.8em; color: #7f8c8d;">(Based on historical patterns)</p>
            `;
        }
    } catch (error) {
        alert("Error connecting to API!");
        console.error(error);
    }
}

// Initialize on Load
document.addEventListener("DOMContentLoaded", updateConstraints);
