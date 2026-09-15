/**
 * Lung Cancer Prediction - Client-Side Interactive Logic
 * Handles interactive scale chips, profile presets, form validation, loading states, and printing.
 */

document.addEventListener('DOMContentLoaded', () => {
  initScaleSelectors();
  initPresetButtons();
  initFormSubmission();
  initPrintButton();
});

/**
 * Initializes interactive scale chips (Rating 1 to N)
 */
function initScaleSelectors() {
  const scaleGroups = document.querySelectorAll('.scale-selector');

  scaleGroups.forEach(group => {
    const hiddenInput = group.querySelector('input[type="hidden"]');
    const chips = group.querySelectorAll('.scale-chip');

    chips.forEach(chip => {
      chip.addEventListener('click', () => {
        const val = chip.getAttribute('data-value');
        hiddenInput.value = val;

        chips.forEach(c => c.classList.remove('selected'));
        chip.classList.add('selected');

        // Trigger change event for any listeners
        hiddenInput.dispatchEvent(new Event('change', { bubbles: true }));
      });
    });
  });
}

/**
 * Quick Profile Presets for testing and clinical demonstration
 */
const PRESETS = {
  low_risk: {
    'Age': 27,
    'Gender': 'Male',
    'Level': 'Low',
    'Air Pollution': 3,
    'Alcohol use': 1,
    'Dust Allergy': 4,
    'chronic Lung Disease': 3,
    'Balanced Diet': 4,
    'Obesity': 3,
    'Smoking': 1,
    'Passive Smoker': 4,
    'Chest Pain': 3,
    'Coughing of Blood': 1,
    'Fatigue': 3,
    'Weight Loss': 2,
    'Shortness of Breath': 2,
    'Wheezing': 4,
    'Swallowing Difficulty': 2,
    'Clubbing of Finger Nails': 2,
    'Frequent Cold': 3,
    'Dry Cough': 4,
    'Snoring': 3
  },
  high_risk: {
    'Age': 52,
    'Gender': 'Female',
    'Level': 'High',
    'Air Pollution': 7,
    'Alcohol use': 7,
    'Dust Allergy': 7,
    'chronic Lung Disease': 6,
    'Balanced Diet': 2,
    'Obesity': 6,
    'Smoking': 8,
    'Passive Smoker': 7,
    'Chest Pain': 8,
    'Coughing of Blood': 8,
    'Fatigue': 8,
    'Weight Loss': 7,
    'Shortness of Breath': 8,
    'Wheezing': 7,
    'Swallowing Difficulty': 6,
    'Clubbing of Finger Nails': 8,
    'Frequent Cold': 6,
    'Dry Cough': 6,
    'Snoring': 5
  }
};

function applyPreset(profileKey) {
  const profile = PRESETS[profileKey];
  if (!profile) return;

  // Set standard inputs (Age)
  const ageInput = document.querySelector('input[name="Age"]');
  if (ageInput && profile['Age']) {
    ageInput.value = profile['Age'];
  }

  // Set Radios (Gender, Level)
  ['Gender', 'Level'].forEach(field => {
    const val = profile[field];
    const radio = document.querySelector(`input[name="${field}"][value="${val}"]`);
    if (radio) {
      radio.checked = true;
    }
  });

  // Set Scale Selectors
  Object.keys(profile).forEach(field => {
    if (field === 'Age' || field === 'Gender' || field === 'Level') return;
    const targetVal = String(profile[field]);

    const hiddenInput = document.querySelector(`input[name="${field}"]`);
    if (hiddenInput) {
      hiddenInput.value = targetVal;
      const group = hiddenInput.closest('.scale-selector');
      if (group) {
        const chips = group.querySelectorAll('.scale-chip');
        chips.forEach(chip => {
          if (chip.getAttribute('data-value') === targetVal) {
            chip.classList.add('selected');
          } else {
            chip.classList.remove('selected');
          }
        });
      }
    }
  });

  showToast(`Loaded ${profileKey === 'low_risk' ? 'Low Risk' : 'High Risk'} demonstration profile`);
}

function initPresetButtons() {
  const btnLow = document.getElementById('btn-preset-low');
  const btnHigh = document.getElementById('btn-preset-high');
  const btnClear = document.getElementById('btn-reset-form');

  if (btnLow) {
    btnLow.addEventListener('click', (e) => {
      e.preventDefault();
      applyPreset('low_risk');
    });
  }

  if (btnHigh) {
    btnHigh.addEventListener('click', (e) => {
      e.preventDefault();
      applyPreset('high_risk');
    });
  }

  if (btnClear) {
    btnClear.addEventListener('click', (e) => {
      e.preventDefault();
      const form = document.getElementById('prediction-form');
      if (form) {
        form.reset();
        // Clear all scale chips
        document.querySelectorAll('.scale-chip').forEach(c => c.classList.remove('selected'));
        document.querySelectorAll('.scale-selector input[type="hidden"]').forEach(inp => inp.value = '');
        showToast('Form cleared');
      }
    });
  }
}

/**
 * Form submission handling with validation and loading indicator
 */
function initFormSubmission() {
  const form = document.getElementById('prediction-form');
  const submitBtn = document.getElementById('submit-btn');

  if (!form || !submitBtn) return;

  form.addEventListener('submit', (e) => {
    // Check if hidden scale inputs are populated
    let missing = [];
    const scaleInputs = form.querySelectorAll('.scale-selector input[type="hidden"]');
    scaleInputs.forEach(inp => {
      if (!inp.value) {
        const fieldName = inp.name;
        missing.push(fieldName);
      }
    });

    if (missing.length > 0) {
      e.preventDefault();
      alert(`Please rate all clinical symptoms before submitting (missing ${missing.length} indicator(s)).`);
      return;
    }

    // Enter loading state
    submitBtn.disabled = true;
    submitBtn.innerHTML = `
      <span class="spinner"></span>
      Analyzing Clinical Indicators...
    `;
  });
}

/**
 * Print Summary report functionality
 */
function initPrintButton() {
  const printBtn = document.getElementById('btn-print-report');
  if (printBtn) {
    printBtn.addEventListener('click', () => {
      window.print();
    });
  }
}

/**
 * Non-intrusive Toast notification helper
 */
function showToast(message) {
  let toast = document.getElementById('app-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'app-toast';
    toast.style.position = 'fixed';
    toast.style.bottom = '24px';
    toast.style.right = '24px';
    toast.style.backgroundColor = '#0f172a';
    toast.style.color = '#fff';
    toast.style.padding = '12px 20px';
    toast.style.borderRadius = '8px';
    toast.style.boxShadow = '0 10px 25px rgba(0,0,0,0.2)';
    toast.style.zIndex = '9999';
    toast.style.fontSize = '0.9rem';
    toast.style.fontWeight = '500';
    toast.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    document.body.appendChild(toast);
  }

  toast.textContent = message;
  toast.style.opacity = '1';
  toast.style.transform = 'translateY(0)';

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
  }, 2500);
}
