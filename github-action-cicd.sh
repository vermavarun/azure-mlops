#!/usr/bin/env bash
# Replicates the GitHub Actions "test" job locally.
# Usage:
#   ./run_test_job.sh
# Optional:
#   PYTHON_VERSION=3.10 ./run_test_job.sh
#   VENV_DIR=.venv ./run_test_job.sh

set -euo pipefail

# ---- Config ----
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
WORKDIR="${WORKDIR:-./LinearRegression/pipeline}"
VENV_DIR="${VENV_DIR:-.venv}"   # set to "" to disable venv creation

echo "==> Python version: ${PYTHON_VERSION}"
echo "==> Working directory: ${WORKDIR}"
echo "==> VENV dir: ${VENV_DIR:-<disabled>}"

# ---- Pre-flight checks ----
if [[ ! -d "${WORKDIR}" ]]; then
  echo "ERROR: Working directory '${WORKDIR}' not found."
  exit 1
fi

if [[ ! -f "${WORKDIR}/requirements.txt" ]]; then
  echo "WARNING: requirements.txt not found in ${WORKDIR}. Continuing..."
fi

# ---- Move to working directory (like defaults.run.working-directory) ----
cd "${WORKDIR}"

# ---- Setup Python / venv (local) ----
# If a venv folder is desired, create and use it
if [[ -n "${VENV_DIR}" ]]; then
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "==> Creating virtual environment at ${VENV_DIR}"
    python${PYTHON_VERSION} -m venv "${VENV_DIR}" 2>/dev/null || python -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  PYTHON_BIN="python"
  PIP_BIN="pip"
else
  # Fall back to system python
  PYTHON_BIN="python${PYTHON_VERSION}"
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="python"
  fi
  PIP_BIN="pip"
fi

echo "==> Using Python: $(${PYTHON_BIN} -V)"

# ---- Install dependencies (mirrors Actions) ----
echo "==> Installing dependencies"
${PYTHON_BIN} -m pip install --upgrade pip
if [[ -f requirements.txt ]]; then
  ${PIP_BIN} install -r requirements.txt
fi
${PIP_BIN} install pytest pytest-cov black pylint

# ---- Format check with Black ----
echo "==> Running Black format check"
set +e
${PYTHON_BIN} -m black . --check --line-length 100
BLACK_STATUS=$?
set -e
if [[ ${BLACK_STATUS} -ne 0 ]]; then
  echo "!! Black check reported issues (continuing, as in CI)"
fi

# ---- Lint with Pylint (errors & fatals only) ----
echo "==> Running Pylint for errors/fatals"
set +e
# lint all tracked .py files (safer than '*.py' in one folder)
PY_FILES=$(git ls-files '*.py' 2>/dev/null || true)
if [[ -z "${PY_FILES}" ]]; then
  echo "No Python files tracked by git to lint. Skipping pylint."
  PYLINT_STATUS=0
else
  pylint --disable=all --enable=E,F ${PY_FILES}
  PYLINT_STATUS=$?
fi
set -e
if [[ ${PYLINT_STATUS} -ne 0 ]]; then
  echo "!! Pylint reported issues (continuing, as in CI)"
fi

# ---- Import check ----
echo "==> Import check"
${PYTHON_BIN} - <<'PY'
import app, config, data_handler, train, evaluate
print('✓ All imports successful')
PY

# ---- Test data handler (returns DataFrame) ----
echo "==> Test data handler"
${PYTHON_BIN} - <<'PY'
from data_handler import generate_synthetic_data
df = generate_synthetic_data()
assert 'target' in df.columns, "Expected 'target' column in returned DataFrame"
X = df.drop(columns=['target'])
y = df['target']
print(f"✓ Generated {len(df)} samples, {X.shape[1]} features; y shape={y.shape}")
PY

# ---- Test training ----
echo "==> Test training"
${PYTHON_BIN} - <<'PY'

from data_handler import generate_synthetic_data
from train import train_model

df = generate_synthetic_data()
X = df.drop(columns=['target']).values
y = df['target'].values

res = train_model(X, y, model_type='linear')

# Accept either signature: model OR (model, score)
if isinstance(res, tuple) and len(res) == 2:
    model, score = res
else:
    model = res
    try:
        score = model.score(X, y)
    except Exception:
        score = float('nan')

print(f"✓ Model trained with R² = {score:.3f}")
PY


# ---- Test evaluation ----
echo "==> Test evaluation (NOT WORKING YET) ----"
${PYTHON_BIN} - <<'PY'

from data_handler import load_data
from train import train_model
from evaluate import evaluate_model

X_train, X_test, y_train, y_test = load_data()

res = train_model(X_train, y_train)

# Accept either signature: model OR (model, score)
if isinstance(res, tuple) and len(res) == 2:
    model, _ = res
else:
    model = res

metrics = evaluate_model(model, X_test, y_test)
print(f"✓ Evaluation complete: R² = {metrics['r2']:.3f}")
PY


echo "==> All local test steps completed."