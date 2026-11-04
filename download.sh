#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------
# Download CIFAR-10 (python version) and extract test_batch
# Save as: dataset/test_bench
# ---------------------------------------------

URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
OUT_DIR="dataset"
OUT_FILE="${OUT_DIR}/test_batch"

TARBALL="cifar-10-python.tar.gz"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

mkdir -p "${OUT_DIR}"

if [[ -f "${OUT_FILE}" ]]; then
  echo "[INFO] ${OUT_FILE} already exists. Skipping download."
  exit 0
fi

echo "[INFO] Downloading CIFAR-10 tarball..."
if command -v curl >/dev/null 2>&1; then
  curl -L --fail -o "${TMP_DIR}/${TARBALL}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${TMP_DIR}/${TARBALL}" "${URL}"
else
  echo "[ERROR] Neither curl nor wget is installed. Please install one of them."
  exit 1
fi

echo "[INFO] Extracting..."
tar -xzf "${TMP_DIR}/${TARBALL}" -C "${TMP_DIR}"

SRC_TEST_BATCH="${TMP_DIR}/cifar-10-batches-py/test_batch"
if [[ ! -f "${SRC_TEST_BATCH}" ]]; then
  echo "[ERROR] test_batch not found after extraction: ${SRC_TEST_BATCH}"
  exit 1
fi

echo "[INFO] Copying test_batch -> ${OUT_FILE}"
cp "${SRC_TEST_BATCH}" "${OUT_FILE}"

echo "[DONE] Saved CIFAR-10 test set to: ${OUT_FILE}"
