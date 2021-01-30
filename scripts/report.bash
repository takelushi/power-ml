#!/bin/bash

readonly REPORT_DIR='./report'
status=0

function log() {
   date_str=$(date '+%Y-%m-%dT%H:%M:%S')
   echo "[${date_str}]${1}"
}

rm -r -d -f "${REPORT_DIR}"
mkdir -p "${REPORT_DIR}"

exec 1> >(tee -a -i "${REPORT_DIR}/stdout.txt")
exec 2> >(tee -a -i "${REPORT_DIR}/stderr.txt")

log "Create test report..."

echo ''
echo '============================================================'
echo ' flake8'
echo '============================================================'
flake8 --format=html "--htmldir=${REPORT_DIR}/flake-report" src/ tests/
exit_code=$?
if [ $exit_code -ne 0 ]; then
   status=1
fi
echo ''
log "flake8 exit code: ${exit_code}"

echo ''
echo '============================================================'
echo ' mypy'
echo '============================================================'
mypy src/ tests/ --html-report "${REPORT_DIR}/mypy"
exit_code=$?
if [ $exit_code -ne 0 ]; then
   status=1
fi
echo ''
log "mypy exit code: ${exit_code}"

echo ''
echo '============================================================'
echo ' pytest'
echo '============================================================'
pytest \
   "--html=${REPORT_DIR}/pytest/index.html" \
   --cov-report "html:${REPORT_DIR}/coverage"
exit_code=$?
if [ $exit_code -ne 0 ]; then
   status=1
fi
echo ''
log "pytest exit code: ${exit_code}"

echo ''
log "The report was exported. See ${REPORT_DIR} directory."
echo "$(ls "${REPORT_DIR}")"
log "Status was ${status}"

exit $status
