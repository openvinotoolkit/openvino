#!/usr/bin/env bash

export WORK_DIR=/opt
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH}"

. penv/bin/activate

for s in tests/test_*.py; do
    if [ -z ${PYTEST_FILTER} ]; then
        pytest -x -r 'A' -s --verbose ${s}
    else
        pytest -x -r 'A' -s --verbose -k "${PYTEST_FILTER}" ${s}
    fi
    VAL=$?
    if [ $VAL != 0 ]
    then
        RET=$VAL
    fi
done

exit $RET
