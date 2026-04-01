# Coverage Helpers

Coverage CI is driven by `.github/workflows/coverage.yml`.

`tools/coverage` now contains only the helper code still used by that workflow:
- suite definitions in `config/`
- suite test runners in `steps/`
- native C/C++ coverage collection in `steps/collect_cpp_coverage.py`
- summary and duration aggregation in `ci_reports.py`

Main entrypoint:
```bash
python3 tools/coverage/coverage.py step <run-cpp-tests|run-python-tests|run-js-tests|collect-cpp-coverage>
```

Additional helper commands:
```bash
python3 tools/coverage/coverage.py list-tests --suite <cpp|python|js> --profile <cpu|gpu|npu>
python3 tools/coverage/coverage.py validate-config
```

Important environment variables used by the workflow:
- `TEST_PROFILE`
- `BUILD_DIR`
- `INSTALL_PKG_DIR`
- `BIN_DIR`
- `CXX_TEST_NAMES`
- `PY_TEST_NAMES`
- `JS_TEST_NAMES`
- `COVERAGE_WRITE_STEP_SUMMARY=false`
