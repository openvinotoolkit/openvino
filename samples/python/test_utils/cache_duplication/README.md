# Cache Duplication Test for Issue #31672

This directory contains a Python script that serves as a regression test for the bug identified in GitHub Issue [#31672](https://github.com/openvinotoolkit/openvino/issues/31672).

## Description

The bug caused multiple, non-identical cache files (`.blob`) to be generated when `core.compile_model` was called repeatedly with the same model and configuration.

This script (`reproduce_cache_bug_31672.py`) automates the following process:
1.  Creates a clean cache directory.
2.  Downloads and converts a simple Hugging Face model to OpenVINO IR.
3.  Calls `core.compile_model` in a loop.
4.  Calculates the hash of every generated `.blob` file.
5.  Passes if only one unique hash is found, and fails otherwise.

This test confirms that the caching mechanism is deterministic and prevents unnecessary storage use.
