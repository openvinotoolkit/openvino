# Transformation profiling

Enables profiling of transformation passes to log their execution times.

## Usage

Set the `OV_ENABLE_PROFILE_PASS` environment variable to `"true"` to enable profiling.
Alternatively, specify a file path where the execution times will be saved.

Example:
```
export OV_ENABLE_PROFILE_PASS=true
export OV_ENABLE_PROFILE_PASS="/path/to/save/profiling/results"
```
