# BTL Function Database

This directory contains the Balanced Ternary Logic (BTL) function database used by the TSSN kernel.

## `btl_function_database.json`

This file contains the definitions of the BTL functions, including their truth tables, algebraic properties, and semantic profiles.

**NOTE:** The current version of this file contains a **subset** of the full 1,444 NPN classes for demonstration and development purposes.

### For Full Production:
To enable the full power of the Gradient-Guided Ternary Function Library (GGTFL), you must replace `btl_function_database.json` with the full version generated from the research data.

1.  Locate the full `btl_function_database.json` (approx 1.3 MB).
2.  Place it in this directory (`src/custom_ops/btl/`) or in the root of your execution directory.
3.  Alternatively, set the `CYBERSPORE_BTL_DB_PATH` environment variable to point to the full file.

## `btl_function_library.cpp`

The C++ library is designed to load this JSON file at runtime. It uses `nlohmann/json` for parsing.
If the file cannot be loaded, it falls back to a minimal embedded set of functions.
