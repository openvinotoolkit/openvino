// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// List of functions to call on library unload.
// 
// To add a new unload function:
// 1. Add UNLOAD_FUNC(your_function_name) below
// 2. Define the function in your module: extern "C" void your_function_name() { /* cleanup code */ }
//
// Default weak implementations are auto-generated, so the function will do nothing if not defined.

UNLOAD_FUNC(itt_shutdown)
UNLOAD_FUNC(test_lib_shutdown)

