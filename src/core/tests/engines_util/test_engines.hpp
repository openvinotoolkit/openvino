// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Builds a class name for a given backend prefix
// The prefix should come from cmake
// Example: INTERPRETER -> INTERPRETER_Engine
// Example: IE_CPU -> IE_CPU_Engine
#define ENGINE_CLASS_NAME(backend) backend##_Engine
