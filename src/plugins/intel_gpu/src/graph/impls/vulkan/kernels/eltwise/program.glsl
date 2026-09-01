// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// The include order follows the shader's dependency graph. Each module owns one
// concern so variants share implementation without duplicating configuration.
#include "configuration.glsl"
#include "bindings.glsl"
#include "abi.glsl"
#include "metadata.glsl"
#include "storage.glsl"
#include "integer_math.glsl"
#include "broadcasting.glsl"
#include "operations.glsl"
#include "evaluation.glsl"
#include "output.glsl"
#include "dispatch.glsl"
