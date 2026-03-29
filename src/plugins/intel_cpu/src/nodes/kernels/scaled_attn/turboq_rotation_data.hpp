// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace Extensions {
namespace Cpu {

// Returns pointer to the cached dim×dim rotation matrix Q (row-major).
// Allocated on first call for a given dim; read-only after init.
const float* turboq_get_rotation_matrix(int dim);

// Returns pointer to the cached dim×dim transpose Q^T (row-major).
const float* turboq_get_rotation_matrix_t(int dim);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov
