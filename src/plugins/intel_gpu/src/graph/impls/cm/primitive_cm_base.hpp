// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "impls/ocl_v2/primitive_ocl_base.hpp"

namespace ov::intel_gpu::cm {

// No changes are needed from implementation standpoint, so just use alias in case if we need some changes in the future
using PrimitiveImplCM = ov::intel_gpu::ocl::PrimitiveImplOCL;
using Stage = ov::intel_gpu::ocl::Stage;

}  // namespace ov::intel_gpu::cm
