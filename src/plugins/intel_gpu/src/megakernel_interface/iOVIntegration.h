// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "iMegakernelRuntime.h"

namespace ov {
class Model;
}

namespace cldnn {
class primitive_inst;
}

namespace mk {

// Creates transformation that fuses eligible ops into a MegaKernel within the
// given model.
bool InsertMegakernelTransformation(const std::shared_ptr<ov::Model>& model);

// Implements binding between the MegaKernel runtime and the fused mk node
// created by the transformation. Returns constant parameters for
// the MegaKernel.
std::shared_ptr<const mk::IConstantParams> GetMegaKernelConstantParams(
    cldnn::primitive_inst* prim);

// Implements binding between the MegaKernel runtime and the fused mk node
// created by the transformation. Returns runtime parameters for
// the MegaKernel.
std::shared_ptr<const mk::IRuntimeParams> GetMegaKernelRuntimeParams(
    cldnn::primitive_inst* prim);
}  // namespace mk