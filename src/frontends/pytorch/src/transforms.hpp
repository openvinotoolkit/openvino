// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/pytorch/decoder.hpp"
#include "openvino/frontend/pytorch/frontend.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

// Apply all transformations for finalize PT model conversion to OV
// This transformaitons cannot be implemented during on-the-fly 1:n translation logic so they are applied in separate
// round Input model is a partiall converted model with PT FW internal ops and FW Nodes, the result of the first round
// of translation.
void apply_pytorch_conversion_transforms(std::shared_ptr<ov::Model> model);

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov