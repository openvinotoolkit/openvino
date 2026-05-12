// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/model.hpp"
#include "openvino/util/env_util.hpp"
#include "transformations_visibility.hpp"

namespace ov {

namespace pass {

inline bool is_mlir_transform_enabled() {
    return !util::getenv_string("OV_MLIR_MODE").empty() && util::getenv_bool("OV_MLIR", true);
}

void TRANSFORMATIONS_API transformMLIR(std::shared_ptr<ov::Model> model,
                                       std::shared_ptr<ov::EvaluationContext> loweringContext);

}
}
