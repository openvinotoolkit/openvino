// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/model.hpp"
#include "transformations_visibility.hpp"

namespace ov {

namespace pass {

void TRANSFORMATIONS_API transformMLIR(std::shared_ptr<ov::Model> model,
                                       std::shared_ptr<ov::EvaluationContext> loweringContext);

}
}
