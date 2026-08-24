// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/// \brief Prevents generic precision conversion from changing SelectiveSSM operations and their inputs.
class TRANSFORMATIONS_API PreserveSelectiveSSMPrecision final : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("PreserveSelectiveSSMPrecision");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::pass
