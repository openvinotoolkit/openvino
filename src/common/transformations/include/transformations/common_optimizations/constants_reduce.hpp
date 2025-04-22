// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include "transformations_visibility.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::pass {

class TRANSFORMATIONS_API ConstantsReduce : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ConstantsReduce");
    ConstantsReduce() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace ov::pass
