// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/multi_matcher.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FuseMOEExperts : public ov::pass::MultiMatcher {
public:
    OPENVINO_RTTI("FuseMOEExperts");
    FuseMOEExperts();
};

// class TRANSFORMATIONS_API FuseMOE : public ModelPass {
// public:
//     OPENVINO_MODEL_PASS_RTTI("FuseMOE");
//     bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
// };

}  // namespace pass
}  // namespace ov

