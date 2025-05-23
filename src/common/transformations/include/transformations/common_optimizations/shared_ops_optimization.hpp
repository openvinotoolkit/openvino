// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SharedOpOptimization;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief SharedOpOptimization optimizes operations which are
 * sourcing from same Output<Node> and perform the same action on the same data
 */
class ov::pass::SharedOpOptimization : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SharedOpOptimization");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
