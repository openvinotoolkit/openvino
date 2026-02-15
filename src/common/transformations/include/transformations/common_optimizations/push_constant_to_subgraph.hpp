// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief PushConstantToSubgraph transformation detects MultiSubGraphOp inputs
 * that can be constfoldable pushes that inputs to subgraphs.
 */
class TRANSFORMATIONS_API PushConstantToSubgraph : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("PushConstantToSubgraph");
    bool run_on_model(const std::shared_ptr<Model>& model) override;
};

}  // namespace pass
}  // namespace ov
