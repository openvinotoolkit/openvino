// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API AlignMixedFP32FP16Types;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief AlignMixedFP32FP16Types adds Converts to keep mixed FP16/FP32 graph type consistent
 */
class ov::pass::AlignMixedFP32FP16Types : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("AlignMixedFP32FP16Types");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
