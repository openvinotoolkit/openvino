// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/model.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ReverseInputChannelsFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ReverseInputChannelsFusion
 */

class ov::pass::ReverseInputChannelsFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ReverseInputChannelsFusion");
    bool run_on_model(const std::shared_ptr<ov::Model>&) override;
};
