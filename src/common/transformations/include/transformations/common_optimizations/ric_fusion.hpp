// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <openvino/core/model.hpp>
#include <openvino/pass/pass.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ReverseInputChannelsFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ReverseInputChannelsFusion
 */

class ngraph::pass::ReverseInputChannelsFusion : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ReverseInputChannelsFusion", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>&) override;
};
