// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "ngraph/opsets/opset.hpp"
#include "ngraph/pass/pass.hpp"
#include "transformations_visibility.hpp"
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ScaleInputs;

}  // namespace pass
}  // namespace ngraph


/**
 * @ingroup ie_transformation_common_api
 * @brief Adds 'scale' operation after each input
 */
class ngraph::pass::ScaleInputs : public ngraph::pass::MatcherPass {
public:
    enum class Version {
        IR_V10
    };
    NGRAPH_RTTI_DECLARATION;

    ScaleInputs(float scale_factor = 1.0f);

    ScaleInputs(const std::map<std::string, std::vector<float>> &scale_map);

private:
    void register_scale_matcher(ngraph::matcher_pass_callback callback);
};