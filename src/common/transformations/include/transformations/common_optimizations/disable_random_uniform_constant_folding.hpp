// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API DisableRandomUniformConstantFolding;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Disables ConstantFolding for RandomUniform operation. It is required as RandomUniform
 * should generate new sequence each run.
 */
class ngraph::pass::DisableRandomUniformConstantFolding : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("DisableRandomUniformConstantFolding", "0");
    DisableRandomUniformConstantFolding();
};
