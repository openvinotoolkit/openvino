// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API UnrollIf;

}  // namespace pass
}  // namespace ngraph

// clang-format off
/**
 * @ingroup ie_transformation_common_api
 * @brief The transformation replaces 'If' operations with one of the internal functions (bodies) if the provided condition is constant.
 * The condition is true: 'If' op is replaced with then_body
 * The condition is false 'If' op is replaced with else_body
 */
// clang-format on

class ngraph::pass::UnrollIf : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("UnrollIf", "0");
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};
