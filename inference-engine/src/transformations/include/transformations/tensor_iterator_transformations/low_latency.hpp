// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/opsets/opset4.hpp>
#include <transformations_visibility.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

    class TRANSFORMATIONS_API LSTMLowLatency;
    class TRANSFORMATIONS_API GRULowLatency;
    class TRANSFORMATIONS_API RNNLowLatency;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief
 * todo: fill
 */

class ngraph::pass::LSTMLowLatency: public ngraph::pass::FunctionPass {
public:
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

class ngraph::pass::GRULowLatency: public ngraph::pass::FunctionPass {
public:
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

class ngraph::pass::RNNLowLatency: public ngraph::pass::FunctionPass {
public:
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};