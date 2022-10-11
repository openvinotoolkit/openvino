// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pass/manager.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class InitConstMask;
class InitMasks;
class PropagateMasks;
class ShrinkWeights;

class Pruning;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Initialising masks for pruned operations
 */
class ngraph::pass::InitMasks : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("InitMasks", "0");
    InitMasks();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Check Constant operation values by given dimensions and set
 * masks according to results that are bases on `condition` lambda function.
 * Works for Constant with floating point type (f16, f32, f64).
 */
class ngraph::pass::InitConstMask : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("InitConstMask", "0");
    explicit InitConstMask(
        const ngraph::AxisSet& dims,
        const std::function<bool(const double& value)>& condition = [](const double& value) {
            return value == 0;
        });
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Contains several MatcherPasses that initialize and propagate
 * masks from Constant operation to the network output.
 */
class ngraph::pass::PropagateMasks : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("PropagateMasks", "0");
    PropagateMasks();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Based on masks in Constant operation it inserts Gather operations
 * to shrink them. After this pass execution ConstantFolding is required.
 */
class ngraph::pass::ShrinkWeights : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("ShrinkWeights", "0");
    bool run_on_model(const std::shared_ptr<ngraph::Function>&) override;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief This is just a sequence of passes that performs pruning transformations pipeline
 */
class ngraph::pass::Pruning : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("Pruning", "0");
    bool run_on_model(const std::shared_ptr<Function>&) override;
};
