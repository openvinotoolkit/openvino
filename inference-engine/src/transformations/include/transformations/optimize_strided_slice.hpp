// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/slice_plan.hpp>
#include <ngraph/util.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API StridedSliceOptimization;
class TRANSFORMATIONS_API UselessStridedSliceEraser;
class TRANSFORMATIONS_API SharedStridedSliceEraser;
class TRANSFORMATIONS_API GroupedStridedSliceOptimizer;

}  // namespace pass
}  // namespace ngraph


/**
 * @ingroup ie_transformation_common_api
 * @brief UselessStridedSliceEraser transformation removes StridedSlice operations
 * with equal input and output shapes.
 */
class ngraph::pass::UselessStridedSliceEraser: public ngraph::pass::FunctionPass {
public:
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief SharedStridedSliceEraser transformation replaces group of StridedSlice
 * operations with first StridedSlice in this group. All SrtideSlices in this group
 * must be equal and consume the same output port.
 */
class ngraph::pass::SharedStridedSliceEraser: public ngraph::pass::FunctionPass {
public:
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GroupedStridedSliceOptimizer transformation replaces group of StridedSlice
 * operations with VariadicSplit. All StridedSlice operations must slice data
 * with the same axis and stride = 1.
 */
class ngraph::pass::GroupedStridedSliceOptimizer: public ngraph::pass::FunctionPass {
public:
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief StridedSliceOptimization transformation executes all transformations
 * related to StridedSlice optimizations.
 */
class ngraph::pass::StridedSliceOptimization: public ngraph::pass::FunctionPass {
public:
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override {
        bool rewritten = UselessStridedSliceEraser().run_on_function(f);
        rewritten |= SharedStridedSliceEraser().run_on_function(f);
        rewritten |= GroupedStridedSliceOptimizer().run_on_function(f);
        return rewritten;
    }
};
