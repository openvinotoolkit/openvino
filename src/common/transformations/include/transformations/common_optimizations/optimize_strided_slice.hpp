// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API StridedSliceOptimization;
class TRANSFORMATIONS_API UselessStridedSliceEraser;
class TRANSFORMATIONS_API SharedStridedSliceEraser;
class TRANSFORMATIONS_API GroupedStridedSliceOptimizer;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief UselessStridedSliceEraser transformation removes StridedSlice operations
 * with equal input and output shapes.
 */
class ov::pass::UselessStridedSliceEraser : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("UselessStridedSliceEraser", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief SharedStridedSliceEraser transformation replaces group of StridedSlice
 * operations with first StridedSlice in this group. All SrtideSlices in this group
 * must be equal and consume the same output port.
 */
class ov::pass::SharedStridedSliceEraser : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("SharedStridedSliceEraser", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GroupedStridedSliceOptimizer transformation replaces group of StridedSlice
 * operations with VariadicSplit. All StridedSlice operations must slice data
 * with the same axis and stride = 1.
 */
class ov::pass::GroupedStridedSliceOptimizer : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("GroupedStridedSliceOptimizer", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief StridedSliceOptimization transformation executes all transformations
 * related to StridedSlice optimizations.
 */
class ov::pass::StridedSliceOptimization : public ov::pass::ModelPass {
public:
    StridedSliceOptimization(bool use_shapes = true);

    OPENVINO_RTTI("StridedSliceOptimization", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    bool m_use_shapes = true;
};

namespace ngraph {
namespace pass {
using ov::pass::GroupedStridedSliceOptimizer;
using ov::pass::SharedStridedSliceEraser;
using ov::pass::StridedSliceOptimization;
using ov::pass::UselessStridedSliceEraser;
}  // namespace pass
}  // namespace ngraph
