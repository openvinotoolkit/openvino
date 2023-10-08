// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API StridedSliceOptimization;
class TRANSFORMATIONS_API UselessSliceEraser;
class TRANSFORMATIONS_API SharedStridedSliceEraser;
class TRANSFORMATIONS_API GroupedStridedSliceOptimizer;
class TRANSFORMATIONS_API GroupedSliceToVSplitOptimization;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief UselessSliceEraser transformation removes Slice/StridedSlice operations
 * with equal input and output shapes.
 */
class ov::pass::UselessSliceEraser : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("UselessSliceEraser", "0");
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
 * @brief GroupedSliceToVSplitOptimization transformation replaces group of Slice
 * operations with VariadicSplit. All Slice operations must slice data
 * with the same axis and step = 1.
 */
class ov::pass::GroupedSliceToVSplitOptimization : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("GroupedSliceToVSplitOptimization", "0");
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
