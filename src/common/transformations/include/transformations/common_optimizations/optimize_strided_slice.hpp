// Copyright (C) 2018-2025 Intel Corporation
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
class TRANSFORMATIONS_API GroupedStridedSliceOptimizer;
class TRANSFORMATIONS_API GroupedSliceToVSplitOptimization;
class TRANSFORMATIONS_API SliceSequenceToSingleSlice;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief UselessSliceEraser transformation removes Slice/StridedSlice operations
 * with equal input and output shapes.
 */
class ov::pass::UselessSliceEraser : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("UselessSliceEraser");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

/**
 * @ingroup ov_transformation_common_api
 * @brief GroupedStridedSliceOptimizer transformation replaces group of StridedSlice
 * operations with VariadicSplit. All StridedSlice operations must slice data
 * with the same axis and stride = 1.
 */
class ov::pass::GroupedStridedSliceOptimizer : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("GroupedStridedSliceOptimizer");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

/**
 * @ingroup ov_transformation_common_api
 * @brief GroupedSliceToVSplitOptimization transformation replaces group of Slice
 * operations with VariadicSplit. All Slice operations must slice data
 * with the same axis and step = 1.
 */
class ov::pass::GroupedSliceToVSplitOptimization : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("GroupedSliceToVSplitOptimization");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

/**
 * @ingroup ov_transformation_common_api
 * @brief SliceSequenceToSingleSlice transformation replaces group of Slice
 * operations with single Slice. All Slice operations must slice data
 * with the different axis.
 *
 * Before:
 * data (shape: 2, 3, 4) -> Slice (axis 0) -> Slice (axis 1) -> Slice (axis 2)
 *
 * After:
 * data (shape: 2, 3, 4) -> Slice (axes: 0, 1, 2)
 */
class ov::pass::SliceSequenceToSingleSlice : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SliceSequenceToSingleSlice");
    SliceSequenceToSingleSlice();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief StridedSliceOptimization transformation executes all transformations
 * related to StridedSlice optimizations.
 */
class ov::pass::StridedSliceOptimization : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("StridedSliceOptimization");
    StridedSliceOptimization(bool use_shapes = true);

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    bool m_use_shapes = true;
};
