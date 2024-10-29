// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API EliminateConcat;
class TRANSFORMATIONS_API EliminateConvert;
class TRANSFORMATIONS_API EliminateConvertNonZero;
class TRANSFORMATIONS_API EliminateEltwise;
class TRANSFORMATIONS_API EliminateScatterUpdate;
class TRANSFORMATIONS_API EliminateReduceReshape;
class TRANSFORMATIONS_API EliminatePad;
class TRANSFORMATIONS_API EliminateSplit;
class TRANSFORMATIONS_API EliminateSplitConcat;
class TRANSFORMATIONS_API EliminateSqueeze;
class TRANSFORMATIONS_API EliminateUnsqueeze;
class TRANSFORMATIONS_API EliminateTranspose;
class TRANSFORMATIONS_API EliminateNopBroadcast;
class TRANSFORMATIONS_API EliminateSliceBeforeGatherElements;
class TRANSFORMATIONS_API EliminateStridedSlice;
class TRANSFORMATIONS_API EliminateSlice;
class TRANSFORMATIONS_API EliminateStridedSliceByShape;
class TRANSFORMATIONS_API NopElimination;
class TRANSFORMATIONS_API PrepareShapeOpsForEliminationAroundBE;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateReduceReshape eliminates Reshape from Reduce -> Reshape pattern
 */
class ov::pass::EliminateReduceReshape : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateReduceReshape", "0");
    EliminateReduceReshape();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminatePad eliminates pad that does nothing
 */
class ov::pass::EliminatePad : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminatePad", "0", ov::pass::MatcherPass);
    EliminatePad();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateConvert eliminates convert that does nothing
 */
class ov::pass::EliminateConvert : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateConvert", "0", ov::pass::MatcherPass);
    EliminateConvert();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateConvertNonZero eliminates convert before NonZero
 */
class ov::pass::EliminateConvertNonZero : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateConvertNonZero", "0", ov::pass::MatcherPass);
    EliminateConvertNonZero();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateConcat eliminates concat that does nothing
 */
class ov::pass::EliminateConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateConcat", "0", ov::pass::MatcherPass);
    EliminateConcat();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateSplit eliminates split that does nothing
 */
class ov::pass::EliminateSplit : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateSplit", "0", ov::pass::MatcherPass);
    EliminateSplit();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateSqueeze eliminates squeeze that does nothing
 */
class ov::pass::EliminateSqueeze : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateSqueeze", "0", ov::pass::MatcherPass);
    EliminateSqueeze();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateUnsqueeze eliminates squeeze that does nothing
 */
class ov::pass::EliminateUnsqueeze : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateUnsqueeze", "0", ov::pass::MatcherPass);
    EliminateUnsqueeze();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateTranspose eliminates transpose that does nothing
 */
class ov::pass::EliminateTranspose : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateTranspose", "0", ov::pass::MatcherPass);
    EliminateTranspose();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateEltwise eliminates eltwise ops that do nothing
 */
class ov::pass::EliminateEltwise : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateEltwise", "0", ov::pass::MatcherPass);
    EliminateEltwise();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateScatterUpdate eliminates scatter ops that do nothing (updates/indices are empty)
 */
class ov::pass::EliminateScatterUpdate : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateScatterUpdate", "0", ov::pass::MatcherPass);
    EliminateScatterUpdate();
};

class ov::pass::NopElimination : public GraphRewrite {
public:
    OPENVINO_RTTI("NopElimination", "0", GraphRewrite);
    NopElimination(bool use_shape_for_elimination = true);
};

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateSplit eliminates split+concat pairs which do nothing
 */
class ov::pass::EliminateSplitConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateSplitConcat", "0", ov::pass::MatcherPass);
    EliminateSplitConcat();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateNopBroadcast eliminates broadcast or tile with all ones on the second input
 */
class ov::pass::EliminateNopBroadcast : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateNopBroadcast", "0", ov::pass::MatcherPass);
    EliminateNopBroadcast();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateSliceBeforeGatherElements eliminates slice before GElements if slicing from 0
 * It is valid since GatherElements doesn't support negative indices and Slice won't affect
 * indexing of elements in the original tensor that GatherElements would like to take
 */
class ov::pass::EliminateSliceBeforeGatherElements : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateSliceBeforeGatherElements", "0", ov::pass::MatcherPass);
    EliminateSliceBeforeGatherElements();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateStridedSlice eliminates Strided Slice in case
 * tensors were not changed
 */
class ov::pass::EliminateStridedSlice : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateStridedSlice", "0", ov::pass::MatcherPass);
    EliminateStridedSlice();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateSlice eliminates Slice in case
 * tensors were not changed
 */
class ov::pass::EliminateSlice : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateSlice", "0", ov::pass::MatcherPass);
    EliminateSlice();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateStridedSlice eliminates Strided Slice in case
 * tensors were not changed
 */
class ov::pass::EliminateStridedSliceByShape : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateStridedSliceByShape", "0", ov::pass::MatcherPass);
    EliminateStridedSliceByShape();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief PrepareShapeOpsForEliminationAroundBE works on the subgraph like
 *  Reshape/Squeeze/Unsqueeze -> BinaryElementwiseOperation -> Reshape/Squeeze/Unsqueeze
 *  and prepares it for the following optimizations by moving bottom op up through Binary op
 */
class ov::pass::PrepareShapeOpsForEliminationAroundBE : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("PrepareShapeOpsForEliminationAroundBE", "0", ov::pass::MatcherPass);
    PrepareShapeOpsForEliminationAroundBE();
};
