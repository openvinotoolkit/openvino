// Copyright (C) 2018-2023 Intel Corporation
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
class TRANSFORMATIONS_API EliminatePad;
class TRANSFORMATIONS_API EliminateSplit;
class TRANSFORMATIONS_API EliminateSplitConcat;
class TRANSFORMATIONS_API EliminateSqueeze;
class TRANSFORMATIONS_API EliminateTranspose;
class TRANSFORMATIONS_API EliminateNopBroadcast;
class TRANSFORMATIONS_API NopSliceBeforeGatherElements;
class TRANSFORMATIONS_API NopElimination;
class TRANSFORMATIONS_API PrepareShapeOpsForEliminationAroundBE;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminatePad eliminates pad that does nothing
 */
class ov::pass::EliminatePad : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminatePad", "0");
    EliminatePad();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateConvert eliminates convert that does nothing
 */
class ov::pass::EliminateConvert : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateConvert", "0");
    EliminateConvert();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateConvertNonZero eliminates convert before NonZero
 */
class ov::pass::EliminateConvertNonZero : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateConvertNonZero", "0");
    EliminateConvertNonZero();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateConcat eliminates concat that does nothing
 */
class ov::pass::EliminateConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateConcat", "0");
    EliminateConcat();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateSplit eliminates split that does nothing
 */
class ov::pass::EliminateSplit : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateSplit", "0");
    EliminateSplit();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateSqueeze eliminates squeeze that does nothing
 */
class ov::pass::EliminateSqueeze : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateSqueeze", "0");
    EliminateSqueeze();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateTranspose eliminates transpose that does nothing
 */
class ov::pass::EliminateTranspose : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateTranspose", "0");
    EliminateTranspose();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateEltwise eliminates eltwise ops that do nothing
 */
class ov::pass::EliminateEltwise : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateEltwise", "0");
    EliminateEltwise();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateScatterUpdate eliminates scatter ops that do nothing (updates/indices are empty)
 */
class ov::pass::EliminateScatterUpdate : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateScatterUpdate", "0");
    EliminateScatterUpdate();
};

class ov::pass::NopElimination : public GraphRewrite {
public:
    OPENVINO_RTTI("NopElimination", "0");
    NopElimination(bool use_shape_for_elimination = true);
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateSplit eliminates split+concat pairs which do nothing
 */
class ov::pass::EliminateSplitConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateSplitConcat", "0");
    EliminateSplitConcat();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateNopBroadcast eliminates broadcast or tile with all ones on the second input
 */
class ov::pass::EliminateNopBroadcast : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateNopBroadcast", "0");
    EliminateNopBroadcast();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief NopSliceBeforeGatherElements eliminates slice before GElements if slicing from 0
 * It is valid since GatherElements doesn't support negative indices and Slice won't affect
 * indexing of elements in the original tensor that GatherElements would like to take
 */
class ov::pass::NopSliceBeforeGatherElements : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("NopSliceBeforeGatherElements", "0");
    NopSliceBeforeGatherElements();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief PrepareShapeOpsForEliminationAroundBE works on the subgraph like
 *  Reshape/Squeeze/Unsqueeze -> BinaryElementwiseOperation -> Reshape/Squeeze/Unsqueeze
 *  and prepares it for the following optimizations by moving bottom op up through Binary op
 */
class ov::pass::PrepareShapeOpsForEliminationAroundBE : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("PrepareShapeOpsForEliminationAroundBE", "0");
    PrepareShapeOpsForEliminationAroundBE();
};
