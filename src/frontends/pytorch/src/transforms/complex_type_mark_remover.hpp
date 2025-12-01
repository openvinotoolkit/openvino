// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

/**
 * ComplexTypeMarkRemover transformation
 *
 * Removes ComplexTypeMark nodes from the graph by replacing them with their
 * underlying data representation (floating-point tensor with shape [..., 2]).
 *
 * ComplexTypeMark nodes are created during translation to track complex tensors,
 * since OpenVINO doesn't have native complex type support. These marker nodes
 * should be removed before model finalization, otherwise they will be detected
 * as unconverted operations by the validation logic in frontend.cpp.
 *
 * This is similar to QuantizedNodeRemover which removes QuantizedPtNode markers.
 */
class ComplexTypeMarkRemover : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::ComplexTypeMarkRemover");
    ComplexTypeMarkRemover();
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
