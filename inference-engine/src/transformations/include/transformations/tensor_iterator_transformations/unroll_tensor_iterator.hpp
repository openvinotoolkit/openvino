// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API UnrollTensorIterator;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Unrolls the body of the TensorIterator layer. Multiple body copies, the number of which is determined by
 * the number of iterations of the TensorIterator layer, are created and connected to each other and to the external
 * network. If the number of TensorIterator iterations is greater than 1, then additional Concat and Split layers
 * are added to the network.
 */

class ngraph::pass::UnrollTensorIterator: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    UnrollTensorIterator();
};
