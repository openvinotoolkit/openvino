// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API UnrollTensorIterator;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Unrolls the body of the TensorIterator layer. Multiple body copies, the number of which is determined by
 * the number of iterations of the TensorIterator layer, are created and connected to each other and to the external
 * network. If the number of TensorIterator iterations is greater than 1, then additional Concat and Split layers
 * are added to the network.
 */

class ov::pass::UnrollTensorIterator: public ov::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<Function>) override;
};
