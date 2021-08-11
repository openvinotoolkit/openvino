// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>


namespace ov {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ReshapeFullyConnected);

}  // namespace pass
}  // namespace ov

/*
 * Description:
 *     ReshapeFullyConnected transformation detects FullyConnected operations
 *     and for each operation where input shape is greater than 2 inserts Reshape
 *     operations before and after FullyConnected operation. This transformation is
 *     required because of IE restrictions.
 *
 * Parametrization:
 *     This transformation can be parametrize with callback. If you dont want to apply
 *     this transformation for some particular FullyConnected operations you can use
 *     setCallback method. See example below.
 *
 * Callback example:
 *
 *     // This callback disables ReshapeFullyConnected for FC with 3D input shapes
 *     auto callback = [](const std::shared_ptr<const ov::Node> & node) -> bool {
 *         if (auto fc_op = std::dynamic_pointer_cast<const ov::op::FullyConnected>(node)) {
 *             return fc_op->input_value(0).get_shape().size() == 3;
 *         }
 *     };
 *
 */

class ov::pass::ReshapeFullyConnected: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ReshapeFullyConnected();
};
