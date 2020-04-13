// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

#include "transformations/utils/pass_param.hpp"

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ReshapeFullyConnected);

}  // namespace pass
}  // namespace ngraph

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
 *     auto callback = [](const std::shared_ptr<const ngraph::Node> & node) -> bool {
 *         if (auto fc_op = std::dynamic_pointer_cast<const ngraph::op::FullyConnected>(node)) {
 *             return fc_op->input_value(0).get_shape().size() == 3;
 *         }
 *     };
 *
 *     auto p = ngraph::pass::ReshapeFullyConnected();
 *     p.setCallback(callback);
 *     p.run_on_function(f);
 *
 */

class ngraph::pass::ReshapeFullyConnected: public ngraph::pass::GraphRewrite, public ngraph::pass::PassParam {
public:
    ReshapeFullyConnected() : GraphRewrite(), PassParam() {
        reshape_fully_connected();
    }

private:
    void reshape_fully_connected();
};
