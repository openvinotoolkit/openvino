// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class NGRAPH_API LSTMStatesBroadcast;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief In case LSTMCell has constant initial hidden and cell state with single batch size
 * we make them broadcast-able by batch
 */

class ngraph::pass::LSTMStatesBroadcast : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("LSTMStatesBroadcast", "0");
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};
