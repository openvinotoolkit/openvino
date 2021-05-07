// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include "squeeze.hpp"
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph {
    namespace frontend {
        namespace pdpd {
            namespace op {

                NamedOutputs squeeze (const NodeContext& node) {
                    auto data = node.get_ng_input("X");
                    auto axes = node.get_attribute<std::vector<int32_t>>("axes");
                    PDPD_ASSERT(data.get_partial_shape().rank().is_static(), "squeeze: X rank must be static!");

                    auto shape = data.get_partial_shape().to_shape();
                    for (auto &&i : axes) {
                        auto idx = i;
                        if (idx < 0) {
                            idx = i + shape.size();
                        }
                        PDPD_ASSERT(shape[idx] == 1, "squeeze: the specified dimension is not equal to one.");
                    }

                    auto axesNode = ngraph::opset6::Constant::create(ngraph::element::i32, {axes.size()}, axes);
                    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Squeeze>(data, axesNode)}, {"Out"});
                }

            }}}}