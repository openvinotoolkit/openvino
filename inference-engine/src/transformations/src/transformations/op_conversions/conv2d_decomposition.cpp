// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/conv2d_decomposition.hpp"

#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::Conv2dDecomposition, "Conv2dDecomposition", 0);
bool ngraph::pass::Conv2dDecomposition::run_on_function(std::shared_ptr<ngraph::Function> f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto conv = std::dynamic_pointer_cast<ngraph::opset1::Convolution> (node);
        if (nullptr == conv || transformation_callback(conv)) {
            return false;
        }
        const Output<Node>& input = conv->input_value(0);
        const Output<Node>& kernels = conv->input_value(1);

        bool isValid = true;
        for (auto p : conv->get_pads_begin())
            isValid &= (p == 0);
        for (auto p : conv->get_pads_end())
            isValid &= (p == 0);

        int input_height = 1;
        int input_channel_count = 1;

        if (input_height == 1) {
            // valid convolution is supported by GNA
            if (isValid) continue;
            std::shared_ptr<ngraph::opset1::Concat> padded_row_concat;
            if (0 == input_channel_count % 32)
            {
                // calculate padding
                size_t flat_left_padding = input_channel_count * pads_begin_x;
                size_t flat_right_padding = input_channel_count * pads_end_x;
                // check if zero const of that size exists
                // if not create zero const of size equal to left padding
                // the same for right padding

                // 
                //
            } else {

            }
            //number of channels is k * 32
            //just concat
            //Variadic split

        } else {
            OutputVector concat_inputs;
            for (int h = 0; h < input_height; h++) {

            }
            concat
        }

        ngraph::copy_runtime_info(conv, nullptr);
        ngraph::replace_node(conv, result);
        is_graph_modfied = true;;
    }
    return is_graph_modfied;
}
