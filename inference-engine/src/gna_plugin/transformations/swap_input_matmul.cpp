// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "swap_input_matmul.hpp"

#include <memory>
#include <vector>

#include <ngraph/pass/manager.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>


NGRAPH_RTTI_DEFINITION(ngraph::pass::SwapInputMatMul, "SwapInputMatMul", 0);

ngraph::pass::SwapInputMatMul::SwapInputMatMul() {
    auto matmul = pattern::wrap_type<opset1::MatMul>({pattern::any_input(pattern::has_static_shape()),
                                                      pattern::any_input(pattern::has_static_shape())},
                                                     pattern::has_static_shape());
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto matmul = std::dynamic_pointer_cast<ngraph::opset1::MatMul>(m.get_match_root());
        if (!matmul) {
            return false;
        }

        auto input_a = matmul->input(0).get_source_output();
        auto input_b = matmul->input(1).get_source_output();

        NodeVector new_ops;

        if (std::dynamic_pointer_cast<opset1::Constant>(input_a.get_node_shared_ptr())  ||
         std::dynamic_pointer_cast<opset1::FakeQuantize>(input_a.get_node_shared_ptr())) {
            auto new_matmul = std::make_shared<ngraph::opset1::MatMul>(input_b, input_a, matmul->get_transpose_b(), matmul->get_transpose_a());
            new_matmul->set_friendly_name(matmul->get_friendly_name());
            new_ops.push_back(new_matmul);

            ngraph::copy_runtime_info(matmul, new_ops);
            ngraph::replace_node(matmul, new_matmul);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matmul, "SwapInputMatMul");
    this->register_matcher(m, callback);
}
