g// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <numeric>
#include <legacy/ngraph_ops/fully_connected.hpp>
#include <ngraph/rt_info.hpp>
#include "handle_gemm_layer.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::HandleGemmLayerPass, "HandleGemmLayerPass", 0);

ngraph::pass::HandleGemmLayerPass::HandleGemmLayerPass() {
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

        auto shape_a = input_a.get_shape();
        auto shape_b = input_b.get_shape();
        auto output_shape = matmul->get_shape();

        auto fc_input_a = input_a, fc_input_b = input_b;
        NodeVector new_ops;

        auto get_aligned_shapes = [shape_a, shape_b, &matmul]() -> std::pair<Shape, Shape> {
            Shape shape_a_aligned(shape_a), shape_b_aligned(shape_b);
            size_t max_size = std::max(shape_a_aligned.size(), shape_b_aligned.size());
            for (size_t i = 0, cnt = max_size - shape_a_aligned.size(); i < cnt; ++i)
                shape_a_aligned.insert(shape_a_aligned.begin(), 1);
            for (size_t i = 0, cnt = max_size - shape_b_aligned.size(); i < cnt; ++i)
                shape_b_aligned.insert(shape_b_aligned.begin(), 1);

            if (matmul->get_transpose_a()) {
                std::swap(*(shape_a_aligned.end() - 1), *(shape_a_aligned.end() - 2));
            }
            if (matmul->get_transpose_b()) {
                std::swap(*(shape_b_aligned.end() - 1), *(shape_b_aligned.end() - 2));
            }

            for (size_t i = 0; i < max_size - 2; ++i) {
                if (shape_a_aligned[i] != shape_b_aligned[i] && shape_a_aligned[i] > 1 && shape_b_aligned[i] > 1) {
                    std::ostringstream stream;
                    stream << "Shapes can't be aligned: " << shape_a_aligned << " " << shape_b_aligned;
                    throw ngraph_error(stream.str());
                }
                size_t max_value = std::max(shape_a_aligned[i], shape_b_aligned[i]);
                shape_a_aligned[i] = shape_b_aligned[i] = max_value;
            }

            return {shape_a_aligned, shape_b_aligned};
        };

        auto create_transpose = [this](Output<Node> node, const std::string& transpose_name) -> std::shared_ptr<Node> {
            Shape output_shape = node.get_node_shared_ptr()->get_shape();

            std::vector<size_t> transpose_order(output_shape.size());
            std::iota(transpose_order.begin(), transpose_order.end(), 0);
            std::swap(*(transpose_order.end() - 1), *(transpose_order.end() - 2));

            auto transpose = register_new_node<ngraph::opset1::Transpose>(
                    node, opset1::Constant::create(element::i64, Shape {transpose_order.size()}, transpose_order));
            transpose->set_friendly_name(transpose_name);
            return transpose;
        };

        if ((std::dynamic_pointer_cast<opset1::Constant>    (fc_input_a.get_node_shared_ptr())  ||
             std::dynamic_pointer_cast<opset1::FakeQuantize>(fc_input_a.get_node_shared_ptr())) &&
            std::count_if(shape_b.begin(), shape_b.end(), [](size_t x) {
                return x != 1;
            }) <= 2) {
            Shape shape_a_aligned, shape_b_aligned;
            std::tie(shape_a_aligned, shape_b_aligned) = get_aligned_shapes();

            if (shape_a_aligned.size() < 2 || shape_b_aligned.size() < 2) {
                throw ngraph_error("MatMul " + matmul->get_friendly_name() + " shapes are inconsistent.");
            }

//            size_t K = *(shape_a_aligned.end() - 1);
            size_t O = *(shape_b_aligned.end() - 1);
            Shape B(shape_a_aligned.begin(), shape_a_aligned.end() - 2);

            fc_input_b = create_transpose(fc_input_b, matmul->get_friendly_name() + "/transpose_b");
            new_ops.push_back(fc_input_b.get_node_shared_ptr());

            fc_input_a = create_transpose(fc_input_a, matmul->get_friendly_name() + "/transpose_a");
            new_ops.push_back(fc_input_a.get_node_shared_ptr());
            std::vector<float> bias_value(O, 0);
            auto fc_bias = opset1::Constant::create(matmul->get_output_element_type(0), Shape {O}, bias_value);
            auto fc = std::make_shared<op::FullyConnected>(fc_input_b, fc_input_a, fc_bias, output_shape, matmul->output(0).get_element_type());
            fc->set_friendly_name(matmul->get_friendly_name());
            new_ops.push_back(fc);

            auto transpose_out = create_transpose(fc, matmul->get_friendly_name() + "/out_transpose");
            new_ops.push_back(transpose_out);

            ngraph::copy_runtime_info(matmul, new_ops);
            ngraph::replace_node(matmul, transpose_out);
            return true;
        }
        return false;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(matmul, "HandleGemmLayerPass");
    this->register_matcher(m, callback);
}