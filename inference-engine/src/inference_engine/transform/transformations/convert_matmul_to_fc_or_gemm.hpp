// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph_ops/fully_connected.hpp>
#include <ngraph_ops/gemm.hpp>
#include <string>
#include <utility>
#include <vector>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/experimental/dyn_reshape.hpp"
#include "ngraph/op/experimental/transpose.hpp"
#include "ngraph/op/fused/matmul.hpp"

namespace ngraph {
namespace pass {

class ConvertMatMulToFCorGemm;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertMatMulToFCorGemm: public ngraph::pass::GraphRewrite {
public:
    ConvertMatMulToFCorGemm(): GraphRewrite() {
        convert_matmul();
    }

private:
    void convert_matmul() {
        auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape {1, 1});
        auto input_1 = std::make_shared<pattern::op::Label>(element::f32, Shape {1, 1});
        auto matmul = std::make_shared<ngraph::op::MatMul>(input_0, input_1);

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            auto matmul = std::dynamic_pointer_cast<ngraph::op::MatMul>(m.get_match_root());
            if (!matmul) {
                return false;
            }

            auto input_a = matmul->input(0).get_source_output();
            auto input_b = matmul->input(1).get_source_output();

            auto shape_a = input_a.get_shape();
            auto shape_b = input_b.get_shape();
            auto output_shape = matmul->get_shape();

            /*
             *  get_aligned_shapes function align two input shapes to have the same size and
             *  the same batch dimensions (last two dimensions are not comparable).
             *  It also checks that dimensions are compatible so in case with two shapes
             *  for example: [2, 32, 64] [3, 64, 64] it will raise an exception.
             */

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
                        THROW_IE_EXCEPTION << "Shapes can't be aligned: " << shape_a_aligned << " " << shape_b_aligned;
                    }
                    size_t max_value = std::max(shape_a_aligned[i], shape_b_aligned[i]);
                    shape_a_aligned[i] = shape_b_aligned[i] = max_value;
                }

                return {shape_a_aligned, shape_b_aligned};
            };

            /*
             *  create_transpose function return Transpose operation to replace transpose_a or transpose_b
             *  arguments with an operation. In other words in this function we create Transpose operation
             *  with order length equal to output_shape length of given node and fill order with increasing
             *  sequence starting from 0 and replace last two dimension. For example for length = 4  the
             *  order will be [0, 1, 3, 2] that emulates transpose_a or transpose_b attribute.
             */

            auto create_transpose = [](Output<Node> node, const std::string& transpose_name) -> std::shared_ptr<Node> {
                Shape output_shape = node.get_node_shared_ptr()->get_shape();

                std::vector<size_t> transpose_order(output_shape.size());
                std::iota(transpose_order.begin(), transpose_order.end(), 0);
                std::swap(*(transpose_order.end() - 1), *(transpose_order.end() - 2));

                auto transpose = std::make_shared<ngraph::op::Transpose>(
                    node, op::Constant::create(element::i64, Shape {transpose_order.size()}, transpose_order));
                transpose->set_friendly_name(transpose_name);
                return transpose;
            };

            // fc_input_a and fc_input_b - are the final inputs that will be set to FullyConnected of GemmIE operations.
            // So in case of adding new operations that takes matmul inputs we need keep update fc_input_a and
            // fc_input_b updated.
            auto fc_input_a = input_a, fc_input_b = input_b;

            // Check that if second inputs is Constant operation and it's shape without ones dimensions has length <= 2
            // we replace MatMul with FullyConnected operation.
            // Otherwise we replace MatMul with Gemm.
            if ((std::dynamic_pointer_cast<op::Constant>    (fc_input_b.get_node_shared_ptr())  ||
                 std::dynamic_pointer_cast<op::FakeQuantize>(fc_input_b.get_node_shared_ptr())) &&
                std::count_if(shape_b.begin(), shape_b.end(), [](size_t x) {
                    return x != 1;
                }) <= 2) {
                Shape shape_a_aligned, shape_b_aligned;
                std::tie(shape_a_aligned, shape_b_aligned) = get_aligned_shapes();

                if (shape_a_aligned.size() < 2 || shape_b_aligned.size() < 2) {
                    THROW_IE_EXCEPTION << "MatMul " << matmul->get_friendly_name() << " shapes are inconsistent.";
                }

                // Transferring from MatMul representation: [B, I, K] * [B, K, O] = [B, I, O]
                // to FullyConnected representation: [I, K] * [O, K] = [I, O]
                size_t K = *(shape_a_aligned.end() - 1);
                size_t O = *(shape_b_aligned.end() - 1);
                Shape B(shape_a_aligned.begin(), shape_a_aligned.end() - 2);

                // Weights normalization
                if (!matmul->get_transpose_b()) {
                    fc_input_b = create_transpose(fc_input_b, matmul->get_friendly_name() + "/transpose_b");
                }

                if (shape_b.size() != 2) {
                    auto reshape_shape =
                        op::Constant::create<int64_t>(element::i64, Shape {2}, {-1ll, static_cast<int64_t>(K)});
                    fc_input_b = std::make_shared<op::v1::Reshape>(fc_input_b, reshape_shape, true);
                }

                // Input normalization
                if (matmul->get_transpose_a()) {
                    fc_input_a = create_transpose(fc_input_a, matmul->get_friendly_name() + "/transpose_a");
                }

                if (shape_a.size() != 2) {
                    auto reshape_shape =
                        op::Constant::create<int64_t>(element::i64, Shape {2}, {-1ll, static_cast<int64_t>(K)});
                    fc_input_a = std::make_shared<op::v1::Reshape>(fc_input_a, reshape_shape, true);
                }

                // Create FullyConnected
                std::vector<float> bias_value(O, 0);
                auto fc_bias = op::Constant::create(matmul->get_input_element_type(0), Shape {O}, bias_value);

                std::shared_ptr<Node> last = std::make_shared<op::FullyConnected>(fc_input_a, fc_input_b, fc_bias);
                if (last != nullptr) {
                    last->set_friendly_name(matmul->get_friendly_name());
                }

                // Output normalization
                if (output_shape.size() != 2) {
                    last = std::make_shared<op::v1::Reshape>(
                        last, op::Constant::create(element::i64, Shape {output_shape.size()}, output_shape), true);
                }
                ngraph::replace_node(matmul, last);
            } else {
                // WA for IE that Gemm must have inputs with the same length.
                if (shape_b.size() != shape_a.size()) {
                    auto & reshape_port = (shape_a.size() < shape_b.size() ? fc_input_a : fc_input_b);
                    Shape to_insert(shape_a.size() < shape_b.size() ? shape_b.size() - shape_a.size() : shape_a.size() - shape_b.size(), 1);
                    Shape reshape_shape(reshape_port.get_shape());
                    reshape_shape.insert(reshape_shape.begin(), to_insert.begin(), to_insert.end());

                    auto reshape_shape_const =
                        op::Constant::create(element::i64, Shape {reshape_shape.size()}, reshape_shape);
                    reshape_port = std::make_shared<op::v1::Reshape>(reshape_port, reshape_shape_const, true);
                    reshape_port.get_node_shared_ptr()->set_friendly_name(matmul->get_friendly_name() + "/reshape");
                }
                auto gemm = std::make_shared<op::GemmIE>(fc_input_a, fc_input_b, matmul->get_transpose_a(),
                                                         matmul->get_transpose_b(), output_shape);
                gemm->set_friendly_name(matmul->get_friendly_name());
                ngraph::replace_node(matmul, gemm);
            }

            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(matmul, "ConvertMatMulToFCorGemm");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
