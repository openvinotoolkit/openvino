// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/fully_connected.hpp"
#include "vpu/ngraph/transformations/convert_matmul_to_fc.hpp"
#include <ngraph/rt_info.hpp>

#include <algorithm>
#include <utility>
#include <memory>
#include <vector>
#include <string>
#include <numeric>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <transformations/utils/utils.hpp>

NGRAPH_RTTI_DEFINITION(vpu::ConvertMatMulToFC, "ConvertMatMulToFC", 0);

namespace vpu {
ConvertMatMulToFC::ConvertMatMulToFC() {
    auto matmul = ngraph::pattern::wrap_type<ngraph::opset1::MatMul>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                      ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                      ngraph::pattern::has_static_shape());

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher& m) {
        auto matmul = std::dynamic_pointer_cast<ngraph::opset1::MatMul>(m.get_match_root());
        if (!matmul || transformation_callback(matmul)) {
            return false;
        }

        auto input_a = matmul->input(0).get_source_output();
        auto input_b = matmul->input(1).get_source_output();

        auto shape_a = input_a.get_shape();
        auto shape_b = input_b.get_shape();
        auto output_shape = matmul->get_shape();

        // Transformation to FC is not supported for 1D second input
        if (shape_b.size() == 1) {
            return false;
        }

        /*
         *  get_aligned_shapes function align two input shapes to have the same size and
         *  the same batch dimensions (last two dimensions are not comparable).
         *  It also checks that dimensions are compatible so in case with two shapes
         *  for example: [2, 32, 64] [3, 64, 64] it will raise an exception.
         */

        auto get_aligned_shapes = [shape_a, shape_b, &matmul]() -> std::pair<ngraph::Shape, ngraph::Shape> {
            ngraph::Shape shape_a_aligned(shape_a), shape_b_aligned(shape_b);
            size_t max_size = std::max(shape_a_aligned.size(), shape_b_aligned.size());
            for (size_t i = 0, cnt = max_size - shape_a_aligned.size(); i < cnt; ++i)
                shape_a_aligned.insert(shape_a_aligned.begin(), 1);
            for (size_t i = 0, cnt = max_size - shape_b_aligned.size(); i < cnt; ++i)
                shape_b_aligned.insert(shape_b_aligned.begin(), 1);

            if (matmul->get_transpose_a() && shape_a.size() != 1) {
                std::swap(*(shape_a_aligned.end() - 1), *(shape_a_aligned.end() - 2));
            }
            if (matmul->get_transpose_b()) {
                std::swap(*(shape_b_aligned.end() - 1), *(shape_b_aligned.end() - 2));
            }

            for (size_t i = 0; i < max_size - 2; ++i) {
                if (shape_a_aligned[i] != shape_b_aligned[i] && shape_a_aligned[i] > 1 && shape_b_aligned[i] > 1) {
                    std::ostringstream stream;
                    stream << "Shapes can't be aligned: " << shape_a_aligned << " " << shape_b_aligned;
                    throw ngraph::ngraph_error(stream.str());
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

        auto create_transpose = [this](ngraph::Output<ngraph::Node> node, const std::string& transpose_name) -> std::shared_ptr<ngraph::Node> {
            ngraph::Shape output_shape = node.get_node_shared_ptr()->get_shape();

            std::vector<size_t> transpose_order(output_shape.size());
            std::iota(transpose_order.begin(), transpose_order.end(), 0);
            std::swap(*(transpose_order.end() - 1), *(transpose_order.end() - 2));

            auto transpose = register_new_node<ngraph::opset1::Transpose>(
                    node, ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape {transpose_order.size()}, transpose_order));
            transpose->set_friendly_name(transpose_name);
            return transpose;
        };

        // fc_input_a and fc_input_b - are the final inputs that will be set to FullyConnected of GemmIE operations.
        // So in case of adding new operations that takes matmul inputs we need keep update fc_input_a and
        // fc_input_b updated.
        auto fc_input_a = input_a, fc_input_b = input_b;

        // vector of new nGraph operations
        ngraph::NodeVector new_ops;

        // Check that if second inputs is Constant operation and it's shape without ones dimensions has length <= 2
        // we replace MatMul with FullyConnected operation.
        // Otherwise we replace MatMul with Gemm.
        if ((std::dynamic_pointer_cast<ngraph::opset1::Constant>    (fc_input_b.get_node_shared_ptr())  ||
             std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(fc_input_b.get_node_shared_ptr())) &&
            std::count_if(shape_b.begin(), shape_b.end(), [](size_t x) {
                return x != 1;
            }) <= 2) {
            ngraph::Shape shape_a_aligned, shape_b_aligned;
            std::tie(shape_a_aligned, shape_b_aligned) = get_aligned_shapes();

            if (shape_a_aligned.size() < 2 || shape_b_aligned.size() < 2) {
                throw ngraph::ngraph_error("MatMul " + matmul->get_friendly_name() + " shapes are inconsistent.");
            }

            // Transferring from MatMul representation: [B, I, K] * [B, K, O] = [B, I, O]
            // to FullyConnected representation: [I, K] * [O, K] = [I, O]
            size_t K = *(shape_a_aligned.end() - 1);
            size_t O = *(shape_b_aligned.end() - 1);
            ngraph::Shape B(shape_a_aligned.begin(), shape_a_aligned.end() - 2);

            // Weights normalization
            if (!matmul->get_transpose_b()) {
                fc_input_b = create_transpose(fc_input_b, matmul->get_friendly_name() + "/transpose_b");
                new_ops.push_back(fc_input_b.get_node_shared_ptr());
            }

            if (shape_b.size() != 2) {
                auto reshape_shape =
                        ngraph::opset1::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape {2}, {-1ll, static_cast<int64_t>(K)});
                fc_input_b = std::make_shared<ngraph::opset1::Reshape>(fc_input_b, reshape_shape, true);
                new_ops.push_back(fc_input_b.get_node_shared_ptr());
            }

            // Input normalization
            if (matmul->get_transpose_a() && shape_a.size() != 1) {
                fc_input_a = create_transpose(fc_input_a, matmul->get_friendly_name() + "/transpose_a");
                new_ops.push_back(fc_input_a.get_node_shared_ptr());
            }

            // Create FullyConnected
            std::vector<float> bias_value(O, 0);
            auto fc_bias = ngraph::opset1::Constant::create(matmul->get_output_element_type(0), ngraph::Shape {O}, bias_value);

            auto fc = std::make_shared<ngraph::vpu::op::FullyConnected>(fc_input_a, fc_input_b, fc_bias, output_shape, matmul->output(0).get_element_type());
            fc->set_friendly_name(matmul->get_friendly_name());
            new_ops.push_back(fc);

            ngraph::copy_runtime_info(matmul, new_ops);
            ngraph::replace_node(matmul, fc);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matmul, "ConvertMatMulToFC");
    this->register_matcher(m, callback);
}
}  // namespace vpu