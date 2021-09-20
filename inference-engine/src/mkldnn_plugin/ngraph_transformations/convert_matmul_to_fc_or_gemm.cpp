// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_matmul_to_fc_or_gemm.hpp"
#include "op/fully_connected.hpp"
#include <numeric>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::ConvertMatMulToFC, "ConvertMatMulToFC", 0);

MKLDNNPlugin::ConvertMatMulToFC::ConvertMatMulToFC() {
    auto matmul = ngraph::pattern::wrap_type<ngraph::opset1::MatMul>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                                      ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                                      ngraph::pattern::has_static_shape());

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher& m) {
        auto matmul = std::dynamic_pointer_cast<ngraph::opset1::MatMul>(m.get_match_root());
        if (!matmul) {
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

            auto transpose = ngraph::pass::MatcherPass::register_new_node<ngraph::opset1::Transpose>(
                    node, ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{transpose_order.size()}, transpose_order));
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
        if ((std::dynamic_pointer_cast<ngraph::opset1::Constant>(fc_input_b.get_node_shared_ptr()) ||
             std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(fc_input_b.get_node_shared_ptr())) &&
             std::count_if(shape_b.begin(), shape_b.end(), [](size_t x) { return x != 1; }) <= 2) {
            ngraph::Shape shape_a_aligned, shape_b_aligned;
            std::tie(shape_a_aligned, shape_b_aligned) = get_aligned_shapes();

            if (shape_a_aligned.size() < 2 || shape_b_aligned.size() < 2) {
                throw ngraph::ngraph_error("MatMul " + matmul->get_friendly_name() + " shapes are inconsistent.");
            }

            // Transferring from MatMul representation: [B, I, K] * [B, K, O] = [B, I, O]
            // to FullyConnected representation: [I, K] * [K, O] = [I, O]
            size_t K = *(shape_a_aligned.end() - 1);
            ngraph::Shape B(shape_a_aligned.begin(), shape_a_aligned.end() - 2);

            // Weights normalization
            if (!matmul->get_transpose_b()) {
                fc_input_b = create_transpose(fc_input_b, matmul->get_friendly_name() + "/transpose_b");
                new_ops.push_back(fc_input_b.get_node_shared_ptr());
            }

            if (shape_b.size() != 2) {
                auto reshape_shape =
                        ngraph::opset1::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{2}, {-1ll, static_cast<int64_t>(K)});
                fc_input_b = std::make_shared<ngraph::opset1::Reshape>(fc_input_b, reshape_shape, true);
                new_ops.push_back(fc_input_b.get_node_shared_ptr());
            }

            // Input normalization
            if (matmul->get_transpose_a() && shape_a.size() != 1) {
                fc_input_a = create_transpose(fc_input_a, matmul->get_friendly_name() + "/transpose_a");
                new_ops.push_back(fc_input_a.get_node_shared_ptr());
            }

            // Create FullyConnected
            auto fc = std::make_shared<MKLDNNPlugin::FullyConnectedNode>(fc_input_a, fc_input_b, output_shape, matmul->output(0).get_element_type());
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

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::ConvertMatMulToGemm, "ConvertMatMulToGemm", 0);

MKLDNNPlugin::ConvertMatMulToGemm::ConvertMatMulToGemm() {
    auto matmul = ngraph::pattern::wrap_type<ngraph::opset1::MatMul>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                                      ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                                      ngraph::pattern::has_static_shape());

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher& m) {
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
        ngraph::NodeVector new_ops;

        if (shape_a.size() == 1) {
            // If the first input is 1D tensor, it is unsqueezed to 2D tensor (row vector)
            // by adding axes with size 1 at ROW_INDEX_DIM, to the left of the shape.
            // For example {S} will be reshaped to {1, S}.
            fc_input_a = std::make_shared<ngraph::opset1::Unsqueeze>(fc_input_a,
                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));
            shape_a = fc_input_a.get_shape();
            new_ops.push_back(fc_input_a.get_node_shared_ptr());
            // For 1D inputs transpose flag is expected to always act like `false`
            matmul->set_transpose_a(false);
        }
        if (shape_b.size() == 1) {
            // If the second input is 1D tensor, it is unsqueezed to 2D tensor (column vector)
            // by adding axes with size 1 at COL_INDEX_DIM, to the right of the shape.
            // For example {S} will be reshaped to {S, 1}.
            fc_input_b = std::make_shared<ngraph::opset1::Unsqueeze>(fc_input_b,
                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1}));
            shape_b = fc_input_b.get_shape();
            new_ops.push_back(fc_input_b.get_node_shared_ptr());
            // For 1D inputs transpose flag is expected to always act like `false`
            matmul->set_transpose_b(false);
        }

        // WA for IE that Gemm must have inputs with the same length.
        // If ranks of input arguments are still different,
        // the smaller tensor is unsqueezed from the left side of the shape
        // by necessary number of axes to make both shapes of the same rank.
        if (shape_a.size() < shape_b.size()) {
            // Reshape first input (fc_input_a)
            ngraph::Shape reshape_shape(shape_b.size() - shape_a.size(), 1);
            reshape_shape.insert(reshape_shape.end(), shape_a.begin(), shape_a.end());
            fc_input_a = ngraph::op::util::reshapeTo(fc_input_a, reshape_shape);
            new_ops.push_back(fc_input_a.get_node_shared_ptr());
        } else if (shape_b.size() < shape_a.size()) {
            // Reshape second input (fc_input_b)
            ngraph::Shape reshape_shape(shape_a.size() - shape_b.size(), 1);
            reshape_shape.insert(reshape_shape.end(), shape_b.begin(), shape_b.end());
            fc_input_b = ngraph::op::util::reshapeTo(fc_input_b, reshape_shape);
            new_ops.push_back(fc_input_b.get_node_shared_ptr());
        }

        auto gemm = matmul->copy_with_new_inputs({ fc_input_a, fc_input_b });
        new_ops.push_back(gemm);

        if (gemm->get_shape() != output_shape) {
            // This case is possible when one of the inputs has exactly 1 dimension (that is not supported by GEMM operation)
            // So to preserve output shape we insert additional reshape operation
            std::shared_ptr<ngraph::Node> reshape_output;
            if (output_shape.size() == 0) {
                std::vector<int64_t> dim_indices(gemm->get_shape().size());
                std::iota(dim_indices.begin(), dim_indices.end(), 0);
                reshape_output = std::make_shared<ngraph::opset1::Squeeze>(gemm,
                    ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{dim_indices.size()}, dim_indices));
            } else {
                reshape_output = ngraph::op::util::reshapeTo(gemm, output_shape);
            }

            new_ops.push_back(reshape_output);
            gemm->set_friendly_name(matmul->get_friendly_name() + "/gemm");
            reshape_output->set_friendly_name(matmul->get_friendly_name());
            ngraph::copy_runtime_info(matmul, new_ops);
            ngraph::replace_node(matmul, reshape_output);
        } else {
            gemm->set_friendly_name(matmul->get_friendly_name());
            ngraph::copy_runtime_info(matmul, new_ops);
            ngraph::replace_node(matmul, gemm);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matmul, "ConvertMatMulToGemm");
    this->register_matcher(m, callback);
}
