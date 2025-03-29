// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "align_matmul_input_ranks.hpp"

#include <algorithm>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::intel_cpu::AlignMatMulInputRanks::AlignMatMulInputRanks() {
    MATCHER_SCOPE(AlignMatMulInputRanks);
    ov::OutputVector twoInputs = {ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank()),
                                  ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank())};

    auto matmulPattern = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>(twoInputs);

    ov::matcher_pass_callback callback = [this](ov::pass::pattern::Matcher& m) {
        auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_match_root());

        if (!matmul || transformation_callback(matmul)) {
            return false;
        }

        const auto& input0 = matmul->input_value(0);
        const auto& input1 = matmul->input_value(1);
        const auto& input0shape = input0.get_partial_shape();
        const auto& input1shape = input1.get_partial_shape();
        const auto& output_shape = matmul->get_output_partial_shape(0);

        assert(input0shape.rank().is_static());
        assert(input1shape.rank().is_static());

        const bool transposedUnsqueeze = input1shape.size() == 1;

        if (input0shape.size() == input1shape.size() && input0shape.size() != 1) {
            return false;  // nothing to do
        }

        auto getUnsqueeze = [&](const ov::Output<ov::Node>& nodeFrom, const ov::Output<ov::Node>& nodeTo) {
            auto rankFrom = nodeFrom.get_partial_shape().size();
            auto rankTo = nodeTo.get_partial_shape().size();

            std::vector<int64_t> unsqueeze_axes;
            for (int64_t j = 0; j < static_cast<int64_t>(rankTo - rankFrom); ++j) {
                unsqueeze_axes.push_back(j);
            }

            if (transposedUnsqueeze) {  // special case for one-dimensional second input
                unsqueeze_axes[unsqueeze_axes.size() - 1]++;
            }

            auto unsqueeze = std::make_shared<ov::opset1::Unsqueeze>(
                nodeFrom,
                ov::opset1::Constant::create(ov::element::i64, ov::Shape{unsqueeze_axes.size()}, unsqueeze_axes));

            unsqueeze->set_friendly_name(nodeFrom.get_node()->get_friendly_name() + "/Unsqueeze");

            return unsqueeze;
        };

        auto matmul_new_inputs = matmul->input_values();
        ov::NodeVector new_ops;

        if (input0shape.size() == 1 && input1shape.size() == 1) {
            // If the input is 1D tensor, it is unsqueezed to 2D tensor (row vector)
            // for the first input:  by adding axes with size 1 at ROW_INDEX_DIM
            //                       to the left of the shape {S} -> {1, S}
            // for the second input: by adding axes with size 1 at COL_INDEX_DIM
            //                       to the right of the shape {S} -> {S, 1}
            const auto unsqueezeInput0 = std::make_shared<ov::opset1::Unsqueeze>(
                input0,
                ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
            const auto unsqueezeInput1 = std::make_shared<ov::opset1::Unsqueeze>(
                input1,
                ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {1}));

            matmul_new_inputs[0] = unsqueezeInput0;
            new_ops.push_back(unsqueezeInput0);
            matmul_new_inputs[1] = unsqueezeInput1;
            new_ops.push_back(unsqueezeInput1);
            // For 1D inputs transpose flag is expected to always act like `false`
            matmul->set_transpose_a(false);
            matmul->set_transpose_b(false);
        } else if (input0shape.size() < input1shape.size()) {
            std::shared_ptr<ov::Node> unsqueezeInput0 = getUnsqueeze(input0, input1);
            matmul_new_inputs[0] = unsqueezeInput0;
            new_ops.push_back(unsqueezeInput0);

            if (input0shape.size() == 1) {
                matmul->set_transpose_a(false);
            }
        } else if (input0shape.size() > input1shape.size()) {
            std::shared_ptr<ov::Node> unsqueezeInput1 = getUnsqueeze(input1, input0);
            matmul_new_inputs[1] = unsqueezeInput1;
            new_ops.push_back(unsqueezeInput1);

            if (input1shape.size() == 1) {
                matmul->set_transpose_b(false);
            }
        }

        std::shared_ptr<ov::Node> matmul_new = matmul->clone_with_new_inputs(matmul_new_inputs);
        new_ops.push_back(matmul_new);

        if (matmul_new->get_output_partial_shape(0) != output_shape) {
            // When one of the inputs is one-dimensional tensor, ngraph shrinks the output node by 1
            // For example: C * AxBxCxD -> AxBxD (instead of AxBx1xD)
            // Insert additional squeeze operation to preserve output shape
            const auto new_out_shape_size = matmul_new->get_output_partial_shape(0).size();
            size_t squeeze_axis = 0;
            std::shared_ptr<ov::Node> squeeze_output;
            // If output data is scalar && new_out_shape is [1 1 .. 1], squeeze all the axis to produce a scalar
            auto& new_output_partial_shape = matmul_new->get_output_partial_shape(0);
            const bool can_squeeze_scalar =
                new_output_partial_shape.is_static() ? ov::shape_size(new_output_partial_shape.to_shape()) == 1 : false;
            if (ov::is_scalar(output_shape) && can_squeeze_scalar) {
                squeeze_output = std::make_shared<ov::op::v0::Squeeze>(matmul_new);
            } else {
                if (input0shape.size() == 1) {
                    squeeze_axis = new_out_shape_size - 2;
                } else if (input1shape.size() == 1) {
                    squeeze_axis = new_out_shape_size - 1;
                }
                squeeze_output = std::make_shared<ov::op::v0::Squeeze>(
                    matmul_new,
                    ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {squeeze_axis}));
            }
            new_ops.push_back(squeeze_output);
            matmul_new->set_friendly_name(matmul->get_friendly_name() + "/MM");
            // Set the name of the last node after transformation to initial node name
            // (in case initial node was an output node)
            squeeze_output->set_friendly_name(matmul->get_friendly_name());
            ov::copy_runtime_info(matmul, new_ops);
            ov::replace_node(matmul, squeeze_output);
        } else {
            matmul_new->set_friendly_name(matmul->get_friendly_name());
            ov::copy_runtime_info(matmul, new_ops);
            ov::replace_node(matmul, matmul_new);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmulPattern, matcher_name);
    this->register_matcher(m, callback);
}
