// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/matmul_horizontal_fusing.hpp"
#include "transformations/utils/utils.hpp"
#include "ngraph/validation_util.hpp"

#include "itt.hpp"
#include "transformations/init_node_info.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
namespace {
std::shared_ptr<ngraph::Node> broadcast_if_necessary(
    const std::shared_ptr<ngraph::Node>& constant,
    const ngraph::PartialShape& data_shape,
    const size_t bcast_axis) {
    auto const_shape = constant->get_output_shape(0);
    if (shape_size(const_shape) > 1) {
        return constant;
    }

    const_shape = ngraph::Shape(data_shape.size(), 1);
    const_shape[bcast_axis] = data_shape[bcast_axis].get_length();
    auto reshape_constant = ngraph::opset8::Constant::create(ngraph::element::i64, { const_shape.size() }, const_shape);
    return ngraph::op::util::make_try_fold<ngraph::opset8::Broadcast>(constant, reshape_constant);
}

bool fill_weights_path(const std::shared_ptr<ngraph::Node>& weights_input,
                       std::shared_ptr<ngraph::Node>& weights_mul_const,
                       std::shared_ptr<ngraph::Node>& weights_sub_const,
                       std::shared_ptr<ngraph::Node>& weights_convert,
                       std::shared_ptr<ngraph::Node>& weights) {
    std::shared_ptr<ngraph::Node> weights_path = weights_input;
    if (ngraph::is_type<ngraph::opset8::Multiply>(weights_path)) {
        auto mul_const = weights_path->get_input_node_shared_ptr(1);
        if (!ngraph::is_type<ngraph::opset8::Constant>(mul_const)) {
            return false;
        }

        weights_mul_const = mul_const;
        weights_path = weights_path->get_input_node_shared_ptr(0);
    }

    if (ngraph::is_type<ngraph::opset8::Subtract>(weights_path)) {
        auto sub_const = weights_path->get_input_node_shared_ptr(1);
        if (!ngraph::is_type<ngraph::opset8::Constant>(sub_const)) {
            return false;
        }

        weights_sub_const = sub_const;
        weights_path = weights_path->get_input_node_shared_ptr(0);
    }

    if (ngraph::is_type<ngraph::opset8::Convert>(weights_path)) {
        weights_convert = weights_path;
        weights_path = weights_path->get_input_node_shared_ptr(0);
    }
    if (ngraph::is_type<ngraph::opset8::Constant>(weights_path)) {
        weights = weights_path;
        return true;
    }

    return false;
}

bool consumers_contain_result(const std::shared_ptr<ngraph::Node>& node) {
    const auto target_inputs = node->output(0).target_inputs();
    return std::any_of(target_inputs.begin(), target_inputs.end(),
        [](const std::shared_ptr<ngraph::Node>& n) { return ngraph::is_type<ngraph::opset8::Result>(n); });
}

size_t get_normalized_axis(const std::shared_ptr<ngraph::Node>& split, const std::shared_ptr<ngraph::opset8::Constant>& axis_node) {
    const auto split_input_pshape = split->get_input_partial_shape(0);
    const auto split_axis = axis_node->cast_vector<std::int64_t>()[0];
    return ngraph::normalize_axis(split->get_friendly_name(), split_axis, split_input_pshape.rank());
}

bool validate_eltwise_const(const std::shared_ptr<ngraph::Node>& constant, const ngraph::PartialShape& data_pshape, const size_t fuse_axis) {
    const auto const_shape = constant->get_output_shape(0);
    const auto const_rank = const_shape.size();
    const auto rank = data_pshape.size();

    // ONNX case: 1D constant
    if (const_shape.size() == 1 && const_shape[0] > 1) {
        return data_pshape[fuse_axis].is_static() && static_cast<std::int64_t>(const_shape[0]) == data_pshape[fuse_axis].get_length();
    }

    return (ngraph::shape_size(const_shape) == 1) || ((const_rank == rank) && (const_shape[fuse_axis] > 1) &&
           (std::count_if(const_shape.begin(), const_shape.end(), [](const size_t elem) { return elem > 1; }) == 1));
}

bool fuse_eltwises(const ngraph::Output<ngraph::Node>& data,
                   const std::shared_ptr<ngraph::Node>& constant,
                   const std::shared_ptr<ngraph::Node>& main_eltwise) {
    const auto split = ngraph::as_type_ptr<ngraph::opset8::Split>(data.get_node_shared_ptr());
    if (!split || !constant || !main_eltwise) {
        return false;
    }

    if (data.get_element_type() != main_eltwise->get_output_element_type(0)) {
        return false;
    }

    const auto split_axis = ngraph::as_type_ptr<ngraph::opset8::Constant>(split->get_input_node_shared_ptr(1));
    if (!split_axis) {
        return false;
    }

    const size_t normalized_axis = get_normalized_axis(split, split_axis);
    if (!validate_eltwise_const(constant, main_eltwise->get_input_partial_shape(0), normalized_axis)) {
        return false;
    }

    const auto split_outputs = split->outputs();
    for (const auto& output : split_outputs) {
        const auto target_inputs = output.target_inputs();
        if (target_inputs.size() != 1) {
            return false;
        }
    }

    const auto const_shape = constant->get_output_shape(0);
    const auto& main_type_info = main_eltwise->get_type_info();

    ngraph::OutputVector data_to_fuse;
    ngraph::NodeVector eltwises_to_fuse, constants_to_fuse;

    // fill eltwises to fuse
    const auto outputs = split->outputs();
    for (const auto& output : outputs) {
        const auto cur_eltwise = output.target_inputs()[0];
        if (cur_eltwise->get_type_info() != main_type_info || consumers_contain_result(cur_eltwise)) {
            return false;
        }

        const auto cur_eltwise_const = cur_eltwise->get_input_node_shared_ptr(1);
        if (!ngraph::is_type<ngraph::opset8::Constant>(cur_eltwise_const)) {
            return false;
        }

        if (output.get_element_type() != cur_eltwise->get_output_element_type(0)) {
            return false;
        }

        const auto cur_eltwise_const_shape = cur_eltwise->get_input_shape(1);
        if (ngraph::shape_size(cur_eltwise_const_shape) > 1 && cur_eltwise_const_shape != const_shape) {
            return false;
        }

        data_to_fuse.emplace_back(cur_eltwise->input_value(0));
        eltwises_to_fuse.emplace_back(cur_eltwise);
        constants_to_fuse.emplace_back(broadcast_if_necessary(cur_eltwise_const,
            cur_eltwise->get_input_partial_shape(0),
            normalized_axis));
    }

    // handled ONNX case (1D constant)
    const auto concat_axis = constants_to_fuse[0]->get_output_partial_shape(0).size() > 1 ? normalized_axis : 0;

    const auto new_constant = ngraph::op::util::make_try_fold<ngraph::opset8::Concat>(constants_to_fuse, concat_axis);
    const auto new_data_node = split->input_value(0);
    const auto new_eltwise = main_eltwise->clone_with_new_inputs({ new_data_node, new_constant });
    ngraph::copy_runtime_info(eltwises_to_fuse, new_eltwise);

    auto split_inputs = split->input_values();
    split_inputs[0] = new_eltwise;
    const auto new_split = split->clone_with_new_inputs(split_inputs);
    ngraph::copy_runtime_info(split, new_split);

    for (size_t i = 0; i < eltwises_to_fuse.size(); ++i) {
        eltwises_to_fuse[i]->output(0).replace(new_split->output(i));
    }

    return true;
}
} // namespace

NGRAPH_RTTI_DEFINITION(ngraph::pass::MatMulHorizontalFusing, "MatMulHorizontalFusing", 0);
ngraph::pass::MatMulHorizontalFusing::MatMulHorizontalFusing() {
    auto activations_m = ngraph::pattern::any_input(ngraph::pattern::has_static_rank());
    // constant path => static shape
    auto weights_path_m = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto matmul_m = ngraph::pattern::wrap_type<ngraph::opset8::MatMul>({ activations_m, weights_path_m }, ngraph::pattern::consumers_count(1));

    ngraph::graph_rewrite_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto main_matmul = as_type_ptr<ngraph::opset8::MatMul>(pattern_map.at(matmul_m).get_node_shared_ptr());

        if (!main_matmul || transformation_callback(main_matmul)) {
            return false;
        }

        const auto activations_input = pattern_map.at(activations_m);
        const auto activations_node = activations_input.get_node_shared_ptr();
        if (!activations_node) {
            return false;
        }

        const auto target_inputs = activations_input.target_inputs();
        if (target_inputs.size() == 1) {
            return false;
        }

        const auto weights_path = pattern_map.at(weights_path_m);
        const auto weights_path_node = weights_path.get_node_shared_ptr();
        if (!weights_path_node) {
            return false;
        }

        const auto transpose_a = main_matmul->get_transpose_a();
        const auto transpose_b = main_matmul->get_transpose_b();

        std::shared_ptr<ngraph::Node> weights_mul_const, weights_sub_const, weights_convert, weights;
        if (!fill_weights_path(weights_path_node, weights_mul_const, weights_sub_const, weights_convert, weights)) {
            return false;
        }

        auto weights_shape = weights_path.get_partial_shape();
        const size_t weights_fuse_axis = transpose_b ? weights_shape.size() - 2 : weights_shape.size() - 1;

        ngraph::OutputVector activations;
        ngraph::NodeVector matmuls_to_fuse, mul_const_to_fuse, sub_const_to_fuse, weights_to_fuse;
        auto add_matmul_to_fuse_if_possible = [&](const std::shared_ptr<ngraph::opset8::MatMul>& cur_matmul) {
            if (cur_matmul->get_transpose_a() != transpose_a ||
                cur_matmul->get_transpose_b() != transpose_b) {
                return false;
            }

            const auto cur_weights_path = cur_matmul->get_input_node_shared_ptr(1);
            if (consumers_contain_result(cur_matmul) || cur_weights_path->get_output_partial_shape(0).is_dynamic()) {
                return false;
            }

            std::shared_ptr<ngraph::Node> cur_weights_mul_const, cur_weights_sub_const, cur_weights_convert, cur_weights;
            if (!fill_weights_path(cur_weights_path, cur_weights_mul_const, cur_weights_sub_const, cur_weights_convert, cur_weights)) {
                return false;
            }

            if ((weights_sub_const && !cur_weights_sub_const) || (!weights_sub_const && cur_weights_sub_const) ||
                (weights_mul_const && !cur_weights_mul_const) || (!weights_mul_const && cur_weights_mul_const) ||
                (weights_convert && !cur_weights_convert) || (!weights_convert && cur_weights_convert)) {
                return false;
            }

            if (weights->get_output_element_type(0) != cur_weights->get_output_element_type(0) ||
                weights->get_output_partial_shape(0) != cur_weights->get_output_partial_shape(0)) {
                return false;
            }

            matmuls_to_fuse.emplace_back(cur_matmul);
            weights_to_fuse.emplace_back(cur_weights);
            activations.emplace_back(cur_matmul->input_value(0));
            if (weights_mul_const) {
                const auto weights_path_shape = cur_matmul->get_input_partial_shape(1);
                mul_const_to_fuse.emplace_back(broadcast_if_necessary(cur_weights_mul_const, weights_path_shape, weights_fuse_axis));
            }
            if (weights_sub_const) {
                const auto weights_path_shape = cur_matmul->get_input_partial_shape(1);
                sub_const_to_fuse.emplace_back(broadcast_if_necessary(cur_weights_sub_const, weights_path_shape, weights_fuse_axis));
            }

            return true;
        };

        // fill weights path to fuse
        for (const auto& input : target_inputs) {
            const auto cur_matmul = as_type_ptr<ngraph::opset8::MatMul>(input);
            if (cur_matmul) {
                add_matmul_to_fuse_if_possible(cur_matmul);
            }
        }

        if (matmuls_to_fuse.size() < 2) {
            return false;
        }

        auto new_weights = ngraph::op::util::make_try_fold<ngraph::opset8::Concat>(weights_to_fuse, weights_fuse_axis);
        ngraph::copy_runtime_info(weights, new_weights);

        if (weights_convert) {
            new_weights = weights_convert->clone_with_new_inputs({ new_weights });
            ngraph::copy_runtime_info(weights_convert, new_weights);
        }

        if (!sub_const_to_fuse.empty()) {
            const auto new_sub_const = ngraph::op::util::make_try_fold<ngraph::opset8::Concat>(sub_const_to_fuse, weights_fuse_axis);
            new_weights = std::make_shared<ngraph::opset8::Subtract>(new_weights, new_sub_const);
        }

        if (!mul_const_to_fuse.empty()) {
            const auto new_mul_const = ngraph::op::util::make_try_fold<ngraph::opset8::Concat>(mul_const_to_fuse, weights_fuse_axis);
            new_weights = std::make_shared<ngraph::opset8::Multiply>(new_weights, new_mul_const);
        }

        const auto new_matmul = main_matmul->clone_with_new_inputs({ activations_input, new_weights });
        ngraph::copy_runtime_info(matmuls_to_fuse, new_matmul);

        const auto matmul_out_rank = main_matmul->get_output_partial_shape(0).size();
        const size_t split_axis_value = matmul_out_rank - 1;
        const auto split_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { split_axis_value });
        const auto new_split = std::make_shared<ngraph::opset8::Split>(new_matmul, split_axis, matmuls_to_fuse.size());
        ngraph::copy_runtime_info(matmuls_to_fuse, new_split);

        for (size_t i = 0; i < matmuls_to_fuse.size(); ++i) {
            auto output = matmuls_to_fuse[i]->output(0);
            output.replace(new_split->output(i));
        }

        return true;
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(matmul_m);
    this->register_matcher(matcher, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::SubtractHorizontalFusing, "SubtractHorizontalFusing", 0);
ngraph::pass::SubtractHorizontalFusing::SubtractHorizontalFusing() {
    auto input_m = ngraph::pattern::any_input(ngraph::pattern::has_static_rank());
    auto split_constant_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto split_m = ngraph::pattern::wrap_type<ngraph::opset8::Split>({ input_m, split_constant_m });
    auto sub_const_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto sub_m = ngraph::pattern::wrap_type<ngraph::opset8::Subtract>({ split_m, sub_const_m }, ngraph::pattern::consumers_count(1));

    ngraph::graph_rewrite_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto split = pattern_map.at(split_m);
        const auto constant = pattern_map.at(sub_const_m).get_node_shared_ptr();
        const auto main_sub = pattern_map.at(sub_m).get_node_shared_ptr();

        if (!main_sub || transformation_callback(main_sub)) {
            return false;
        }

        return fuse_eltwises(split, constant, main_sub);
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(sub_m);
    this->register_matcher(matcher, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::MultiplyHorizontalFusing, "MultiplyHorizontalFusing", 0);
ngraph::pass::MultiplyHorizontalFusing::MultiplyHorizontalFusing() {
    auto input_m = ngraph::pattern::any_input(ngraph::pattern::has_static_rank());
    auto split_constant_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto split_m = ngraph::pattern::wrap_type<ngraph::opset8::Split>({ input_m, split_constant_m });
    auto mul_const_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto mul_m = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({ split_m, mul_const_m }, ngraph::pattern::consumers_count(1));

    ngraph::graph_rewrite_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto data = pattern_map.at(split_m);
        const auto constant = pattern_map.at(mul_const_m).get_node_shared_ptr();
        const auto main_mul = pattern_map.at(mul_m).get_node_shared_ptr();

        if (!main_mul || transformation_callback(main_mul)) {
            return false;
        }

        return fuse_eltwises(data, constant, main_mul);
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(mul_m);
    this->register_matcher(matcher, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::AddHorizontalFusing, "AddHorizontalFusing", 0);
ngraph::pass::AddHorizontalFusing::AddHorizontalFusing() {
    auto input_m = ngraph::pattern::any_input(ngraph::pattern::has_static_rank());
    auto split_constant_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto split_m = ngraph::pattern::wrap_type<ngraph::opset8::Split>({ input_m, split_constant_m });
    auto add_const_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto add_m = ngraph::pattern::wrap_type<ngraph::opset8::Add>({ split_m, add_const_m }, ngraph::pattern::consumers_count(1));

    ngraph::graph_rewrite_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto data = pattern_map.at(split_m);
        const auto constant = pattern_map.at(add_const_m).get_node_shared_ptr();
        const auto main_add = pattern_map.at(add_m).get_node_shared_ptr();

        if (!main_add || transformation_callback(main_add)) {
            return false;
        }

        return fuse_eltwises(data, constant, main_add);
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(add_m);
    this->register_matcher(matcher, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::FakeQuantizeHorizontalFusing, "FakeQuantizeHorizontalFusing", 0);
ngraph::pass::FakeQuantizeHorizontalFusing::FakeQuantizeHorizontalFusing() {
    auto input = ngraph::pattern::any_input(ngraph::pattern::has_static_rank());
    auto split_const_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto split_m = ngraph::pattern::wrap_type<ngraph::opset8::Split>({ input, split_const_m });

    auto il_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto ih_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto ol_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto oh_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto fq_m = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({ split_m, il_m, ih_m, ol_m, oh_m }, ngraph::pattern::consumers_count(1));

    ngraph::graph_rewrite_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto data = pattern_map.at(input);
        const auto split = as_type_ptr<ngraph::opset8::Split>(pattern_map.at(split_m).get_node_shared_ptr());
        const auto split_axis = ngraph::as_type_ptr<ngraph::opset8::Constant>(pattern_map.at(split_const_m).get_node_shared_ptr());
        const auto il = as_type_ptr<ngraph::opset8::Constant>(pattern_map.at(il_m).get_node_shared_ptr());
        const auto ih = as_type_ptr<ngraph::opset8::Constant>(pattern_map.at(ih_m).get_node_shared_ptr());
        const auto ol = as_type_ptr<ngraph::opset8::Constant>(pattern_map.at(ol_m).get_node_shared_ptr());
        const auto oh = as_type_ptr<ngraph::opset8::Constant>(pattern_map.at(oh_m).get_node_shared_ptr());
        const auto main_fq = as_type_ptr<ngraph::opset8::FakeQuantize>(pattern_map.at(fq_m).get_node_shared_ptr());

        if (!main_fq || !split || !il || !ih || !ol || !oh || !split_axis || transformation_callback(main_fq)) {
            return false;
        }

        const auto split_outputs = split->outputs();
        for (const auto& output : split_outputs) {
            const auto target_inputs = output.target_inputs();
            if (target_inputs.size() > 1) {
                return false;
            }
        }

        const auto il_shape = il->get_shape();
        const auto ih_shape = ih->get_shape();
        const auto ol_shape = ol->get_shape();
        const auto oh_shape = oh->get_shape();
        const size_t levels = main_fq->get_levels();
        const auto out_et = main_fq->get_output_element_type(0);
        ngraph::OutputVector data_to_fuse;
        ngraph::NodeVector fq_to_fuse, il_to_fuse, ih_to_fuse, ol_to_fuse, oh_to_fuse;


        const size_t normalized_axis = get_normalized_axis(split, split_axis);
        auto fq_match = [&](const std::shared_ptr<ngraph::Node>& cur_fq_node) {
            const auto cur_fq = ngraph::as_type_ptr<ngraph::opset8::FakeQuantize>(cur_fq_node);
            if (!cur_fq || consumers_contain_result(cur_fq)) {
                return false;
            }

            const auto cur_levels = cur_fq->get_levels();
            const auto cur_out_et = cur_fq->get_output_element_type(0);
            if (cur_levels != levels || cur_out_et != out_et) {
                return false;
            }

            if (!ngraph::is_type<ngraph::opset8::Constant>(cur_fq->get_input_node_shared_ptr(1)) ||
                !ngraph::is_type<ngraph::opset8::Constant>(cur_fq->get_input_node_shared_ptr(2)) ||
                !ngraph::is_type<ngraph::opset8::Constant>(cur_fq->get_input_node_shared_ptr(3)) ||
                !ngraph::is_type<ngraph::opset8::Constant>(cur_fq->get_input_node_shared_ptr(4))) {
                return false;
            }

            const auto cur_il_shape = cur_fq->get_input_shape(1);
            const auto cur_ih_shape = cur_fq->get_input_shape(2);
            const auto cur_ol_shape = cur_fq->get_input_shape(3);
            const auto cur_oh_shape = cur_fq->get_input_shape(4);

            auto check_quantization_axis = [&](const ngraph::Shape& const_shape) {
                if (ngraph::shape_size(const_shape) == 1) {
                    return cur_fq->get_output_partial_shape(0)[normalized_axis].is_static();
                }

                for (size_t i = 0; i < const_shape.size(); ++i) {
                    if (const_shape[i] > 1 && i != normalized_axis) {
                        return false;
                    }
                }

                return true;
            };

            if (!check_quantization_axis(cur_il_shape) || !check_quantization_axis(cur_ih_shape) ||
                !check_quantization_axis(cur_ol_shape) || !check_quantization_axis(cur_oh_shape)) {
                return false;
            }

            if ((ngraph::shape_size(cur_il_shape) > 1 && cur_il_shape != il_shape) ||
                (ngraph::shape_size(cur_ih_shape) > 1 && cur_ih_shape != ih_shape) ||
                (ngraph::shape_size(cur_ol_shape) > 1 && cur_ol_shape != ol_shape) ||
                (ngraph::shape_size(cur_oh_shape) > 1 && cur_oh_shape != oh_shape)) {
                return false;
            }

            return true;
        };

        // fill fake quantizes to fuse
        const auto outputs = split->outputs();
        for (const auto& output : outputs) {
            const auto cur_fq = output.target_inputs()[0];
            if (!fq_match(cur_fq)) {
                return false;
            }

            fq_to_fuse.emplace_back(cur_fq);
            data_to_fuse.emplace_back(cur_fq->input_value(0));
            il_to_fuse.emplace_back(cur_fq->get_input_node_shared_ptr(1));
            ih_to_fuse.emplace_back(cur_fq->get_input_node_shared_ptr(2));
            ol_to_fuse.emplace_back(cur_fq->get_input_node_shared_ptr(3));
            oh_to_fuse.emplace_back(cur_fq->get_input_node_shared_ptr(4));
        }

        auto get_fused_constant = [&](ngraph::NodeVector& constants) {
            const auto constant = ngraph::as_type_ptr<ngraph::opset8::Constant>(constants[0]);
            if (ngraph::shape_size(constant->get_shape()) == 1) {
                bool constants_are_identical = true;
                const float value = ngraph::as_type_ptr<ngraph::opset8::Constant>(constants[0])->cast_vector<float>()[0];
                for (size_t i = 1; i < constants.size(); ++i) {
                    if (ngraph::shape_size(constant->get_output_shape(0)) != 1 ||
                        ngraph::as_type_ptr<ngraph::opset8::Constant>(constants[i])->cast_vector<float>()[0] != value) {
                        constants_are_identical = false;
                        break;
                    }
                }

                if (constants_are_identical) {
                    std::shared_ptr<ngraph::Node> result = ngraph::opset8::Constant::create(
                        constant->get_element_type(), constant->get_shape(), { value });
                    return result;
                }
            }

            for (size_t i = 0; i < constants.size(); ++i) {
                constants[i] = broadcast_if_necessary(constants[i], fq_to_fuse[i]->get_input_partial_shape(0), normalized_axis);
            }

            return ngraph::op::util::make_try_fold<ngraph::opset8::Concat>(constants, normalized_axis);
        };

        const auto new_data = split->input_value(0);
        const auto new_il = get_fused_constant(il_to_fuse);
        const auto new_ih = get_fused_constant(ih_to_fuse);
        const auto new_ol = get_fused_constant(ol_to_fuse);
        const auto new_oh = get_fused_constant(oh_to_fuse);
        const auto new_fq = main_fq->clone_with_new_inputs({ data, new_il, new_ih, new_ol, new_oh });
        ngraph::copy_runtime_info(fq_to_fuse, new_fq);

        auto new_split_inputs = split->input_values();
        new_split_inputs[0] = new_fq;
        const auto new_split = split->clone_with_new_inputs(new_split_inputs);
        ngraph::copy_runtime_info(split, new_split);

        for (size_t i = 0; i < fq_to_fuse.size(); ++i) {
            fq_to_fuse[i]->output(0).replace(new_split->output(i));
        }

        return true;
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(fq_m);
    this->register_matcher(matcher, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::ReshapeHorizontalFusing, "ReshapeHorizontalFusing", 0);
ngraph::pass::ReshapeHorizontalFusing::ReshapeHorizontalFusing() {
    auto input = ngraph::pattern::any_input(ngraph::pattern::has_static_rank());
    auto split_constant_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto split_m = ngraph::pattern::wrap_type<ngraph::opset8::Split>({ input, split_constant_m });
    auto reshape_const_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto reshape_m = ngraph::pattern::wrap_type<ngraph::opset8::Reshape>({ split_m, reshape_const_m }, ngraph::pattern::consumers_count(1));

    ngraph::graph_rewrite_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto data = pattern_map.at(input);
        const auto split = as_type_ptr<ngraph::opset8::Split>(pattern_map.at(split_m).get_node_shared_ptr());
        const auto split_axis = as_type_ptr<ngraph::opset8::Constant>(pattern_map.at(split_constant_m).get_node_shared_ptr());
        const auto constant = as_type_ptr<ngraph::opset8::Constant>(pattern_map.at(reshape_const_m).get_node_shared_ptr());
        const auto main_reshape = as_type_ptr<ngraph::opset8::Reshape>(pattern_map.at(reshape_m).get_node_shared_ptr());

        if (!split || !split_axis || !constant || !main_reshape || transformation_callback(main_reshape)) {
            return false;
        }

        const auto split_outputs = split->outputs();
        for (const auto& output : split_outputs) {
            const auto target_inputs = output.target_inputs();
            if (target_inputs.size() > 1) {
                return false;
            }
        }

        const size_t normalized_axis = get_normalized_axis(split, split_axis);
        auto input_pshape = main_reshape->get_input_partial_shape(0);
        auto input_rank = input_pshape.size();
        auto output_pshape = main_reshape->get_output_partial_shape(0);
        const auto main_reshape_pattern = constant->cast_vector<std::int64_t>();

        auto check_reshape_pattern = [&]() {
            if (main_reshape_pattern.size() <= normalized_axis) {
                return false;
            }

            if (main_reshape_pattern[normalized_axis] == 0 && main_reshape->get_special_zero()) {
                return true;
            }

            if (main_reshape_pattern.size() == input_rank) {
                const auto splitted_in_dim = input_pshape[normalized_axis];
                const auto splitted_out_dim = output_pshape[normalized_axis];
                if (splitted_in_dim.is_static() && splitted_out_dim.is_static() && splitted_in_dim == splitted_out_dim) {
                    return true;
                }
            }

            if (main_reshape_pattern.size() == input_rank + 1 && normalized_axis == input_rank - 1) {
                const auto splitted_in_dim = input_pshape[normalized_axis];
                const auto splitted_out_dim = output_pshape[normalized_axis];
                const auto post_splitted_out_dim = output_pshape[normalized_axis + 1];
                if (splitted_in_dim.is_static() && splitted_out_dim.is_static() && post_splitted_out_dim.is_static() &&
                    splitted_in_dim == splitted_out_dim * post_splitted_out_dim) {
                    return true;
                }
            }

            if (main_reshape_pattern.size() == input_rank - 1 && normalized_axis == input_rank - 2) {
                const auto splitted_in_dim = input_pshape[normalized_axis];
                const auto post_splitted_in_dim = input_pshape[normalized_axis + 1];
                const auto splitted_out_dim = output_pshape[normalized_axis];
                if (splitted_in_dim.is_static() && post_splitted_in_dim.is_static() && splitted_out_dim.is_static() &&
                    splitted_in_dim * post_splitted_in_dim == splitted_out_dim) {
                    return true;
                }
            }

            return false;
        };
        if (!check_reshape_pattern()) {
            return false;
        }

        ngraph::OutputVector data_to_fuse;
        ngraph::NodeVector reshapes_to_fuse;

        auto reshape_match = [&main_reshape_pattern](const std::shared_ptr<ngraph::Node>& cur_reshape_node) {
            const auto cur_reshape = ngraph::as_type_ptr<ngraph::opset8::Reshape>(cur_reshape_node);
            if (!cur_reshape || consumers_contain_result(cur_reshape)) {
                return false;
            }

            const auto cur_reshape_const = ngraph::as_type_ptr<ngraph::opset8::Constant>(cur_reshape->get_input_node_shared_ptr(1));
            return cur_reshape_const && cur_reshape_const->cast_vector<std::int64_t>() == main_reshape_pattern;
        };

        const auto outputs = split->outputs();
        for (const auto& output : outputs) {
            const auto cur_reshape = output.target_inputs()[0];
            if (!reshape_match(cur_reshape)) {
                return false;
            }

            data_to_fuse.emplace_back(cur_reshape->input_value(0));
            reshapes_to_fuse.emplace_back(cur_reshape);
        }

        auto new_pattern = main_reshape_pattern;
        new_pattern[normalized_axis] *= reshapes_to_fuse.size();
        const auto new_reshape_const = ngraph::opset8::Constant::create(constant->get_element_type(), { new_pattern.size() }, new_pattern);
        const auto reshape_special_zero = main_reshape->get_special_zero();

        const auto new_reshape = std::make_shared<ngraph::opset8::Reshape>(data, new_reshape_const, reshape_special_zero);
        ngraph::copy_runtime_info(reshapes_to_fuse, new_reshape);

        auto split_inputs = split->input_values();
        split_inputs[0] = new_reshape;
        const auto new_split = split->clone_with_new_inputs(split_inputs);
        ngraph::copy_runtime_info(split, new_split);

        for (size_t i = 0; i < reshapes_to_fuse.size(); ++i) {
            reshapes_to_fuse[i]->output(0).replace(new_split->output(i));
        }

        return true;
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(reshape_m);
    this->register_matcher(matcher, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::OptimizeTransposePairsBeforeMatMul, "OptimizeTransposePairsBeforeMatMul", 0);
ngraph::pass::OptimizeTransposePairsBeforeMatMul::OptimizeTransposePairsBeforeMatMul() {
    MATCHER_SCOPE(OptimizeBTransposeBeforeMatMul);
    auto a_transpose_constant_m = pattern::wrap_type<opset4::Constant>();
    auto a_transpose_m = pattern::wrap_type<opset4::Transpose>({ pattern::any_input(), a_transpose_constant_m });

    auto b_transpose_constant_m = pattern::wrap_type<opset4::Constant>();
    auto b_transpose_m = pattern::wrap_type<opset4::Transpose>({ pattern::any_input(), b_transpose_constant_m }, pattern::consumers_count(1));

    auto b_mul_const_m = pattern::wrap_type<opset4::Constant>();
    auto b_mul_m = pattern::wrap_type<opset4::Multiply>({ b_transpose_m, b_mul_const_m }, pattern::consumers_count(1));

    auto b_input_m = std::make_shared<pattern::op::Or>(ngraph::OutputVector{ b_mul_m, b_transpose_m });
    auto matmul_label = pattern::wrap_type<opset4::MatMul>({ a_transpose_m, b_input_m });

    ngraph::graph_rewrite_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto a_transpose_const = ngraph::as_type_ptr<opset4::Constant>(pattern_map.at(a_transpose_constant_m).get_node_shared_ptr());
        auto a_transpose = ngraph::as_type_ptr<opset4::Transpose>(pattern_map.at(a_transpose_m).get_node_shared_ptr());
        auto b_transpose_const = ngraph::as_type_ptr<opset4::Constant>(pattern_map.at(b_transpose_constant_m).get_node_shared_ptr());
        auto b_transpose = ngraph::as_type_ptr<opset4::Transpose>(pattern_map.at(b_transpose_m).get_node_shared_ptr());
        auto matmul = ngraph::as_type_ptr<opset4::MatMul>(pattern_map.at(matmul_label).get_node_shared_ptr());

        if (!a_transpose || !a_transpose_const || !b_transpose || !b_transpose_const || !matmul) {
            return false;
        }

        bool mul_before_transpose_b = pattern_map.count(b_mul_m);
        std::shared_ptr<ngraph::Node> b_mul = mul_before_transpose_b ? pattern_map.at(b_mul_m).get_node_shared_ptr() : nullptr;
        std::shared_ptr<ngraph::Node> b_mul_const = mul_before_transpose_b ? pattern_map.at(b_mul_const_m).get_node_shared_ptr() : nullptr;
        if (mul_before_transpose_b && (!b_mul || !b_mul_const)) {
            return false;
        }

        auto a_transpose_vals = a_transpose_const->cast_vector<size_t>();
        auto b_transpose_vals = b_transpose_const->cast_vector<size_t>();
        std::swap(b_transpose_vals[b_transpose_vals.size() - 1], b_transpose_vals[b_transpose_vals.size() - 2]);
        if (a_transpose_vals != b_transpose_vals) {
            return false;
        }

        if (b_mul) {
            const auto b_mul_const_shape = b_mul->get_input_shape(1);
            if (ngraph::shape_size(b_mul_const_shape) > 1) {
                if (b_mul_const_shape.size() != b_transpose_vals.size() ||
                    b_mul_const_shape[b_transpose_vals.size() - 1] > 1 ||
                    b_mul_const_shape[b_transpose_vals.size() - 2] > 1) {
                    return false;
                }
            }
        }

        auto new_b_transpose_const = opset4::Constant::create(element::i64, { b_transpose_vals.size() }, b_transpose_vals);
        auto new_b_transpose = pass::MatcherPass::register_new_node<opset8::Transpose>(b_transpose->input_value(0), new_b_transpose_const);
        new_b_transpose->set_friendly_name(b_transpose->get_friendly_name());
        copy_runtime_info(b_transpose, new_b_transpose);

        std::shared_ptr<ngraph::Node> matmul_input_b = new_b_transpose;
        if (mul_before_transpose_b) {
            auto new_b_mul = b_mul->clone_with_new_inputs({ new_b_transpose, b_mul_const });
            new_b_mul->set_friendly_name(b_mul->get_friendly_name());
            copy_runtime_info(b_mul, new_b_mul);
            matmul_input_b = new_b_mul;
        }

        const auto new_matmul = std::make_shared<opset8::MatMul>(a_transpose, matmul_input_b, matmul->get_transpose_a(), !matmul->get_transpose_b());
        new_matmul->set_friendly_name(matmul->get_friendly_name());
        copy_runtime_info(matmul, new_matmul);
        replace_node(matmul, new_matmul);

        return true;
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(matmul_label);
    this->register_matcher(matcher, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::TransposeHorizontalFusing, "TransposeHorizontalFusing", 0);
ngraph::pass::TransposeHorizontalFusing::TransposeHorizontalFusing() {
    auto input = ngraph::pattern::any_input();
    auto split_constant_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto split_m = ngraph::pattern::wrap_type<ngraph::opset8::Split>({ input, split_constant_m });
    auto transpose_const_m = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto transpose_m = ngraph::pattern::wrap_type<ngraph::opset8::Transpose>({ split_m, transpose_const_m }, ngraph::pattern::consumers_count(1));

    ngraph::graph_rewrite_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto data = pattern_map.at(input);
        const auto split = as_type_ptr<ngraph::opset8::Split>(pattern_map.at(split_m).get_node_shared_ptr());
        const auto split_axis = as_type_ptr<ngraph::opset8::Constant>(pattern_map.at(split_constant_m).get_node_shared_ptr());
        const auto constant = as_type_ptr<ngraph::opset8::Constant>(pattern_map.at(transpose_const_m).get_node_shared_ptr());
        const auto main_transpose = pattern_map.at(transpose_m).get_node_shared_ptr();

        if (!split || !split_constant_m || !constant || !main_transpose || transformation_callback(main_transpose)) {
            return false;
        }

        const auto split_outputs = split->outputs();
        for (const auto& output : split_outputs) {
            const auto target_inputs = output.target_inputs();
            if (target_inputs.size() > 1) {
                return false;
            }
        }

        const auto transpose_values = constant->cast_vector<size_t>();
        auto transpose_match = [&transpose_values](const std::shared_ptr<ngraph::Node>& transpose_node) {
            const auto cur_transpose = as_type_ptr<ngraph::opset8::Transpose>(transpose_node);
            if (!cur_transpose || consumers_contain_result(cur_transpose)) {
                return false;
            }

            const auto cur_transpose_const = as_type_ptr<ngraph::opset8::Constant>(cur_transpose->get_input_node_shared_ptr(1));
            return cur_transpose_const && cur_transpose_const->cast_vector<size_t>() == transpose_values;
        };

        ngraph::OutputVector data_to_fuse;
        ngraph::NodeVector transposes_to_fuse;
        const auto outputs = split->outputs();
        for (const auto& output : outputs) {
            const auto cur_transpose = output.target_inputs()[0];
            if (!transpose_match(cur_transpose)) {
                return false;
            }

            data_to_fuse.emplace_back(cur_transpose->input_value(0));
            transposes_to_fuse.emplace_back(cur_transpose);
        }

        if (transposes_to_fuse.size() < 2) {
            return false;
        }

        const auto new_transpose = std::make_shared<ngraph::opset8::Transpose>(data, constant);
        ngraph::copy_runtime_info(transposes_to_fuse, new_transpose);

        const size_t normalized_axis = get_normalized_axis(split, split_axis);
        // Change split axis according to the transpose values
        size_t new_split_axis = normalized_axis;
        for (size_t i = 0; i < transpose_values.size(); ++i) {
            if (transpose_values[i] == normalized_axis) {
                new_split_axis = i;
                break;
            }
        }

        const auto new_split_axis_node = ngraph::opset8::Constant::create(split_axis->get_element_type(), Shape{}, { new_split_axis });
        const auto new_split = std::make_shared<ngraph::opset8::Split>(new_transpose, new_split_axis_node, split->get_num_splits());
        ngraph::copy_runtime_info(split, new_split);

        for (size_t i = 0; i < transposes_to_fuse.size(); ++i) {
            transposes_to_fuse[i]->output(0).replace(new_split->output(i));
        }

        return true;
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(transpose_m);
    this->register_matcher(matcher, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::HorizontalFusings, "HorizontalFusings", 0);
ngraph::pass::HorizontalFusings::HorizontalFusings() {
    add_matcher<MatMulHorizontalFusing>();
    add_matcher<SubtractHorizontalFusing>();
    add_matcher<MultiplyHorizontalFusing>();
    add_matcher<AddHorizontalFusing>();
    add_matcher<FakeQuantizeHorizontalFusing>();
    add_matcher<ReshapeHorizontalFusing>();
    add_matcher<OptimizeTransposePairsBeforeMatMul>();
    add_matcher<TransposeHorizontalFusing>();
}
