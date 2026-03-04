// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_gated_delta_net.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/util/op_types.hpp"
#include "transformations/utils/gen_pattern.hpp"

using namespace ov::pass;

namespace {

using InputDesc = ov::op::util::MultiSubGraphOp::InputDescription;

bool is_slice_desc(const std::shared_ptr<InputDesc>& desc,
				   uint64_t input_index,
				   int64_t axis,
				   int64_t part_size) {
	auto slice = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::SliceInputDescription>(desc);
	if (!slice) {
		return false;
	}
	return slice->m_input_index == input_index && slice->m_axis == axis && slice->m_part_size == part_size &&
		   slice->m_stride == 1 && slice->m_start == 0 && slice->m_end == -1;
}

bool is_merged_desc(const std::shared_ptr<InputDesc>& desc, uint64_t input_index) {
	auto merged = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp::MergedInputDescription>(desc);
	return merged && merged->m_input_index == input_index;
}


bool matches_linear_attention_loop(const std::shared_ptr<ov::op::v5::Loop>& loop) {
	if (!loop) {
		return false;
	}

	if (loop->get_input_size() < 9 || loop->get_output_size() != 2) {
		return false;
	}

	const auto& body = loop->get_function();
	if (!body) {
		return false;
	}

	if (body->get_parameters().size() < 8 || body->get_results().size() < 3) {
		return false;
	}

	const auto& input_descs = loop->get_input_descriptions();
	if (input_descs.size() < 7) {
		return false;
	}

	for (uint64_t idx = 2; idx <= 6; ++idx) {
		bool found_slice = false;
		for (const auto& desc : input_descs) {
			if (is_slice_desc(desc, idx, 1, 1)) {
				found_slice = true;
				break;
			}
		}
		if (!found_slice) {
			return false;
		}
	}

	bool has_merged_state = false;
	bool has_merged_output = false;
	for (const auto& desc : input_descs) {
		if (is_merged_desc(desc, 7)) {
			has_merged_state = true;
		}
		if (is_merged_desc(desc, 8)) {
			has_merged_output = true;
		}
	}
	if (!has_merged_state || !has_merged_output) {
		return false;
	}

	const auto& output_descs = loop->get_output_descriptions();
	if (output_descs.size() != 2) {
		return false;
	}

	// bool has_scatter = false;
	// bool has_add = false;
	// for (const auto& result : body->get_results()) {
	// 	auto result_node = result->input_value(0).get_node_shared_ptr();
	// 	if (std::dynamic_pointer_cast<ov::op::v3::ScatterUpdate>(result_node) != nullptr) {
	// 		has_scatter = true;
	// 	}
	// 	if (is_add_node(result_node)) {
	// 		has_add = true;
	// 	}
	// }
	// if (!has_scatter || !has_add) {
	// 	return false;
	// }

	// if (!has_body_node_type(body, ov::op::v0::Exp::get_type_info_static()) ||
	// 	!has_body_node_type(body, ov::op::v1::ReduceSum::get_type_info_static()) ||
	// 	!has_body_node_type(body, ov::op::v1::Multiply::get_type_info_static())) {
	// 	return false;
	// }

	return true;
}

template <typename T>
std::shared_ptr<T> get_single_consumer_as(const ov::Output<ov::Node>& output) {
	const auto& targets = output.get_target_inputs();
	if (targets.size() != 1) {
		return nullptr;
	}
	auto target_node = targets.begin()->get_node()->shared_from_this();
	return ov::as_type_ptr<T>(target_node);
}

std::shared_ptr<ov::opset1::ReduceProd> find_reduce_prod_from_slice_like(
	const std::shared_ptr<ov::Node>& slice) {
	if (!slice) {
		return nullptr;
	}
	auto try_inputs = [&](const std::shared_ptr<ov::Node>& node) {
		for (size_t i = 1; i <= 2; ++i) {
			if (node->get_input_size() <= i) {
				continue;
			}
			auto reduce_prod = std::dynamic_pointer_cast<ov::opset1::ReduceProd>(
				node->input_value(i).get_node_shared_ptr());
			if (reduce_prod) {
				return reduce_prod;
			}
		}
		return std::shared_ptr<ov::opset1::ReduceProd>{};
	};

	if (std::dynamic_pointer_cast<ov::opset8::Slice>(slice) ||
		std::dynamic_pointer_cast<ov::opset1::StridedSlice>(slice)) {
		return try_inputs(slice);
	}
	return nullptr;
}

bool uses_reduce_prod_at(const std::shared_ptr<ov::Node>& slice,
																const std::shared_ptr<ov::opset1::ReduceProd>& reduce_prod,
																size_t idx) {
	if (!slice || !reduce_prod) {
		return false;
	}
	if (slice->get_input_size() <= idx) {
		return false;
	}
	return slice->input_value(idx).get_node_shared_ptr() == reduce_prod;
}

bool replace_concat_slice_with_linear_attention(
	const std::shared_ptr<ov::op::v5::Loop>& loop,
	const std::shared_ptr<ov::op::GatedDeltaNet>& linear_attn) {
	if (!loop || !linear_attn) {
		return false;
	}

	auto reshape_out0 = get_single_consumer_as<ov::opset1::Reshape>(loop->output(0));
	auto reshape_out1 = get_single_consumer_as<ov::opset1::Reshape>(loop->output(1));
	if (!reshape_out0 || !reshape_out1) {
		return false;
	}
	auto concat0 = get_single_consumer_as<ov::opset1::Concat>(reshape_out0->output(0));
	auto concat1 = get_single_consumer_as<ov::opset1::Concat>(reshape_out1->output(0));
	if (!concat0 || concat0 != concat1) {
		return false;
	}
	auto concat = concat0;
	const auto& concat_targets = concat->output(0).get_target_inputs();
	if (concat_targets.size() != 2) {
		return false;
	}
	std::vector<std::shared_ptr<ov::Node>> slices;
	slices.reserve(2);
	for (const auto& input : concat_targets) {
		auto slice_node = input.get_node()->shared_from_this();
		if (!std::dynamic_pointer_cast<ov::opset8::Slice>(slice_node) &&
			!std::dynamic_pointer_cast<ov::opset1::StridedSlice>(slice_node)) {
			return false;
		}
		slices.push_back(slice_node);
	}

	auto reduce_prod = find_reduce_prod_from_slice_like(slices[0]);
	if (!reduce_prod) {
		reduce_prod = find_reduce_prod_from_slice_like(slices[1]);
	}
	if (!reduce_prod) {
		return false;
	}
	const auto is_first_part = [&](const std::shared_ptr<ov::Node>& slice) {
		return uses_reduce_prod_at(slice, reduce_prod, 2) && !uses_reduce_prod_at(slice, reduce_prod, 1);
	};
	const auto is_second_part = [&](const std::shared_ptr<ov::Node>& slice) {
		return uses_reduce_prod_at(slice, reduce_prod, 1) && !uses_reduce_prod_at(slice, reduce_prod, 2);
	};

	std::shared_ptr<ov::Node> slice_value;
	std::shared_ptr<ov::Node> slice_state;
	for (const auto& slice : slices) {
		if (is_first_part(slice)) {
			slice_value = slice;
		}
		if (is_second_part(slice)) {
			slice_state = slice;
		}
	}
	if (!slice_value || !slice_state) {
		return false;
	}

	auto reshape_value = get_single_consumer_as<ov::opset1::Reshape>(slice_value->output(0));
	auto reshape_state = get_single_consumer_as<ov::opset1::Reshape>(slice_state->output(0));
	if (!reshape_value || !reshape_state) {
		return false;
	}
	if (!ov::replace_output_update_name(reshape_value->output(0), linear_attn->output(0))) {
		reshape_value->output(0).replace(linear_attn->output(0));
	}
	if (!ov::replace_output_update_name(reshape_state->output(0), linear_attn->output(1))) {
		reshape_state->output(0).replace(linear_attn->output(1));
	}
	std::cout << "Fused GatedDeltaNet Replace Concat//Slice\n";
	return true;
}

}  // namespace
using namespace ov::gen_pattern;
using namespace ov::pass::pattern;
ov::pass::GatedDeltaNetFusion::GatedDeltaNetFusion() {
	auto key = ov::pass::pattern::any_input();
	auto query = ov::pass::pattern::any_input();
	auto value = ov::pass::pattern::any_input();
	auto axis_q_const = pattern::wrap_type<opset1::Constant>();
	auto axis_q_convert = pattern::wrap_type<opset1::Convert>({axis_q_const});
	auto axis_q = std::make_shared<pattern::op::Or>(OutputVector{axis_q_const, axis_q_convert});

	auto eps_q_const = pattern::wrap_type<opset1::Constant>();
	auto eps_q_convert = pattern::wrap_type<opset1::Convert>({eps_q_const});
	auto eps_q = std::make_shared<pattern::op::Or>(OutputVector{eps_q_const, eps_q_convert});

	auto inv_const_q_const = pattern::wrap_type<opset1::Constant>();
	auto inv_const_q_convert = pattern::wrap_type<opset1::Convert>({inv_const_q_const});
	auto inv_const_q = std::make_shared<pattern::op::Or>(OutputVector{inv_const_q_const, inv_const_q_convert});

	auto axis_k_const = pattern::wrap_type<opset1::Constant>();
	auto axis_k_convert = pattern::wrap_type<opset1::Convert>({axis_k_const});
	auto axis_k = std::make_shared<pattern::op::Or>(OutputVector{axis_k_const, axis_k_convert});

	auto eps_k_const = pattern::wrap_type<opset1::Constant>();
	auto eps_k_convert = pattern::wrap_type<opset1::Convert>({eps_k_const});
	auto eps_k = std::make_shared<pattern::op::Or>(OutputVector{eps_k_const, eps_k_convert});

	auto inv_const_k_const = pattern::wrap_type<opset1::Constant>();
	auto inv_const_k_convert = pattern::wrap_type<opset1::Convert>({inv_const_k_const});
	auto inv_const_k = std::make_shared<pattern::op::Or>(OutputVector{inv_const_k_const, inv_const_k_convert});

	auto minus_one = pattern::wrap_type<opset1::Constant>();

	auto Multiply_14 = pattern::wrap_type<opset1::Multiply>({query, query}, {{"auto_broadcast", "numpy"}});
	auto ReduceSum_15 = pattern::wrap_type<opset1::ReduceSum>({Multiply_14, axis_q->output(0)}, {{"keep_dims", true}});
	auto Add_18 = pattern::wrap_type<opset1::Add>({ReduceSum_15, eps_q->output(0)}, {{"auto_broadcast", "numpy"}});
	auto Sqrt_19 = pattern::wrap_type<opset1::Sqrt>({Add_18});
	auto Divide_20 = pattern::wrap_type<opset1::Divide>({inv_const_q->output(0), Sqrt_19}, {{"auto_broadcast", "numpy"}});
	auto Power_20 = pattern::wrap_type<opset1::Power>({Sqrt_19, minus_one}, {{"auto_broadcast", "numpy"}});
	auto inv_sqrt_q = std::make_shared<pattern::op::Or>(OutputVector{Divide_20, Power_20});
	auto Multiply_21 = pattern::wrap_type<opset1::Multiply>({query, inv_sqrt_q->output(0)}, {{"auto_broadcast", "numpy"}});
	// q / sqrt(d)
	auto Multiply_32 = pattern::wrap_type<opset1::Multiply>({Multiply_21, any_input()}, {{"auto_broadcast", "numpy"}});
	// q * scale (when no per-token normalization is present)
	auto Multiply_q_simple = pattern::wrap_type<opset1::Multiply>({query, any_input()}, {{"auto_broadcast", "numpy"}});
	auto q_candidate = std::make_shared<pattern::op::Or>(OutputVector{Multiply_32, Multiply_21, Multiply_q_simple});

	auto q_candidate_compressed_to_f16 =
		pattern::wrap_type<op::v0::Convert>({q_candidate->output(0)}, {{"destination_type", "f16"}});

	auto Multiply_22 = pattern::wrap_type<opset1::Multiply>({key, key}, {{"auto_broadcast", "numpy"}});
	auto ReduceSum_23 = pattern::wrap_type<opset1::ReduceSum>({Multiply_22, axis_k->output(0)}, {{"keep_dims", true}});
	auto Add_26 = pattern::wrap_type<opset1::Add>({ReduceSum_23, eps_k->output(0)}, {{"auto_broadcast", "numpy"}});
	auto Sqrt_27 = pattern::wrap_type<opset1::Sqrt>({Add_26});
	auto Divide_28 = pattern::wrap_type<opset1::Divide>({inv_const_k->output(0), Sqrt_27}, {{"auto_broadcast", "numpy"}});
	auto Power_28 = pattern::wrap_type<opset1::Power>({Sqrt_27, minus_one}, {{"auto_broadcast", "numpy"}});
	auto inv_sqrt_k = std::make_shared<pattern::op::Or>(OutputVector{Divide_28, Power_28});
	auto Multiply_29 = pattern::wrap_type<opset1::Multiply>({key, inv_sqrt_k->output(0)}, {{"auto_broadcast", "numpy"}});
	auto Multiply_29_compressed_to_f16 = pattern::wrap_type<op::v0::Convert>({Multiply_29}, {{"destination_type", "f16"}});

	auto q_in = std::make_shared<pattern::op::Or>(OutputVector{q_candidate, q_candidate_compressed_to_f16});
	auto k_in = std::make_shared<pattern::op::Or>(OutputVector{Multiply_29, Multiply_29_compressed_to_f16});

	auto loop_label = ov::pass::pattern::wrap_type<ov::op::v5::Loop>(OutputVector{
		any_input(),
		any_input(),
		q_in->output(0),
		k_in->output(0),
		value,
		any_input(),
		any_input(),
		any_input(),
		any_input()
	});

	matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
		const auto& pattern_map = m.get_pattern_value_map();
		auto loop = std::dynamic_pointer_cast<ov::op::v5::Loop>(m.get_match_root());
		if (!matches_linear_attention_loop(loop)) {
			return false;
		}

		std::vector<std::shared_ptr<ov::Node>> rt_nodes{loop};

		ov::Output<Node> query_in, key_in;
		if (pattern_map.count(q_candidate_compressed_to_f16)) {
			query_in = pattern_map.at(q_candidate_compressed_to_f16);
		} else if (pattern_map.count(Multiply_32)) {
			query_in = pattern_map.at(Multiply_32);
		} else if (pattern_map.count(Multiply_21)) {
			query_in = pattern_map.at(Multiply_21);
		} else if (pattern_map.count(Multiply_q_simple)) {
			query_in = pattern_map.at(Multiply_q_simple);
		} else {
			query_in = pattern_map.at(query);
		}

		if (pattern_map.count(Multiply_29_compressed_to_f16)) {
			key_in = pattern_map.at(Multiply_29_compressed_to_f16);
		} else if (pattern_map.count(Multiply_29)) {
			key_in = pattern_map.at(Multiply_29);
		} else {
			key_in = pattern_map.at(key);
		}
		auto value_in = loop->input_value(4);
		ov::OutputVector inputs;
		inputs.reserve(6);

		// Loop inputs layout in the target subgraph:
		// 0: trip_count, 1: execution_condition, 2: query, 3: key, 4: value, 5: g, 6: beta, 7: initial_state, 8: init_output
		inputs.push_back(query_in);  // query
		inputs.push_back(key_in);    // key
		inputs.push_back(value_in);  // value
		inputs.push_back(loop->input_value(7));  // initial_state
		inputs.push_back(loop->input_value(5));  // g
		inputs.push_back(loop->input_value(6));  // beta

		auto linear_attn = std::make_shared<ov::op::GatedDeltaNet>(inputs);
		linear_attn->set_friendly_name(loop->get_friendly_name());

		ov::copy_runtime_info(rt_nodes, linear_attn);
		replace_concat_slice_with_linear_attention(loop, linear_attn);
		ov::replace_node(loop, linear_attn);
        std::cout << "GatedDeltaNetFusion applied." << std::endl;
		register_new_node(linear_attn);
		return true;
	};

	auto m = std::make_shared<ov::pass::pattern::Matcher>(loop_label, "GatedDeltaNetFusion");
	register_matcher(m, callback);
}