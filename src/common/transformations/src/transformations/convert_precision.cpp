// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_precision.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/reference/convert.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/fp16_compression/align_mixed_fp32_fp16_types.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/fp16_compression/mark_subgraphs_to_keep_in_mixed_precision.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

bool fuse_type_to_parameter(const std::shared_ptr<ov::Node>& node,
                            const precisions_map& precisions,
                            bool convert_input_precision);

bool fuse_type_to_constant(const std::shared_ptr<ov::Node>& node,
                           const precisions_map& precisions,
                           const std::vector<ov::Input<ov::Node>>& consumers);
bool fuse_type_to_shapeof(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool fuse_type_to_shapeof_v0(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool fuse_type_to_random_uniform_v8(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool fuse_type_to_unique_v10(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool fuse_type_to_range_v4(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool fuse_type_to_eye_v9(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool fuse_type_to_convert(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool fuse_type_to_nms3(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool fuse_type_to_nms4(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool fuse_type_to_nms5(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool fuse_type_to_nms9(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool fuse_type_to_matrix_nms(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool fuse_type_to_multiclass_nms(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool fuse_type_to_generate_proposals(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool fuse_type_to_topk(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool fuse_type_to_maxpool(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool fuse_type_to_nonzero(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool fuse_type_to_bucketize(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool fuse_type_to_ctc_greedy_decoder_seq_len(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);

bool fuse_type_to_random_uniform_v8(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);

bool extend_select_type(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);
bool extend_reverse_type(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions);

template <typename T>
bool fuse_type_to_binary_comparision(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto it = precisions.find(node->get_output_element_type(0));
    if (it == precisions.end())
        return false;
    const auto& to = it->second;
    if (auto type_relaxed = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(node)) {
        type_relaxed->set_overridden_output_type(to);
        return true;
    } else if (auto casted = std::dynamic_pointer_cast<T>(node)) {
        auto relaxed_op =
            std::make_shared<ov::op::TypeRelaxed<T>>(*casted, ov::element::TypeVector{}, ov::element::TypeVector{to});
        replace_node(node, relaxed_op);
        return true;
    }
    return false;
}

template <typename T>
bool fuse_type_to_logical(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto it = precisions.find(node->get_output_element_type(0));
    if (it == precisions.end())
        return false;
    const auto& to = it->second;
    if (auto type_relaxed = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(node)) {
        type_relaxed->set_overridden_output_type(to);
        for (size_t i = 0; i < node->get_input_size(); ++i)
            type_relaxed->set_origin_input_type(ov::element::boolean, i);
        return true;
    } else if (auto casted = std::dynamic_pointer_cast<T>(node)) {
        ov::element::TypeVector input_types(node->get_input_size(), ov::element::boolean);
        auto relaxed_op = std::make_shared<ov::op::TypeRelaxed<T>>(*casted, input_types, ov::element::TypeVector{to});
        replace_node(node, relaxed_op);
        return true;
    }
    return false;
}

template <class T>
bool fuse_type_to_reduce_logical(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto it = precisions.find(node->get_output_element_type(0));
    if (it == precisions.end())
        return false;
    const auto& to = it->second;
    if (auto type_relaxed = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(node)) {
        type_relaxed->set_overridden_output_type(to);
        type_relaxed->set_origin_input_type(ov::element::boolean, 0);
        return true;
    } else if (auto casted = std::dynamic_pointer_cast<T>(node)) {
        auto relaxed_op = std::make_shared<ov::op::TypeRelaxed<T>>(*casted,
                                                                   ov::element::TypeVector{ov::element::boolean},
                                                                   ov::element::TypeVector{to});
        replace_node(node, relaxed_op);
        return true;
    }
    return false;
}

namespace {

bool node_is_replaced(const std::shared_ptr<Node>& node) {
    const auto outputs = node->outputs();
    bool has_consumers = std::all_of(outputs.begin(), outputs.end(), [](const Output<Node>& output) {
        return output.get_target_inputs().size() == 0;
    });
    return has_consumers && !(is_type<op::v0::Result>(node) || is_type<op::Sink>(node));
}

bool convert_node_output_precision(
    const std::shared_ptr<ov::Node>& node,
    const precisions_map& precisions,
    const type_to_fuse_map& type_to_fuse,
    const std::unordered_map<const ov::Node*, std::vector<Input<Node>>>& const_to_internal_output,
    bool function_changed) {
    bool node_changed = false;
    // Handle case with Constants as they can have consumers from other ov::Model object
    const auto constant = ov::as_type_ptr<opset10::Constant>(node);
    const auto it = const_to_internal_output.find(node.get());
    if (constant && it != const_to_internal_output.end()) {
        return fuse_type_to_constant(node, precisions, it->second);
    }

    // Check that node type exists in map and we can fuse type into node
    const auto t2f_it = type_to_fuse.find(node->get_type_info());
    if (t2f_it != type_to_fuse.end()) {
        node_changed = t2f_it->second(node, precisions);
    }

    if ((function_changed || node_changed) && !node_is_replaced(node)) {
        node->revalidate_and_infer_types();
    }

    return node_changed;
}

bool convert_node_input_precision(const std::shared_ptr<ov::Node>& node,
                                  const precisions_map& precisions,
                                  const type_to_fuse_map& type_to_extend) {
    // For some operations we need to extend their input types to support new type
    auto it = type_to_extend.find(node->get_type_info());
    if (it != type_to_extend.end()) {
        return it->second(node, precisions);
    }
    return false;
}

bool convert_function_precision(const std::shared_ptr<Model>& f,
                                const type_to_fuse_map& type_to_fuse,
                                const type_to_fuse_map& type_to_extend,
                                const precisions_map& precisions,
                                std::unordered_map<const ov::Node*, std::vector<Input<Node>>>& const_to_internal_output,
                                bool has_fp16_compression,
                                bool skip_precision_sensitive,
                                bool is_changed,
                                bool is_subgraph,
                                bool convert_input_output_precision) {
    bool is_output_precision_changed = false;

    ov::element::TypeVector orig_result_types;
    if (!convert_input_output_precision) {
        const auto& results = f->get_results();
        orig_result_types.reserve(results.size());
        for (const auto& result : results) {
            orig_result_types.push_back(result->get_input_element_type(0));
        }
    }

    // Iterate over all nodes in topological order and then iterate over node outputs.
    // If output type mismatch given type we try to fuse type into this operation
    // otherwise we insert Convert operation.
    auto ops = f->get_ordered_ops();
    for (auto& node : ops) {
        if (skip_precision_sensitive && fp16_compression_is_disabled(node) && has_fp16_compression)
            continue;
        is_changed |= convert_node_input_precision(node, precisions, type_to_extend);
    }

    for (const auto& param : f->get_parameters()) {
        if (skip_precision_sensitive && fp16_compression_is_disabled(param) && has_fp16_compression)
            continue;
        is_changed |= fuse_type_to_parameter(param, precisions, convert_input_output_precision);
    }

    if (is_changed)
        ops = f->get_ordered_ops();

    auto register_constants = [&const_to_internal_output](const std::vector<std::shared_ptr<Node>>& ops) {
        for (auto& node : ops) {
            for (auto& input : node->inputs()) {
                if (auto const_node =
                        std::dynamic_pointer_cast<opset4::Constant>(input.get_source_output().get_node_shared_ptr())) {
                    const_to_internal_output[const_node.get()].emplace_back(input);
                }
            }
        }
    };

    // Register internal constants only after fixing input type that could lead to nodes
    // replacement
    register_constants(ops);

    for (auto& node : ops) {
        // skip precision sensitive nodes
        if (skip_precision_sensitive && fp16_compression_is_disabled(node) && has_fp16_compression)
            continue;
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::MultiSubGraphOp>(node)) {
            size_t sub_graphs_num = sub_graph_node->get_internal_subgraphs_size();
            for (size_t sub_graph_ind = 0; sub_graph_ind < sub_graphs_num; ++sub_graph_ind) {
                is_changed |= convert_function_precision(sub_graph_node->get_function(static_cast<int>(sub_graph_ind)),
                                                         type_to_fuse,
                                                         type_to_extend,
                                                         precisions,
                                                         const_to_internal_output,
                                                         has_fp16_compression,
                                                         skip_precision_sensitive,
                                                         is_changed || is_output_precision_changed,
                                                         true,
                                                         true);
            }
        }
        is_output_precision_changed |= convert_node_output_precision(node,
                                                                     precisions,
                                                                     type_to_fuse,
                                                                     const_to_internal_output,
                                                                     is_changed || is_output_precision_changed);
    }

    if (is_output_precision_changed) {
        ops = f->get_ordered_ops();
        is_changed |= is_output_precision_changed;
    }

    if (!is_subgraph) {
        // TODO: we need to split NopElimination pass to separate MatcherPasses and call
        // Convert elimination here
        for (auto& node : ops) {
            if (auto convert = std::dynamic_pointer_cast<opset4::Convert>(node)) {
                if (pass::constant_folding_is_disabled(node))
                    continue;
                // WA for topK, dont remove fake convert
                if (convert->input(0).get_element_type() == convert->get_convert_element_type() &&
                    convert->input_value(0).get_node_shared_ptr()->get_output_size() == 1) {
                    replace_output_update_name(convert->output(0), convert->input_value(0));
                }
            }
        }
    }

    if (is_changed && !convert_input_output_precision) {
        auto& results = f->get_results();
        for (size_t i = 0; i < results.size(); i++) {
            auto& result = results[i];
            if (result->get_input_element_type(0) != orig_result_types[i]) {
                auto result_input = result->input_value(0);
                const auto convert = std::make_shared<ov::op::v0::Convert>(result_input, orig_result_types[i]);
                if (result_input.get_node()->get_output_size() > 1) {
                    convert->set_friendly_name(result_input.get_node()->get_friendly_name() + "." +
                                               std::to_string(result_input.get_index()));
                } else {
                    convert->set_friendly_name(result_input.get_node()->get_friendly_name());
                    result_input.get_node()->set_friendly_name("");
                }

                auto& convert_output_tensor = convert->get_output_tensor(0);
                convert_output_tensor.set_names(result_input.get_names());
                OPENVINO_SUPPRESS_DEPRECATED_START
                const auto& legacy_name = ov::descriptor::get_ov_tensor_legacy_name(result_input.get_tensor());
                if (!legacy_name.empty()) {
                    ov::descriptor::set_ov_tensor_legacy_name(convert_output_tensor, legacy_name);
                }
                OPENVINO_SUPPRESS_DEPRECATED_END

                result_input.set_names({});
                result->input(0).replace_source_output(convert->output(0));
                result->revalidate_and_infer_types();
            }
        }
    }

    return is_changed;
}

bool convert_precision(ov::pass::PassBase& pass,
                       const std::shared_ptr<ov::Model>& f,
                       const type_to_fuse_map& type_to_fuse,
                       const type_to_fuse_map& type_to_extend,
                       const precisions_map& precisions,
                       bool has_fp16_compression,
                       bool skip_precision_sensitive,
                       bool convert_input_output_precision) {
    // As Constant operations can be shared between multiple ov::Models so before
    // changing precision we need to understand which Constant consumers belongs
    // to the current ov::Model
    std::unordered_map<const ov::Node*, std::vector<Input<Node>>> const_to_internal_output;
    return convert_function_precision(f,
                                      type_to_fuse,
                                      type_to_extend,
                                      precisions,
                                      const_to_internal_output,
                                      has_fp16_compression,
                                      skip_precision_sensitive,
                                      false,
                                      false,
                                      convert_input_output_precision);
}

using precisions_set_t = std::unordered_set<ov::element::Type_t, EnumClassHash>;

precisions_set_t find_all_used_precisions(const std::shared_ptr<ov::Model>& fn) {
    precisions_set_t used_precisions;

    ov::traverse_nodes(fn, [&](const std::shared_ptr<ov::Node>& node) {
        for (const auto& output : node->outputs()) {
            used_precisions.emplace(output.get_element_type());
        }
        if (auto sub_graph_node = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp>(node)) {
            size_t sub_graphs_num = sub_graph_node->get_internal_subgraphs_size();
            for (size_t sub_graph_ind = 0; sub_graph_ind < sub_graphs_num; ++sub_graph_ind) {
                auto sub_graph_precisions =
                    find_all_used_precisions(sub_graph_node->get_function(static_cast<int>(sub_graph_ind)));
                used_precisions.insert(sub_graph_precisions.begin(), sub_graph_precisions.end());
            }
        }
    });

    return used_precisions;
}

}  // namespace

bool ov::pass::ConvertPrecision::run_on_model(const std::shared_ptr<ov::Model>& f) {
    const auto used_precisions_set = find_all_used_precisions(f);
    precisions_map used_precisions;
    for (const auto& p : used_precisions_set) {
        auto it = m_precisions.find(p);
        if (it != m_precisions.end())
            used_precisions.insert(*it);
    }

    if (used_precisions.empty())
        return false;

    bool has_fp16_compression = m_precisions.count(element::f32) > 0 && m_precisions[element::f32] == element::f16;

    if (m_keep_precision_sensitive_in_fp32 && has_fp16_compression) {
        pass::Manager manager(get_pass_config());
        // Mark subgraphs with disable_fp16_compression to keep them in FP32
        manager.register_pass<pass::MarkSugraphsToKeepInMixedPrecision>();
        manager.register_pass<pass::AlignMixedFP32FP16Types>();
        manager.run_passes(f);
    }

    type_to_fuse_map type_to_fuse{
        {opset4::Convert::get_type_info_static(), fuse_type_to_convert},
        {opset4::ShapeOf::get_type_info_static(), fuse_type_to_shapeof},
        {opset3::NonMaxSuppression::get_type_info_static(), fuse_type_to_nms3},
        {opset4::NonMaxSuppression::get_type_info_static(), fuse_type_to_nms4},
        {opset5::NonMaxSuppression::get_type_info_static(), fuse_type_to_nms5},
        {opset9::NonMaxSuppression::get_type_info_static(), fuse_type_to_nms9},
        {opset8::MatrixNms::get_type_info_static(), fuse_type_to_matrix_nms},
        {opset8::MulticlassNms::get_type_info_static(), fuse_type_to_multiclass_nms},
        {opset9::MulticlassNms::get_type_info_static(), fuse_type_to_multiclass_nms},
        {opset9::GenerateProposals::get_type_info_static(), fuse_type_to_generate_proposals},
        {opset6::CTCGreedyDecoderSeqLen::get_type_info_static(), fuse_type_to_ctc_greedy_decoder_seq_len},
        {opset1::TopK::get_type_info_static(), fuse_type_to_topk},
        {opset4::TopK::get_type_info_static(), fuse_type_to_topk},
        {opset11::TopK::get_type_info_static(), fuse_type_to_topk},
        {opset8::MaxPool::get_type_info_static(), fuse_type_to_maxpool},
        {opset4::NonZero::get_type_info_static(), fuse_type_to_nonzero},
        {opset4::Bucketize::get_type_info_static(), fuse_type_to_bucketize},
        {opset4::Equal::get_type_info_static(), fuse_type_to_binary_comparision<opset4::Equal>},
        {opset4::NotEqual::get_type_info_static(), fuse_type_to_binary_comparision<opset4::NotEqual>},
        {opset4::Greater::get_type_info_static(), fuse_type_to_binary_comparision<opset4::Greater>},
        {opset4::GreaterEqual::get_type_info_static(), fuse_type_to_binary_comparision<opset4::GreaterEqual>},
        {opset4::Less::get_type_info_static(), fuse_type_to_binary_comparision<opset4::Less>},
        {opset4::LessEqual::get_type_info_static(), fuse_type_to_binary_comparision<opset4::LessEqual>},
        {opset10::IsFinite::get_type_info_static(), fuse_type_to_binary_comparision<opset10::IsFinite>},
        {opset10::IsNaN::get_type_info_static(), fuse_type_to_binary_comparision<opset10::IsNaN>},
        {opset10::IsInf::get_type_info_static(), fuse_type_to_binary_comparision<opset10::IsInf>},
        {opset4::LogicalAnd::get_type_info_static(), fuse_type_to_logical<opset4::LogicalAnd>},
        {opset4::LogicalOr::get_type_info_static(), fuse_type_to_logical<opset4::LogicalOr>},
        {opset4::LogicalXor::get_type_info_static(), fuse_type_to_logical<opset4::LogicalXor>},
        {opset4::LogicalNot::get_type_info_static(), fuse_type_to_logical<opset4::LogicalNot>},
        {opset1::Xor::get_type_info_static(), fuse_type_to_logical<opset1::Xor>},
        {opset4::ReduceLogicalAnd::get_type_info_static(), fuse_type_to_reduce_logical<opset4::ReduceLogicalAnd>},
        {opset4::ReduceLogicalOr::get_type_info_static(), fuse_type_to_reduce_logical<opset4::ReduceLogicalOr>},
        {opset1::ShapeOf::get_type_info_static(), fuse_type_to_shapeof_v0},
        {opset4::Range::get_type_info_static(), fuse_type_to_range_v4},
        {opset9::Eye::get_type_info_static(), fuse_type_to_eye_v9},
        {opset10::Unique::get_type_info_static(), fuse_type_to_unique_v10},
        {opset8::RandomUniform::get_type_info_static(), fuse_type_to_random_uniform_v8}};

    for (const auto& it : m_additional_type_to_fuse_map) {
        type_to_fuse[it.first] = it.second;
    }

    type_to_fuse.insert(m_additional_type_to_fuse_map.begin(), m_additional_type_to_fuse_map.end());

    static type_to_fuse_map type_to_extend{
        {opset4::Select::get_type_info_static(), extend_select_type},
        {opset1::Reverse::get_type_info_static(), extend_reverse_type},
    };

    bool is_changed = convert_precision(*this,
                                        f,
                                        type_to_fuse,
                                        type_to_extend,
                                        used_precisions,
                                        has_fp16_compression,
                                        m_keep_precision_sensitive_in_fp32,
                                        m_convert_input_output_precision);

    // to remove extra converts
    if (m_keep_precision_sensitive_in_fp32) {
        pass::Manager manager(get_pass_config());
        manager.register_pass<pass::EnableDecompressionConvertConstantFolding>();
        manager.register_pass<pass::ConstantFolding>();
        manager.run_passes(f);
    }

    (void)is_changed;  // ignored

    // Returning value is false because pass::Manager always apply Validation pass
    // if function was changed. This helps to avoid excess Validations after applying
    // this pass. In future when we will return more meaningful status code it will be
    // replaced with real status reported by manager.run_passes() method call.
    return false;
}

bool fuse_type_to_shapeof(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto it = precisions.find(node->get_output_element_type(0));
    if (it == precisions.end())
        return false;
    const auto& to = it->second;
    if (auto shapeof = ov::as_type_ptr<opset4::ShapeOf>(node)) {
        if (to == ov::element::i32 || to == ov::element::i64) {
            shapeof->set_output_type(to);
            return true;
        }
    }
    return false;
}

bool fuse_type_to_random_uniform_v8(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto it = precisions.find(node->get_output_element_type(0));
    if (it == precisions.end())
        return false;
    const auto& to = it->second;
    if (auto random_uniform = ov::as_type_ptr<opset8::RandomUniform>(node)) {
        if (to.is_integral_number() || to.is_real()) {
            random_uniform->set_out_type(to);
            return true;
        }
    }
    return false;
}

bool fuse_type_to_unique_v10(const std::shared_ptr<Node>& node, const precisions_map& precisions) {
    bool res = false;
    if (auto unique = ov::as_type_ptr<opset10::Unique>(node)) {
        auto it = precisions.find(node->get_output_element_type(1));
        if (it != precisions.end()) {
            unique->set_index_element_type(it->second);
            res = true;
        }
        it = precisions.find(node->get_output_element_type(3));
        if (it != precisions.end()) {
            unique->set_count_element_type(it->second);
            res = true;
        }
    }
    return res;
}

bool fuse_type_to_range_v4(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto it = precisions.find(node->get_output_element_type(0));
    if (it == precisions.end())
        return false;
    const auto& to = it->second;
    if (auto range = ov::as_type_ptr<opset4::Range>(node)) {
        if (to.is_integral_number() || to.is_real()) {
            range->set_output_type(to);
            return true;
        }
    }
    return false;
}

bool fuse_type_to_eye_v9(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto it = precisions.find(node->get_output_element_type(0));
    if (it == precisions.end())
        return false;
    const auto& to = it->second;
    if (auto eye_node = ov::as_type_ptr<opset9::Eye>(node)) {
        if (to.is_integral() || to.is_real()) {
            eye_node->set_out_type(to);
            return true;
        }
    }
    return false;
}

bool fuse_type_to_parameter(const std::shared_ptr<ov::Node>& node,
                            const precisions_map& precisions,
                            bool convert_input_precision) {
    auto it = precisions.find(node->get_output_element_type(0));
    if (it == precisions.end())
        return false;
    bool changed = false;
    const auto& to = it->second;
    if (auto param = ov::as_type_ptr<opset4::Parameter>(node)) {
        if (convert_input_precision) {
            param->set_element_type(to);
            param->validate_and_infer_types();
            changed = true;
        } else {
            auto param_consumers = param->output(0).get_target_inputs();
            auto convert = std::make_shared<opset4::Convert>(param, to);
            for (auto& input : param_consumers) {
                const auto consumer = input.get_node();
                if (ov::is_type<ov::op::v0::Result>(consumer) || ov::is_type<ov::op::v0::Convert>(consumer)) {
                    continue;
                }
                input.replace_source_output(convert);
                changed = true;
            }
        }
    }
    return changed;
}

bool fuse_type_to_convert(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto it = precisions.find(node->get_output_element_type(0));
    if (it == precisions.end())
        return false;
    const auto& to = it->second;
    if (auto convert = ov::as_type_ptr<opset4::Convert>(node)) {
        convert->set_convert_element_type(to);
        return true;
    }
    return false;
}

bool fuse_type_to_nms3(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto it = precisions.find(node->get_output_element_type(0));
    if (it == precisions.end())
        return false;
    const auto& to = it->second;
    if (auto nms = ov::as_type_ptr<opset3::NonMaxSuppression>(node)) {
        if (to == ov::element::i32 || to == ov::element::i64) {
            nms->set_output_type(to);
        } else {
            OPENVINO_THROW("Type: " + to.get_type_name() + " is not supported for NMS3");
        }
        return true;
    }
    return false;
}

bool fuse_type_to_nms4(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto it = precisions.find(node->get_output_element_type(0));
    if (it == precisions.end())
        return false;
    const auto& to = it->second;
    if (auto nms = ov::as_type_ptr<opset4::NonMaxSuppression>(node)) {
        if (to == ov::element::i32 || to == ov::element::i64) {
            nms->set_output_type(to);
        } else {
            OPENVINO_THROW("Type: " + to.get_type_name() + " is not supported for NMS4");
        }
        return true;
    }
    return false;
}

bool fuse_type_to_nms5(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto nms = ov::as_type_ptr<opset5::NonMaxSuppression>(node);
    if (!nms) {
        return false;
    }

    bool res = false;
    auto it = precisions.find(node->get_output_element_type(0));
    if (it != precisions.end()) {
        const auto& to = it->second;
        if (to == ov::element::i32 || to == ov::element::i64) {
            nms->set_output_type(to);
            res = true;
            if (precisions.count(node->get_output_element_type(1)) == 0) {
                return res;
            }
        }
    }

    auto type_relaxed = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(node);
    ov::element::TypeVector output_types;
    for (size_t i = 0; i < node->get_output_size(); i++) {
        it = precisions.find(node->get_output_element_type(i));
        if (it == precisions.end()) {
            output_types.push_back(node->get_output_element_type(i));
            continue;
        }
        const auto& to = it->second;
        if (type_relaxed) {
            type_relaxed->set_overridden_output_type(to, i);
            res = true;
        }
        output_types.push_back(to);
    }

    if (!type_relaxed) {
        auto relaxed_op = std::make_shared<ov::op::TypeRelaxed<opset5::NonMaxSuppression>>(*nms,
                                                                                           ov::element::TypeVector{},
                                                                                           output_types);
        replace_node(node, relaxed_op);
        res = true;
    }

    return res;
}

bool fuse_type_to_nms9(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto nms = ov::as_type_ptr<opset9::NonMaxSuppression>(node);
    if (!nms) {
        return false;
    }

    bool res = false;
    auto type_relaxed = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(node);
    if (type_relaxed) {
        for (size_t i = 0; i < node->get_output_size(); i++) {
            auto it = precisions.find(node->get_output_element_type(i));
            if (it == precisions.end()) {
                continue;
            }
            const auto& to = it->second;
            type_relaxed->set_overridden_output_type(to, i);
            res = true;
        }
        return res;
    }
    auto it = precisions.find(node->get_output_element_type(0));
    if (it != precisions.end()) {
        const auto& to = it->second;
        if (to == ov::element::i32 || to == ov::element::i64) {
            nms->set_output_type(to);
            res = true;
            if (precisions.count(node->get_output_element_type(1)) == 0) {
                return res;
            }
        }
    }

    ov::element::TypeVector output_types;
    for (size_t i = 0; i < node->get_output_size(); i++) {
        it = precisions.find(node->get_output_element_type(i));
        if (it == precisions.end()) {
            output_types.push_back(node->get_output_element_type(i));
            continue;
        }
        const auto& to = it->second;
        output_types.push_back(to);
    }

    auto relaxed_op =
        std::make_shared<ov::op::TypeRelaxed<opset9::NonMaxSuppression>>(*nms, ov::element::TypeVector{}, output_types);
    replace_node(node, relaxed_op);
    return true;
}

namespace {

bool update_type(size_t idx,
                 const std::shared_ptr<ov::Node>& node,
                 const precisions_map& precisions,
                 std::function<void(const element::Type&)> update_method) {
    auto it = precisions.find(node->get_output_element_type(idx));
    if (it != precisions.end()) {
        const auto& to = it->second;
        if (to == ov::element::i32 || to == ov::element::i64) {
            update_method(to);
            return true;
        }
    }
    return false;
}

}  // namespace

bool fuse_type_to_matrix_nms(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto nms = ov::as_type_ptr<opset8::MatrixNms>(node);
    if (!nms) {
        return false;
    }

    return update_type(1, node, precisions, [&](const element::Type& to) {
        nms->set_output_type(to);
    });
}

bool fuse_type_to_multiclass_nms(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    std::shared_ptr<ov::op::util::MulticlassNmsBase> nms;
    if (ov::is_type<ov::op::v8::MulticlassNms>(node)) {
        nms = ov::as_type_ptr<opset8::MulticlassNms>(node);
    } else {
        nms = ov::as_type_ptr<opset9::MulticlassNms>(node);
    }
    if (!nms) {
        return false;
    }

    return update_type(1, node, precisions, [&](const element::Type& to) {
        nms->set_output_type(to);
    });
}

bool fuse_type_to_generate_proposals(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto generate_proposals = ov::as_type_ptr<opset9::GenerateProposals>(node);
    if (!generate_proposals) {
        return false;
    }

    return update_type(2, node, precisions, [&](const element::Type& to) {
        generate_proposals->set_roi_num_type(to);
    });
}

bool fuse_type_to_topk(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    if (auto topk = ov::as_type_ptr<ov::op::util::TopKBase>(node)) {
        return update_type(1, node, precisions, [&](const element::Type& to) {
            topk->set_index_element_type(to);
        });
    }
    return false;
}

bool fuse_type_to_maxpool(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    if (auto maxpool = ov::as_type_ptr<opset8::MaxPool>(node)) {
        return update_type(1, node, precisions, [&](const element::Type& to) {
            maxpool->set_index_element_type(to);
        });
    }
    return false;
}

bool fuse_type_to_ctc_greedy_decoder_seq_len(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    bool res = false;
    if (auto ctc_decoder = ov::as_type_ptr<opset6::CTCGreedyDecoderSeqLen>(node)) {
        res = update_type(0, node, precisions, [&](const element::Type& to) {
            ctc_decoder->set_classes_index_type(to);
        });
        res = update_type(1,
                          node,
                          precisions,
                          [&](const element::Type& to) {
                              ctc_decoder->set_sequence_length_type(to);
                          }) ||
              res;
    }
    return res;
}

bool fuse_type_to_nonzero(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    if (auto nonzero = ov::as_type_ptr<opset4::NonZero>(node)) {
        return update_type(0, node, precisions, [&](const element::Type& to) {
            nonzero->set_output_type(to);
        });
    }
    return false;
}

bool fuse_type_to_bucketize(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    if (auto b = ov::as_type_ptr<opset4::Bucketize>(node)) {
        return update_type(0, node, precisions, [&](const element::Type& to) {
            b->set_output_type(to);
        });
    }
    return false;
}

bool fuse_type_to_shapeof_v0(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto it = precisions.find(node->get_output_element_type(0));
    if (it == precisions.end())
        return false;
    const auto& to = it->second;
    if (auto type_relaxed = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(node)) {
        type_relaxed->set_overridden_output_type(to);
        return true;
    } else if (auto casted = std::dynamic_pointer_cast<opset1::ShapeOf>(node)) {
        auto relaxed_op = std::make_shared<ov::op::TypeRelaxed<opset1::ShapeOf>>(*casted,
                                                                                 ov::element::TypeVector{},
                                                                                 ov::element::TypeVector{to});
        replace_node(node, relaxed_op);
        return true;
    }
    return false;
}

bool extend_select_type(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    if (auto type_relaxed = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(node)) {
        type_relaxed->set_origin_input_type(ov::element::boolean, 0);
        return true;
    } else if (auto casted = std::dynamic_pointer_cast<opset4::Select>(node)) {
        auto relaxed_op =
            std::make_shared<op::TypeRelaxed<opset4::Select>>(*casted,
                                                              ov::element::TypeVector{ov::element::boolean},
                                                              ov::element::TypeVector{});
        replace_node(node, relaxed_op);
        return true;
    }
    return false;
}

bool extend_reverse_type(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    if (const auto casted = std::dynamic_pointer_cast<opset1::Reverse>(node)) {
        if (casted->get_mode() == ov::op::v1::Reverse::Mode::MASK) {
            auto relaxed_op = std::make_shared<op::TypeRelaxed<opset1::Reverse>>(
                *casted,
                ov::element::TypeVector{casted->get_input_element_type(0), ov::element::boolean},
                ov::element::TypeVector{casted->get_output_element_type(0)});
            replace_node(node, relaxed_op);
        }
        return true;
    }
    return false;
}

template <typename src_type, typename dst_type>
inline dst_type convert_value(src_type val) {
    if (val > std::numeric_limits<dst_type>::max()) {
        return std::numeric_limits<dst_type>::max();
    } else if (val < std::numeric_limits<dst_type>::lowest()) {
        return std::numeric_limits<dst_type>::lowest();
    }
    return static_cast<dst_type>(val);
}

// We need to treat U64->I32 and U32->I32 as a separate case, because of C++'s implicit promotion
// from signed to unsigned, and we don't need to compare and clamp the input to
// std::numeric_limits<int32_t>::lowest()
template <>
inline int32_t convert_value<uint64_t, int32_t>(uint64_t val) {
    if (val > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return std::numeric_limits<int32_t>::max();
    }
    return static_cast<int32_t>(val);
}

template <>
inline int32_t convert_value<uint32_t, int32_t>(uint32_t val) {
    if (val > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        return std::numeric_limits<int32_t>::max();
    }
    return static_cast<int32_t>(val);
}

namespace {
template <ov::element::Type_t PREC_FROM, ov::element::Type_t PREC_TO>
std::shared_ptr<ov::Node> change_constant_precision(std::shared_ptr<opset4::Constant>& constant) {
    using src_type = typename element_type_traits<PREC_FROM>::value_type;
    using dst_type = typename element_type_traits<PREC_TO>::value_type;

    const auto* src_data = constant->get_data_ptr<src_type>();
    const auto size = shape_size(constant->get_shape());

    auto new_constant = std::make_shared<opset4::Constant>(PREC_TO, constant->get_shape());
    new_constant->output(0).set_names(constant->output(0).get_names());
    auto* dst_data = const_cast<dst_type*>(reinterpret_cast<const dst_type*>(new_constant->get_data_ptr()));
    if (dst_data == nullptr)
        OPENVINO_THROW("Can't get destination data pointer");

    for (size_t i = 0; i < size; ++i) {
        dst_data[i] = convert_value<src_type, dst_type>(src_data[i]);
    }
    return new_constant;
}

template <>
std::shared_ptr<Node> change_constant_precision<ov::element::Type_t::f32, ov::element::Type_t::f16>(
    std::shared_ptr<opset4::Constant>& constant) {
    using src_type = typename element_type_traits<ov::element::Type_t::f32>::value_type;
    using dst_type = typename element_type_traits<ov::element::Type_t::f16>::value_type;

    const auto* src_data = constant->get_data_ptr<src_type>();
    const auto size = shape_size(constant->get_shape());

    auto new_constant = std::make_shared<opset4::Constant>(ov::element::Type_t::f16, constant->get_shape());
    new_constant->output(0).set_names(constant->output(0).get_names());
    auto* dst_data = const_cast<dst_type*>(reinterpret_cast<const dst_type*>(new_constant->get_data_ptr()));
    if (dst_data == nullptr)
        OPENVINO_THROW("Can't get destination data pointer");

    ov::reference::convert_from_f32_to_f16_with_clamp(src_data, dst_data, size);

    return new_constant;
}

template <>
std::shared_ptr<Node> change_constant_precision<ov::element::Type_t::f16, ov::element::Type_t::f32>(
    std::shared_ptr<opset4::Constant>& constant) {
    using src_type = typename element_type_traits<ov::element::Type_t::f16>::value_type;
    using dst_type = typename element_type_traits<ov::element::Type_t::f32>::value_type;

    const auto* src_data = constant->get_data_ptr<src_type>();
    const auto size = shape_size(constant->get_shape());

    auto new_constant = std::make_shared<opset4::Constant>(ov::element::Type_t::f32, constant->get_shape());
    new_constant->output(0).set_names(constant->output(0).get_names());
    auto* dst_data = const_cast<dst_type*>(reinterpret_cast<const dst_type*>(new_constant->get_data_ptr()));
    if (dst_data == nullptr)
        OPENVINO_THROW("Can't get destination data pointer");

    ov::reference::convert<src_type, dst_type>(src_data, dst_data, size);

    return new_constant;
}

/**
 * @brief Method converts low precision integer types
 * The method uses the next logic for conversion:
 *  * For unsigned types we just copy all bits to destination type (which is bigger):
 *    int4 [1011] -> int8 [00001011]
 *  * For signed types we copy all bits (except sign bit) to destination type and after
 *    that for negative values we set to 1 all higher bits:
 *    int4 [1011] -> int8 [11111011]
 *
 * @param src source value      !!! the type must be unsigned !!!
 * @param dst destination value !!! the type must be unsigned !!!
 * @param src_offset source offset (for custom data types)
 * @param src_size source size (for custom data types)
 * @param dst_offset destination offset
 * @param dst_size destination size
 * @param is_signed the type of source data
 */
template <class SRC, class DST>
void convert_lp_value(const SRC& src,
                      DST& dst,
                      size_t src_offset,
                      size_t src_size,
                      size_t dst_offset,
                      size_t dst_size,
                      bool is_signed) {
    constexpr SRC src_max = std::numeric_limits<SRC>::max();
    constexpr DST dst_max = std::numeric_limits<DST>::max();
    // Make a shift for the source value
    // src [11101000] offset 2, size 4
    // val [00011101]
    SRC val = src >> src_offset;
    // dst     [10001111 00000100] offset 5 size 9
    // new_val [00000000 00000000]
    DST new_val = 0;

    // Calculate diff in order to clean bits which don't exist in the source value
    // diff 4
    size_t diff = sizeof(SRC) * 8 - src_size;
    // Clean unnecessary bits
    // val [11010000]
    val = val << diff;
    // val [00001101]
    val = val >> diff;

    // Get the sign of value
    // sign [00000001]
    SRC sign = (val >> (src_size - 1)) & 0b1;

    // If source type is signed and negative
    if (is_signed && sign) {
        // val [11111101]
        val |= src_max << diff;
        // new_val [00000001 11111111]
        new_val = dst_max >> (sizeof(DST) * 8 - dst_size);
        // new_val [00000001 11111101]
        new_val &= (dst_max << sizeof(SRC) * 8) | val;
    } else {
        // new_val [00000000 00001101]
        new_val = val;
    }

    // Make a mask in order to save other values if DST contains several values
    // mask [11000000 00011111]
    DST mask = 0;
    if (dst_offset + dst_size < sizeof(DST) * 8)
        mask = (dst_max << (dst_offset + dst_size));
    if (dst_offset != 0)
        mask |= (dst_max >> (sizeof(DST) * 8 - dst_offset));

    // Add mask to our converted value
    // signed:   new_val [11100000 10111111]
    // unsigned: new_val [11000001 10111111]
    new_val = mask | (new_val << dst_offset);

    // Add our value to destination
    // dst: [10111111 11100100]
    dst |= ~mask;
    // signed:   dst [10100000 10100100]
    // unsigned: dst [10000001 10100100]
    dst &= new_val;
}

std::shared_ptr<Node> convert_low_precisions_int(std::shared_ptr<opset4::Constant>& constant, ov::element::Type to) {
    // Supported integer precisions
    static const precisions_set_t supported_integer_precisions = {ov::element::i4, ov::element::u4, ov::element::u1};
    // Get source element type and source data
    auto src_type = constant->get_element_type();
    const auto* src_data = reinterpret_cast<const uint8_t*>(constant->get_data_ptr());

    // We support conversion only if several elements can be represented in one instance of some
    // C++ common data type without any exception, destination data type should be bigger than
    // source and destination data type should be real
    if (!supported_integer_precisions.count(src_type) || (src_type.size() * 8) % src_type.bitwidth() ||
        (to.size() * 8) % to.bitwidth() || to.is_real() || to.bitwidth() < src_type.bitwidth())
        OPENVINO_THROW("Convert low precision for " + constant->get_element_type().get_type_name() + " to " +
                       to.get_type_name() + " is not implemented!");

    // Create a new constant operation and get destination data
    auto new_constant = std::make_shared<opset4::Constant>(to, constant->get_shape());
    auto* dst_data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(new_constant->get_data_ptr()));
    // Check pointers
    if (src_data == nullptr || dst_data == nullptr)
        OPENVINO_THROW("Can't get data pointer");

    // Convert values
    const auto size = shape_size(constant->get_shape());
    size_t src_idx(0), dst_idx(0), dst_off(0), src_off(0);
    if (src_type.bitwidth() < 8) {
        src_off = 8 - src_type.bitwidth();
    }

    if (to.bitwidth() < 8) {
        dst_off = 8 - to.bitwidth();
    }

    for (size_t i = 0; i < size; i++) {
        // Source type at the current moment always less than 1 byte
        // Select the right destination type
        switch (to.size()) {
        case 1:
            convert_lp_value<uint8_t, uint8_t>(src_data[src_idx],
                                               dst_data[dst_idx],
                                               src_off,
                                               src_type.bitwidth(),
                                               dst_off,
                                               to.bitwidth(),
                                               src_type.is_signed());
            break;
        case 2:
            convert_lp_value<uint8_t, uint16_t>(src_data[src_idx],
                                                reinterpret_cast<uint16_t*>(dst_data)[dst_idx],
                                                src_off,
                                                src_type.bitwidth(),
                                                dst_off,
                                                to.bitwidth(),
                                                src_type.is_signed());
            break;
        case 4:
            convert_lp_value<uint8_t, uint32_t>(src_data[src_idx],
                                                reinterpret_cast<uint32_t*>(dst_data)[dst_idx],
                                                src_off,
                                                src_type.bitwidth(),
                                                dst_off,
                                                to.bitwidth(),
                                                src_type.is_signed());
            break;
        case 8:
            convert_lp_value<uint8_t, uint64_t>(src_data[src_idx],
                                                reinterpret_cast<uint64_t*>(dst_data)[dst_idx],
                                                src_off,
                                                src_type.bitwidth(),
                                                dst_off,
                                                to.bitwidth(),
                                                src_type.is_signed());
            break;
        default:
            OPENVINO_THROW("Unsupported element size!");
        }
        // Calculate offsets and indexes
        if (src_type.bitwidth() < 8) {
            if (src_off == 0) {
                src_off = 8;
                src_idx++;
            }
            src_off -= src_type.bitwidth();
        } else {
            src_idx++;
        }
        if (to.bitwidth() < 8) {
            if (dst_off == 0) {
                dst_off = 8;
                dst_idx++;
            }
            dst_off -= to.bitwidth();
        } else {
            dst_idx++;
        }
    }

    return new_constant;
}

}  // namespace

bool fuse_type_to_constant(const std::shared_ptr<ov::Node>& node,
                           const precisions_map& precisions,
                           const std::vector<Input<Node>>& consumers) {
    // Consts marked with is_keep_const_precision should be kept in their own precision until they reach the plugin
    if (is_keep_const_precision(node))
        return false;

    auto from = node->get_element_type();
    auto it = precisions.find(from);
    if (it == precisions.end())
        return false;
    const auto& to = it->second;
    if (auto constant = ov::as_type_ptr<opset4::Constant>(node)) {
        std::shared_ptr<ov::Node> new_const;
        if (from == ov::element::u64 && to == ov::element::i32) {
            new_const = change_constant_precision<ov::element::Type_t::u64, ov::element::Type_t::i32>(constant);
        } else if (from == ov::element::i64 && to == ov::element::i32) {
            new_const = change_constant_precision<ov::element::Type_t::i64, ov::element::Type_t::i32>(constant);
        } else if (from == ov::element::u8 && to == ov::element::i32) {
            new_const = change_constant_precision<ov::element::Type_t::u8, ov::element::Type_t::i32>(constant);
        } else if (from == ov::element::u16 && to == ov::element::i32) {
            new_const = change_constant_precision<ov::element::Type_t::u16, ov::element::Type_t::i32>(constant);
        } else if (from == ov::element::i16 && to == ov::element::i32) {
            new_const = change_constant_precision<ov::element::Type_t::i16, ov::element::Type_t::i32>(constant);
        } else if (from == ov::element::u32 && to == ov::element::i32) {
            new_const = change_constant_precision<ov::element::Type_t::u32, ov::element::Type_t::i32>(constant);
        } else if (from == ov::element::f64 && to == ov::element::f32) {
            new_const = change_constant_precision<ov::element::Type_t::f64, ov::element::Type_t::f32>(constant);
        } else if (from == ov::element::bf16 && to == ov::element::f32) {
            new_const = change_constant_precision<ov::element::Type_t::bf16, ov::element::Type_t::f32>(constant);
        } else if (from == ov::element::f32 && to == ov::element::f16) {
            new_const = change_constant_precision<ov::element::Type_t::f32, ov::element::Type_t::f16>(constant);
        } else if (from == ov::element::f16 && to == ov::element::f32) {
            new_const = change_constant_precision<ov::element::Type_t::f16, ov::element::Type_t::f32>(constant);
        } else if (from == ov::element::boolean && to == ov::element::u8) {
            new_const = change_constant_precision<ov::element::Type_t::boolean, ov::element::Type_t::u8>(constant);
        } else if (from == ov::element::boolean && to == ov::element::i32) {
            new_const = change_constant_precision<ov::element::Type_t::boolean, ov::element::Type_t::i32>(constant);
        } else if (from == ov::element::i4 || from == ov::element::u4 || from == ov::element::u1) {
            new_const = convert_low_precisions_int(constant, to);
        } else {
            OPENVINO_THROW("Precision conversion from " + from.get_type_name() + " to " + to.get_type_name() +
                           " is not supported");
        }
        for (auto& output : consumers) {
            output.replace_source_output(new_const);
        }

        new_const->validate_and_infer_types();
        new_const->set_friendly_name(constant->get_friendly_name());
        ov::copy_runtime_info(constant, new_const);
        return true;
    }
    return false;
}
