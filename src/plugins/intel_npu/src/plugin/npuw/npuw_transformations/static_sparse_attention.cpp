// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "static_sparse_attention.hpp"

#include <optional>
#include <regex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../logging.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/node_util.hpp"

namespace {

namespace op = ov::op;

constexpr const char* kBlockIndexKey = "block_index_key";

struct QSAState {
    std::string kind;   // block_index_key | incomplete_index_key | incomplete_index_rope
    std::string index;  // layer index
};

// StatefulToStateless keeps the QSA variables unrecognized, as
//     {input|output}_restored.cache_params.past.<kind>.N cache_params.present.<kind>.N
std::optional<QSAState> qsa_state(const ov::descriptor::Tensor& tensor, const std::string& io) {
    static const std::regex pattern(
        R"(^(input|output)_restored\.cache_params\.past\.(block_index_key|incomplete_index_key|incomplete_index_rope)\.(\d+)cache_params\.present\.\2\.\3$)");
    for (const auto& name : tensor.get_names()) {
        std::smatch match;
        if (std::regex_match(name, match, pattern) && match[1].str() == io) {
            return QSAState{match[2].str(), match[3].str()};
        }
    }
    return std::nullopt;
}

std::optional<int64_t> scalar_value(const ov::Output<ov::Node>& output) {
    const auto constant = ov::as_type_ptr<op::v0::Constant>(output.get_node_shared_ptr());
    if (!constant || ov::shape_size(constant->get_shape()) != 1) {
        return std::nullopt;
    }
    return constant->cast_vector<int64_t>()[0];
}

bool is_scalar(const ov::Output<ov::Node>& output, int64_t value) {
    const auto v = scalar_value(output);
    return v.has_value() && *v == value;
}

// num_blocks = ReduceMax(Divide(key_length, r), 0) cuts the block_index_key state of QSAPrepare:
//     Parameter -> Concat(past, filler) -> Slice(stop = Unsqueeze(Add(num_blocks, 1)))
std::shared_ptr<op::v1::ReduceMax> find_num_blocks(const std::shared_ptr<op::v0::Parameter>& past_state) {
    for (const auto& concat_in : past_state->output(0).get_target_inputs()) {
        const auto concat = ov::as_type<op::v0::Concat>(concat_in.get_node());
        if (!concat || concat_in.get_index() != 0) {
            continue;
        }
        for (const auto& slice_in : concat->output(0).get_target_inputs()) {
            const auto slice = ov::as_type<op::v8::Slice>(slice_in.get_node());
            if (!slice || slice_in.get_index() != 0) {
                continue;
            }
            const auto stop = ov::as_type_ptr<op::v0::Unsqueeze>(slice->get_input_node_shared_ptr(2));
            const auto add = stop ? ov::as_type_ptr<op::v1::Add>(stop->get_input_node_shared_ptr(0)) : nullptr;
            if (!add) {
                continue;
            }
            for (size_t i = 0; i < 2; ++i) {
                const auto reduce = ov::as_type_ptr<op::v1::ReduceMax>(add->get_input_node_shared_ptr(i));
                if (reduce && is_scalar(add->input_value(1 - i), 1) && !reduce->get_keep_dims() &&
                    is_scalar(reduce->input_value(1), 0) &&
                    ov::is_type<op::v1::Divide>(reduce->get_input_node_ptr(0))) {
                    return reduce;
                }
            }
        }
    }
    return nullptr;
}

// sequence_positions = Unsqueeze(Range(0, S, 1), 0) - Unsqueeze(Subtract(S, key_length), 1)
bool is_sequence_positions(const std::shared_ptr<ov::Node>& node, const ov::Output<ov::Node>& key_length) {
    const auto sub = ov::as_type_ptr<op::v1::Subtract>(node);
    if (!sub) {
        return false;
    }
    const auto slots = ov::as_type_ptr<op::v0::Unsqueeze>(sub->get_input_node_shared_ptr(0));
    const auto padding = ov::as_type_ptr<op::v0::Unsqueeze>(sub->get_input_node_shared_ptr(1));
    if (!slots || !padding || !is_scalar(slots->input_value(1), 0) || !is_scalar(padding->input_value(1), 1)) {
        return false;
    }
    const auto range = slots->get_input_node_shared_ptr(0);
    if (!ov::is_type<op::v4::Range>(range) && !ov::is_type<op::v0::Range>(range)) {
        return false;
    }
    const auto pad = ov::as_type_ptr<op::v1::Subtract>(padding->get_input_node_shared_ptr(0));
    return pad && is_scalar(range->input_value(0), 0) && is_scalar(range->input_value(2), 1) &&
           range->input_value(1) == pad->input_value(0) && pad->input_value(1) == key_length;
}

// Logical position of every key slot: CumSum(mask) - 1 for the valid slots, -1 for the masked ones.
ov::Output<ov::Node> positions_from_mask(const ov::Output<ov::Node>& attention_mask) {
    auto mask = attention_mask;
    if (mask.get_element_type() != ov::element::i64) {
        mask = std::make_shared<op::v0::Convert>(mask, ov::element::i64);
    }
    const auto i64 = [](int64_t v) {
        return op::v0::Constant::create(ov::element::i64, ov::Shape{}, {v});
    };
    const auto valid = std::make_shared<op::v1::Greater>(mask, i64(0));
    const auto count =
        std::make_shared<op::v0::CumSum>(std::make_shared<op::v0::Convert>(valid, ov::element::i64), i64(1));
    const auto positions =
        std::make_shared<op::v1::Select>(valid, std::make_shared<op::v1::Subtract>(count, i64(1)), i64(-1));
    positions->set_friendly_name("npuw/qsa_sequence_positions");
    return positions;
}

// past_length = key_length - query_length assumes that the current tokens have padding only without past
// tokens. A partial NPUW prefill chunk is left-padded after the past ones, so count the past tokens instead:
//     sum(attention_mask[:, :mask_length - query_length])
ov::Output<ov::Node> past_from_mask(const ov::Output<ov::Node>& attention_mask,
                                    const ov::Output<ov::Node>& query_length,
                                    const ov::element::Type& type) {
    const auto i64 = [](std::vector<int64_t> v) {
        return op::v0::Constant::create(ov::element::i64, ov::Shape{v.size()}, v);
    };
    const auto mask_length = std::make_shared<op::v8::Gather>(std::make_shared<op::v3::ShapeOf>(attention_mask),
                                                              i64({1}),
                                                              i64({0}));
    auto query = query_length;
    if (query.get_element_type() != ov::element::i64) {
        query = std::make_shared<op::v0::Convert>(query, ov::element::i64);
    }
    query = std::make_shared<op::v1::Reshape>(query, i64({1}), false);
    const auto stop = std::make_shared<op::v1::Subtract>(mask_length, query);
    const auto past_mask = std::make_shared<op::v8::Slice>(attention_mask, i64({0}), stop, i64({1}), i64({1}));
    ov::Output<ov::Node> past = std::make_shared<op::v1::ReduceSum>(past_mask, i64({1}), false);
    if (past.get_element_type() != type) {
        past = std::make_shared<op::v0::Convert>(past, type);
    }
    past.get_node()->set_friendly_name("npuw/qsa_past_length");
    return past;
}

// past_length = Maximum(key_length - query_length, 0)
bool is_past_length(const std::shared_ptr<ov::Node>& node, const ov::Output<ov::Node>& key_length) {
    const auto maximum = ov::as_type_ptr<op::v1::Maximum>(node);
    if (!maximum || !is_scalar(maximum->input_value(1), 0)) {
        return false;
    }
    const auto sub = ov::as_type_ptr<op::v1::Subtract>(maximum->get_input_node_shared_ptr(0));
    return sub && sub->input_value(0) == key_length;
}

}  // namespace

bool ov::npuw::MakeSparseAttentionStatic::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::pair<QSAState, std::shared_ptr<op::v0::Parameter>>> past;
    std::vector<std::pair<QSAState, std::shared_ptr<op::v0::Result>>> present;
    std::shared_ptr<op::v0::Parameter> attention_mask;
    for (const auto& param : model->get_parameters()) {
        if (auto state = qsa_state(param->get_output_tensor(0), "input")) {
            past.emplace_back(std::move(*state), param);
        } else if (param->get_output_tensor(0).get_names().count("attention_mask")) {
            attention_mask = param;
        }
    }
    for (const auto& result : model->get_results()) {
        if (auto state = qsa_state(result->get_output_tensor(0), "output")) {
            present.emplace_back(std::move(*state), result);
        }
    }
    if (past.empty()) {
        return false;
    }
    OPENVINO_ASSERT(past.size() == present.size(),
                    "MakeSparseAttentionStatic: ",
                    past.size(),
                    " past QSA states but ",
                    present.size(),
                    " present ones");
    OPENVINO_ASSERT(attention_mask, "MakeSparseAttentionStatic: the model has no attention_mask input");

    // 1. num_blocks -> max_kv_len / r
    std::set<std::shared_ptr<op::v1::ReduceMax>> num_blocks;
    for (const auto& [state, param] : past) {
        if (state.kind == kBlockIndexKey) {
            const auto reduce = find_num_blocks(param);
            OPENVINO_ASSERT(reduce,
                            "MakeSparseAttentionStatic: the number of blocks of block_index_key.",
                            state.index,
                            " is not recognized");
            num_blocks.insert(reduce);
        }
    }
    OPENVINO_ASSERT(!num_blocks.empty(), "MakeSparseAttentionStatic: no block_index_key state");

    std::optional<ov::Output<ov::Node>> key_length;
    std::optional<int64_t> ratio;
    for (const auto& reduce : num_blocks) {
        const auto divide = reduce->get_input_node_shared_ptr(0);
        const auto r = scalar_value(divide->input_value(1));
        OPENVINO_ASSERT(r.has_value() && *r > 0, "MakeSparseAttentionStatic: the key block size is not a constant");
        OPENVINO_ASSERT(!ratio || (*ratio == *r && *key_length == divide->input_value(0)),
                        "MakeSparseAttentionStatic: the QSA layers disagree on key_length or the key block size");
        ratio = r;
        key_length = divide->input_value(0);
    }
    const int64_t max_blocks = static_cast<int64_t>(m_max_kv_len) / *ratio;
    OPENVINO_ASSERT(max_blocks > 0, "MakeSparseAttentionStatic: the KV cache is shorter than one key block");
    for (const auto& reduce : num_blocks) {
        OPENVINO_ASSERT(reduce->get_output_partial_shape(0).rank().get_length() == 0,
                        "MakeSparseAttentionStatic: the number of blocks is expected to be a scalar");
        auto constant = op::v0::Constant::create(reduce->get_output_element_type(0), ov::Shape{}, {max_blocks});
        constant->set_friendly_name(reduce->get_friendly_name() + "/static");
        ov::replace_node(reduce, constant);
    }

    // 2. key slot -> sequence position from attention_mask
    std::vector<std::shared_ptr<ov::Node>> sequence_positions;
    for (const auto& node : model->get_ordered_ops()) {
        if (is_sequence_positions(node, *key_length)) {
            sequence_positions.push_back(node);
        }
    }
    OPENVINO_ASSERT(!sequence_positions.empty(),
                    "MakeSparseAttentionStatic: the sequence positions of SparseAttention are not recognized");
    const auto positions = positions_from_mask(attention_mask->output(0));
    for (const auto& node : sequence_positions) {
        OPENVINO_ASSERT(node->get_output_element_type(0) == ov::element::i64,
                        "MakeSparseAttentionStatic: unexpected sequence positions type ",
                        node->get_output_element_type(0));
        node->output(0).replace(positions);
    }

    // 3. past length from attention_mask, for partially filled prefill chunks
    std::vector<std::shared_ptr<ov::Node>> past_lengths;
    for (const auto& node : model->get_ordered_ops()) {
        if (is_past_length(node, *key_length)) {
            past_lengths.push_back(node);
        }
    }
    OPENVINO_ASSERT(!past_lengths.empty(), "MakeSparseAttentionStatic: the QSA past length is not recognized");
    for (const auto& node : past_lengths) {
        const auto query_length = node->get_input_node_shared_ptr(0)->input_value(1);
        node->output(0).replace(
            past_from_mask(attention_mask->output(0), query_length, node->get_output_element_type(0)));
    }

    // 4. static block_index_key state and lincache-like names
    for (const auto& [state, param] : past) {
        if (state.kind == kBlockIndexKey) {
            auto shape = param->get_partial_shape();
            OPENVINO_ASSERT(shape.rank().is_static() && shape.rank().get_length() == 3,
                            "MakeSparseAttentionStatic: block_index_key.",
                            state.index,
                            " is expected to be rank-3");
            shape[1] = max_blocks;
            param->set_partial_shape(shape);
        }
        ov::op::util::set_name(*param, "cache_params.past." + state.kind + "." + state.index);
    }
    for (const auto& [state, result] : present) {
        ov::op::util::set_name(*result, "cache_params.present." + state.kind + "." + state.index);
    }
    model->validate_nodes_and_infer_types();

    for (const auto& [state, result] : present) {
        if (state.kind == kBlockIndexKey) {
            const auto& shape = result->get_input_partial_shape(0);
            OPENVINO_ASSERT(shape.rank().is_static() && shape.rank().get_length() == 3 && shape[1] == max_blocks,
                            "MakeSparseAttentionStatic: present block_index_key.",
                            state.index,
                            " got shape ",
                            shape,
                            ", expected ",
                            max_blocks,
                            " blocks");
        }
    }

    LOG_INFO("Sparse attention made static: " << past.size() << " QSA states, " << sequence_positions.size()
                                              << " SparseAttention layers, " << past_lengths.size()
                                              << " past lengths, " << max_blocks << " key blocks of " << *ratio
                                              << " tokens");
    return true;
}
