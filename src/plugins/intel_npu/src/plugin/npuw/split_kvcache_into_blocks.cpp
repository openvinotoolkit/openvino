// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "split_kvcache_into_blocks.hpp"

#include <vector>

#include "logging.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {
namespace pass {

namespace {

// Information collected in the scan phase; all fields are pre-computed so the
// transform phase never needs to re-query the parameter name.
struct KVCacheTransformInfo {
    std::shared_ptr<ov::op::v0::Parameter> param;
    std::shared_ptr<ov::op::v0::Concat> concat;
    ov::Output<ov::Node> present_kv_input;
    std::shared_ptr<ov::Node> convert_node;  // nullptr when Parameter→Concat directly
    bool is_key;                             // true = K tensor
    bool is_value;                           // true = V tensor
    int64_t concat_axis;                     // sequence dimension index in the concat
};

// Walk a parameter's consumers to find the Concat it feeds (directly or via Convert).
// Returns {concat, convert_node}; concat is nullptr if the pattern is not found.
std::pair<std::shared_ptr<ov::op::v0::Concat>, std::shared_ptr<ov::Node>> find_concat_for_param(
    const std::shared_ptr<ov::op::v0::Parameter>& param) {
    for (const auto& output : param->outputs()) {
        for (const auto& target : output.get_target_inputs()) {
            auto node = target.get_node()->shared_from_this();
            if (auto concat = ov::as_type_ptr<ov::op::v0::Concat>(node)) {
                return {concat, nullptr};
            }
            if (ov::is_type<ov::op::v0::Convert>(node)) {
                for (const auto& cvt_out : node->outputs()) {
                    for (const auto& cvt_target : cvt_out.get_target_inputs()) {
                        auto next = cvt_target.get_node()->shared_from_this();
                        if (auto concat = ov::as_type_ptr<ov::op::v0::Concat>(next)) {
                            return {concat, node};
                        }
                    }
                }
            }
        }
    }
    return {nullptr, nullptr};
}

// Build a 4D block shape.  seq_dim is the axis holding the sequence (2 or 3);
// all other dims are copied from orig_shape.
ov::Shape make_block_shape(const ov::PartialShape& orig_shape, int64_t seq_dim, size_t seq_size) {
    ov::Shape s(4);
    for (int i = 0; i < 4; ++i) {
        s[i] = (i == seq_dim) ? seq_size : static_cast<size_t>(orig_shape[i].get_length());
    }
    return s;
}

}  // namespace

SplitKVCacheIntoBlocks::SplitKVCacheIntoBlocks(uint32_t block_size, bool v_transposed)
    : m_block_size(block_size),
      m_v_transposed(v_transposed) {}

bool SplitKVCacheIntoBlocks::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool model_changed = false;

    // --- Phase 1: Scan — collect parameters that need to be split ----------
    std::vector<KVCacheTransformInfo> params_to_transform;

    for (const auto& param : model->get_parameters()) {
        const std::string& name = param->get_friendly_name();

        const bool is_key = ov::npuw::util::isPastKeyParamContiguous(name);
        const bool is_value = ov::npuw::util::isPastValueParamContiguous(name);
        if (!is_key && !is_value) {
            continue;  // not a contiguous KV cache parameter
        }

        // Shape must be 4D and fully static before we can proceed.
        const auto& orig_shape = param->get_partial_shape();
        if (orig_shape.rank().is_dynamic() || orig_shape.rank().get_length() != 4) {
            LOG_WARN("SplitKVCacheIntoBlocks: Skipping " << name << " — expected 4D shape, got " << orig_shape);
            continue;
        }
        if (!orig_shape[0].is_static() || !orig_shape[1].is_static() || !orig_shape[2].is_static() ||
            !orig_shape[3].is_static()) {
            LOG_WARN("SplitKVCacheIntoBlocks: Skipping " << name << " — dynamic dimensions not supported");
            continue;
        }

        // Locate the Concat this parameter feeds (directly or via Convert).
        auto [concat, convert_node] = find_concat_for_param(param);
        if (!concat) {
            continue;
        }

        // Locate the "present_kv" input — the non-param input of the Concat.
        ov::Output<ov::Node> present_kv_input;
        bool found = false;
        for (size_t i = 0; i < concat->get_input_size(); ++i) {
            auto src = concat->input(i).get_source_output().get_node_shared_ptr();
            bool from_param = (src == param) || (ov::is_type<ov::op::v0::Convert>(src) &&
                                                 src->input(0).get_source_output().get_node_shared_ptr() == param);
            if (!from_param) {
                present_kv_input = concat->input(i).get_source_output();
                found = true;
                break;
            }
        }
        if (!found) {
            continue;
        }

        // Pre-compute the concat axis (sequence dimension) once.
        const int64_t concat_axis = (is_key || (is_value && !m_v_transposed)) ? 2 : 3;

        params_to_transform.push_back({param, concat, present_kv_input, convert_node, is_key, is_value, concat_axis});
    }

    // --- Phase 2: Transform — replace each collected parameter with blocks --
    for (auto& info : params_to_transform) {
        auto& param = info.param;
        auto& concat = info.concat;

        const auto& orig_shape = param->get_partial_shape();
        const int64_t seq_len = orig_shape[info.concat_axis].get_length();
        const uint32_t num_full_blocks = static_cast<uint32_t>(seq_len) / m_block_size;
        const uint32_t tail_size = static_cast<uint32_t>(seq_len) % m_block_size;
        const uint32_t total_blocks = num_full_blocks + (tail_size > 0 ? 1 : 0);

        LOG_INFO("SplitKVCacheIntoBlocks: Transforming " << param->get_friendly_name() << " shape=" << orig_shape
                                                         << " → " << total_blocks << " blocks (tail=" << tail_size
                                                         << ")");

        // Build block Parameter nodes.
        ov::OutputVector block_outputs;
        std::vector<std::shared_ptr<ov::op::v0::Parameter>> new_params;
        block_outputs.reserve(total_blocks);
        new_params.reserve(total_blocks);

        auto make_block_param = [&](const std::string& suffix, size_t seq_size) {
            auto block_shape = make_block_shape(orig_shape, info.concat_axis, seq_size);
            auto p = std::make_shared<ov::op::v0::Parameter>(param->get_element_type(), block_shape);
            const std::string block_name = param->get_friendly_name() + suffix;
            p->set_friendly_name(block_name);
            p->output(0).set_names({block_name});
            return p;
        };

        for (uint32_t i = 0; i < num_full_blocks; ++i) {
            new_params.push_back(make_block_param("_block_" + std::to_string(i), m_block_size));
            block_outputs.push_back(new_params.back());
        }
        if (tail_size > 0) {
            new_params.push_back(make_block_param("_block_tail", tail_size));
            block_outputs.push_back(new_params.back());
        }

        model->add_parameters(new_params);

        // If the original path had a Convert, insert one after each block param.
        ov::OutputVector inputs_for_concat;
        inputs_for_concat.reserve(total_blocks + 1);

        if (info.convert_node) {
            const auto target_type = info.convert_node->get_output_element_type(0);
            for (const auto& blk : block_outputs) {
                auto cvt = std::make_shared<ov::op::v0::Convert>(blk, target_type);
                cvt->set_friendly_name(blk.get_node()->get_friendly_name() + "_convert");
                inputs_for_concat.push_back(cvt);
            }
        } else {
            inputs_for_concat = block_outputs;
        }
        inputs_for_concat.push_back(info.present_kv_input);

        // Replace the old Concat.
        auto new_concat = std::make_shared<ov::op::v0::Concat>(inputs_for_concat, info.concat_axis);
        new_concat->set_friendly_name(concat->get_friendly_name());

        ov::NodeVector new_nodes{new_concat};
        for (const auto& p : new_params) {
            new_nodes.push_back(p);
        }
        copy_runtime_info({param, concat}, new_nodes);

        concat->output(0).replace(new_concat->output(0));
        model->remove_parameter(param);

        LOG_DEBUG("SplitKVCacheIntoBlocks: new concat shape " << new_concat->get_output_partial_shape(0));

        model_changed = true;
    }

    return model_changed;
}

}  // namespace pass
}  // namespace npuw
}  // namespace ov
