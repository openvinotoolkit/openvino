// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_node.h"

#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include <algorithm>
#include <cctype>

using namespace cldnn;

namespace cldnn {

namespace {

std::string to_lower_copy(const std::string& s) {
    std::string out;
    out.resize(s.size());
    std::transform(s.begin(), s.end(), out.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return out;
}

bool wildcard_match(const std::string& pattern, const std::string& value) {
    const auto lower_pattern = to_lower_copy(pattern);
    const auto lower_value = to_lower_copy(value);

    size_t p = 0;
    size_t v = 0;
    size_t star = std::string::npos;
    size_t match = 0;

    while (v < lower_value.size()) {
        if (p < lower_pattern.size() && lower_pattern[p] == lower_value[v]) {
            ++p;
            ++v;
        } else if (p < lower_pattern.size() && lower_pattern[p] == '*') {
            star = p++;
            match = v;
        } else if (star != std::string::npos) {
            p = star + 1;
            v = ++match;
        } else {
            return false;
        }
    }

    while (p < lower_pattern.size() && lower_pattern[p] == '*') {
        ++p;
    }

    return p == lower_pattern.size();
}

bool is_pattern_key(const primitive_id& key) {
    return key.find('*') != primitive_id::npos;
}

bool has_primitive_suffix(const primitive_id& key) {
    return key.find(':') != primitive_id::npos;
}

bool is_layer_name_match(const primitive_id& key, const primitive_id& node_id) {
    const auto lower_key = to_lower_copy(key);
    const auto lower_node = to_lower_copy(node_id);

    if (lower_node == lower_key) {
        return true;
    }

    const std::string prefix = lower_key + ":";
    if (lower_node.rfind(prefix, 0) == 0) {
        return true;
    }

    return lower_node.find(lower_key) != std::string::npos;
}

}  // namespace

void graph_initializations::set_outputs(program& p) {
    auto custom_outputs = p.get_config().get_custom_outputs();
    if (!custom_outputs.empty()) {
        for (auto const& output : custom_outputs) {
            OPENVINO_ASSERT(p.has_node(output), "not found custom output node in current cldnn::program: ", output);
            auto o_node = p.get_node_ptr(output);
            o_node->set_output(true);
            p.outputs.push_back(o_node.get());
        }
    } else {
        for (auto& node : p.nodes_map)
            if (node.second->is_endpoint() && !node.second->is_type<data>()) {
                node.second->set_output(true);
                p.outputs.push_back(node.second.get());
            }
    }
}

void graph_initializations::run(program& p) {
    set_outputs(p);

    auto forcing_map = p.get_config().get_force_implementations();
    std::vector<primitive_id> missing_forced_nodes;
    for (auto& kv : forcing_map) {
        bool found_match = false;

        if (p.has_node(kv.first)) {
            p.get_node(kv.first).set_forced_impl_type(kv.second.impl_type);
            found_match = true;
        }

        for (auto& node_kv : p.nodes_map) {
            const auto& node_id = node_kv.first;
            if (node_id == kv.first) {
                continue;
            }

            bool matched = false;
            if (is_pattern_key(kv.first)) {
                matched = wildcard_match(kv.first, node_id);
            } else if (!has_primitive_suffix(kv.first)) {
                matched = is_layer_name_match(kv.first, node_id);
            }

            if (!matched) {
                continue;
            }

            node_kv.second->set_forced_impl_type(kv.second.impl_type);
            found_match = true;
        }

        if (!found_match) {
            missing_forced_nodes.push_back(kv.first);
        }
    }

    if (!missing_forced_nodes.empty() && !p.is_internal_program()) {
        std::string missing_ids;
        for (size_t i = 0; i < missing_forced_nodes.size(); ++i) {
            missing_ids += missing_forced_nodes[i];
            if (i + 1 < missing_forced_nodes.size()) {
                missing_ids += ", ";
            }
        }

        OPENVINO_THROW("[GPU] force_implementations contains primitive ids that are not present in graph: ", missing_ids);
    }

    p.get_processing_order().calc_processing_order(p);
}
}  // namespace cldnn
