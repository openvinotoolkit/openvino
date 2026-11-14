// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/extension/weight_sharing.hpp"

#include <cstring>
#include <unordered_map>

#include "openvino/core/graph_util.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"
#include "shared_weights.hpp"

namespace ov::frontend::gguf {

namespace {

using CopyConstant = std::function<std::shared_ptr<Node>(const ov::op::v0::Constant&)>;

// Copy of the weight subgraph ending at `output`, node by node with names and runtime info.
// Constants go through `copy_constant` when given; otherwise Constant::clone_with_new_inputs
// shares their data.
Output<Node> copy_weight_subgraph(const Output<Node>& output, const CopyConstant& copy_constant = {}) {
    std::unordered_map<const Node*, std::shared_ptr<Node>> copies;
    for (const auto& node : ov::topological_sort(NodeVector{output.get_node_shared_ptr()})) {
        FRONT_END_GENERAL_CHECK(!ov::is_type<ov::op::v0::Parameter>(node),
                                "[GGUF] a shared weight must not depend on a model input: ",
                                node->get_friendly_name());
        std::shared_ptr<Node> copy;
        const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
        if (constant && copy_constant) {
            copy = copy_constant(*constant);
        } else {
            OutputVector inputs;
            for (const auto& input : node->input_values())
                inputs.push_back(copies.at(input.get_node())->output(input.get_index()));
            copy = node->clone_with_new_inputs(inputs);
        }
        copy->get_rt_info() = node->get_rt_info();
        copy->set_friendly_name(node->get_friendly_name());
        for (size_t i = 0; i < node->get_output_size(); ++i) {
            copy->output(i).get_rt_info() = node->output(i).get_rt_info();
            copy->output(i).set_names(node->output(i).get_names());
        }
        copies[node.get()] = std::move(copy);
    }
    return copies.at(output.get_node())->output(output.get_index());
}

}  // namespace

Output<Node> SharedWeights::get(const GgufWeight& source, const std::function<Output<Node>()>& build) {
    const Key key{source.data.data(), source.data.get_byte_size(), source.qtype, source.shape, source.source_name};
    std::lock_guard<std::mutex> lock(m_mutex);
    auto found = m_weights.find(key);
    if (found == m_weights.end()) {
        auto kept = copy_weight_subgraph(build(), [this](const ov::op::v0::Constant& constant) {
            return make_shared_constant(constant);
        });
        found = m_weights.emplace(key, std::move(kept)).first;
    }
    return copy_weight_subgraph(found->second);
}

ov::AnyMap SharedWeights::get_compile_properties() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return {ov::internal::model_sharing_context(std::make_shared<const ov::weight_sharing::Context>(*m_context))};
}

std::shared_ptr<Node> SharedWeights::make_shared_constant(const ov::op::v0::Constant& constant) {
    const auto size = constant.get_byte_size();
    // An empty weight has nothing to share; keep it on the plugin's ordinary constant path.
    if (size == 0)
        return constant.clone_with_new_inputs({});
    const auto page = static_cast<size_t>(ov::util::get_system_page_size());
    auto storage = std::make_shared<ov::AlignedBuffer>(ov::util::align_size_up(size, page), page);
    std::memcpy(storage->get_ptr(), constant.get_data_ptr(), size);

    // One source per constant, so the constant sits at offset 0 of its own buffer.
    const auto source_id = static_cast<ov::weight_sharing::DataID>(m_sources.size() + 1);
    auto source = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
        storage->get_ptr<char>(),
        storage->size(),
        storage,
        ov::create_base_descriptor(source_id, 0, nullptr));
    auto view = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
        storage->get_ptr<char>(),
        size,
        source,
        ov::create_base_descriptor(source_id, 0, source));
    auto shared = std::make_shared<ov::op::v0::Constant>(constant.get_element_type(), constant.get_shape(), view);
    FRONT_END_GENERAL_CHECK(ov::weight_sharing::set_weight_source(*m_context, source) &&
                                ov::weight_sharing::set_constant(*m_context, *shared),
                            "[GGUF] failed to register a shared weight");
    m_sources.push_back(std::move(source));
    return shared;
}

WeightSharingExtension::WeightSharingExtension() : m_weights(std::make_shared<SharedWeights>()) {}

WeightSharingExtension::~WeightSharingExtension() = default;

ov::AnyMap WeightSharingExtension::get_compile_properties() const {
    return m_weights->get_compile_properties();
}

}  // namespace ov::frontend::gguf
