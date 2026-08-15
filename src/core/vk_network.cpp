// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vk_network.hpp"

#include "runtime/except.hpp"

namespace ov::core::vulkan {
namespace cross_platform {

vk_network::vk_network(vk_engine& engine, const ExecutionConfig& config, const std::shared_ptr<vk_program>& program)
    : _config(config)
    , _stream(engine.create_stream(config))
    , _program(program) {
    OPENVINO_ASSERT(program != nullptr, "[GPU] vk_network: null program");
}

void vk_network::set_input_data(const std::string& id, vk_memory_ptr mem) {
    _input_overrides[id] = mem;
}

std::vector<vk_event_ptr> vk_network::set_output_memory(const std::string& id, vk_memory_ptr mem, bool) {
    _output_overrides[id] = mem;
    return {};
}

std::map<std::string, network_output> vk_network::execute(const std::vector<vk_event_ptr>&) {
    for (const auto& node : _program->nodes) {
        kernel_arguments_desc desc;
        desc.workGroups = node->wgs;
        desc.scalars = node->scalars;

        kernel_arguments_data data;
        data.inputs.reserve(node->inputs.size());
        for (size_t i = 0; i < node->inputs.size(); ++i) {
            auto it = _input_overrides.find(node->input_ids[i]);
            data.inputs.push_back(it != _input_overrides.end() ? it->second : node->inputs[i]);
        }
        data.outputs.reserve(node->outputs.size());
        for (size_t i = 0; i < node->outputs.size(); ++i) {
            auto mem = node->outputs[i];
            if (!node->output_id.empty() && i == 0) {
                auto ov = _output_overrides.find(node->output_id);
                if (ov != _output_overrides.end())
                    mem = ov->second;
            }
            data.outputs.push_back(mem);
        }
        data.weights = node->weights;
        data.bias = node->bias;

        _stream->enqueue_kernel(*node->k, desc, data);
    }

    std::map<std::string, network_output> outputs;
    for (const auto& [port, id] : _program->output_port_to_id) {
        (void)port;
        auto mem_it = _program->memories.find(id);
        OPENVINO_ASSERT(mem_it != _program->memories.end(), "[GPU] vk_network: output buffer ", id, " not found");
        // The overridden buffer carries the user's layout (may differ from the
        // program buffer only in shape metadata).
        auto out_it = _output_overrides.find(id);
        const auto& mem = out_it != _output_overrides.end() ? out_it->second : mem_it->second;
        outputs[id] = network_output(mem, mem->get_layout());
    }
    return outputs;
}

}  // namespace cross_platform
}  // namespace ov::core::vulkan
