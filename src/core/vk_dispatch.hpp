// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// vk_dispatch: the single entry point of the standalone core. Routes an
// ir_graph to its executor by device name, replacing the routing the OV
// plugin used to do:
//
//   "CPU"      → native cpu_engine executor (pure C++, no Vulkan required).
//   "GPU[.N]"  → Vulkan runtime (vk_engine → vk_program → vk_network);
//                ".N" picks the Nth matching physical device.
//   "NPU"      → served by the same Vulkan path; fails with a clean
//                "No Vulkan device matching" error until a Vulkan SC-class
//                device exists.
//
// One ir_graph, many executors: the graph format (FB/PB/GGUF/Paddle) and the
// executor choice are orthogonal.

#pragma once

#include "cpu_engine.hpp"
#include "runtime/except.hpp"
#include "vk_device_detector.hpp"
#include "vk_engine_factory.hpp"
#include "vk_ir.hpp"
#include "vk_memory.hpp"
#include "vk_network.hpp"
#include "vk_platform.hpp"
#include "vk_program.hpp"

#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ov::core {
namespace vulkan {
namespace cross_platform {

// Physical devices visible to the core, as routable names ("GPU", "GPU.1",
// "CPU.0", ...). Indexes follow physical enumeration order per class.
inline std::vector<std::string> vk_available_devices() {
    std::vector<std::string> gpu;
    std::vector<std::string> cpu;
    for (const auto& [index, dev] : vk_device_detector().get_available_devices()) {
        auto& list = dev->is_cpu_device() ? cpu : gpu;
        list.push_back(std::string(dev->is_cpu_device() ? "CPU." : "GPU.") + index);
    }
    std::vector<std::string> names = std::move(gpu);
    names.insert(names.end(), cpu.begin(), cpu.end());
    return names;
}

namespace {

size_t dispatch_element_count(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (const size_t d : shape)
        n *= d;
    return n;
}

// Runs |graph| on a freshly created Vulkan engine for |device_name|.
std::map<std::string, std::vector<float>> vk_execute_on_vulkan(const ir_graph& graph,
                                                               const std::map<std::string, std::vector<float>>& inputs,
                                                               const std::string& device_name,
                                                               const vk_platform_config& platform) {
    vk_platform_config cfg = platform;
    cfg.device_name = device_name;
    auto engine = vk_engine::create(cfg);

    ExecutionConfig config;
    vk_program_builder builder(*engine, config);
    auto prog = builder.build(graph);
    vk_network net(*engine, config, prog);

    // Upload inputs into host-visible buffers (zero-copy bindings happen
    // through set_input_data overrides).
    for (const auto& [id, vals] : inputs) {
        auto it = graph.tensor_shapes.find(id);
        OPENVINO_ASSERT(it != graph.tensor_shapes.end(), "[GPU] vk_execute: unknown input ", id);
        const layout lay(it->second, 4);
        auto mem = engine->allocate_memory(lay, allocation_type::usm_host, true);
        void* p = mem->lock();
        std::memcpy(p, vals.data(), vals.size() * sizeof(float));
        mem->unlock();
        net.set_input_data(id, mem);
    }

    // Output buffers are device-local; register one host-visible override per
    // distinct producer output id so results can be locked after execution.
    std::map<std::string, vk_memory_ptr> host_out;
    for (const auto& [port, buf_id] : prog->output_port_to_id) {
        if (host_out.count(buf_id) != 0)
            continue;
        OPENVINO_ASSERT(port < graph.outputs.size(), "[GPU] vk_execute: output port out of range");
        const auto& shape = graph.tensor_shapes.at(graph.outputs.at(port));
        host_out[buf_id] = engine->allocate_memory(layout(shape, 4), allocation_type::usm_host, true);
    }
    for (const auto& [buf_id, mem] : host_out)
        net.set_output_memory(buf_id, mem, false);

    net.execute({});

    std::map<std::string, std::vector<float>> result;
    for (size_t port = 0; port < graph.outputs.size(); ++port) {
        const auto& buf_id = prog->output_port_to_id.at(port);
        auto& mem = host_out.at(buf_id);
        const float* p = static_cast<const float*>(mem->lock());
        result[graph.outputs[port]] =
            std::vector<float>(p, p + dispatch_element_count(graph.tensor_shapes.at(graph.outputs[port])));
        mem->unlock();
    }
    return result;
}

}  // namespace

// Executes |graph| with |inputs| on |device_name| and returns the model
// outputs keyed by their graph output ids.
inline std::map<std::string, std::vector<float>> vk_execute(
    const ir_graph& graph,
    const std::map<std::string, std::vector<float>>& inputs,
    const std::string& device_name = "GPU",
    const vk_platform_config& platform = {}) {
    if (device_name == "CPU")
        return cpu_execute(graph, inputs);
    return vk_execute_on_vulkan(graph, inputs, device_name, platform);
}

}  // namespace cross_platform
}  // namespace vulkan
}  // namespace ov::core
