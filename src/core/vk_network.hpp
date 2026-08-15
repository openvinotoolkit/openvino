// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// vk_network: executes a compiled vk_program on the Vulkan runtime. Buffers
// are dispatched to their kernels in program order and the model outputs are
// collected into a map keyed by internal buffer id.

#pragma once

#include "runtime/execution_config.hpp"
#include "vk_common.hpp"
#include "vk_event.hpp"
#include "vk_memory.hpp"
#include "vk_program.hpp"
#include "vk_stream.hpp"
#include "vk_types.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ov::core::vulkan {
namespace cross_platform {

using ov::intel_gpu::ExecutionConfig;

// A single model output: the mnemonic get_memory(bool) matches the legacy
// network::output API shape.
struct network_output {
    network_output() = default;
    network_output(vk_memory_ptr mem, layout lay)
        : _mem(std::move(mem)), _lay(std::move(lay)) {}

    vk_memory_ptr get_memory(bool) const { return _mem; }
    const layout& get_layout() const { return _lay; }

private:
    vk_memory_ptr _mem;
    layout _lay;
};

class vk_network {
public:
    vk_network(vk_engine& engine, const ExecutionConfig& config, const std::shared_ptr<vk_program>& program);
    ~vk_network() = default;

    vk_network(const vk_network&) = delete;
    vk_network& operator=(const vk_network&) = delete;

    // Runs every node in program order (in-order queue) and returns the model
    // outputs keyed by internal buffer id (output_port_to_id values).
    std::map<std::string, network_output> execute(const std::vector<vk_event_ptr>& deps);

    vk_stream& get_stream() const { return *_stream; }
    const ExecutionConfig& get_config() const { return _config; }

    // Buffer overrides: bind a user-provided buffer for a given internal id
    // instead of the program's own buffer (zero-copy where the underlying
    // allocation can be shared).
    void set_input_data(const std::string& id, vk_memory_ptr mem);
    std::vector<vk_event_ptr> set_output_memory(const std::string& id, vk_memory_ptr mem, bool);

private:
    const ExecutionConfig _config;
    vk_stream_ptr _stream;
    std::shared_ptr<vk_program> _program;
    std::map<std::string, vk_memory_ptr> _input_overrides;
    std::map<std::string, vk_memory_ptr> _output_overrides;
};

}  // namespace cross_platform
}  // namespace ov::core::vulkan
