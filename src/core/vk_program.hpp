// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// vk_program: lowers a ir_graph into a flat list of native SPIR-V kernel
// invocations plus pre-allocated buffers. There is no kernel_selector, no
// OpenCL C source and no runtime shader compiler involved. Every op is bound
// to a kernel from the builtin native store (see vk_kernel_builder) and all
// tensor/setup parameters travel as push constants.
//
// The IR (it is what the runtime core consumes) is produced on the plugin
// side by VkModelConverter: this file has no openvino core graph dependency.

#pragma once

#include "runtime/execution_config.hpp"
#include "vk_engine.hpp"
#include "vk_ir.hpp"
#include "vk_kernel.hpp"
#include "vk_kernel_builder.hpp"
#include "vk_memory.hpp"
#include "vk_stream.hpp"
#include "vk_types.hpp"

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ov::core {
namespace vulkan {
namespace cross_platform {

using ov::intel_gpu::ExecutionConfig;

struct vk_program_node {
    std::string id;  // unique internal id (used by the network for I/O binding)
    vk_kernel_ptr k;
    std::vector<std::string> input_ids;  // canonical buffer ids (for I/O overrides)
    std::string output_id;               // canonical buffer id of the produced output
    std::vector<vk_memory_ptr> inputs;   // bound first (shader bindings 0..N-1)
    std::vector<vk_memory_ptr> outputs;  // bound next
    vk_memory_ptr weights;               // bound after outputs (conv2d only)
    vk_memory_ptr bias;                  // bound after weights (conv2d only)
    scalars_desc scalars;                // written into push constants in order
    work_group_sizes wgs;
};

struct vk_program {
    // Execution order (topologically sorted ops).
    std::vector<std::shared_ptr<vk_program_node>> nodes;
    // All buffers by internal id.
    std::map<std::string, vk_memory_ptr> memories;
    // Model input port index -> internal buffer id.
    std::map<size_t, std::string> input_port_to_id;
    // Model output port index -> internal buffer id.
    std::map<size_t, std::string> output_port_to_id;
    // Internal buffer id -> produced node (for set_input_data/set_output_memory plumbing).
    std::map<std::string, std::shared_ptr<vk_program_node>> producer;

    vk_memory_ptr get_memory(const std::string& id) const {
        auto it = memories.find(id);
        OPENVINO_ASSERT(it != memories.end(), "[GPU] vk_program: unknown buffer id ", id);
        return it->second;
    }
};

class vk_program_builder {
public:
    explicit vk_program_builder(vk_engine& engine, const ExecutionConfig& config);

    // Throws ov::Exception for unsupported IR.
    std::shared_ptr<vk_program> build(const ir_graph& graph);

    vk_program_builder(const vk_program_builder&) = delete;
    vk_program_builder& operator=(const vk_program_builder&) = delete;

private:
    vk_engine& _engine;
    vk_stream_ptr _stream;
    std::shared_ptr<vk_kernel_builder> _kern_builder;
    // op type -> kernel (pipelines are shared between nodes of the same type)
    std::map<std::string, vk_kernel_ptr> _kernels;

    vk_memory_ptr make_memory(const std::vector<size_t>& shape, const std::string& id, bool host_visible);
    vk_memory_ptr make_memory_bytes(size_t bytes, const std::string& id, bool host_visible);

    // Returns (or lazily builds and caches) the pipeline for the given native
    // kernel id. Throws ov::Exception for unknown ids.
    vk_kernel_ptr get_build_kernel(std::string_view kernel_id);
};

}  // namespace cross_platform
}  // namespace vulkan
}  // namespace ov
