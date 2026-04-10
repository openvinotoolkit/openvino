// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/kernel.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "primitive_inst.h"

#include <mutex>
#include <unordered_map>

namespace cldnn {

#ifdef GPU_DEBUG_CONFIG

/// @brief Enqueues named no-op OpenCL kernels at network execution boundaries
/// so that tools like CLIntercept can identify network start/finish in traces.
/// Kernel names follow the pattern: network_marker_{start|finish}_p{prog_id}_n{net_id}
class NetworkMarkerHelper {
public:
    /// @brief Enqueue a start marker kernel for the given network.
    static void enqueue_start_marker(network& net);

    /// @brief Enqueue a finish marker kernel for the given network.
    static void enqueue_finish_marker(network& net);

private:
    static kernel::ptr get_or_compile_marker(network& net, const std::string& kernel_name);

    static std::mutex _mutex;
    static std::unordered_map<std::string, kernel::ptr> _compiled_kernels;
};

class NodeDebugHelper {
public:
    NodeDebugHelper(const primitive_inst& inst);
    ~NodeDebugHelper();

private:
    std::string get_iteration_prefix() {
        if (m_iter < 0)
            return std::string("");
        return std::to_string(m_iter) + "_";
    }

    std::string get_file_prefix() {
        auto prog_id = ((m_program != nullptr) ? m_program->get_id() : 0);
        auto net_id = m_network.get_id();

        return "program" + std::to_string(prog_id) + "_network" + std::to_string(net_id) + "_" + get_iteration_prefix() + m_inst.id();
    }


    const primitive_inst& m_inst;
    stream& m_stream;
    const network& m_network;
    const program* m_program;
    const size_t m_iter;
};

class NetworkDebugHelper {
public:
    NetworkDebugHelper(network& net);
    ~NetworkDebugHelper();

private:
    void dump_memory_pool(std::string dump_path, int64_t curr_iter) const;
    network& m_network;
    const size_t m_iter;
};

#define NETWORK_DEBUG(net) NetworkDebugHelper __network_debug_helper(net)
#define NODE_DEBUG(inst) NodeDebugHelper __node_debug_helper(inst)

#else

#define NETWORK_DEBUG(...)
#define NODE_DEBUG(...)

#endif  // GPU_DEBUG_CONFIG

}  // namespace cldnn
