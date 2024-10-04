// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "primitive_inst.h"

namespace cldnn {

#ifdef GPU_DEBUG_CONFIG

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

    const debug_configuration* debug_config = cldnn ::debug_configuration ::get_instance();
};

class NetworkDebugHelper {
public:
    NetworkDebugHelper(const network& net);
    ~NetworkDebugHelper();

private:
    void dump_memory_pool(std::string dump_path, int64_t curr_iter) const;
    const network& m_network;
    const size_t m_iter;

    const debug_configuration* debug_config = cldnn ::debug_configuration ::get_instance();
};

#define NETWORK_DEBUG(net) NetworkDebugHelper __network_debug_helper(net)
#define NODE_DEBUG(inst) NodeDebugHelper __node_debug_helper(inst)

#else

#define NETWORK_DEBUG(...)
#define NODE_DEBUG(...)

#endif  // GPU_DEBUG_CONFIG

}  // namespace cldnn
