// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <set>

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/moe_router_fused.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "primitive_inst.h"

namespace ov::intel_gpu::ocl::moe {

class LookaheadRouter {
public:
    struct Result {
        cldnn::memory::ptr indices;
        cldnn::memory::ptr weights;
        cldnn::event::ptr ready;
    };

    LookaheadRouter(cldnn::network& owner,
                    const cldnn::primitive_id& hidden,
                    const cldnn::primitive_id& router)
        : m_hidden(hidden),
          m_router(router),
          m_input_layout(owner.get_program()->get_node(hidden).get_output_layout()) {
        auto shape = m_input_layout.get_partial_shape();
        OPENVINO_ASSERT(shape.rank().is_static() && shape.size() >= 2 && shape[shape.size() - 1].is_static(),
                        "Lookahead router requires a static hidden width");
        for (size_t axis = 0; axis + 1 < shape.size(); ++axis) {
            shape[axis] = 1;
        }
        m_input_layout.set_partial_shape(shape);
        cldnn::topology topology;
        std::set<cldnn::primitive_id> visited;
        std::function<void(const cldnn::program_node&)> append = [&](const cldnn::program_node& node) {
            if (!visited.insert(node.id()).second) {
                return;
            }
            if (node.id() == m_hidden) {
                topology.add(cldnn::input_layout(m_hidden, m_input_layout));
                return;
            }
            if (node.is_type<cldnn::data>()) {
                topology.add(cldnn::data(node.id(), owner.get_primitive(node.id())->output_memory_ptr()));
                return;
            }
            OPENVINO_ASSERT(!node.get_dependencies().empty() && !node.has_fused_primitives(),
                            "Unsupported lookahead router dependency: ", node.id());
            for (const auto& dependency : node.get_dependencies()) {
                append(*dependency.first);
            }
            auto primitive = node.get_primitive()->clone();
            const auto& dependencies = node.get_dependencies();
            for (size_t index = 0; index < dependencies.size(); ++index) {
                primitive->get_dependency(index) = cldnn::input_info(dependencies[index].first->id(),
                                                                    dependencies[index].second);
            }
            topology.add_primitive(std::move(primitive));
        };
        const auto& router_node = owner.get_program()->get_node(router);
        OPENVINO_ASSERT(router_node.is_type<cldnn::moe_router_fused>(), "Lookahead entry must be a fused router");
        append(router_node);
        OPENVINO_ASSERT(visited.count(hidden), "Lookahead router does not depend on the supplied hidden input");
        auto config = owner.get_config().clone();
        config.set_property(ov::intel_gpu::moe_lookahead_prefetch(false),
                            ov::intel_gpu::offload_ratio(0),
                            ov::intel_gpu::custom_outputs(std::vector<std::string>{router}));
        auto program = cldnn::program::build_program(owner.get_engine(), topology, config, true);
        m_network = cldnn::network::allocate_network(owner.get_stream_ptr(), std::move(program), true);
    }

    Result enqueue(const cldnn::memory::ptr& hidden, const cldnn::event::ptr& ready) {
        OPENVINO_ASSERT(hidden && ready && hidden->count() == m_input_layout.count() &&
                            hidden->get_layout().data_type == m_input_layout.data_type,
                        "Invalid lookahead hidden input");
        auto input = m_network->get_engine().reinterpret_buffer(*hidden, m_input_layout);
        m_network->set_input_data(m_hidden, std::move(input));
        m_network->execute({ready});
        auto router = m_network->get_primitive(m_router);
        return {router->output_memory_ptr(1), router->output_memory_ptr(0),
                m_network->get_stream().enqueue_marker({}, true)};
    }

private:
    cldnn::primitive_id m_hidden;
    cldnn::primitive_id m_router;
    cldnn::layout m_input_layout;
    cldnn::network::ptr m_network;
};

}  // namespace ov::intel_gpu::ocl::moe