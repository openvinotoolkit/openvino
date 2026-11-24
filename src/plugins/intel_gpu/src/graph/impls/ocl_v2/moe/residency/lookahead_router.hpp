// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <memory>
#include <vector>

#include "intel_gpu/graph/network.hpp"
#include "impls/onednn/fully_connected_onednn.hpp"
#include "../moe_router_fused_opt.hpp"
#include "intel_gpu/primitives/moe_router_fused.hpp"
#include "ocl/ocl_event.hpp"
#include "ocl/ocl_memory.hpp"
#include "ocl/ocl_stream.hpp"
#include "primitive_inst.h"

namespace ov::intel_gpu::ocl::moe {

class LookaheadRouter {
public:
    struct Result {
        cldnn::memory::ptr indices;
        cldnn::memory::ptr weights;
        cldnn::event::ptr ready;
        std::shared_ptr<std::vector<uint32_t>> cpu_indices;
    };

    LookaheadRouter(cldnn::network& owner,
                    const cldnn::primitive_id& hidden,
                                        const cldnn::primitive_id& router,
                                        const cldnn::primitive_id& projection)
                : m_engine(owner.get_engine()),
                    m_stream(owner.get_stream_ptr()),
                    m_projection_instance(owner.get_primitive(projection).get()),
                    m_router_instance(owner.get_primitive(router).get()),
          m_input_layout(owner.get_program()->get_node(hidden).get_output_layout()) {
        validate();
    }

    // Projection and router instances may belong to a separate internal network that shares the compute stream.
    LookaheadRouter(cldnn::network& stream_owner,
                    const cldnn::layout& input_layout,
                    cldnn::primitive_inst* projection,
                    cldnn::primitive_inst* router)
        : m_engine(stream_owner.get_engine()),
          m_stream(stream_owner.get_stream_ptr()),
          m_projection_instance(projection),
          m_router_instance(router),
          m_input_layout(input_layout) {
        validate();
    }

    ~LookaheadRouter() {
        for (const auto& pending : m_pending) {
            try {
                pending.ready->wait();
            } catch (...) {
            }
        }
    }

    Result enqueue(const cldnn::memory::ptr& hidden, const cldnn::event::ptr& ready) {
        OPENVINO_ASSERT(hidden && ready && hidden->count() == m_input_layout.count() &&
                            hidden->get_layout().data_type == m_input_layout.data_type &&
                            !hidden->get_layout().data_padding,
                        "Invalid lookahead hidden input");
        if (!m_projection.enqueue)
            initialize();
        while (!m_pending.empty()) {
            const auto events = cldnn::ocl::utils::get_cl_events({m_pending.front().ready});
            const auto status = events.front().getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
            if (status > CL_COMPLETE)
                break;
            m_pending.pop_front();
            OPENVINO_ASSERT(status == CL_COMPLETE, "Lookahead readback failed");
        }
        auto input = m_engine.reinterpret_buffer(*hidden, m_projection.input_layout);
        const auto projected = m_projection.enqueue(input, m_logits, {ready});
        std::vector<cldnn::memory::ptr> inputs{m_logits};
        for (size_t index = 1; index < m_router_instance->inputs_memory_count(); ++index)
            inputs.push_back(m_router_instance->input_memory_ptr(index));
        const auto routed = m_router(inputs, {m_weights, m_indices}, {projected});
        const auto top_k = m_router_instance->get_typed_desc<cldnn::moe_router_fused>()->_config.top_k;
        auto cpu_indices = std::make_shared<std::vector<uint32_t>>(top_k);
        auto& stream = dynamic_cast<cldnn::ocl::ocl_stream&>(*m_stream);
        auto& queue = stream.get_cl_queue();
        auto wait_list = cldnn::ocl::utils::get_cl_events({routed});
        cl::Event readback;
        try {
            stream.get_usm_helper().enqueue_memcpy(queue, cpu_indices->data(), m_indices->buffer_ptr(),
                                                  top_k * sizeof(uint32_t), false, &wait_list, &readback);
            wait_list = {readback};
            cl::Event complete;
            queue.enqueueBarrierWithWaitList(&wait_list, &complete);
            auto completed = std::make_shared<cldnn::ocl::ocl_event>(complete);
            m_pending.push_back({completed, cpu_indices});
            return {m_indices, m_weights, completed, std::move(cpu_indices)};
        } catch (...) {
            if (readback()) {
                try {
                    readback.wait();
                } catch (...) {
                }
            }
            throw;
        }
    }

private:
    struct Pending {
        cldnn::event::ptr ready;
        std::shared_ptr<std::vector<uint32_t>> cpu_indices;
    };

    void validate() {
        auto shape = m_input_layout.get_partial_shape();
        OPENVINO_ASSERT(shape.rank().is_static() && shape.size() >= 2 && shape[shape.size() - 1].is_static(),
                        "Lookahead router requires a static hidden width");
        for (size_t axis = 0; axis + 1 < shape.size(); ++axis) {
            shape[axis] = 1;
        }
        m_input_layout.set_partial_shape(shape);
        OPENVINO_ASSERT(m_projection_instance->type() == cldnn::fully_connected::type_id() &&
                            m_router_instance->type() == cldnn::moe_router_fused::type_id(),
                        "Lookahead requires the next-layer FC and fused router");
    }

    void initialize() {
        m_projection = cldnn::onednn::make_fc_prediction_entry(*m_projection_instance, &m_input_layout);
        m_router = make_router_prediction_entry(*m_router_instance);
        m_logits = m_engine.allocate_memory(m_projection.output_layout, cldnn::allocation_type::usm_device, false);
        const auto top_k = m_router_instance->get_typed_desc<cldnn::moe_router_fused>()->_config.top_k;
        const ov::PartialShape topk_shape{1, static_cast<int64_t>(top_k)};
        m_weights = m_engine.allocate_memory(cldnn::layout{topk_shape, m_projection.output_layout.data_type, cldnn::format::bfyx},
                                             cldnn::allocation_type::usm_device, false);
        m_indices = m_engine.allocate_memory(cldnn::layout{topk_shape, cldnn::data_types::i32, cldnn::format::bfyx},
                                             cldnn::allocation_type::usm_device, false);
    }

    cldnn::engine& m_engine;
    cldnn::stream::ptr m_stream;
    cldnn::primitive_inst* m_projection_instance;
    cldnn::primitive_inst* m_router_instance;
    cldnn::layout m_input_layout;
    cldnn::onednn::FullyConnectedPredictionEntry m_projection;
    RouterPredictionEntry m_router;
    cldnn::memory::ptr m_logits;
    cldnn::memory::ptr m_weights;
    cldnn::memory::ptr m_indices;
    std::deque<Pending> m_pending;
};

}  // namespace ov::intel_gpu::ocl::moe