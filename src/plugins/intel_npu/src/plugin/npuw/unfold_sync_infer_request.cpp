// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unfold_sync_infer_request.hpp"

#include "attn/attn_subgraph.hpp"
#include "compiled_model.hpp"
#include "logging.hpp"
#include "openvino/core/parallel.hpp"

ov::npuw::UnfoldInferRequest::UnfoldInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model)
    : ov::npuw::IBaseInferRequest(compiled_model) {

    if (m_npuw_model->m_using_pipeline_model) {
        m_pipeline_request = m_npuw_model->m_compiled_pipeline_model->create_infer_request();
        m_pipeline_request_device = "NPU";

        auto& inputs{m_npuw_model->m_compiled_pipeline_model->inputs()};

        auto bind_all_shared_ports = [&](const std::string& port_name) {
            auto& kv_tiles{m_npuw_model->m_pipeline_global_parameters[port_name]};

            if (kv_tiles.size()) {
                auto& ref_port{inputs[kv_tiles.front()]};
                auto tensor{allocOut(ref_port, "NPU")};

                for (auto& idx : kv_tiles) {
                    auto& port{inputs[idx]};
                    m_pipeline_request->set_tensor(port, tensor);
                }
            }
        };

        auto& pipeline_connected_inputs{m_npuw_model->m_pipeline_connected_inputs};

        for (auto& in : pipeline_connected_inputs) {
            bind_all_shared_ports(in.first);
        }

        const auto& outputs{m_npuw_model->outputs()};
        size_t out_idx{};

        // Bind and allocate outputs early
        for (auto& out : outputs) {
            auto& tensor{get_tensor(out)};

            auto subm_out_idx{out_idx};

            auto pipeline_out_port_idx{m_npuw_model->m_pipeline_global_outputs.find(static_cast<uint32_t>(out_idx))};
            if (pipeline_out_port_idx != std::end(m_npuw_model->m_pipeline_global_outputs)) {
                subm_out_idx = pipeline_out_port_idx->second;
            }

            const auto& s_port = m_pipeline_request->get_outputs()[subm_out_idx];
            m_pipeline_request->set_tensor(s_port, tensor);
            out_idx++;
        }
    }
    else {
        // Create infer requests
        // Preallocate funcall tensors & substitute function call requests
        for (std::size_t i = 0; i < m_num_submodels; i++) {
            LOG_INFO("Creating infer request for Subgraph[" << i << "]...");
            LOG_BLOCK();
            auto& comp_model_desc = m_npuw_model->m_compiled_submodels[i];

            if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
                // no model & no funcall - optimized out, do nothing
                LOG_INFO("OPTIMIZED OUT");
                continue;
            }

            if (comp_model_desc.replaced_by) {
                // Pre-allocate output tensors for this function call
                const auto real_idx = comp_model_desc.replaced_by.value();
                auto& proto_comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
                if (proto_comp_model_desc.spatial) {
                    NPUW_ASSERT(false && "Spatial is not supported in unfold");
                }
                if (ov::npuw::attn::get_compiled_dynamic(proto_comp_model_desc.pipeline.context) != nullptr) {
                    NPUW_ASSERT(false && "Dynamic is not supported in unfold");
                }
            }  // if(replaced_by)

            const auto real_idx = comp_model_desc.replaced_by.value_or(i);
            auto& proto_comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
            // NB: UnfoldInferRequest is _NOT_ fail-safe! Fail means fail here
           
            m_subrequests[i] = proto_comp_model_desc.compiled_model->create_infer_request();
            
            LOG_INFO("DONE");
        }  // for(submodels)
    
    
        

        LOG_INFO("Connecting subrequests...");
        LOG_BLOCK();
        for (const auto& kvp : m_npuw_model->m_submodels_input_to_prev_output) {
            const auto& subm_idx_to = kvp.first.first;
            const auto& port_idx_to = kvp.first.second;
            const auto& subm_idx_from = kvp.second.first;
            const auto& port_idx_from = kvp.second.second;

            LOG_DEBUG("Subgraph[" << subm_idx_from << "]/" << port_idx_from << " --> " << "Subgraph[" << subm_idx_to
                                  << "]/" << port_idx_to);
            NPUW_ASSERT(m_subrequests[subm_idx_from]);  // prod request is created
            NPUW_ASSERT(m_subrequests[subm_idx_to]);    // cons request is created
            NPUW_ASSERT(m_subrequests[subm_idx_from]._ptr != m_subrequests[subm_idx_to]._ptr);

            const auto& iport = m_subrequests[subm_idx_to]->get_compiled_model()->inputs()[port_idx_to];
            const auto& oport = m_subrequests[subm_idx_from]->get_compiled_model()->outputs()[port_idx_from];
            const auto& tensor = m_subrequests[subm_idx_from]->get_tensor(oport);
            LOG_DEBUG("Set Subgraph[" << subm_idx_to << "]/" << iport << " to Subgraph[" << subm_idx_from << "]/"
                                      << oport);
            m_subrequests[subm_idx_to]->set_tensor(iport, tensor);
        }  // for(map)

        LOG_INFO("Done");        
    }

    alloc_quant_gather();
    init_gio();

    for (size_t i = 0; i < m_num_submodels; i++) {
        LOG_VERB("Trying to preemptively set tensors for Subgraph[" << i << "]...");
        LOG_BLOCK();
        auto& comp_model_desc = m_npuw_model->m_compiled_submodels[i];
        if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
            continue;  // Optimized out
        }
        if (comp_model_desc.replaced_by) {
            if (!m_npuw_model->m_using_pipeline_model) {
                unpack_closure(i, m_subrequests[i]);
            } else {
                unpack_closure(i, m_pipeline_request);
            }
        }
        LOG_VERB("Done");
    }
}

bool ov::npuw::UnfoldInferRequest::valid_subrequest(std::size_t idx) const {
    return m_subrequests.at(idx) != nullptr;
}

void ov::npuw::UnfoldInferRequest::infer() {
    const bool do_async = m_npuw_model->m_cfg.get<::intel_npu::NPUW_FUNCALL_ASYNC>();

    auto prepare = [&](std::size_t idx) {
        if (idx >= m_subrequests.size()) {
            return;
        }

        if (!m_npuw_model->m_using_pipeline_model) {
            bind_global_params(idx, m_subrequests[idx]);
            bind_global_results(idx, m_subrequests[idx]);
        } else {
            bind_global_params(idx, m_pipeline_request);
            bind_global_results(idx, m_pipeline_request);
        }
    };
    auto wait_and_clear = [](RqPtrs& rqs) {
        for (auto&& r : rqs) {
            r->wait();
        }
        rqs.clear();
    };

    if (do_async) {
        if (!m_npuw_model->m_using_pipeline_model) {
            std::size_t past_repl_id = 0u;
            RqPtrs previous_requests;

            prepare(0);
            for (std::size_t idx = 0; idx < m_num_submodels; idx++) {
                auto& subr = m_subrequests[idx];
                if (!subr) {
                    prepare(idx + 1);
                    continue;
                }
                auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
                const auto this_repl_id = comp_model_desc.replaced_by.value_or(idx);
                if (this_repl_id != past_repl_id) {
                    // For non-repeating blocks, the above value_or returns idx
                    // For repeating blocks, it returns the function group id
                    // If either is not equal to the past_repl_id, make a barrier here
                    wait_and_clear(previous_requests);
                    past_repl_id = this_repl_id;
                }
                subr->start_async();
                previous_requests.push_back(subr);
                prepare(idx + 1);
            }
            wait_and_clear(previous_requests);
        } else {
            for (std::size_t idx = 0; idx < m_num_submodels; idx++) {
                prepare(idx);
            }
            m_pipeline_request->infer();
        }
    } else {
        prepare(0);
        for (std::size_t idx = 0; idx < m_num_submodels; idx++) {
            auto& subr = m_subrequests[idx];
            if (!subr) {
                prepare(idx + 1);
                continue;
            }
            subr->start_async();
            prepare(idx + 1);
            subr->wait();
        }
    }  // (async)
}
