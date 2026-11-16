// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unfold_sync_infer_request.hpp"

#include "attn/attn_subgraph.hpp"
#include "compiled_model.hpp"
#include "logging.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "util.hpp"

ov::npuw::UnfoldInferRequest::UnfoldInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model)
    : ov::npuw::IBaseInferRequest(compiled_model) {
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
        LOG_EXEC(m_npuw_model->m_name << " Subgraph[" << i << "] on " << m_npuw_model->submodel_device(i)
                                      << ": create_infer_request() -> ireq@" << m_subrequests[i]._ptr.get()
                                      << (comp_model_desc.replaced_by ? " (from func Subgraph[" +
                                                                            std::to_string(real_idx) + "])"
                                                                      : std::string{}));
        LOG_INFO("DONE");
    }  // for(submodels)

    alloc_quant_gather();

    LOG_INFO("Connecting subrequests...");
    LOG_BLOCK();
    for (const auto& kvp : m_npuw_model->m_submodels_input_to_prev_output) {
        const auto& subm_idx_to = kvp.first.first;
        const auto& port_idx_to = kvp.first.second;
        const auto& subm_idx_from = kvp.second.first;
        const auto& port_idx_from = kvp.second.second;

        LOG_DEBUG("Subgraph[" << subm_idx_from << "]/" << port_idx_from << " --> " << "Subgraph[" << subm_idx_to << "]/"
                              << port_idx_to);

        const auto is_optimized_out = [&](std::size_t idx) {
            const auto& desc = m_npuw_model->m_compiled_submodels[idx];
            return !desc.compiled_model && !desc.replaced_by;
        };
        if (is_optimized_out(subm_idx_from) || is_optimized_out(subm_idx_to)) {
            LOG_DEBUG("Skip: connection involves an optimized-out subgraph");
            continue;
        }

        NPUW_ASSERT(m_subrequests[subm_idx_from]);  // prod request is created
        NPUW_ASSERT(m_subrequests[subm_idx_to]);    // cons request is created
        NPUW_ASSERT(m_subrequests[subm_idx_from]._ptr != m_subrequests[subm_idx_to]._ptr);

        const auto& iport = m_subrequests[subm_idx_to]->get_compiled_model()->inputs()[port_idx_to];
        const auto& oport = m_subrequests[subm_idx_from]->get_compiled_model()->outputs()[port_idx_from];
        const auto& tensor = m_subrequests[subm_idx_from]->get_tensor(oport);
        LOG_DEBUG("Set Subgraph[" << subm_idx_to << "]/" << iport << " to Subgraph[" << subm_idx_from << "]/" << oport);
        m_subrequests[subm_idx_to]->set_tensor(iport, tensor);
    }  // for(map)
    LOG_INFO("Done");

    init_gio();

    for (size_t i = 0; i < m_num_submodels; i++) {
        LOG_VERB("Trying to preemptively set tensors for Subgraph[" << i << "]...");
        LOG_BLOCK();
        auto& comp_model_desc = m_npuw_model->m_compiled_submodels[i];
        if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
            continue;  // Optimized out
        }
        if (comp_model_desc.replaced_by) {
            unpack_closure(i, m_subrequests[i]);
        }
        LOG_VERB("Done");
    }

    if (ov::npuw::util::submit_all_enabled()) {
        m_submit_all = true;
        for (std::size_t i = 0; i < m_num_submodels; i++) {
            if (m_subrequests[i] && !ov::npuw::util::starts_with(m_npuw_model->submodel_device(i), "NPU")) {
                m_submit_all = false;
                break;
            }
        }
        LOG_EXEC(m_npuw_model->m_name << ": OPENVINO_NPUW_SUBMIT_ALL "
                                      << (m_submit_all ? "active: no host barriers between subgraphs"
                                                       : "ignored: not all subgraphs run on NPU")
                                      << ", max in flight=" << ov::npuw::util::submit_all_depth() << " (0 = unlimited)");
    }

    if (ov::npuw::util::exec_list_enabled_for(m_npuw_model->m_name)) {
        create_execution_lists();
    }
}

void ov::npuw::UnfoldInferRequest::create_execution_lists() {
    const auto max_executes = ov::npuw::util::exec_list_max();

    // Check every subgraph first: falling back is only possible before the first append(), since an
    // appended request is owned by its list and can't run on its own anymore.
    std::vector<std::pair<std::size_t, std::shared_ptr<::intel_npu::IExecutionListFactory>>> members;
    try {
        for (std::size_t idx = 0; idx < m_num_submodels; idx++) {
            if (!m_subrequests[idx]) {
                continue;
            }
            const auto context = m_subrequests[idx]->get_compiled_model()->get_context();
            auto factory = std::dynamic_pointer_cast<::intel_npu::IExecutionListFactory>(context._ptr);
            OPENVINO_ASSERT(factory != nullptr, "Subgraph[", idx, "] is not on an NPU supporting execution lists");
            members.emplace_back(idx, std::move(factory));
        }
    } catch (const std::exception& ex) {
        LOG_EXEC(m_npuw_model->m_name << ": OPENVINO_NPUW_EXEC_LIST not used: " << ex.what());
        return;
    }

    // No prepare() here: it would allocate device-visible buffers for every global parameter that is
    // not bound yet (e.g. a full KV cache per attention call). The lists record on their first
    // start_async(), after infer() has bound everything.
    for (const auto& [idx, factory] : members) {
        if (m_exec_lists.empty() || (max_executes > 0 && m_exec_lists.back().executes == max_executes)) {
            // All lists of this request share one command queue, owned by the request
            m_exec_lists.push_back({idx, idx, 0u, factory->create_execution_list(this)});
        }
        m_exec_lists.back().list->append(m_subrequests[idx]._ptr);
        m_exec_lists.back().last = idx;
        m_exec_lists.back().executes++;
    }
    for (const auto& el : m_exec_lists) {
        LOG_EXEC(m_npuw_model->m_name << ": execution list Subgraph[" << el.first << ".." << el.last << "], "
                                      << el.executes << " graph executes (recorded on first infer)");
    }
    LOG_EXEC(m_npuw_model->m_name << ": OPENVINO_NPUW_EXEC_LIST active, " << m_exec_lists.size()
                                  << " execution list(s), max graph executes per list=" << max_executes
                                  << " (0 = unlimited)");
}

bool ov::npuw::UnfoldInferRequest::valid_subrequest(std::size_t idx) const {
    return m_subrequests.at(idx) != nullptr;
}

void ov::npuw::UnfoldInferRequest::infer() {
    exec_trace_begin("UnfoldInferRequest");
    const bool do_async = m_npuw_model->m_cfg.get<::intel_npu::NPUW_FUNCALL_ASYNC>();

    auto prepare = [&](std::size_t idx) {
        if (idx >= m_subrequests.size()) {
            return;
        }
        bind_global_params(idx, m_subrequests[idx]);
        bind_global_results(idx, m_subrequests[idx]);
    };
    // Subgraph index of each request passed to wait_and_clear(), for exec tracing only
    std::vector<std::size_t> previous_idx;
    auto wait_and_clear = [&](RqPtrs& rqs) {
        for (std::size_t i = 0; i < rqs.size(); i++) {
            rqs[i]->wait();
            exec_trace(previous_idx[i], rqs[i], "wait() returned");
        }
        rqs.clear();
        previous_idx.clear();
    };

    if (!m_exec_lists.empty()) {
        // PROTOTYPE: each list is one submission on its own queue; barriers inside it order the executes
        for (auto& el : m_exec_lists) {
            for (std::size_t idx = el.first; idx <= el.last; idx++) {
                if (m_subrequests[idx]) {
                    prepare(idx);
                }
            }
            const std::string tag = " [execution list Subgraph[" + std::to_string(el.first) + ".." +
                                    std::to_string(el.last) + "], " + std::to_string(el.executes) + " graph executes]";
            exec_trace(el.first, m_subrequests[el.first], "start_async()", tag);
            el.list->start_async();
            el.list->wait();
            exec_trace(el.last, m_subrequests[el.last], "wait() returned", tag);
        }
    } else if (m_submit_all) {
        // PROTOTYPE: all subgraphs sit on one in-order NPU queue (NPU_SHARED_COMMON_QUEUE +
        // NPU_RUN_INFERENCES_SEQUENTIALLY) fed by one start thread, so the device orders every
        // producer before its consumers. Submit everything, then wait once in submission order.
        // OPENVINO_NPUW_SUBMIT_ALL_DEPTH optionally caps the number of requests in flight.
        const std::size_t max_in_flight = ov::npuw::util::submit_all_depth();
        RqPtrs submitted_requests;
        std::size_t next_to_wait = 0u;
        auto wait_oldest = [&]() {
            submitted_requests[next_to_wait]->wait();
            exec_trace(previous_idx[next_to_wait], submitted_requests[next_to_wait], "wait() returned");
            next_to_wait++;
        };
        prepare(0);
        for (std::size_t idx = 0; idx < m_num_submodels; idx++) {
            auto& subr = m_subrequests[idx];
            if (!subr) {
                prepare(idx + 1);
                continue;
            }
            if (max_in_flight > 0 && submitted_requests.size() - next_to_wait >= max_in_flight) {
                wait_oldest();
            }
            exec_trace(idx, subr, "start_async()");
            subr->start_async();
            submitted_requests.push_back(subr);
            previous_idx.push_back(idx);
            prepare(idx + 1);
        }
        while (next_to_wait < submitted_requests.size()) {
            wait_oldest();
        }
        previous_idx.clear();
    } else if (do_async) {
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
            exec_trace(idx, subr, "start_async()");
            subr->start_async();
            previous_requests.push_back(subr);
            previous_idx.push_back(idx);
            prepare(idx + 1);
        }
        wait_and_clear(previous_requests);
    } else {
        prepare(0);
        for (std::size_t idx = 0; idx < m_num_submodels; idx++) {
            auto& subr = m_subrequests[idx];
            if (!subr) {
                prepare(idx + 1);
                continue;
            }
            exec_trace(idx, subr, "start_async()");
            subr->start_async();
            prepare(idx + 1);
            subr->wait();
            exec_trace(idx, subr, "wait() returned");
        }
    }  // (async)
    exec_trace_end();
}
