// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unfold_sync_infer_request.hpp"

#include <atomic>
#include <chrono>
#include <memory>

#include "attn/attn_subgraph.hpp"
#include "compiled_model.hpp"
#include "logging.hpp"
#include "openvino/core/parallel.hpp"

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
}

bool ov::npuw::UnfoldInferRequest::valid_subrequest(std::size_t idx) const {
    return m_subrequests.at(idx) != nullptr;
}

void ov::npuw::UnfoldInferRequest::infer() {
    const bool do_async = m_npuw_model->m_cfg.get<::intel_npu::NPUW_FUNCALL_ASYNC>();
    // In the async mode subrequests of one group are in flight at once, so their
    // individual intervals genuinely overlap and no residual row is meaningful
    m_perf.execution.children_overlap = do_async;

    auto prepare = [&](std::size_t idx) {
        if (idx >= m_subrequests.size()) {
            return;
        }
        bind_global_params(idx, m_subrequests[idx]);
        bind_global_results(idx, m_subrequests[idx]);
    };

    // Note: in the async mode subrequests of one group run overlapped, so their individual
    // times overlap as well
    using Clock = std::chrono::steady_clock;
    using Finish = std::atomic<Clock::duration::rep>;
    // A submitted subrequest pending its wait(): request + submodel id + submission time
    struct Pending {
        RqPtr rq;
        std::size_t idx;
        ov::npuw::perf::sample timer;
        std::shared_ptr<Finish> finish;
    };
    auto wait_and_clear = [&](std::vector<Pending>& pending) {
        try {
            for (auto&& p : pending) {
                p.rq->wait();
                if (p.finish) {
                    p.rq->set_callback({});
                    const auto finish = p.finish->load(std::memory_order_relaxed);
                    p.timer.commit_at(finish == 0 ? Clock::now() : Clock::time_point(Clock::duration(finish)));
                }
                perf_record_unfold(p.idx, p.timer);
            }
        } catch (...) {
            for (auto&& p : pending) {
                if (p.finish) {
                    try {
                        p.rq->wait();
                    } catch (...) {
                    }
                    p.rq->set_callback({});
                }
            }
            throw;
        }
        pending.clear();
    };

    if (do_async) {
        std::size_t past_repl_id = 0u;
        std::vector<Pending> previous_requests;

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
            ov::npuw::perf::sample timer(perf_device_handle(idx));
            std::shared_ptr<Finish> finish;
            if (m_perf.execution.enabled()) {
                finish = std::make_shared<Finish>(0);
                subr->set_callback([finish](std::exception_ptr) {
                    finish->store(Clock::now().time_since_epoch().count(), std::memory_order_relaxed);
                });
            }
            try {
                subr->start_async();
            } catch (...) {
                if (finish) {
                    subr->set_callback({});
                }
                throw;
            }
            previous_requests.push_back(Pending{subr, idx, std::move(timer), std::move(finish)});
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
            ov::npuw::perf::sample timer(perf_device_handle(idx));
            subr->start_async();
            prepare(idx + 1);
            subr->wait();
            perf_record_unfold(idx, timer);
        }
    }  // (async)
}
