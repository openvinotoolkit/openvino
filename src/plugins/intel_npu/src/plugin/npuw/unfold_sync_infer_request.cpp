// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unfold_sync_infer_request.hpp"

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
        }  // if(replaced_by)

        const auto real_idx = comp_model_desc.replaced_by.value_or(i);
        auto& proto_comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
        // NB: UnfoldInferRequest is _NOT_ fail-safe! Fail means fail here
        m_subrequests[i] = proto_comp_model_desc.compiled_model->create_infer_request();
        m_subrequest_devices[i] = *proto_comp_model_desc.device_it;

        if (m_npuw_model->m_acc_check) {
            LOG_INFO("Create reference subrequest for Subgraph[" << i << "] on " << m_npuw_model->m_ref_device << "...");
            LOG_BLOCK();
            if (m_npuw_model->submodel_device(i) != m_npuw_model->m_ref_device) {
                auto& ref_submodel = m_npuw_model->m_compiled_submodels.at(real(i)).ref_compiled_model;
                ov::SoPtr<ov::IAsyncInferRequest> ref_infer_request = {ref_submodel->create_infer_request(),
                                                                       ref_submodel._so};
                NPUW_ASSERT(ref_infer_request);
                m_ref_subrequests.at(i) = std::move(ref_infer_request);
                LOG_INFO("Done");
            } else {
                LOG_INFO("Skip creation of reference subrequest for Subgraph["
                        << i << "] on reference device: " << m_npuw_model->m_ref_device << ", as actual subrequest ["
                        << i << "] has been already created on "
                        << "it .");
            }
        }

        LOG_INFO("DONE");
    }  // for(submodels)

    alloc_io();

    LOG_INFO("Connecting subrequests...");
    LOG_BLOCK();
    for (const auto& kvp : m_npuw_model->m_submodels_input_to_prev_output) {
        const auto& subm_idx_to = kvp.first.first;
        const auto& port_idx_to = kvp.first.second;
        const auto& subm_idx_from = kvp.second.first;
        const auto& port_idx_from = kvp.second.second;

        LOG_DEBUG("Subgraph[" << subm_idx_from << "]/" << port_idx_from << " --> "
                              << "Subgraph[" << subm_idx_to << "]/" << port_idx_to);
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
    bool accuracy_failover = false;

    auto prepare = [&](std::size_t idx) {
        if (idx >= m_subrequests.size()) {
            return;
        }
        bind_global_params(idx, m_subrequests[idx]);
        bind_global_results(idx, m_subrequests[idx]);
    };
    auto wait_and_clear = [&](std::vector<std::size_t> rqs_ids) {
        for (auto&& r_id : rqs_ids) {
            try_accurate_subwait(r_id, accuracy_failover);
        }
        rqs_ids.clear();
    };

    if (do_async) {
        std::size_t past_repl_id = 0u;
        std::vector<std::size_t> previous_requests;

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
            try_accurate_substart_async(idx);
            previous_requests.push_back(idx);
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
            try_accurate_substart_async(idx);
            prepare(idx + 1);
            try_accurate_subwait(idx, accuracy_failover);
        }
    }  // (async)

    if (accuracy_failover) {
        LOG_ERROR("Refined device distribution:");
        LOG_BLOCK();
        m_npuw_model->log_device_dist(ov::npuw::LogLevel::Error);
    }
}

namespace {
    void set_inputs(const ov::SoPtr<ov::IAsyncInferRequest>& from, ov::SoPtr<ov::IAsyncInferRequest>& to) {
        const auto& from_comp_model = from->get_compiled_model();
        const auto& to_comp_model = to->get_compiled_model();
        for (size_t i = 0; i < from_comp_model->inputs().size(); i++) {
            const auto& itnsr = from->get_tensor(from_comp_model->inputs()[i]);
            to->set_tensor(to_comp_model->inputs()[i], itnsr);
        }
    }

    void copy_results(const ov::SoPtr<ov::IAsyncInferRequest>& from, ov::SoPtr<ov::IAsyncInferRequest>& to) {
        const auto& from_comp_model = from->get_compiled_model();
        const auto& to_comp_model = to->get_compiled_model();
        for (size_t i = 0; i < to_comp_model->outputs().size(); i++) {
            const auto& from_tnsr = from->get_tensor(from_comp_model->outputs()[i]);
            const auto& to_tnsr = to->get_tensor(to_comp_model->outputs()[i]);
            from_tnsr->copy_to(to_tnsr._ptr);
        }
    }

    std::stringstream create_launch_msg(std::size_t idx,  std::size_t real_idx) {
        std::stringstream log_msg_stream;
        log_msg_stream << "Launching subrequest[" << idx << "]" <<
        ((real_idx == idx) ? std::string("...").c_str() :
                             std::string(std::string(", which is actually subrequest[") +
                                std::to_string(real_idx) + "]").c_str());
        return log_msg_stream;
    }
} // anonymous namespace

void ov::npuw::UnfoldInferRequest::try_accurate_substart_async(std::size_t subidx) {
    auto& act_subr = m_subrequests.at(subidx);
    if (!m_npuw_model->m_acc_check) {
        act_subr->start_async();
        return;
    }

    std::stringstream log_msg_stream = create_launch_msg(subidx, subidx);
    log_msg_stream << "...";
    LOG_INFO(log_msg_stream.str());
    LOG_BLOCK();

    if (m_npuw_model->m_compiled_submodels[real(subidx)].switched_to_ref) {
        LOG_INFO("Subrequest was inaccurate somewhere before, launching it on reference device.");

        auto& act_subr = m_subrequests.at(subidx);
        auto& ref_subr = m_ref_subrequests.at(subidx);

        set_inputs(act_subr, ref_subr);
        ref_subr->start_async();
    } else {
        act_subr->start_async();
    }
}

void ov::npuw::UnfoldInferRequest::try_accurate_subwait(std::size_t subidx, bool& accuracy_failover) {
    auto& act_subr = m_subrequests.at(subidx);
    if (!m_npuw_model->m_acc_check) {
        act_subr->wait();
        return;
    }

    LOG_BLOCK();

    if (m_npuw_model->m_compiled_submodels[real(subidx)].switched_to_ref) {
        auto& act_subr = m_subrequests.at(subidx);
        auto& ref_subr = m_ref_subrequests.at(subidx);

        ref_subr->wait();
        copy_results(ref_subr, act_subr);
    } else {
        act_subr->wait();
        ensure_subrequest_is_accurate(subidx, accuracy_failover);
    }
}

void ov::npuw::UnfoldInferRequest::ensure_subrequest_is_accurate(std::size_t idx, bool& accuracy_failover) {
    if (!m_npuw_model->m_acc_check) {
         return;
    }

    LOG_INFO("Check if subrequest[" << idx << "] is accurate...");
    LOG_BLOCK();

    std::size_t real_idx = real(idx);
    OPENVINO_ASSERT(m_npuw_model->m_compiled_submodels[real_idx].switched_to_ref == false);

    if (m_npuw_model->submodel_device(idx) == m_npuw_model->m_ref_device) {
        LOG_INFO("Skipped, subrequest[" << idx << "] is launched on reference device.");
        return;
    }

    accuracy_failover = false;
    auto& actual_subr = m_subrequests.at(idx);
    auto& ref_subr = m_ref_subrequests.at(idx);

    // Setting inputs:
    set_inputs(actual_subr, ref_subr);

    // Running inference:
    ref_subr->infer();

    // Comparing results of actual and reference inferfences:
    LOG_INFO("Compare actual outputs against references:");
    bool tensors_converge = true;
    const auto& actual_comp_model = actual_subr->get_compiled_model();
    const auto& ref_comp_model = ref_subr->get_compiled_model();
    std::vector<bool> converges(actual_comp_model->outputs().size());
    std::vector<double> metrics(actual_comp_model->outputs().size());
    for (size_t i = 0; i < actual_comp_model->outputs().size(); i++) {
        const auto& actual_tensor = actual_subr->get_tensor(actual_comp_model->outputs()[i]);
        const auto& ref_tensor = ref_subr->get_tensor(ref_comp_model->outputs()[i]);
        converges[i] = m_npuw_model->m_acc_check(actual_tensor, ref_tensor, &metrics[i]);
        tensors_converge &= converges[i];
    }
    if (tensors_converge == false) {
        if (ov::npuw::get_log_level() == ov::npuw::LogLevel::Error) {
            // For just log level error print header message:
            LOG_ERROR("Check if subrequest[" << idx << "] is accurate...");
        }
    }
    // Log comparison details:
    for (size_t i = 0; i < actual_comp_model->outputs().size(); i++) {
        if (converges[i]) {
            LOG_INFO(" - " << actual_comp_model->outputs()[i]);
            LOG_BLOCK();
            LOG_INFO(m_npuw_model->m_acc_check_name << " loss: " << metrics[i] <<
                      ", threshold: " << m_npuw_model->m_acc_check_threshold << ".");
            LOG_INFO("PASS");
        } else {
            LOG_ERROR(" - " << actual_comp_model->outputs()[i]);
            LOG_BLOCK();
            LOG_ERROR(m_npuw_model->m_acc_check_name << " loss: " << metrics[i] <<
                      ", threshold: " << m_npuw_model->m_acc_check_threshold << ".");
            LOG_ERROR("FAIL");
        }
    }

    // If comparison fails, copy reference results to original tensors and mark subgraph as
    // switched to reference:
    if (tensors_converge) {
        LOG_INFO("PASS");
    } else {
        LOG_ERROR("FAIL");
        LOG_ERROR("Subrequest[" << idx << "] is inaccurate, failover to reference results.");
        if (idx != real_idx) {
            LOG_ERROR("As subrequest[" << idx << "] is actually " << "subrequest[" << real_idx <<
                       "], all subrequests, corresponding to last, will be further " <<
                       "launched on " << m_npuw_model->m_ref_device << ".'");
        } else if (m_npuw_model->m_compiled_submodels[real_idx].replaced_by) {
            LOG_ERROR("As subrequest[" << real_idx << "] is actually " << "a function, all " <<
                      "subrequests, corresponding to it, will be further launched on " <<
                      m_npuw_model->m_ref_device << ".");
        }

        if (m_npuw_model->m_cfg.get<::intel_npu::NPUW_ACC_DUMP_FAILS>()) {
            // Not here anymore due to optimizations.
            const auto model = m_npuw_model->m_compiled_submodels[real_idx].model;
            const auto model_path = std::string("inaccurate_") + model->get_friendly_name() + std::string(".xml");
            ov::save_model(model, model_path);
            dump_input_tensors(idx, true);
            dump_output_tensors(idx, true);
        }

        // Due to complex memory management logic it is safe to just copy
        // results back to already properly allocated and linked tensors:
        copy_results(ref_subr, actual_subr);
        m_npuw_model->m_compiled_submodels[real_idx].switched_to_ref = true;
        accuracy_failover = true;
    }

    LOG_INFO("Done");
}
