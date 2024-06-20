// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base_sync_infer_request.hpp"

#include "compiled_model.hpp"
#include "intel_npu/al/config/npuw.hpp"
#include "logging.hpp"
#include "util.hpp"

ov::npuw::IBaseInferRequest::IBaseInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_npuw_model(compiled_model),
      m_num_submodels(m_npuw_model->m_compiled_submodels.size()) {
    m_subrequests.resize(m_num_submodels, {});
    m_subrequest_devices.resize(m_num_submodels, {});
    m_completion_cbs.resize(m_num_submodels, {});
    if (m_npuw_model->m_acc_check) {
        m_ref_subrequests.resize(m_num_submodels);
    }
}

ov::npuw::IBaseInferRequest::RqPtrs ov::npuw::IBaseInferRequest::create_infer_requests(std::size_t id,
                                                                                       std::size_t nireq,
                                                                                       bool* recompiled) {
    NPUW_ASSERT(nireq > 0);
    RqPtrs rqs;
    rqs.reserve(nireq);

    // See explanation in the class definition
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[id];
    NPUW_ASSERT(comp_model_desc.replaced_by.value_or(id) == id);

    bool successful = false;
    bool can_try_again = true;

    // Altering iterators here!! Contracts should be changed!
    while (!successful && can_try_again) {
        bool should_recompile = false;
        try {
            // FIXME: As the model may recompile, reference
            // shouldn't be lifted from the loop
            auto& comp_model = comp_model_desc.compiled_model;
            rqs.clear();
            for (std::size_t i = 0u; i < nireq; i++) {
                rqs.emplace_back(comp_model->create_infer_request(), comp_model._so);
            }
            successful = true;
        } catch (const std::exception& ex) {
            LOG_WARN("Subgraph [" << id << "] - Failed to create infer request:" << std::endl << ex.what());
            should_recompile = true;
        } catch (...) {
            LOG_WARN("Subgraph [" << id << "] - Failed to create infer request: REASON UNKNOWN");
            should_recompile = true;
        }
        if (should_recompile) {
            LOG_INFO("- Trying next device...");
            comp_model_desc.device_it++;
            can_try_again = m_npuw_model->compile_for_success(id);
            if (can_try_again) {
                if (recompiled)
                    *recompiled = true;
                // Probably shouldn't be called all the time, but only if
                // I/O submodel is affected
                m_npuw_model->reset_io();
            }
        }
    }  // while(!new_ireq && can_try_again)
    if (!successful) {
        OPENVINO_THROW("NPUW: Fatal - couldn't create infer request for Subgraph[", id, "]");
    }
    NPUW_ASSERT(rqs.size() == nireq);

    // TODO: Support creation and return of multiple infer requests
    if (m_npuw_model->m_acc_check && m_ref_subrequests.at(id) == nullptr) {
        if (nireq > 1) {
            OPENVINO_THROW("NPUW: TEMPORARY LIMITATION: Couldn't create reference infer "
                           "requests if 'nireq' is set to > 1!");
        }
        LOG_INFO("Create reference subrequest for submodel [" << id << "] on " << m_npuw_model->m_ref_device << "...");
        LOG_BLOCK();
        if (m_npuw_model->submodel_device(id) != m_npuw_model->m_ref_device) {
            auto& ref_submodel = m_npuw_model->m_compiled_submodels.at(id).ref_compiled_model;
            ov::SoPtr<ov::IAsyncInferRequest> ref_infer_request = {ref_submodel->create_infer_request(),
                                                                   ref_submodel._so};
            NPUW_ASSERT(ref_infer_request);
            m_ref_subrequests.at(id) = std::move(ref_infer_request);
            LOG_INFO("Done");
        } else {
            LOG_INFO("Skip creation of reference subrequest for submodule["
                     << id << "] on reference device: " << m_npuw_model->m_ref_device << ", as actual subrequest ["
                     << id << "] has been already created on "
                     << "it .");
        }
    }

    return rqs;
}

void ov::npuw::IBaseInferRequest::ensure_subrequest_is_accurate(std::size_t idx, bool& failover) {
    LOG_INFO("Check if subrequest[" << idx << "] is accurate...");
    LOG_BLOCK();
    failover = false;
    if (m_ref_subrequests.at(idx) != nullptr && m_subrequests.at(idx)._ptr != m_ref_subrequests.at(idx)._ptr) {
        NPUW_ASSERT(m_npuw_model->m_compiled_submodels.at(idx).switched_to_ref == false);
        NPUW_ASSERT(m_npuw_model->m_compiled_submodels.at(idx).replaced_by.value_or(idx) == idx);

        const auto& ref_comp_model = m_ref_subrequests.at(idx)->get_compiled_model();
        const auto& actual_comp_model = m_subrequests.at(idx)->get_compiled_model();
        NPUW_ASSERT(actual_comp_model->inputs().size() == ref_comp_model->inputs().size());
        // Setting inputs:
        for (size_t i = 0; i < actual_comp_model->inputs().size(); i++) {
            const auto& itensor = m_subrequests.at(idx)->get_tensor(actual_comp_model->inputs()[i]);
            m_ref_subrequests.at(idx)->set_tensor(ref_comp_model->inputs()[i], itensor);
        }
        m_ref_subrequests.at(idx)->infer();

        LOG_INFO("Compare actual outputs against references:");
        bool tensors_converge = true;
        for (size_t i = 0; i < actual_comp_model->outputs().size(); i++) {
            LOG_INFO(" - " << actual_comp_model->outputs()[i]);
            const auto& actual_tensor = m_subrequests.at(idx)->get_tensor(actual_comp_model->outputs()[i]);
            const auto& ref_tensor = m_ref_subrequests.at(idx)->get_tensor(ref_comp_model->outputs()[i]);
            LOG_BLOCK();
            tensors_converge &= m_npuw_model->m_acc_check(actual_tensor, ref_tensor);
        }
        LOG_INFO((tensors_converge ? "PASS" : "FAIL"));

        if (!tensors_converge) {
            LOG_INFO("Subrequest is inaccurate, failover to reference.");
            // FIXME: We need to copy reference tensors to actual only in single-model-inference mode
            //        or if our subgraph is last in the chain.
            for (size_t i = 0; i < actual_comp_model->outputs().size(); i++) {
                const auto& actual_tensor = m_subrequests.at(idx)->get_tensor(actual_comp_model->outputs()[i]);
                const auto& ref_tensor = m_ref_subrequests.at(idx)->get_tensor(ref_comp_model->outputs()[i]);
                ref_tensor->copy_to(actual_tensor._ptr);
            }
            m_npuw_model->m_compiled_submodels.at(idx).compiled_model =
                m_npuw_model->m_compiled_submodels.at(idx).ref_compiled_model;
            m_npuw_model->m_compiled_submodels.at(idx).switched_to_ref = true;
            m_subrequests.at(idx) = m_ref_subrequests.at(idx);
            update_subrequest_links(idx);
            failover = true;
        }

        LOG_INFO("Done");
    } else {
        LOG_INFO("Skipped, subrequest is launched on reference device.");
    }
}

ov::SoPtr<ov::ITensor> ov::npuw::IBaseInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    // assert(persistent)
    return m_port_to_tensor.at(port).tensor;
}

void ov::npuw::IBaseInferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                             const ov::SoPtr<ov::ITensor>& tensor) {
    // Assigning via .at() to ensure it is a known port
    // assert(persistent)
    m_port_to_tensor.at(port).tensor = tensor;
}

std::vector<ov::SoPtr<ov::ITensor>> ov::npuw::IBaseInferRequest::get_tensors(
    const ov::Output<const ov::Node>& port) const {
    return {};  // NB: Comment why it is empty, and not { get_tensor(port); }
}

void ov::npuw::IBaseInferRequest::set_tensors(const ov::Output<const ov::Node>&,
                                              const std::vector<ov::SoPtr<ov::ITensor>>&) {
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::npuw::IBaseInferRequest::check_tensors() const {
    // Ignore `check_tensor` of inputs and outputs of Hetero Compiled Model because
    // `m_tensors` are not allocated
    return;
}

void ov::npuw::IBaseInferRequest::infer() {
    m_now_idx.reset();
    prepare_for_infer();
    bool failover_happened = false;
    for (std::size_t idx = 0u; idx < m_num_submodels; idx++) {
        m_now_idx = idx;
        if (!valid_subrequest(idx)) {
            continue;
        }
        subscribe_subrequest(idx, [](std::exception_ptr) {});
        bool failover = false;
        run_subrequest_for_success(idx, failover);
        failover_happened |= failover;
        complete_subrequest(idx);
        if (m_npuw_model->m_acc_check) {
            ensure_subrequest_is_accurate(idx, failover);
            failover_happened |= failover;
        }
    }

    // Increment counter regardless if dumps etc are enabled or not.
    m_run_iter++;

    if (failover_happened) {
        LOG_INFO("Refined device distribution:");
        LOG_BLOCK();
        m_npuw_model->log_device_dist();
    }
    m_now_idx.reset();
}

void ov::npuw::IBaseInferRequest::dump_input_tensors(std::size_t idx) {
    auto real_idx = m_npuw_model->m_compiled_submodels[idx].replaced_by.value_or(idx);
    const auto& comp_submodel = m_npuw_model->m_compiled_submodels[real_idx].compiled_model;

    // Note: keep using the absolute `idx` for identififaction and printing
    // Note:
    // - _name is used for the user option (no leading 00s for indices)
    // - _path is used for disk dump (will have leading 00s for indices)
    const auto comp_submodel_name = subgr_name(idx);
    const auto comp_submodel_path = m_npuw_model->m_name + subgr_path_suffix(idx) + iter_path_suffix(idx);

    const std::string dump_ios_opt = m_npuw_model->m_cfg.get<::intel_npu::NPUW_DUMP_IO>();
    if (ov::npuw::util::is_set(idx, dump_ios_opt)) {
        std::vector<std::string> in_base_names;
        for (std::size_t i = 0u, num_inputs = comp_submodel->inputs().size(); i < num_inputs; i++) {
            const auto& port = comp_submodel->inputs()[i];
            const auto& tnsr = m_subrequests[real_idx]->get_tensor(port);
            std::string in_base_name = comp_submodel_path + "_input_" + ov::npuw::util::fmt(i, num_inputs);
            ov::npuw::dump_tensor(tnsr, in_base_name);
            in_base_names.push_back(std::move(in_base_name));
        }
        ov::npuw::dump_input_list(comp_submodel_path, in_base_names);
    }
}

void ov::npuw::IBaseInferRequest::dump_output_tensors(std::size_t idx) {
    auto real_idx = m_npuw_model->m_compiled_submodels[idx].replaced_by.value_or(idx);
    const auto& comp_submodel = m_npuw_model->m_compiled_submodels[real_idx].compiled_model;

    // Note: keep using the absolute `idx` for identififaction and printing
    // Note:
    // - _name is used for the user option (no leading 00s for indices)
    // - _path is used for disk dump (will have leading 00s for indices)
    // FIXME: Duplication is evil
    const auto comp_submodel_name = subgr_name(idx);
    const auto comp_submodel_path = m_npuw_model->m_name + subgr_path_suffix(idx) + iter_path_suffix(idx);

    const std::string dump_ios_opt = m_npuw_model->m_cfg.get<::intel_npu::NPUW_DUMP_IO>();
    if (ov::npuw::util::is_set(idx, dump_ios_opt)) {
        std::vector<std::string> out_base_names;
        for (std::size_t i = 0u, num_outputs = comp_submodel->outputs().size(); i < num_outputs; i++) {
            const auto& port = comp_submodel->outputs()[i];
            const auto& tnsr = m_subrequests[real_idx]->get_tensor(port);
            std::string out_base_name = comp_submodel_path + "_output_" + ov::npuw::util::fmt(i, num_outputs);
            ov::npuw::dump_tensor(tnsr, out_base_name);
            out_base_names.push_back(std::move(out_base_name));
        }
        ov::npuw::dump_output_list(comp_submodel_path, out_base_names);
    }
}

std::string ov::npuw::IBaseInferRequest::subgr_name(std::size_t idx) const {
    return m_npuw_model->m_name + "_" + std::to_string(idx);
}

std::string ov::npuw::IBaseInferRequest::subgr_path_suffix(std::size_t idx) const {
    return "_" + ov::npuw::util::fmt(idx, m_npuw_model->m_compiled_submodels.size());
}

std::string ov::npuw::IBaseInferRequest::iter_path_suffix(std::size_t idx) const {
    // Check if per-iter dump is required - and provide the suffix if necessary
    if (!m_iter_suffix_required.has_value()) {
        m_iter_suffix_required = m_npuw_model->m_cfg.get<::intel_npu::NPUW_DUMP_IO_ITERS>();
    }

    if (!m_iter_suffix_required.value()) {
        return "";  // no need to dump individual iterations - keep suffix empty
    }

    // Hope alignment to 4 digits is fine to this case (no problem if this number
    // is exceeded)
    return "_iter_" + ov::npuw::util::fmt(m_run_iter, 1000);
}

bool ov::npuw::IBaseInferRequest::needs_copy(std::size_t idx) const {
    // Answer if the given subgraph needs copy for I/O or tolerates
    // the set/get_ tensor API
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    const auto real_idx = comp_model_desc.replaced_by.value_or(idx);
    if (ov::npuw::util::starts_with(m_subrequest_devices[real_idx], "CPU")) {
        return false;
    }

    // Assume all others prefer copy unless remote tensors are supported
    return true;
}

std::size_t ov::npuw::IBaseInferRequest::next(std::size_t idx_base) const {
    // Answer the next valid subrequest which is possible to prepare
    // FIXME: this could be a predefined map, not a lookup
    for (std::size_t idx = idx_base; idx < m_num_submodels; idx++) {
        auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
        if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
            continue;
        }
        return idx;
    }

    // went over entire list and nothing found?
    // NOTE: this recursive call is a short-cut and may enter the recursion
    // if all the subgraphs are OPTIMIZED OUT (shouldn't be possible but
    // there's a Murphy's law on this).
    return next(0);
}

std::size_t ov::npuw::IBaseInferRequest::real(std::size_t idx) const {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    return comp_model_desc.replaced_by.value_or(idx);
}

ov::npuw::IBaseInferRequest::now_t ov::npuw::IBaseInferRequest::now_idx() const {
    return m_now_idx;
}
