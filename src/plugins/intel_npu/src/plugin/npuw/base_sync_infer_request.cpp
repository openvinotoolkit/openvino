// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base_sync_infer_request.hpp"

#include "compiled_model.hpp"
#include "intel_npu/config/npuw.hpp"
#include "logging.hpp"
#include "openvino/core/parallel.hpp"
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
            if (can_try_again && recompiled) {
                *recompiled = true;
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

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::IBaseInferRequest::query_state() const {
    std::vector<ov::SoPtr<ov::IVariableState>> variable_states = {};
    for (const auto& request : m_subrequests) {
        if (!request)  // optimized out
            continue;
        for (auto&& state : request->query_state()) {
            if (!state._so)
                state._so = request._so;
            variable_states.emplace_back(state);
        }
    }
    return variable_states;
}

std::vector<ov::ProfilingInfo> ov::npuw::IBaseInferRequest::get_profiling_info() const {
    std::vector<ov::ProfilingInfo> info;
    for (size_t i = 0; i < m_subrequests.size(); ++i) {
        if (!m_subrequests[i])  // optimized out
            continue;
        auto&& subreq_info = m_subrequests[i]->get_profiling_info();
        for (auto&& rec : subreq_info)
            rec.node_name = std::string("subgraph") + std::to_string(i) + ": " + rec.node_name;
        info.insert(info.end(), subreq_info.begin(), subreq_info.end());
    }
    return info;
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

std::size_t ov::npuw::IBaseInferRequest::total_subrequests() const {
    return m_subrequests.size();
}

ov::npuw::TensorPtr ov::npuw::IBaseInferRequest::allocMem(const ov::element::Type type,
                                                          const ov::Shape& shape,
                                                          const std::string& device) {
    if (device == "CPU" || ov::shape_size(shape) == 0) {
        return ov::get_tensor_impl(ov::Tensor(type, shape));
    }

    auto remote_ctx = m_npuw_model->get_plugin()->get_core()->get_default_context(device)._ptr;
    auto remote_tensor = remote_ctx->create_host_tensor(type, shape);
    return ov::get_tensor_impl(ov::make_tensor(remote_tensor));
}

ov::npuw::TensorPtr ov::npuw::IBaseInferRequest::allocOut(const ov::Output<const ov::Node>& node,
                                                          const std::string& device) {
    return allocMem(node.get_element_type(), node.get_shape(), device);
}

void ov::npuw::IBaseInferRequest::alloc_io() {
    // Preallocate input tensors
    LOG_INFO("Preallocating input tensors...");
    for (size_t i = 0; i < m_npuw_model->inputs().size(); i++) {
        const auto& port = m_npuw_model->inputs()[i];
        ov::SoPtr<ov::ITensor> allocated = allocOut(port, m_npuw_model->global_mem_device());
        m_input_tensors.push_back(allocated);
        m_input_allocated.insert(allocated->data());
        m_port_to_tensor[port] = TensorStorage{m_input_tensors.back(), true};
    }  // for(inputs)

    // Preallocate output tensors
    LOG_INFO("Preallocating output tensors...");
    for (size_t i = 0; i < m_npuw_model->outputs().size(); i++) {
        LOG_BLOCK();
        const auto& port = m_npuw_model->outputs()[i];
        LOG_INFO("Output " << i << " of " << m_npuw_model->outputs().size() << ": " << port);

        // FIXME: Yes, the CompiledModel::ToSubmodel == JustInferRequest::LinkFrom
        const auto& from_submodel = m_npuw_model->m_outputs_to_submodels_outputs.at(i);
        LOG_INFO("Produced by Subgraph[" << from_submodel.first << "] / " << from_submodel.second);

        auto tensor = alloc_global_out(i);
        m_output_tensors.push_back(tensor);
        m_port_to_tensor[port] = TensorStorage{tensor, true};
    }
}

ov::npuw::TensorPtr ov::npuw::IBaseInferRequest::alloc_global_out(std::size_t out_idx) {
    const auto& port = m_npuw_model->outputs().at(out_idx);
    return allocOut(port, m_npuw_model->global_mem_device());
}

void ov::npuw::IBaseInferRequest::init_gio() {
    // Build the parameter/result mapping
    m_subrequests_gio.resize(m_subrequests.size());

    // Parameters: stage 1...
    for (size_t i = 0; i < m_npuw_model->inputs().size(); i++) {
        const auto& to_submodel = m_npuw_model->m_inputs_to_submodels_inputs.at(i);
        if (to_submodel != CompiledModel::NO_LINK) {
            std::size_t sub_idx{}, in_idx{};
            std::tie(sub_idx, in_idx) = to_submodel;
            m_subrequests_gio.at(sub_idx).global_params[i] = in_idx;
        }
    }  // for(inputs)

    // Parameters: stage 2...
    for (auto&& it : m_npuw_model->m_param_subscribers) {
        const auto param_idx = it.first;
        for (auto&& to_submodel : it.second) {
            std::size_t sub_idx{}, in_idx{};
            std::tie(sub_idx, in_idx) = to_submodel;
            m_subrequests_gio.at(sub_idx).global_params[param_idx] = in_idx;
        }
    }

    // Results
    for (size_t i = 0; i < m_npuw_model->outputs().size(); i++) {
        std::size_t sub_idx{}, out_idx{};
        std::tie(sub_idx, out_idx) = m_npuw_model->m_outputs_to_submodels_outputs.at(i);
        m_subrequests_gio.at(sub_idx).global_results[i] = out_idx;
    }
}

void ov::npuw::IBaseInferRequest::unpack_closure(std::size_t idx, RqPtr request) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];

    NPUW_ASSERT(comp_model_desc.replaced_by);
    const auto real_idx = comp_model_desc.replaced_by.value();
    auto& func_desc = m_npuw_model->m_compiled_submodels[real_idx];

    // Bind extra parameters from the function's closure
    // First, do easy things & delay heavy stuff
    std::vector<std::size_t> closure_unpack_required;
    std::vector<std::size_t> closure_copy_required;

    for (std::size_t cidx = 0u; cidx < comp_model_desc.closure.size(); cidx++) {
        auto& closure = comp_model_desc.closure[cidx];
        const auto closure_param_id = comp_model_desc.param_base + cidx;

        if (m_npuw_model->is_gather_closure(idx, cidx)) {
            // No need to set/copy the host_gather's closure tensor int
            // the subrequest - it is just a dummy. host_gather writes
            // to the right buffer directly.
            continue;
        }

        auto& iport = func_desc.compiled_model->inputs()[closure_param_id];
        if (m_npuw_model->unpack_required(idx, cidx)) {
            // Remember where the unpack is required
            closure_unpack_required.push_back(cidx);
        } else {
            if (needs_copy(idx, cidx)) {
                // Remember where copy is requried
                closure_copy_required.push_back(cidx);
            } else {
                // Easy case, just set one to another
                request->set_tensor(iport, ov::get_tensor_impl(closure));
            }
        }
    }  // for(closure)

    // m_ms_unpack += ov::npuw::perf::ms_to_run([&](){
    ov::parallel_for(closure_copy_required.size(), [&](std::size_t j) {
        auto cidx = closure_copy_required[j];
        auto& closure = comp_model_desc.closure[cidx];
        const auto closure_param_id = comp_model_desc.param_base + cidx;
        auto& iport = func_desc.compiled_model->inputs()[closure_param_id];
        auto clparam = request->get_tensor(iport);
        ov::get_tensor_impl(closure)->copy_to(clparam._ptr);
    });
    // }); // ms_to_run

    for (std::size_t j = 0; j != closure_unpack_required.size(); j++) {
        // NB: No need to protect anything here as containers are all
        // preallocated and we only access elements under particular (thread
        // -local) indices.
        auto cidx = closure_unpack_required[j];

        // FIXME: zerops are stored with absolute indexing, this needs to be aligned
        auto& closure = comp_model_desc.closure[cidx];

        const auto closure_param_id = comp_model_desc.param_base + cidx;
        auto& iport = func_desc.compiled_model->inputs()[closure_param_id];
        auto clparam = request->get_tensor(iport);

        if (!comp_model_desc.scales.empty() && comp_model_desc.scales[cidx] && comp_model_desc.zerops[cidx]) {
            // Unpacking this weight requires scaling with zero points...
            ov::npuw::util::unpack(ov::get_tensor_impl(closure),
                                   ov::get_tensor_impl(comp_model_desc.zerops[cidx]),
                                   ov::get_tensor_impl(comp_model_desc.scales[cidx]),
                                   clparam);
        } else if (!comp_model_desc.scales.empty() && comp_model_desc.scales[cidx]) {
            // Unpacking this weight requires scaling
            ov::npuw::util::unpack(ov::get_tensor_impl(closure),
                                   ov::get_tensor_impl(comp_model_desc.scales[cidx]),
                                   clparam);
        } else {
            // Unpacking this weight doesn't require scaling
            ov::npuw::util::unpack(ov::get_tensor_impl(closure), clparam);
        }
    }
}

void ov::npuw::IBaseInferRequest::bind_global_params(std::size_t idx, RqPtr request) {
    LOG_DEBUG("Binding parameters for Subgraph[" << idx << "]");
    LOG_BLOCK();

    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    const auto real_idx = comp_model_desc.replaced_by.value_or(idx);

    const bool do_copy = needs_copy(idx);
    const auto& iodesc = m_subrequests_gio.at(idx);

    const auto& proto_comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
    const bool is_spatial = proto_comp_model_desc.spatial.has_value();

    // a list of ports to copy tensors, if needed: FROM -> TO
    std::vector<std::pair<ov::SoPtr<ov::ITensor>, ov::Output<const ov::Node>>> copy_list;

    // Check if the given subgraph's input is spatial
    auto is_spatial_param = [&](std::size_t sub_in_idx) -> bool {
        if (!is_spatial) {
            return false;  // Early return
        }
        auto& spatial = proto_comp_model_desc.spatial.value();
        return std::any_of(spatial.params.begin(), spatial.params.end(), [&](const auto& p) -> bool {
            return p.idx == sub_in_idx;
        });
    };

    for (auto&& it : iodesc.global_params) {
        std::size_t param_idx{}, sub_in_idx{};
        std::tie(param_idx, sub_in_idx) = it;
        LOG_DEBUG("Processing " << param_idx << " -> " << sub_in_idx << std::endl);

        const auto& g_port = m_npuw_model->inputs()[param_idx];
        const auto& g_tnsr = m_port_to_tensor.at(g_port).tensor;
        const auto& s_port = request->get_inputs()[sub_in_idx];
        LOG_DEBUG("Processing " << g_port << " -> " << s_port << "...");
        LOG_BLOCK();
        if (!is_spatial_param(sub_in_idx)) {
            // Input parameter is non-spatial, do normal handling
            if (m_input_allocated.count(g_tnsr->data()) == 0 && do_copy) {
                LOG_DEBUG("Will be copied");
                copy_list.emplace_back(g_tnsr, s_port);
            } else {
                LOG_DEBUG("Will be set");
                request->set_tensor(s_port, g_tnsr);
            }
        } else {
            // Register for future use
            m_spatial_io[real_idx].inputs.at(sub_in_idx) = g_tnsr;
        }
    }

    LOG_DEBUG("Running copy...");
    ov::parallel_for(copy_list.size(), [&](std::size_t idx) {
        auto& it = copy_list[idx];
        ov::SoPtr<ov::ITensor> dst = request->get_tensor(it.second);
        it.first->copy_to(dst._ptr);
    });

    // Run host-side gather, if required
    if (comp_model_desc.host_gather.dst_idx != -1) {
        const auto& gport = comp_model_desc.compiled_model->inputs()[comp_model_desc.host_gather.dst_idx];
        const auto gather = request->get_tensor(gport);

        const auto& vocab = comp_model_desc.closure[comp_model_desc.host_gather.src_idx - comp_model_desc.param_base];
        const auto& lport = comp_model_desc.compiled_model->inputs()[comp_model_desc.host_gather.idx_idx];
        const auto lookup = request->get_tensor(lport);
        ov::npuw::util::gather(ov::get_tensor_impl(vocab), lookup, gather);
    }

    LOG_DEBUG("Done");
}

void ov::npuw::IBaseInferRequest::bind_global_results(std::size_t idx, RqPtr request) {
    LOG_DEBUG("Binding results for Subgraph[" << idx << "]");
    LOG_BLOCK();

    const auto& iodesc = m_subrequests_gio.at(idx);
    for (auto&& it : iodesc.global_results) {
        std::size_t result_idx{}, sub_out_idx{};
        std::tie(result_idx, sub_out_idx) = it;
        const auto& g_port = m_npuw_model->outputs()[result_idx];
        const auto& s_port = request->get_outputs()[sub_out_idx];
        request->set_tensor(s_port, m_port_to_tensor.at(g_port).tensor);
    }

    LOG_DEBUG("Done");
}

void ov::npuw::IBaseInferRequest::dump_input_tensors(std::size_t idx) {
    const std::string dump_ios_opt = m_npuw_model->m_cfg.get<::intel_npu::NPUW_DUMP_IO>();
    const std::size_t end_idx = m_npuw_model->m_compiled_submodels.size();
    auto real_idx = m_npuw_model->m_compiled_submodels[idx].replaced_by.value_or(idx);

    if (!ov::npuw::util::is_set(idx, dump_ios_opt, real_idx, end_idx)) {
        return;
    }

    const auto& comp_submodel_desc = m_npuw_model->m_compiled_submodels[real_idx];
    const auto& comp_submodel = comp_submodel_desc.compiled_model;

    // Note: keep using the absolute `idx` for identififaction and printing
    // Note:
    // - _name is used for the user option (no leading 00s for indices)
    // - _path is used for disk dump (will have leading 00s for indices)
    const auto& comp_submodel_path = m_npuw_model->m_name + subgr_path_suffix(idx) + iter_path_suffix(idx);
    const auto num_inputs = comp_submodel->inputs().size();

    // There's different approaches to dumping normal and spatial subgraphs.
    if (!comp_submodel_desc.spatial) {
        // In the normal, non-spatial mode, we just dump the current subgrequests
        // pre-set tensors and that's it
        std::vector<std::string> in_base_names;
        for (std::size_t i = 0u; i < num_inputs; i++) {
            const auto& port = comp_submodel->inputs()[i];
            const auto& tnsr = m_subrequests[real_idx]->get_tensor(port);
            std::string in_base_name = comp_submodel_path + "_input_" + ov::npuw::util::fmt(i, num_inputs);
            ov::npuw::dump_tensor(tnsr, in_base_name);
            in_base_names.push_back(std::move(in_base_name));
        }
        ov::npuw::dump_input_list(comp_submodel_path, in_base_names);
    } else {
        const auto& s = comp_submodel_desc.spatial.value();

        std::set<std::size_t> spatial_param_idx;
        std::vector<std::string> in_base_names(num_inputs);

        // First, dump the non-spatial input tensors just once - and remember its names
        for (auto&& p : s.params) {
            spatial_param_idx.insert(p.idx);
        }
        for (std::size_t i = 0u; i < num_inputs; i++) {
            if (spatial_param_idx.count(i)) {
                continue;
            }
            const auto& port = comp_submodel->inputs()[i];
            const auto& tnsr = m_subrequests[real_idx]->get_tensor(port);
            std::string in_base_name = comp_submodel_path + "_input_" + ov::npuw::util::fmt(i, num_inputs);
            ov::npuw::dump_tensor(tnsr, in_base_name);
            in_base_names[i] = std::move(in_base_name);
        }

        // Now iterate over the spatial range and dump the individual tiles
        // For the spatial case, these tiles should've been taken from the special
        // spatial_io tensors
        for (std::size_t offset = 0u; offset < s.range; offset += s.nway) {
            const std::size_t this_len = (offset + s.nway <= s.range) ? s.nway               // the full tile
                                                                      : (s.range - offset);  // the last tile
            if (m_spatial_selector != nullptr && !m_spatial_selector->need_submit(offset, this_len)) {
                continue;
            }

            for (auto&& p : s.params) {
                std::string in_base_name = comp_submodel_path + "_input_" + ov::npuw::util::fmt(p.idx, num_inputs) +
                                           "_d" + ov::npuw::util::fmt(p.dim, 10) + "_" +
                                           ov::npuw::util::fmt(offset, s.range);

                const auto& tnsr = m_spatial_io[real_idx].inputs.at(p.idx);
                const auto& view = ov::npuw::util::view(tnsr, p.dim, offset, this_len);

                ov::npuw::dump_tensor(view, in_base_name);
                in_base_names[p.idx] = std::move(in_base_name);
            }
            // Dump ilist per tile
            std::string tile_ilist_name = comp_submodel_path + "_" + ov::npuw::util::fmt(offset, s.range);
            ov::npuw::dump_input_list(tile_ilist_name, in_base_names);
        }  // for(offset)
    }
}

void ov::npuw::IBaseInferRequest::dump_output_tensors(std::size_t idx) {
    const std::string dump_ios_opt = m_npuw_model->m_cfg.get<::intel_npu::NPUW_DUMP_IO>();
    const std::size_t end_idx = m_npuw_model->m_compiled_submodels.size();
    auto real_idx = m_npuw_model->m_compiled_submodels[idx].replaced_by.value_or(idx);

    if (!ov::npuw::util::is_set(idx, dump_ios_opt, real_idx, end_idx)) {
        return;
    }

    const auto& comp_submodel_desc = m_npuw_model->m_compiled_submodels[real_idx];
    const auto& comp_submodel = comp_submodel_desc.compiled_model;

    // Note: keep using the absolute `idx` for identififaction and printing
    // Note:
    // - _name is used for the user option (no leading 00s for indices)
    // - _path is used for disk dump (will have leading 00s for indices)
    // FIXME: Duplication is evil
    const auto& comp_submodel_path = m_npuw_model->m_name + subgr_path_suffix(idx) + iter_path_suffix(idx);
    const std::size_t num_outputs = comp_submodel->outputs().size();

    // Same approach as in above. Spatial tensors require special handling
    if (!comp_submodel_desc.spatial) {
        std::vector<std::string> out_base_names;
        for (std::size_t i = 0u; i < num_outputs; i++) {
            const auto& port = comp_submodel->outputs()[i];
            const auto& tnsr = m_subrequests[real_idx]->get_tensor(port);
            std::string out_base_name = comp_submodel_path + "_output_" + ov::npuw::util::fmt(i, num_outputs);
            ov::npuw::dump_tensor(tnsr, out_base_name);
            out_base_names.push_back(std::move(out_base_name));
        }
        ov::npuw::dump_output_list(comp_submodel_path, out_base_names);
    } else {
        // All outputs are considered spatial now so it should be easier
        const auto& s = comp_submodel_desc.spatial.value();
        for (std::size_t offset = 0u; offset < s.range; offset += s.nway) {
            const std::size_t this_len = (offset + s.nway <= s.range) ? s.nway               // the full tile
                                                                      : (s.range - offset);  // the last tile
            std::vector<std::string> tile_olist;
            for (std::size_t i = 0u; i < num_outputs; i++) {
                std::string out_base_name = comp_submodel_path + "_output_" + ov::npuw::util::fmt(i, num_outputs) +
                                            "_d" + ov::npuw::util::fmt(s.out_dim, 10) + "_" +
                                            ov::npuw::util::fmt(offset, s.range);
                const auto& tnsr = m_spatial_io[real_idx].outputs.at(i);
                const auto& view = ov::npuw::util::view(tnsr, s.out_dim, offset, this_len);

                ov::npuw::dump_tensor(view, out_base_name);
                tile_olist.push_back(std::move(out_base_name));
            }
            // Dump olist per tile
            std::string tile_olist_name = comp_submodel_path + "_" + ov::npuw::util::fmt(offset, s.range);
            ov::npuw::dump_output_list(tile_olist_name, tile_olist);
        }
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

bool ov::npuw::IBaseInferRequest::needs_copy(std::size_t idx, std::size_t cidx) const {
    if (!needs_copy(idx)) {
        return false;
    }
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    if (comp_model_desc.is_remote[cidx]) {
        // FIXME: Test if the tensor device and the request device are
        // the same or compatible!
        return false;
    }
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
