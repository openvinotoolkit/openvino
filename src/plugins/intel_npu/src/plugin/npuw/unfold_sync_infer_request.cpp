// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/parallel.hpp"

#include "unfold_sync_infer_request.hpp"
#include "compiled_model.hpp"
#include "logging.hpp"

ov::npuw::UnfoldInferRequest::UnfoldInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model)
    : ov::npuw::JustInferRequest(compiled_model, false) {
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
        m_subrequests[i] = proto_comp_model_desc.compiled_model->create_infer_request();
        m_subrequest_devices[i] = *proto_comp_model_desc.device_it;
        LOG_INFO("DONE");
    }  // for(submodels)

    // Preallocate input tensors. Note - there may be
    // multiple subrequest consumers on the same input tensor
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
        const auto& tensor = allocOut(port, m_npuw_model->global_mem_device());

        m_output_tensors.push_back(tensor);
        m_port_to_tensor[port] = TensorStorage{tensor, true};
    }

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

    // Build the parameter/result mapping {{{
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
    // }}}

    for (size_t i = 0; i < m_num_submodels; i++) {
        LOG_VERB("Trying to preemptively set tensors for Subgraph[" << i << "]...");
        LOG_BLOCK();
        auto& comp_model_desc = m_npuw_model->m_compiled_submodels[i];
        if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
            continue;  // Optimized out
        }
        unpack_closure(i, m_subrequests[i]);
        LOG_VERB("Done");
    }
}

void ov::npuw::UnfoldInferRequest::prepare(std::size_t idx) {
    if (idx >= m_subrequests.size()) {
        return;
    }
    auto& subr = m_subrequests.at(idx);
    const bool do_copy = needs_copy(idx);

    std::vector<std::pair<ov::SoPtr<ov::ITensor>, ov::Output<const ov::Node>>> copy_list;

    // bind_global_parameters(), a simplified way
    const auto& iodesc = m_subrequests_gio.at(idx);
    for (auto&& it : iodesc.global_params) {
        std::size_t param_idx{}, sub_in_idx{};
        std::tie(param_idx, sub_in_idx) = it;
        const auto& g_port = m_npuw_model->inputs()[param_idx];
        const auto& g_tnsr = m_port_to_tensor.at(g_port).tensor;
        const auto& s_port = subr->get_inputs()[sub_in_idx];

        if (m_input_allocated.count(g_tnsr->data()) == 0 && do_copy) {
            copy_list.emplace_back(g_tnsr, s_port);
        } else {
            subr->set_tensor(s_port, g_tnsr);
        }
    }

    // bind_global_results, a simplified way
    for (auto&& it : iodesc.global_results) {
        std::size_t result_idx{}, sub_out_idx{};
        std::tie(result_idx, sub_out_idx) = it;
        const auto& g_port = m_npuw_model->outputs()[result_idx];
        const auto& s_port = subr->get_outputs()[sub_out_idx];
        subr->set_tensor(s_port, m_port_to_tensor.at(g_port).tensor);
    }

    // run copy, if required
    ov::parallel_for(copy_list.size(), [&](std::size_t idx) {
        auto& it = copy_list[idx];
        ov::SoPtr<ov::ITensor> dst = subr->get_tensor(it.second);
        it.first->copy_to(dst._ptr);
    });

    // run host gather, if required
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    if (comp_model_desc.host_gather.dst_idx != -1) {
        const auto& gport = comp_model_desc.compiled_model->inputs()[comp_model_desc.host_gather.dst_idx];
        const auto gather = subr->get_tensor(gport);

        const auto& vocab =
            comp_model_desc.closure[comp_model_desc.host_gather.src_idx - comp_model_desc.param_base];
        const auto& lport = comp_model_desc.compiled_model->inputs()[comp_model_desc.host_gather.idx_idx];
        const auto lookup = subr->get_tensor(lport);
        ov::npuw::util::gather(ov::get_tensor_impl(vocab), lookup, gather);
    }
}

void ov::npuw::UnfoldInferRequest::infer() {
    const bool do_async = m_npuw_model->m_cfg.get<::intel_npu::NPUW_FUNCALL_ASYNC>();

    if (do_async) {
        for (std::size_t idx = 0; idx < m_num_submodels; idx++) {
            prepare(idx);
        }
        for (std::size_t idx = 0; idx < m_num_submodels; idx++) {
            auto& subr = m_subrequests[idx];
            if (!subr) {
                continue;
            }
            subr->start_async();
        }
        for (std::size_t idx = 0; idx < m_num_submodels; idx++) {
            auto& subr = m_subrequests[idx];
            if (!subr) {
                continue;
            }
            subr->wait();
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
    } // (async)
}
