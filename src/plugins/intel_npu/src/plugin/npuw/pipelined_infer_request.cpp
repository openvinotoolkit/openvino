// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pipelined_infer_request.hpp"

#include <cstring>

#include "compiled_model.hpp"
#include "logging.hpp"
#include "moe/moe_subgraph.hpp"
#include "openvino/core/parallel.hpp"
#include "util.hpp"

ov::npuw::PipelinedInferRequest::PipelinedInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model)
    : ov::npuw::IBaseInferRequest(compiled_model) {
    NPUW_ASSERT(m_npuw_model->m_compiled_pipeline_model);

    m_pipeline_request = m_npuw_model->m_compiled_pipeline_model->create_infer_request();
    m_subrequests[0] = m_pipeline_request;

    initialize_shared_inputs();
    initialize_hfa_branch_selection();
    initialize_outputs();
    init_gio();
    alloc_quant_gather();

    for (std::size_t idx = 0; idx < m_num_submodels; ++idx) {
        const auto& desc = m_npuw_model->m_compiled_submodels[idx];
        if (desc.replaced_by) {
            unpack_closure(idx, m_pipeline_request);
        }
    }
}

void ov::npuw::PipelinedInferRequest::initialize_shared_inputs() {
    const auto& pipeline_inputs = m_npuw_model->m_compiled_pipeline_model->inputs();
    for (const auto& connected_input : m_npuw_model->m_pipeline_connected_inputs) {
        const auto parameter_it = m_npuw_model->m_pipeline_global_parameters.find(connected_input.first);
        if (parameter_it == std::end(m_npuw_model->m_pipeline_global_parameters) || parameter_it->second.empty()) {
            continue;
        }

        const auto& shared_port = pipeline_inputs.at(parameter_it->second.front());
        auto shared_tensor = allocOut(shared_port, "NPU");
        for (const auto port_index : parameter_it->second) {
            m_pipeline_request->set_tensor(pipeline_inputs.at(port_index), shared_tensor);
        }
    }
}

void ov::npuw::PipelinedInferRequest::initialize_hfa_branch_selection() {
    if (!m_npuw_model->m_pipeline_has_hfa) {
        return;
    }

    const auto parameter_it =
        m_npuw_model->m_pipeline_global_parameters.find(m_npuw_model->m_nlp_branch_select_port_name);
    OPENVINO_ASSERT(
        parameter_it != std::end(m_npuw_model->m_pipeline_global_parameters) && !parameter_it->second.empty(),
        "HFA pipeline branch-selection port is missing");

    const auto& branch_port = m_npuw_model->m_compiled_pipeline_model->inputs().at(parameter_it->second.front());
    auto branch_tensor = allocOut(branch_port, "NPU");
    m_pipeline_request->set_tensor(branch_port, branch_tensor);
    std::memset(branch_tensor->data(), 0, branch_tensor->get_byte_size());

    for (const auto& port_name :
         {m_npuw_model->m_hfa_port_names.K, m_npuw_model->m_hfa_port_names.V, m_npuw_model->m_hfa_port_names.Q}) {
        const auto ports_it = m_npuw_model->m_pipeline_global_parameters.find(port_name);
        if (ports_it == std::end(m_npuw_model->m_pipeline_global_parameters) || ports_it->second.empty()) {
            continue;
        }

        const auto& reference_port = m_npuw_model->m_compiled_pipeline_model->inputs().at(ports_it->second.front());
        auto shared_tensor = allocOut(reference_port, "NPU");
        for (const auto port_index : ports_it->second) {
            m_pipeline_request->set_tensor(m_npuw_model->m_compiled_pipeline_model->inputs().at(port_index),
                                           shared_tensor);
        }
    }
}

void ov::npuw::PipelinedInferRequest::initialize_outputs() {
    const auto& pipeline_outputs = m_npuw_model->m_compiled_pipeline_model->outputs();
    for (std::size_t output_index = 0; output_index < m_npuw_model->outputs().size(); ++output_index) {
        const auto mapping_it = m_npuw_model->m_pipeline_global_outputs.find(static_cast<uint32_t>(output_index));
        const auto pipeline_output_index =
            mapping_it == std::end(m_npuw_model->m_pipeline_global_outputs) ? output_index : mapping_it->second;
        m_pipeline_request->set_tensor(pipeline_outputs.at(pipeline_output_index),
                                       get_tensor(m_npuw_model->outputs()[output_index]));
    }
}

void ov::npuw::PipelinedInferRequest::prepare_for_infer() {
    for (std::size_t idx = 0; idx < m_num_submodels; ++idx) {
        bind_global_params(idx, m_pipeline_request);
    }
}

bool ov::npuw::PipelinedInferRequest::valid_subrequest(std::size_t idx) const {
    return idx == 0 && m_pipeline_request != nullptr;
}

void ov::npuw::PipelinedInferRequest::start_subrequest(std::size_t idx) {
    OPENVINO_ASSERT(idx == 0, "Pipelined request has one executable subrequest");
    m_pipeline_request->start_async();
}

void ov::npuw::PipelinedInferRequest::run_subrequest_for_success(std::size_t idx) {
    OPENVINO_ASSERT(idx == 0, "Pipelined request has one executable subrequest");
    for (std::size_t submodel_idx = 0; submodel_idx < m_num_submodels; ++submodel_idx) {
        bind_global_results(submodel_idx, m_pipeline_request);
    }
    update_hfa_branch_selection();
    m_pipeline_request->infer();
}

void ov::npuw::PipelinedInferRequest::subscribe_subrequest(std::size_t idx, Completed cb) {
    OPENVINO_ASSERT(idx == 0, "Pipelined request has one executable subrequest");
    m_pipeline_request->set_callback(std::move(cb));
}

void ov::npuw::PipelinedInferRequest::complete_subrequest(std::size_t) {}

void ov::npuw::PipelinedInferRequest::cancel_subrequest(std::size_t idx) {
    OPENVINO_ASSERT(idx == 0, "Pipelined request has one executable subrequest");
    m_pipeline_request->cancel();
}

bool ov::npuw::PipelinedInferRequest::supports_async_pipeline() const {
    return false;
}

void ov::npuw::PipelinedInferRequest::update_subrequest_links(std::size_t) {}

void ov::npuw::PipelinedInferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                                 const ov::SoPtr<ov::ITensor>& tensor) {
    std::unique_lock lock(m_io_storages_mutex);
    m_port_to_tensor[port] = TensorStorage{tensor, true};

    for (std::size_t output_index = 0; output_index < m_npuw_model->outputs().size(); ++output_index) {
        if (m_npuw_model->outputs()[output_index] != port) {
            continue;
        }

        const auto mapping_it = m_npuw_model->m_pipeline_global_outputs.find(static_cast<uint32_t>(output_index));
        if (mapping_it != std::end(m_npuw_model->m_pipeline_global_outputs)) {
            m_pipeline_request->set_tensor(m_npuw_model->m_compiled_pipeline_model->outputs().at(mapping_it->second),
                                           tensor);
        }
    }

    handle_set_remote_input(port, tensor);
}

std::optional<ov::Output<const ov::Node>> ov::npuw::PipelinedInferRequest::get_pipeline_port(
    const std::string& port_name,
    std::size_t usage_index) const {
    const auto parameter_it = m_npuw_model->m_pipeline_global_parameters.find(port_name);
    if (parameter_it == std::end(m_npuw_model->m_pipeline_global_parameters) ||
        usage_index >= parameter_it->second.size()) {
        return std::nullopt;
    }
    return m_npuw_model->m_compiled_pipeline_model->inputs().at(parameter_it->second[usage_index]);
}

std::optional<ov::Output<const ov::Node>> ov::npuw::PipelinedInferRequest::get_pipeline_parameter_port(
    std::size_t idx,
    std::size_t closure_param_id) const {
    const auto& desc = m_npuw_model->m_compiled_submodels[idx];
    const auto real_idx = desc.replaced_by.value();
    const auto& function_desc = m_npuw_model->m_compiled_submodels[real_idx];
    const auto base_idx = m_npuw_model->m_pipeline_submodel_base_indices.at(idx);
    const auto usage_index = idx - base_idx;
    const auto& port_name = function_desc.input_port_name[closure_param_id];

    std::size_t offset{};
    const auto offset_it = m_npuw_model->m_pipeline_global_parameters_offset.find(base_idx);
    if (offset_it != std::end(m_npuw_model->m_pipeline_global_parameters_offset)) {
        const auto port_offset_it = offset_it->second.find(port_name);
        if (port_offset_it != std::end(offset_it->second)) {
            offset = port_offset_it->second;
        }
    }
    return get_pipeline_port(port_name, usage_index + offset);
}

void ov::npuw::PipelinedInferRequest::unpack_closure(std::size_t idx, RqPtr request) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    NPUW_ASSERT(comp_model_desc.replaced_by);
    const auto real_idx = comp_model_desc.replaced_by.value();
    auto& func_desc = m_npuw_model->m_compiled_submodels[real_idx];

    if (ov::npuw::moe::has_compiled_experts(func_desc.pipeline)) {
        return;
    }

    std::vector<std::size_t> closure_unpack_required;
    std::vector<std::size_t> closure_copy_required;
    auto& desc_closure = comp_model_desc.closure.get().closure;

    for (std::size_t cidx = 0; cidx < desc_closure.size(); ++cidx) {
        if (m_npuw_model->is_gather_closure(idx, cidx)) {
            continue;
        }

        const auto closure_param_id = comp_model_desc.param_base + cidx;
        const auto pipeline_port = get_pipeline_parameter_port(idx, closure_param_id);
        if (!pipeline_port) {
            continue;
        }

        if (m_npuw_model->unpack_required(idx, cidx)) {
            closure_unpack_required.push_back(cidx);
        } else if (needs_copy(idx, cidx)) {
            closure_copy_required.push_back(cidx);
        } else {
            request->set_tensor(*pipeline_port, ov::get_tensor_impl(desc_closure[cidx]));
        }
    }

    ov::parallel_for(closure_copy_required.size(), [&](std::size_t position) {
        const auto cidx = closure_copy_required[position];
        const auto closure_param_id = comp_model_desc.param_base + cidx;
        const auto pipeline_port = get_pipeline_parameter_port(idx, closure_param_id);
        OPENVINO_ASSERT(pipeline_port, "Pipeline closure parameter port is missing");
        auto destination = request->get_tensor(*pipeline_port);
        ov::get_tensor_impl(desc_closure[cidx])->copy_to(destination._ptr);
    });

    for (const auto cidx : closure_unpack_required) {
        const auto closure_param_id = comp_model_desc.param_base + cidx;
        const auto pipeline_port = get_pipeline_parameter_port(idx, closure_param_id);
        OPENVINO_ASSERT(pipeline_port, "Pipeline closure parameter port is missing");
        auto destination = request->get_tensor(*pipeline_port);
        auto& closure = desc_closure[cidx];

        if (!comp_model_desc.scales.empty() && comp_model_desc.scales[cidx] && comp_model_desc.zerops[cidx]) {
            ov::npuw::util::unpack(ov::get_tensor_impl(closure),
                                   ov::get_tensor_impl(comp_model_desc.zerops[cidx]),
                                   ov::get_tensor_impl(comp_model_desc.scales[cidx]),
                                   destination);
        } else if (!comp_model_desc.scales.empty() && comp_model_desc.scales[cidx]) {
            ov::npuw::util::unpack(ov::get_tensor_impl(closure),
                                   ov::get_tensor_impl(comp_model_desc.scales[cidx]),
                                   destination);
        } else {
            ov::npuw::util::unpack(ov::get_tensor_impl(closure), destination);
        }
    }
}

void ov::npuw::PipelinedInferRequest::bind_global_params(std::size_t idx, RqPtr request) {
    const auto& iodesc = m_subrequests_gio.at(idx);
    const auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    const auto real_idx = comp_model_desc.replaced_by.value_or(idx);
    const auto& proto_desc = m_npuw_model->m_compiled_submodels[real_idx];

    if (ov::npuw::moe::has_compiled_experts(proto_desc.pipeline)) {
        return;
    }

    for (const auto& [param_idx, submodel_input_idx] : iodesc.global_params) {
        const auto mapping_it = m_npuw_model->m_pipeline_global_inputs.find(static_cast<uint32_t>(param_idx));
        const auto pipeline_input_idx = mapping_it == std::end(m_npuw_model->m_pipeline_global_inputs)
                                            ? submodel_input_idx
                                            : static_cast<std::size_t>(mapping_it->second);
        request->set_tensor(m_npuw_model->m_compiled_pipeline_model->inputs().at(pipeline_input_idx),
                            get_tensor(m_npuw_model->inputs().at(param_idx)));
    }

    if (comp_model_desc.host_gather.dst_idx != -1) {
        const auto gather_port = get_pipeline_parameter_port(idx, comp_model_desc.host_gather.dst_idx);
        const auto lookup_port = get_pipeline_parameter_port(idx, comp_model_desc.host_gather.idx_idx);
        OPENVINO_ASSERT(gather_port && lookup_port, "Pipeline host-gather ports are missing");

        const auto& vocab =
            comp_model_desc.closure.get().closure[comp_model_desc.host_gather.src_idx - comp_model_desc.param_base];
        ov::npuw::util::gather(ov::get_tensor_impl(vocab),
                               request->get_tensor(*lookup_port),
                               request->get_tensor(*gather_port));
    }

    handle_quant_host_gather(idx, request);
}

void ov::npuw::PipelinedInferRequest::bind_global_results(std::size_t idx, RqPtr request) {
    const auto& iodesc = m_subrequests_gio.at(idx);
    for (const auto& [result_idx, submodel_output_idx] : iodesc.global_results) {
        const auto mapping_it = m_npuw_model->m_pipeline_global_outputs.find(static_cast<uint32_t>(result_idx));
        const auto pipeline_output_idx = mapping_it == std::end(m_npuw_model->m_pipeline_global_outputs)
                                             ? submodel_output_idx
                                             : static_cast<std::size_t>(mapping_it->second);
        request->set_tensor(m_npuw_model->m_compiled_pipeline_model->outputs().at(pipeline_output_idx),
                            get_tensor(m_npuw_model->outputs().at(result_idx)));
    }
}

void ov::npuw::PipelinedInferRequest::alloc_quant_gather() {
    for (std::size_t idx = 0; idx < m_num_submodels; ++idx) {
        auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
        auto& quant_unpack_gather = comp_model_desc.quant_unpack_gather;
        if (quant_unpack_gather.dst_idx == -1 || !comp_model_desc.replaced_by) {
            continue;
        }

        auto get_port = [&](int64_t port_index) {
            auto port = get_pipeline_parameter_port(idx, static_cast<std::size_t>(port_index));
            OPENVINO_ASSERT(port, "Pipeline quantized-gather port is missing");
            return *port;
        };

        const auto& lookup = m_pipeline_request->get_tensor(get_port(quant_unpack_gather.idx_idx));
        const auto& vocabw = m_pipeline_request->get_tensor(get_port(quant_unpack_gather.src_w_idx));
        const auto ids_shape = lookup->get_shape();
        const auto get_gathered_shape = [&ids_shape](const ov::Shape& shape) {
            return ov::Shape{1, ids_shape[1], shape.size() == 3 ? shape[1] * shape[2] : shape[1]};
        };

        m_quant_gather_tensors.w = ov::Tensor(vocabw->get_element_type(), get_gathered_shape(vocabw->get_shape()));
        if (quant_unpack_gather.src_z_idx != -1 && quant_unpack_gather.src_s_idx != -1) {
            const auto& vocabz = m_pipeline_request->get_tensor(get_port(quant_unpack_gather.src_z_idx));
            const auto& vocabs = m_pipeline_request->get_tensor(get_port(quant_unpack_gather.src_s_idx));
            m_quant_gather_tensors.z = ov::Tensor(vocabz->get_element_type(), get_gathered_shape(vocabz->get_shape()));
            m_quant_gather_tensors.s = ov::Tensor(vocabs->get_element_type(), get_gathered_shape(vocabs->get_shape()));
        } else if (quant_unpack_gather.src_s_idx != -1) {
            const auto& vocabs = m_pipeline_request->get_tensor(get_port(quant_unpack_gather.src_s_idx));
            m_quant_gather_tensors.s = ov::Tensor(vocabs->get_element_type(), get_gathered_shape(vocabs->get_shape()));
        }
    }
}

void ov::npuw::PipelinedInferRequest::handle_quant_host_gather(std::size_t idx, RqPtr request) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    auto& quant_unpack_gather = comp_model_desc.quant_unpack_gather;
    if (quant_unpack_gather.dst_idx == -1 || !comp_model_desc.replaced_by) {
        return;
    }

    auto get_port = [&](int64_t port_index) {
        auto port = get_pipeline_parameter_port(idx, static_cast<std::size_t>(port_index));
        OPENVINO_ASSERT(port, "Pipeline quantized-gather port is missing");
        return *port;
    };

    const auto& lookup = request->get_tensor(get_port(quant_unpack_gather.idx_idx));
    const auto& gather = request->get_tensor(get_port(quant_unpack_gather.dst_idx));
    const auto& vocabw = request->get_tensor(get_port(quant_unpack_gather.src_w_idx));
    ov::npuw::util::gather(vocabw, lookup, ov::get_tensor_impl(m_quant_gather_tensors.w));

    if (quant_unpack_gather.src_z_idx != -1 && quant_unpack_gather.src_s_idx != -1) {
        const auto& vocabz = request->get_tensor(get_port(quant_unpack_gather.src_z_idx));
        const auto& vocabs = request->get_tensor(get_port(quant_unpack_gather.src_s_idx));
        ov::npuw::util::gather(vocabz, lookup, ov::get_tensor_impl(m_quant_gather_tensors.z));
        ov::npuw::util::gather(vocabs, lookup, ov::get_tensor_impl(m_quant_gather_tensors.s));
        ov::npuw::util::unpack(ov::get_tensor_impl(m_quant_gather_tensors.w),
                               ov::get_tensor_impl(m_quant_gather_tensors.z),
                               ov::get_tensor_impl(m_quant_gather_tensors.s),
                               gather);
    } else if (quant_unpack_gather.src_s_idx != -1) {
        const auto& vocabs = request->get_tensor(get_port(quant_unpack_gather.src_s_idx));
        ov::npuw::util::gather(vocabs, lookup, ov::get_tensor_impl(m_quant_gather_tensors.s));
        ov::npuw::util::unpack(ov::get_tensor_impl(m_quant_gather_tensors.w),
                               ov::get_tensor_impl(m_quant_gather_tensors.s),
                               gather);
    } else {
        NPUW_ASSERT(false && "Not supported");
    }
}

void ov::npuw::PipelinedInferRequest::update_hfa_branch_selection() {
    if (!m_npuw_model->m_pipeline_has_hfa) {
        return;
    }

    const auto& branch_ports =
        m_npuw_model->m_pipeline_global_parameters.at(m_npuw_model->m_nlp_branch_select_port_name);
    auto branch_tensor =
        m_pipeline_request->get_tensor(m_npuw_model->m_compiled_pipeline_model->inputs().at(branch_ports.front()));
    auto* selection = branch_tensor->data<uint64_t>();
    const auto iteration = m_npuw_model->get_prefill_iteration();
    if (iteration < m_npuw_model->m_nlp_controlflow_branch_select_size) {
        selection[iteration] = 1ull;
    }
    for (std::size_t i = 0; i < iteration; ++i) {
        selection[i] = 0ull;
    }
}