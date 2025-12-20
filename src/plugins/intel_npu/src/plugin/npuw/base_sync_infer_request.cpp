// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base_sync_infer_request.hpp"

#include "compiled_model.hpp"
#include "infer_request_utils.hpp"  // to utilize copy_tensor_by_dim
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/utils/zero/zero_host_tensor.hpp"
#include "intel_npu/utils/zero/zero_remote_tensor.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
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

    // Initialize profiling
    m_profile.report_on_die = ov::npuw::profiling_enabled();
    m_profile.area = m_npuw_model->m_name + "/performance";

    m_footprint.report_on_die = ov::npuw::profiling_enabled();
    m_footprint.area = m_npuw_model->m_name + "/memory";
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
                     << id << "] has been already created on " << "it .");
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
    std::unique_lock lock(m_io_storages_mutex);

    if (is_stored(port)) {
        return m_port_to_tensor.at(port).tensor;
    }

    // I/O: allocate here on demand (to reduce memory consumption in case some I/O were shared)
    // Input
    for (std::size_t i = 0; i < m_npuw_model->inputs().size(); ++i) {
        if (m_npuw_model->inputs()[i] == port) {
            ov::SoPtr<ov::ITensor> allocated = allocOut(port, global_input_mem_device(i));
            m_input_allocated.insert(allocated->data());
            m_port_to_tensor[port] = TensorStorage{allocated, true};
            return m_port_to_tensor.at(port).tensor;
        }
    }

    // Output
    for (size_t i = 0; i < m_npuw_model->outputs().size(); i++) {
        if (m_npuw_model->outputs()[i] == port) {
            auto tensor = alloc_global_out(i);
            m_port_to_tensor[port] = TensorStorage{tensor, true};
            return m_port_to_tensor.at(port).tensor;
        }
    }

    NPUW_ASSERT(false);
    return {};
}

void ov::npuw::IBaseInferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                             const ov::SoPtr<ov::ITensor>& tensor) {
    std::unique_lock lock(m_io_storages_mutex);

    if (!is_stored(port)) {
        // TODO: might be useful to check if the tensor is allocated on the device
        m_port_to_tensor[port] = TensorStorage{tensor, true};
    } else {
        m_port_to_tensor.at(port).tensor = tensor;
    }

    // Check if setting input tensor
    if (m_port_to_tensor.at(port).persistent) {
        handle_set_remote_input(port, tensor);
    }
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

bool ov::npuw::IBaseInferRequest::is_stored(const ov::Output<const ov::Node>& port) const {
    return m_port_to_tensor.find(port) != m_port_to_tensor.end();
}

void ov::npuw::IBaseInferRequest::handle_set_remote_input(const ov::Output<const ov::Node>& port,
                                                          const ov::SoPtr<ov::ITensor>& tensor) {
    for (std::size_t i = 0; i < m_npuw_model->inputs().size(); ++i) {
        if (m_npuw_model->inputs()[i] == port) {
            // This is a tricky case:
            // 1) We already stored an input tensor ptr in m_input_allocated via FMM
            // 2) We got an input tensor from outside
            // Later in runtime we rely on m_input_allocated to check if the memory is
            // allocated internally to prevent the copy. Here we need to check if the memory
            // is properly allocated externally, to prevent runtime copy as well.
            // Also we can get a strided remote tensor. In this case the copy cannot be avoided for now.
            if (m_npuw_model->global_mem_device() == "NPU") {
                auto remote_ctx =
                    m_npuw_model->get_plugin()->get_core()->get_default_context(m_npuw_model->global_mem_device())._ptr;
                auto zrh = remote_ctx->get_property().at(ov::intel_npu::l0_context.name());
                if (::intel_npu::zeroUtils::get_l0_context_memory_allocation_id(
                        static_cast<ze_context_handle_t>(zrh.as<void*>()),
                        tensor->data()) > 0) {
                    if (tensor->is_continuous()) {
                        // Note: no need for locking as it's internal method that should
                        // only be called from set_tensor()
                        m_input_allocated.insert(tensor->data());
                    } else {
                        LOG_WARN("Strided remote tensor is not supported on the device! Expect worse performance due "
                                 "to CPU runtime copy.");
                    }
                }
            }
        }
    }
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

std::string ov::npuw::IBaseInferRequest::profile_tag(std::size_t idx) const {
    // So far accumulate over devices involved
    const auto& proto_comp_model_desc = m_npuw_model->m_compiled_submodels[real(idx)];
    return *proto_comp_model_desc.device_it;
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
        std::cout << "Running subrequest[" << idx << "] on device " << profile_tag(idx) << "...\n";
        m_profile[profile_tag(idx)].record([&]() {
            run_subrequest_for_success(idx, failover);
        });
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
                                                          const std::string& device) const {
    auto ptr = ov::npuw::util::allocMem(type, shape, device, m_npuw_model->get_plugin());
    m_footprint[device] += ptr->get_byte_size();
    return ptr;
}

ov::npuw::TensorPtr ov::npuw::IBaseInferRequest::allocOut(const ov::Output<const ov::Node>& node,
                                                          const std::string& device) const {
    return allocMem(node.get_element_type(), node.get_shape(), device);
}

std::string ov::npuw::IBaseInferRequest::global_input_mem_device(std::size_t idx) const {
    // Use the consumer subgraph device if it is alone;
    // resort to global if there's many
    if (!m_npuw_model->m_param_subscribers[idx].empty()) {
        // There's subscribers, so resort to global
        return m_npuw_model->global_mem_device();
    }

    const auto& to_submodel = m_npuw_model->m_inputs_to_submodels_inputs.at(idx);
    if (to_submodel != CompiledModel::NO_LINK) {
        const auto& proto_comp_model_desc = m_npuw_model->m_compiled_submodels[real(to_submodel.first)];
        return *proto_comp_model_desc.device_it;
    }

    // Resort to global again
    return m_npuw_model->global_mem_device();
}

std::string ov::npuw::IBaseInferRequest::global_output_mem_device(std::size_t idx) const {
    // Pick the affinitiy based on the producer subgraph
    const auto& from_submodel = m_npuw_model->m_outputs_to_submodels_outputs.at(idx);
    const auto& proto_comp_model_desc = m_npuw_model->m_compiled_submodels[real(from_submodel.first)];
    return *proto_comp_model_desc.device_it;
}

void ov::npuw::IBaseInferRequest::alloc_quant_gather() {
    // Try to allocate intermediate tensors to gather into, when host quant gather is enabled
    for (size_t i = 0; i < m_num_submodels; i++) {
        auto& comp_model_desc = m_npuw_model->m_compiled_submodels[i];
        if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
            continue;  // Optimized out
        }
        alloc_quant_gather_tensors(i, m_subrequests[i]);
    }
}

ov::npuw::TensorPtr ov::npuw::IBaseInferRequest::alloc_global_out(std::size_t out_idx) const {
    const auto& port = m_npuw_model->outputs().at(out_idx);
    return allocOut(port, global_output_mem_device(out_idx));
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

    // Skip MoE expert submodels - they require special handling via unpack_moe_expert_closure()
    if (func_desc.moe_experts.has_value()) {
        return;
    }

    // Bind extra parameters from the function's closure
    // First, do easy things & delay heavy stuff
    std::vector<std::size_t> closure_unpack_required;
    std::vector<std::size_t> closure_copy_required;

    auto& desc_closure = comp_model_desc.closure.get().closure;

    for (std::size_t cidx = 0u; cidx < desc_closure.size(); cidx++) {
        auto& closure = desc_closure[cidx];
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
        auto& closure = desc_closure[cidx];
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
        auto& closure = desc_closure[cidx];

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

ov::Tensor ov::npuw::IBaseInferRequest::slice_expert_weight(const ov::Tensor& batched_weight,
                                                            size_t expert_id,
                                                            size_t num_experts) {
    // Slice weight tensor from batched (num_experts, ...) to single expert (1, ...)
    auto shape = batched_weight.get_shape();

    if (shape.empty() || shape[0] != num_experts) {
        LOG_WARN("Invalid batched weight shape for expert slicing: " << shape);
        return batched_weight;  // Return original if not batched
    }

    // Calculate the slice
    ov::Shape single_expert_shape = shape;
    single_expert_shape[0] = 1;

    // Create new tensor for single expert
    ov::Tensor sliced_weight(batched_weight.get_element_type(), single_expert_shape);

    // Calculate byte offset for the expert_id
    size_t expert_size = batched_weight.get_byte_size() / num_experts;
    size_t offset = expert_id * expert_size;

    // Copy data for this expert
    const uint8_t* src = static_cast<const uint8_t*>(batched_weight.data()) + offset;
    uint8_t* dst = static_cast<uint8_t*>(sliced_weight.data());
    std::memcpy(dst, src, expert_size);

    LOG_DEBUG("Sliced expert " << expert_id << " weight: " << shape << " -> " << single_expert_shape);

    return sliced_weight;
}

ov::Tensor ov::npuw::IBaseInferRequest::slice_batch_expert_weights(const ov::Tensor& batched_weight,
                                                                   const std::vector<size_t>& expert_ids,
                                                                   size_t num_experts) const {
    // Extract multiple experts' weights and concatenate them
    // Input: batched_weight shape [num_experts, ...remaining_dims]
    // Output: sliced_weight shape [K, ...remaining_dims] where K = expert_ids.size()

    auto shape = batched_weight.get_shape();
    if (shape.empty() || shape[0] != num_experts) {
        OPENVINO_THROW("Invalid batched weight shape for expert slicing");
    }

    size_t K = expert_ids.size();
    auto elem_type = batched_weight.get_element_type();

    // Calculate dimensions
    size_t expert_slice_size = 1;
    for (size_t i = 1; i < shape.size(); ++i) {
        expert_slice_size *= shape[i];
    }

    // Create output shape: [K, ...remaining_dims]
    ov::Shape output_shape = shape;
    output_shape[0] = K;

    // Allocate tensor through allocMem for proper memory management
    auto sliced_batch_ptr = allocMem(elem_type, output_shape, "NPU");
    auto sliced_batch = ov::make_tensor(sliced_batch_ptr);

    // Copy each expert's data
    if (elem_type == ov::element::f32) {
        const float* src = batched_weight.data<float>();
        float* dst = sliced_batch.data<float>();
        for (size_t i = 0; i < K; ++i) {
            size_t expert_id = expert_ids[i];
            const float* expert_src = src + expert_id * expert_slice_size;
            float* expert_dst = dst + i * expert_slice_size;
            std::memcpy(expert_dst, expert_src, expert_slice_size * sizeof(float));
        }
    } else if (elem_type == ov::element::f16) {
        const ov::float16* src = batched_weight.data<ov::float16>();
        ov::float16* dst = sliced_batch.data<ov::float16>();
        for (size_t i = 0; i < K; ++i) {
            size_t expert_id = expert_ids[i];
            const ov::float16* expert_src = src + expert_id * expert_slice_size;
            ov::float16* expert_dst = dst + i * expert_slice_size;
            std::memcpy(expert_dst, expert_src, expert_slice_size * sizeof(ov::float16));
        }
    } else if (elem_type == ov::element::i8) {
        const int8_t* src = batched_weight.data<int8_t>();
        int8_t* dst = sliced_batch.data<int8_t>();
        for (size_t i = 0; i < K; ++i) {
            size_t expert_id = expert_ids[i];
            const int8_t* expert_src = src + expert_id * expert_slice_size;
            int8_t* expert_dst = dst + i * expert_slice_size;
            std::memcpy(expert_dst, expert_src, expert_slice_size * sizeof(int8_t));
        }
    } else if (elem_type == ov::element::u8) {
        const uint8_t* src = batched_weight.data<uint8_t>();
        uint8_t* dst = sliced_batch.data<uint8_t>();
        for (size_t i = 0; i < K; ++i) {
            size_t expert_id = expert_ids[i];
            const uint8_t* expert_src = src + expert_id * expert_slice_size;
            uint8_t* expert_dst = dst + i * expert_slice_size;
            std::memcpy(expert_dst, expert_src, expert_slice_size * sizeof(uint8_t));
        }
    } else if (elem_type == ov::element::nf4 || elem_type == ov::element::u4 || elem_type == ov::element::i4) {
        // Handle 4-bit types (nf4, u4, i4) - copy at byte level
        // Note: 4-bit types are packed, so we calculate byte size
        size_t expert_byte_size = batched_weight.get_byte_size() / num_experts;
        const uint8_t* src = static_cast<const uint8_t*>(batched_weight.data());
        uint8_t* dst = static_cast<uint8_t*>(sliced_batch.data());
        for (size_t i = 0; i < K; ++i) {
            size_t expert_id = expert_ids[i];
            const uint8_t* expert_src = src + expert_id * expert_byte_size;
            uint8_t* expert_dst = dst + i * expert_byte_size;
            std::memcpy(expert_dst, expert_src, expert_byte_size);
        }
    } else {
        OPENVINO_THROW("Unsupported element type for batch expert weight slicing: ", elem_type);
    }

    return sliced_batch;
}

std::vector<size_t> ov::npuw::IBaseInferRequest::parse_selected_experts_from_router(
    const ov::SoPtr<ov::ITensor>& router_output,
    size_t num_experts,
    std::map<size_t, std::vector<size_t>>& token_to_experts) {
    std::set<size_t> selected_experts_set;

    if (!router_output) {
        LOG_WARN("Router output is null, selecting all experts");
        std::vector<size_t> all_experts;
        for (size_t i = 0; i < num_experts; ++i) {
            all_experts.push_back(i);
        }
        return all_experts;
    }

    auto shape = router_output->get_shape();
    auto elem_type = router_output->get_element_type();

    // Router output shape: [num_experts, 1, token_num, 1]
    // We need to parse which expert each token selects based on non-zero weights

    if (shape.size() != 4 || shape[0] != num_experts || shape[1] != 1 || shape[3] != 1) {
        LOG_WARN("Unexpected router output shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << ", "
                                                     << shape[3] << "], expected [" << num_experts
                                                     << ", 1, token_num, 1]");
    }

    size_t num_tokens = shape[2];  // token_num from shape

    auto parse_experts = [&](auto* data) {
        // For each token, find which experts have non-zero weights
        for (size_t token_id = 0; token_id < num_tokens; ++token_id) {
            for (size_t expert_id = 0; expert_id < num_experts; ++expert_id) {
                // Index calculation for shape [num_experts, 1, token_num, 1]
                // data[expert_id, 0, token_id, 0]
                size_t idx = expert_id * num_tokens + token_id;

                float value = std::abs(static_cast<float>(data[idx]));
                if (value > 1e-6f) {
                    // This token selected this expert
                    token_to_experts[token_id].push_back(expert_id);
                    selected_experts_set.insert(expert_id);
                }
            }
        }
    };

    if (elem_type == ov::element::f32) {
        parse_experts(router_output->data<float>());
    } else if (elem_type == ov::element::f16) {
        parse_experts(router_output->data<ov::float16>());
    } else {
        LOG_WARN("Unsupported router output element type: " << elem_type << ", selecting all experts");
        std::vector<size_t> all_experts;
        for (size_t i = 0; i < num_experts; ++i) {
            all_experts.push_back(i);
        }
        return all_experts;
    }

    // Convert set to vector (sorted order)
    std::vector<size_t> selected_experts(selected_experts_set.begin(), selected_experts_set.end());
    return selected_experts;
}

void ov::npuw::IBaseInferRequest::relayout_single_expert_output(
    size_t expert_id,
    const ov::SoPtr<ov::ITensor>& expert_output,
    const ov::SoPtr<ov::ITensor>& target_tensor,
    const std::map<size_t, std::vector<size_t>>& token_to_experts,
    size_t num_tokens,
    size_t embed_dim) {
    // Get expert output shape and validate
    auto shape = expert_output->get_shape();
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 1 || shape[2] != num_tokens || shape[3] != embed_dim) {
        LOG_WARN("Expert " << expert_id << " has unexpected output shape: " << shape);
    }

    auto elem_type = target_tensor->get_element_type();

    // Process each token that selected this expert
    for (auto& [token_id, expert_ids] : token_to_experts) {
        // Check if this token selected the current expert and get its slot index
        auto it = std::find(expert_ids.begin(), expert_ids.end(), expert_id);
        if (it == expert_ids.end()) {
            continue;  // This token didn't select this expert
        }

        if (token_id >= num_tokens) {
            LOG_WARN("Token ID " << token_id << " exceeds num_tokens " << num_tokens);
            continue;
        }

        // Calculate expert_slot: position of this expert in the token's expert list
        // This ensures each token's experts are placed in slots [0, 1, 2, 3]
        size_t expert_slot = std::distance(expert_ids.begin(), it);

        // Copy this expert's output for this token to the target tensor
        // Source: expert_output[0, 0, token_id, :]
        // Target: target_tensor[expert_slot, 0, token_id, :]

        if (elem_type == ov::element::f32) {
            const float* src = expert_output->data<float>() + token_id * embed_dim;
            float* dst = target_tensor->data<float>() + (expert_slot * num_tokens * embed_dim + token_id * embed_dim);
            std::memcpy(dst, src, embed_dim * sizeof(float));

        } else if (elem_type == ov::element::f16) {
            const ov::float16* src = expert_output->data<ov::float16>() + token_id * embed_dim;
            ov::float16* dst =
                target_tensor->data<ov::float16>() + (expert_slot * num_tokens * embed_dim + token_id * embed_dim);
            std::memcpy(dst, src, embed_dim * sizeof(ov::float16));

        } else {
            LOG_ERROR("Unsupported element type for MoE output relayout: " << elem_type);
            OPENVINO_THROW("MoE: Unsupported element type for output relayout");
        }
    }
}

void ov::npuw::IBaseInferRequest::unpack_moe_expert_closure(std::size_t idx, RqPtr request, size_t expert_id) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    NPUW_ASSERT(comp_model_desc.replaced_by);

    const auto real_idx = comp_model_desc.replaced_by.value();
    auto& func_desc = m_npuw_model->m_compiled_submodels[real_idx];

    NPUW_ASSERT(func_desc.moe_experts.has_value());
    const auto num_experts = func_desc.moe_experts->num_experts;

    auto& desc_closure = comp_model_desc.closure.get().closure;

    for (std::size_t cidx = 0u; cidx < desc_closure.size(); cidx++) {
        auto& closure = desc_closure[cidx];
        const auto closure_param_id = comp_model_desc.param_base + cidx;

        if (m_npuw_model->is_gather_closure(idx, cidx)) {
            continue;  // Skip gather closures
        }

        auto& iport = func_desc.compiled_model->inputs()[closure_param_id];

        // Check if this weight needs slicing (has num_experts in first dimension)
        auto closure_shape = closure.get_shape();
        bool needs_slicing = !closure_shape.empty() && closure_shape[0] == num_experts;

        if (needs_slicing) {
            // Check cache first
            auto cache_key = cidx;
            if (m_moe_io[idx].expert_weights_cache.find(cache_key) == m_moe_io[idx].expert_weights_cache.end()) {
                // Not in cache, slice it
                ov::Tensor sliced = slice_expert_weight(closure, expert_id, num_experts);
                m_moe_io[idx].expert_weights_cache[cache_key] = sliced;
            }

            auto& sliced_weight = m_moe_io[idx].expert_weights_cache[cache_key];

            // Handle unpacking if needed
            if (m_npuw_model->unpack_required(idx, cidx)) {
                auto clparam = request->get_tensor(iport);

                if (!comp_model_desc.scales.empty() && comp_model_desc.scales[cidx] && comp_model_desc.zerops[cidx]) {
                    // TODO: May need to slice scales/zerops as well if they're batched
                    ov::npuw::util::unpack(ov::get_tensor_impl(sliced_weight),
                                           ov::get_tensor_impl(comp_model_desc.zerops[cidx]),
                                           ov::get_tensor_impl(comp_model_desc.scales[cidx]),
                                           clparam);
                } else if (!comp_model_desc.scales.empty() && comp_model_desc.scales[cidx]) {
                    ov::npuw::util::unpack(ov::get_tensor_impl(sliced_weight),
                                           ov::get_tensor_impl(comp_model_desc.scales[cidx]),
                                           clparam);
                } else {
                    ov::npuw::util::unpack(ov::get_tensor_impl(sliced_weight), clparam);
                }
            } else {
                // Direct set (no unpacking needed)
                request->set_tensor(iport, ov::get_tensor_impl(sliced_weight));
            }
        } else {
            // This closure parameter doesn't need slicing, use original logic
            if (needs_copy(idx, cidx)) {
                auto clparam = request->get_tensor(iport);
                ov::get_tensor_impl(closure)->copy_to(clparam._ptr);
            } else {
                request->set_tensor(iport, ov::get_tensor_impl(closure));
            }
        }
    }
}

void ov::npuw::IBaseInferRequest::unpack_moe_batch_expert_closure(std::size_t idx,
                                                                  RqPtr request,
                                                                  const std::vector<size_t>& expert_ids) {
    // Unpack multiple experts' closures at once for batch inference (decoding mode)
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    NPUW_ASSERT(comp_model_desc.replaced_by);

    const auto real_idx = comp_model_desc.replaced_by.value();
    auto& func_desc = m_npuw_model->m_compiled_submodels[real_idx];

    NPUW_ASSERT(func_desc.moe_experts.has_value());
    const auto num_experts = func_desc.moe_experts->num_experts;
    const size_t K = expert_ids.size();

    auto& desc_closure = comp_model_desc.closure.get().closure;

    for (std::size_t cidx = 0u; cidx < desc_closure.size(); cidx++) {
        auto& closure = desc_closure[cidx];
        const auto closure_param_id = comp_model_desc.param_base + cidx;

        if (m_npuw_model->is_gather_closure(idx, cidx)) {
            continue;  // Skip gather closures
        }

        auto& iport = func_desc.compiled_model->inputs()[closure_param_id];

        // Check if this weight needs slicing (has num_experts in first dimension)
        auto closure_shape = closure.get_shape();
        bool needs_slicing = !closure_shape.empty() && closure_shape[0] == num_experts;

        if (needs_slicing) {
            // Slice K experts' weights at once
            ov::Tensor sliced_batch = slice_batch_expert_weights(closure, expert_ids, num_experts);

            // Handle unpacking if needed
            if (m_npuw_model->unpack_required(idx, cidx)) {
                auto clparam = request->get_tensor(iport);

                if (!comp_model_desc.scales.empty() && comp_model_desc.scales[cidx] && comp_model_desc.zerops[cidx]) {
                    // TODO: May need to handle batched scales/zerops
                    ov::npuw::util::unpack(ov::get_tensor_impl(sliced_batch),
                                           ov::get_tensor_impl(comp_model_desc.zerops[cidx]),
                                           ov::get_tensor_impl(comp_model_desc.scales[cidx]),
                                           clparam);
                } else if (!comp_model_desc.scales.empty() && comp_model_desc.scales[cidx]) {
                    ov::npuw::util::unpack(ov::get_tensor_impl(sliced_batch),
                                           ov::get_tensor_impl(comp_model_desc.scales[cidx]),
                                           clparam);
                } else {
                    ov::npuw::util::unpack(ov::get_tensor_impl(sliced_batch), clparam);
                }
            } else {
                // Direct set (no unpacking needed)
                request->set_tensor(iport, ov::get_tensor_impl(sliced_batch));
            }
        } else {
            // This closure parameter doesn't need slicing, use original logic
            if (needs_copy(idx, cidx)) {
                auto clparam = request->get_tensor(iport);
                ov::get_tensor_impl(closure)->copy_to(clparam._ptr);
            } else {
                request->set_tensor(iport, ov::get_tensor_impl(closure));
            }
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
    const bool is_attention = proto_comp_model_desc.attention.has_value();
    const bool is_pyramid_attention = proto_comp_model_desc.pyramid_attention.has_value();
    const bool is_moe = proto_comp_model_desc.moe_experts.has_value();

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

    // Check if the given subgraph's input is dynamic
    auto is_attn_param = [&](std::size_t sub_in_idx) -> bool {
        if (!is_attention) {
            return false;  // Early return
        }
        auto& attn = proto_comp_model_desc.attention.value();
        return std::any_of(attn.params.begin(), attn.params.end(), [&](const auto& p) -> bool {
            return p.idx == sub_in_idx;
        });
    };

    auto is_pyramid_attn_param = [&](std::size_t sub_in_idx) -> bool {
        if (!is_pyramid_attention) {
            return false;  // Early return
        }

        auto pyramid_id = m_pyramid_selector->pyramid_id();
        auto& pyramid_attn = proto_comp_model_desc.pyramid_attention.value()._attention_infos[pyramid_id];
        return std::any_of(pyramid_attn.params.begin(), pyramid_attn.params.end(), [&](const auto& p) -> bool {
            return p.idx == sub_in_idx;
        });
    };

    for (auto&& it : iodesc.global_params) {
        std::size_t param_idx{}, sub_in_idx{};
        std::tie(param_idx, sub_in_idx) = it;
        LOG_DEBUG("Processing " << param_idx << " -> " << sub_in_idx << std::endl);

        const auto& g_port = m_npuw_model->inputs()[param_idx];
        const auto& g_tnsr = get_tensor(g_port);
        const auto& s_port = request->get_inputs()[sub_in_idx];
        LOG_DEBUG("Processing " << g_port << " -> " << s_port << "...");
        LOG_BLOCK();
        if (is_spatial_param(sub_in_idx)) {
            // Register for future use
            // FIXME: Not sure why this code is here. There should be no
            // spatial global parameters, as this execution mode iterates over
            // the iterations only.
            // Also, it pretty much looks like _io[] should be taken at
            // idx but not real_idx, as referring to real_idx breaks the
            // function pipelining
            NPUW_ASSERT(false && "Global parameter can't be spatial");
            m_spatial_io[real_idx].inputs.at(sub_in_idx) = g_tnsr;
        } else if (is_attn_param(sub_in_idx) || is_pyramid_attn_param(sub_in_idx)) {
            // Register for future use
            m_attention_io[idx].inputs.at(sub_in_idx) = g_tnsr;
        } else if (is_moe) {
            // Register MoE input for future use - will be processed in function_prologue
            LOG_DEBUG("Registering MoE global param " << param_idx << " -> " << sub_in_idx);
            m_moe_io[idx].inputs.at(sub_in_idx) = g_tnsr;
        } else {
            // Lock mutex just in case. m_input_allocated might be altered in parallel in get_tensor()
            std::unique_lock lock(m_io_storages_mutex);
            // Input parameter is non-spatial, do normal handling
            if (m_input_allocated.count(g_tnsr->data()) == 0 && do_copy) {
                LOG_DEBUG("Will be copied");
                copy_list.emplace_back(g_tnsr, s_port);
            } else {
                LOG_DEBUG("Will be set");
                request->set_tensor(s_port, g_tnsr);
            }
        }
    }  // for(global_params)

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

        const auto& vocab =
            comp_model_desc.closure.get().closure[comp_model_desc.host_gather.src_idx - comp_model_desc.param_base];
        const auto& lport = comp_model_desc.compiled_model->inputs()[comp_model_desc.host_gather.idx_idx];
        const auto lookup = request->get_tensor(lport);

        ov::npuw::util::gather(ov::get_tensor_impl(vocab), lookup, gather);
    }

    // Run host-side quantized gather, if required
    handle_quant_host_gather(idx, request);

    // Handle attention inputs, if required
    m_profile["attn(io)"].record([&]() {
        bind_attention_inputs(idx, request);
    });

    // Handle pyramid attention inputs, if required
    m_profile["attn(io)"].record([&]() {
        bind_pyramid_attention_inputs(idx, request);
    });

    LOG_DEBUG("Done");
}

void ov::npuw::IBaseInferRequest::alloc_quant_gather_tensors(std::size_t idx, RqPtr request) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    auto& quant_unpack_gather = comp_model_desc.quant_unpack_gather;

    if (quant_unpack_gather.dst_idx != -1) {
        NPUW_ASSERT(quant_unpack_gather.idx_idx != -1 && quant_unpack_gather.src_w_idx != -1);

        const auto& lport = comp_model_desc.compiled_model->inputs()[quant_unpack_gather.idx_idx];
        const auto& lookup = request->get_tensor(lport);

        const auto& wport = comp_model_desc.compiled_model->inputs()[quant_unpack_gather.src_w_idx];
        const auto& vocabw = request->get_tensor(wport);

        auto ids_shape = lookup->get_shape();

        auto get_gathered_shape = [&ids_shape](const ov::Shape& shape) {
            return ov::Shape{1, ids_shape[1], shape.size() == 3 ? shape[1] * shape[2] : shape[1]};
        };

        m_quant_gather_tensors.w = ov::Tensor(vocabw->get_element_type(), get_gathered_shape(vocabw->get_shape()));

        if (quant_unpack_gather.src_z_idx != -1 && quant_unpack_gather.src_s_idx != -1) {
            const auto& zport = comp_model_desc.compiled_model->inputs()[quant_unpack_gather.src_z_idx];
            const auto& vocabz = request->get_tensor(zport);

            const auto& sport = comp_model_desc.compiled_model->inputs()[quant_unpack_gather.src_s_idx];
            const auto& vocabs = request->get_tensor(sport);

            m_quant_gather_tensors.z = ov::Tensor(vocabz->get_element_type(), get_gathered_shape(vocabz->get_shape()));
            m_quant_gather_tensors.s = ov::Tensor(vocabs->get_element_type(), get_gathered_shape(vocabs->get_shape()));
        } else if (quant_unpack_gather.src_s_idx != -1) {
            const auto& sport = comp_model_desc.compiled_model->inputs()[quant_unpack_gather.src_s_idx];
            const auto& vocabs = request->get_tensor(sport);

            m_quant_gather_tensors.s = ov::Tensor(vocabs->get_element_type(), get_gathered_shape(vocabs->get_shape()));
        }
    }
}

void ov::npuw::IBaseInferRequest::handle_quant_host_gather(std::size_t idx, RqPtr request) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    auto& quant_unpack_gather = comp_model_desc.quant_unpack_gather;

    if (quant_unpack_gather.dst_idx != -1) {
        NPUW_ASSERT(quant_unpack_gather.idx_idx != -1 && quant_unpack_gather.src_w_idx != -1);

        const auto& lport = comp_model_desc.compiled_model->inputs()[quant_unpack_gather.idx_idx];
        const auto& lookup = request->get_tensor(lport);

        const auto& gport = comp_model_desc.compiled_model->inputs()[quant_unpack_gather.dst_idx];
        const auto& gather = request->get_tensor(gport);

        const auto& wport = comp_model_desc.compiled_model->inputs()[quant_unpack_gather.src_w_idx];
        const auto& vocabw = request->get_tensor(wport);

        // Gather weight
        ov::npuw::util::gather(vocabw, lookup, ov::get_tensor_impl(m_quant_gather_tensors.w));

        if (quant_unpack_gather.src_z_idx != -1 && quant_unpack_gather.src_s_idx != -1) {
            const auto& zport = comp_model_desc.compiled_model->inputs()[quant_unpack_gather.src_z_idx];
            const auto& vocabz = request->get_tensor(zport);

            const auto& sport = comp_model_desc.compiled_model->inputs()[quant_unpack_gather.src_s_idx];
            const auto& vocabs = request->get_tensor(sport);

            // Gather first
            ov::npuw::util::gather(vocabz, lookup, ov::get_tensor_impl(m_quant_gather_tensors.z));
            ov::npuw::util::gather(vocabs, lookup, ov::get_tensor_impl(m_quant_gather_tensors.s));

            // Then unpack
            ov::npuw::util::unpack(ov::get_tensor_impl(m_quant_gather_tensors.w),
                                   ov::get_tensor_impl(m_quant_gather_tensors.z),
                                   ov::get_tensor_impl(m_quant_gather_tensors.s),
                                   gather);
        } else if (quant_unpack_gather.src_s_idx != -1) {
            const auto& sport = comp_model_desc.compiled_model->inputs()[quant_unpack_gather.src_s_idx];
            const auto& vocabs = request->get_tensor(sport);

            // Gather first
            ov::npuw::util::gather(vocabs, lookup, ov::get_tensor_impl(m_quant_gather_tensors.s));

            // Then unpack
            ov::npuw::util::unpack(ov::get_tensor_impl(m_quant_gather_tensors.w),
                                   ov::get_tensor_impl(m_quant_gather_tensors.s),
                                   gather);
        } else {
            NPUW_ASSERT(false && "Not supported");
        }
    }
}

void ov::npuw::IBaseInferRequest::bind_attention_inputs(std::size_t idx, RqPtr request) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[real(idx)];
    if (!comp_model_desc.attention) {
        return;
    }

    LOG_DEBUG("Binding Attention inputs...");
    LOG_BLOCK();

    const auto& dynamic = comp_model_desc.attention.value();
    auto& r = request;

    const auto pos_id = m_attention_selector->length();
    if (pos_id == -1) {
        // Dynamic range couldn't be identified - fallback to the default
        // (worst case) behavior
        for (auto&& param : dynamic.params) {
            const auto& iport = comp_model_desc.compiled_model->inputs()[param.idx];
            const auto& input = m_attention_io[idx].inputs.at(param.idx);
            r->set_tensor(iport, input);
        }
    } else {
        const auto past_len = m_attention_selector->past_length();
        const auto do_copy = needs_copy(idx) && !m_npuw_model->m_cfg.get<::intel_npu::NPUW_ATTN_NO_COPY>();

        // Set the past k/v values first
        for (auto&& param : dynamic.params) {
            const auto& iport = comp_model_desc.compiled_model->inputs()[param.idx];
            const auto& input = m_attention_io[idx].inputs.at(param.idx);
            const auto& view = ov::npuw::util::view(input, param.dim, 0, past_len);
            const auto shape = view->get_shape();

            LOG_DEBUG(iport);
            LOG_BLOCK();
            if (do_copy && ov::shape_size(shape) > 0) {
                // FIXME: Same devices that don't tolerate set_, also don't tolerate strided inputs
                const auto& dst = r->get_tensor(iport);
                const auto old_ptr = dst->data();
                dst->set_shape(shape);
                const auto new_ptr = dst->data();
                if (old_ptr != new_ptr) {
                    m_footprint[*comp_model_desc.device_it] += dst->get_byte_size();
                }
                LOG_DEBUG("Do copy: " << shape << "...");
                view->copy_to(dst._ptr);
            } else if (do_copy && ov::shape_size(shape) == 0) {
                // Special case for 0ths chunk.
                // Zero the tensor shape but not set to view
                // (a view tensor can't be extended)
                r->get_tensor(iport)->set_shape(shape);
            } else {
                r->set_tensor(iport, view);
            }
        }  // for(params)
    }

    LOG_DEBUG("Done");
}

void ov::npuw::IBaseInferRequest::bind_pyramid_attention_inputs(std::size_t idx, RqPtr request) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[real(idx)];
    if (!comp_model_desc.pyramid_attention) {
        return;
    }

    LOG_DEBUG("Binding Pyramid Attention inputs...");
    LOG_BLOCK();

    const auto pyramid_id = m_pyramid_selector->pyramid_id();
    const auto& pyramid_attention = comp_model_desc.pyramid_attention.value();
    const auto& attention_info = pyramid_attention._attention_infos[pyramid_id];
    const auto& pyramid_model = pyramid_attention._compiled_models[pyramid_id];

    const auto pos_id = m_pyramid_selector->length();
    if (pos_id == -1) {
        // Pyramid dynamic range couldn't be identified - fallback to the default
        // (worst case) behavior
        for (auto&& param : attention_info.params) {
            const auto& iport = pyramid_model->inputs()[param.idx];
            const auto& input = m_attention_io[idx].inputs.at(param.idx);
            request->set_tensor(iport, input);
        }

        return;
    }

    // Pyramid dynamic range identified
    const auto past_len = m_pyramid_selector->past_length();
    const auto infer_case = m_pyramid_selector->this_case();

    using namespace ov::npuw::runtime;

    // Process each KV parameter based on inference case
    if (infer_case == pyramid_attention::Selector::Case::PREFILL) {
        // PREFILL: Set or copy past KV to destination tensors
        for (auto&& param : attention_info.params) {
            const auto& iport = pyramid_model->inputs()[param.idx];
            const auto& input = m_attention_io[idx].inputs.at(param.idx);
            const auto& input_shape = input->get_shape();

            LOG_DEBUG(iport);
            LOG_BLOCK();

            // Optimization for the last chunk: Direct tensor reuse when shapes match
            if (static_cast<int64_t>(input_shape[param.dim]) == past_len) {
                request->set_tensor(iport, input);
                continue;
            }

            // Create view of past KV data
            const auto& view = ov::npuw::util::view(input, param.dim, 0, past_len);
            const auto& shape = view->get_shape();

            // Handle empty shape case (first chunk)
            if (ov::shape_size(shape) == 0) {
                request->get_tensor(iport)->set_shape(shape);
                continue;
            }

            // Copy past KV to full destination tensor
            LOG_DEBUG("Do copy: " << shape << "...");
            const auto& dst = request->get_tensor(iport);
            ov::npuw::util::copy_tensor_by_dim(view,
                                               dst,
                                               static_cast<uint32_t>(param.dim),
                                               static_cast<uint32_t>(param.dim));
        }
    } else if (infer_case == pyramid_attention::Selector::Case::GENERATE) {
        // GENERATE: Set or copy past KV, preserving existing data
        for (auto&& param : attention_info.params) {
            const auto& iport = pyramid_model->inputs()[param.idx];
            const auto& input = m_attention_io[idx].inputs.at(param.idx);
            const auto& input_shape = input->get_shape();

            LOG_DEBUG(iport);
            LOG_BLOCK();

            // Validation: ensure space for new tokens
            if (static_cast<int64_t>(input_shape[param.dim]) == past_len) {
                NPUW_ASSERT(false && "Past KV is full, no space for generation");
            }

            const auto& dst = request->get_tensor(iport);
            const auto& dst_shape = dst->get_shape();

            // Optimization: Direct tensor reuse when destination matches input
            if (dst_shape == input_shape) {
                request->set_tensor(iport, input);
                continue;
            }

            // FIXME: No need to copy whole past KV, just the new part

            // Create view of past KV data
            const auto& view = ov::npuw::util::view(input, param.dim, 0, past_len);

            // Copy past KV to sliced destination (preserve space for new tokens)
            LOG_DEBUG("Do copy: " << view->get_shape() << "...");
            const auto& dst_slice = ov::npuw::util::view(dst, param.dim, 0, past_len);
            ov::npuw::util::copy_tensor_by_dim(view,
                                               dst_slice,
                                               static_cast<uint32_t>(param.dim),
                                               static_cast<uint32_t>(param.dim));
        }
    } else {
        NPUW_ASSERT(false && "Unsupported pyramid attention case");
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
        request->set_tensor(s_port, get_tensor(g_port));
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
    if (comp_model_desc.closure.get().is_remote[cidx]) {
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
