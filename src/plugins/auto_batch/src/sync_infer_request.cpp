// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "sync_infer_request.hpp"

#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace autobatch_plugin {

inline ov::SoPtr<ov::ITensor> create_shared_tensor_on_batched_tensor(ov::SoPtr<ov::ITensor> batched_tensor,
                                                                     std::size_t port,
                                                                     const std::set<std::size_t>& batched_ports,
                                                                     size_t batch_id,
                                                                     size_t batch_num) {
    auto ptr = static_cast<uint8_t*>(batched_tensor->data());
    auto size_per_batch = batched_tensor->get_byte_size() / batch_num;
    auto batched_shape = batched_tensor->get_shape();
    // for performance reason (copy avoidance) current impl of the auto-batching supports only batching by 0th dim
    if (batched_ports.count(port)) {
        batched_shape[0] = 1;
        return {ov::make_tensor(batched_tensor->get_element_type(), batched_shape, ptr + size_per_batch * batch_id),
                batched_tensor._so};
    } else {
        return {ov::make_tensor(batched_tensor->get_element_type(), batched_shape, ptr), batched_tensor._so};
    }
}

SyncInferRequest::SyncInferRequest(
    const std::shared_ptr<const ov::autobatch_plugin::CompiledModel>& compiled_model,
    const std::shared_ptr<ov::autobatch_plugin::CompiledModel::WorkerInferRequest>& worker_request,
    int batch_id,
    int num_batch,
    const std::set<std::size_t>& batched_inputs,
    const std::set<std::size_t>& batched_outputs)
    : ov::ISyncInferRequest(compiled_model),
      m_batched_request_wrapper(worker_request),
      m_batch_id(batch_id),
      m_batch_size(num_batch) {
    if (m_batched_request_wrapper)
        share_tensors_with_batched_req(batched_inputs, batched_outputs);
}

size_t SyncInferRequest::get_batch_size() const {
    return m_batch_size;
}

void SyncInferRequest::share_tensors_with_batched_req(const std::set<std::size_t>& batched_inputs,
                                                      const std::set<std::size_t>& batched_outputs) {
    const auto inputs = get_inputs();
    for (size_t input_id = 0; input_id < inputs.size(); input_id++) {
        const auto& input = inputs[input_id];
        ov::SoPtr<ov::ITensor> res;
        auto batched_tensor = m_batched_request_wrapper->_infer_request_batched->get_tensor(input);
        if (!batched_tensor._so)
            batched_tensor._so = m_batched_request_wrapper->_infer_request_batched._so;
        res =
            create_shared_tensor_on_batched_tensor(batched_tensor, input_id, batched_inputs, m_batch_id, m_batch_size);
        set_tensor(input, res);
    }

    const auto& outputs = get_outputs();
    for (size_t output_id = 0; output_id < outputs.size(); output_id++) {
        const auto& output = outputs[output_id];
        ov::SoPtr<ov::ITensor> res;
        auto batched_tensor = m_batched_request_wrapper->_infer_request_batched->get_tensor(output);
        if (!batched_tensor._so)
            batched_tensor._so = m_batched_request_wrapper->_infer_request_batched._so;
        res = create_shared_tensor_on_batched_tensor(batched_tensor,
                                                     output_id,
                                                     batched_outputs,
                                                     m_batch_id,
                                                     m_batch_size);
        set_tensor(output, res);
    }
}

void SyncInferRequest::set_tensors_to_another_request(ov::SoPtr<ov::IAsyncInferRequest>& req) {
    for (const auto& it : get_inputs()) {
        // this request is already in BUSY state, so using the internal functions safely
        auto tensor = get_tensor(it);
        OPENVINO_ASSERT(tensor != nullptr, "The tensor is empty!");
        auto type = tensor->get_element_type();
        bool is_remote = std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr) ||
                         std::dynamic_pointer_cast<ov::IRemoteTensor>(req->get_tensor(it)._ptr);
        if (is_remote || req->get_tensor(it)->data(type) != tensor->data(type)) {
            req->set_tensor(it, tensor);
        }
    }
    for (const auto& it : get_outputs()) {
        // this request is already in BUSY state, so using the internal functions safely
        auto tensor = get_tensor(it);
        OPENVINO_ASSERT(tensor != nullptr, "The tensor is empty!");
        auto type = tensor->get_element_type();
        bool is_remote = std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr) ||
                         std::dynamic_pointer_cast<ov::IRemoteTensor>(req->get_tensor(it)._ptr);
        if (is_remote || req->get_tensor(it)->data(type) != tensor->data(type)) {
            req->set_tensor(it, tensor);
        }
    }
}

void SyncInferRequest::copy_inputs_if_needed() {
    for (const auto& it : get_inputs()) {
        // this request is already in BUSY state, so using the internal functions safely
        auto dst_tensor = m_batched_request_wrapper->_infer_request_batched->get_tensor(it);
        copy_tensor_if_needed(get_tensor(it), dst_tensor, true);
    }
}

void SyncInferRequest::copy_tensor_if_needed(const ov::SoPtr<ov::ITensor>& src,
                                             ov::SoPtr<ov::ITensor>& dst,
                                             const bool bInput) {
    auto ptrDst = static_cast<char*>(dst->data());
    auto ptrSrc = static_cast<char*>(src->data());
    ptrdiff_t szDst = dst->get_byte_size();
    ptrdiff_t szSrc = src->get_byte_size();
    if (bInput) {
        ptrdiff_t offset = szSrc != szDst ? m_batch_id * szDst / m_batch_size : 0;
        if ((ptrDst + offset) == ptrSrc)
            return;
        else
            memcpy(ptrDst + offset, ptrSrc, szSrc);
    } else {
        ptrdiff_t offset = szSrc != szDst ? m_batch_id * szSrc / m_batch_size : 0;
        if ((ptrSrc + offset) == ptrDst)
            return;
        else
            memcpy(ptrDst, ptrSrc + offset, szDst);
    }
}

void SyncInferRequest::copy_outputs_if_needed() {
    for (const auto& it : get_outputs()) {
        // this request is already in BUSY state, so using the internal functions safely
        auto dst_tensor = get_tensor(it);
        copy_tensor_if_needed(m_batched_request_wrapper->_infer_request_batched->get_tensor(it), dst_tensor, false);
    }
}

void SyncInferRequest::infer() {
    OPENVINO_NOT_IMPLEMENTED;
}

std::vector<ov::SoPtr<ov::IVariableState>> SyncInferRequest::query_state() const {
    auto states = m_batched_request_wrapper->_infer_request_batched->query_state();
    for (auto&& state : states) {
        if (!state._so)
            state._so = m_batched_request_wrapper->_infer_request_batched._so;
    }
    return states;
}

std::vector<ov::ProfilingInfo> SyncInferRequest::get_profiling_info() const {
    return m_batched_request_wrapper->_infer_request_batched->get_profiling_info();
}
}  // namespace autobatch_plugin
}  // namespace ov
