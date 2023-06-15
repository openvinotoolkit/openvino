// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "auto_batch_sync_infer_request.hpp"

#include "openvino/core/type/element_type_traits.hpp"
#include "transformations/utils/utils.hpp"

template <ov::element::Type_t precision>
ov::Tensor create_shared_blob_on_top_of_batched_blob(ov::Tensor batched_blob,
                                                     std::string name,
                                                     const std::set<std::string>& batched_names,
                                                     size_t batch_id,
                                                     size_t batch_num) {
    using T = ov::fundamental_type_for<precision>;
    auto ptr = static_cast<T*>(batched_blob.data());
    auto sizePerBatch = batched_blob.get_size() / batch_num;
    auto batched_shape = batched_blob.get_shape();
    // for performance reason (copy avoidance) current impl of the auto-batching supports only batching by 0th dim
    if (batched_names.count(name)) {
        batched_shape[0] = 1;
        return ov::Tensor(batched_blob.get_element_type(), batched_shape, ptr + sizePerBatch * batch_id);
    } else {
        // same blob for all requests (e.g. constants)
        return ov::Tensor(batched_blob.get_element_type(), batched_shape, ptr);
    }
}

ov::autobatch_plugin::SyncInferRequest::SyncInferRequest(
    const std::shared_ptr<const ov::autobatch_plugin::CompiledModel>& compiled_model,
    std::shared_ptr<ov::autobatch_plugin::CompiledModel::WorkerInferRequest> workerRequest,
    int batch_id,
    int num_batch,
    const std::set<std::string>& batchedInputs,
    const std::set<std::string>& batchedOutputs)
    : ov::ISyncInferRequest(compiled_model),
      _myBatchedRequestWrapper(workerRequest),
      _batchId(batch_id),
      _batchSize(num_batch) {
    ShareBlobsWithBatchRequest(batchedInputs, batchedOutputs);
}

void ov::autobatch_plugin::SyncInferRequest::SetBlobsToAnotherRequest(std::shared_ptr<ov::IAsyncInferRequest>& req) {
    for (const auto& it : get_inputs()) {
        // this request is already in BUSY state, so using the internal functions safely
        auto tensor = get_tensor(it);
        auto type = tensor.get_element_type();
        if (req->get_tensor(it).data(type) != tensor.data(type)) {
            req->set_tensor(it, ov::Tensor(it, tensor.data()));
        }
    }
    for (const auto& it : get_outputs()) {
        // this request is already in BUSY state, so using the internal functions safely
        auto tensor = get_tensor(it);
        auto type = tensor.get_element_type();
        if (req->get_tensor(it).data(type) != tensor.data(type)) {
            req->set_tensor(it, ov::Tensor(it, tensor.data()));
        }
    }
}

void ov::autobatch_plugin::SyncInferRequest::CopyInputsIfNeeded() {
    for (const auto& it : get_inputs()) {
        // this request is already in BUSY state, so using the internal functions safely
        auto dst_tensor = _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it);
        CopyBlobIfNeeded(get_tensor(it), dst_tensor, true);
    }
}

void ov::autobatch_plugin::SyncInferRequest::CopyOutputsIfNeeded() {
    for (const auto& it : get_outputs()) {
        // this request is already in BUSY state, so using the internal functions safely
        auto dst_tensor = get_tensor(it);
        CopyBlobIfNeeded(_myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it), dst_tensor, false);
    }
}

void ov::autobatch_plugin::SyncInferRequest::CopyBlobIfNeeded(const ov::Tensor& src, ov::Tensor& dst, bool bInput) {
    auto ptrDst = static_cast<char*>(dst.data());
    auto ptrSrc = static_cast<char*>(src.data());
    ptrdiff_t szDst = dst.get_byte_size();
    ptrdiff_t szSrc = src.get_byte_size();
    if (bInput) {
        ptrdiff_t offset = szSrc != szDst ? _batchId * szDst / _batchSize : 0;
        if ((ptrDst + offset) == ptrSrc) {
            return;
        } else
            memcpy(ptrDst + offset, ptrSrc, szSrc);
    } else {
        ptrdiff_t offset = szSrc != szDst ? _batchId * szSrc / _batchSize : 0;
        if ((ptrSrc + offset) == ptrDst) {
            return;
        } else
            memcpy(ptrDst, ptrSrc + offset, szDst);
    }
}

void ov::autobatch_plugin::SyncInferRequest::infer() {
    OPENVINO_NOT_IMPLEMENTED;
}

std::vector<std::shared_ptr<ov::IVariableState>> ov::autobatch_plugin::SyncInferRequest::query_state() const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::vector<ov::ProfilingInfo> ov::autobatch_plugin::SyncInferRequest::get_profiling_info() const {
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::autobatch_plugin::SyncInferRequest::ShareBlobsWithBatchRequest(const std::set<std::string>& batchedInputs,
                                                                        const std::set<std::string>& batchedOutputs) {
    for (const auto& it : get_inputs()) {
        auto blob = _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it);
        ov::Tensor res;
        switch (it.get_tensor_ptr()->get_element_type()) {
        case ov::element::f32:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::f32>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                ov::op::util::get_ie_output_name(it),
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::i32:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::i32>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                ov::op::util::get_ie_output_name(it),
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::i8:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::i8>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                ov::op::util::get_ie_output_name(it),
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::i16:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::i16>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                ov::op::util::get_ie_output_name(it),
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::u16:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::u16>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                ov::op::util::get_ie_output_name(it),
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::u32:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::u32>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                ov::op::util::get_ie_output_name(it),
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::f64:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::f64>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                ov::op::util::get_ie_output_name(it),
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::f16:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::f16>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                ov::op::util::get_ie_output_name(it),
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::bf16:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::bf16>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                ov::op::util::get_ie_output_name(it),
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::u64:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::u64>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                ov::op::util::get_ie_output_name(it),
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::i64:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::i64>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                ov::op::util::get_ie_output_name(it),
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::u8:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::u8>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                ov::op::util::get_ie_output_name(it),
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::boolean:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::boolean>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                ov::op::util::get_ie_output_name(it),
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        default:
            OPENVINO_THROW("Unsupported input precision ", it.get_tensor_ptr()->get_element_type());
        }
        set_tensor(it, res);
    }

    for (const auto& it : get_outputs()) {
        auto blob = _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it);
        auto name = ov::op::util::get_ie_output_name(it.get_node_shared_ptr()->input_value(0));
        ov::Tensor res;
        switch (it.get_tensor_ptr()->get_element_type()) {
        case ov::element::f32:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::f32>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                name,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::i32:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::i32>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                name,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::i8:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::i8>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                name,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::i16:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::i16>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                name,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::u16:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::u16>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                name,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::u32:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::u32>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                name,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::f64:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::f64>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                name,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::f16:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::f16>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                name,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::bf16:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::bf16>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                name,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::u64:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::u64>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                name,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::i64:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::i64>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                name,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::u8:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::u8>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                name,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case ov::element::boolean:
            res = create_shared_blob_on_top_of_batched_blob<ov::element::boolean>(
                _myBatchedRequestWrapper->_inferRequestBatched->get_tensor(it),
                name,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        default:
            OPENVINO_THROW("Unsupported input precision ", it.get_tensor_ptr()->get_element_type());
        }
        set_tensor(it, res);
    }
}