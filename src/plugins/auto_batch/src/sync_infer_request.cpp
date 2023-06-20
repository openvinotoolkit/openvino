// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "sync_infer_request.hpp"

namespace ov {
namespace autobatch_plugin {

using namespace InferenceEngine;

template <Precision::ePrecision precision>
Blob::Ptr create_shared_blob_on_top_of_batched_blob(Blob::Ptr batched_blob,
                                                    std::string name,
                                                    const std::set<std::string>& batched_names,
                                                    size_t batch_id,
                                                    size_t batch_num) {
    typedef typename PrecisionTrait<precision>::value_type TYPE;
    typedef typename std::add_pointer<TYPE>::type TYPEPTR;
    auto ptr = batched_blob->buffer().as<TYPEPTR>();
    auto sizePerBatch = batched_blob->size() / batch_num;
    SizeVector dims = batched_blob->getTensorDesc().getDims();
    // for performance reason (copy avoidance) current impl of the auto-batching supports only batching by 0th dim
    if (batched_names.count(name)) {
        dims[0] = 1;
        return make_shared_blob<TYPE>({precision, dims, batched_blob->getTensorDesc().getLayout()},
                                      ptr + sizePerBatch * batch_id,
                                      sizePerBatch);
    } else {
        // same blob for all requests (e.g. constants)
        return make_shared_blob<TYPE>({precision, dims, batched_blob->getTensorDesc().getLayout()}, ptr);
    }
}

AutoBatchInferRequest::AutoBatchInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                             const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                                             AutoBatchExecutableNetwork::WorkerInferRequest& workerRequest,
                                             int batch_id,
                                             int num_batch,
                                             const std::set<std::string>& batchedInputs,
                                             const std::set<std::string>& batchedOutputs)
    : IInferRequestInternal(inputs, outputs),
      _myBatchedRequestWrapper(workerRequest),
      _batchId(batch_id),
      _batchSize(num_batch) {
    ShareBlobsWithBatchRequest(batchedInputs, batchedOutputs);
}

AutoBatchInferRequest::AutoBatchInferRequest(const InputsDataMap& networkInputs,
                                             const OutputsDataMap& networkOutputs,
                                             AutoBatchExecutableNetwork::WorkerInferRequest& workerRequest,
                                             int batch_id,
                                             int num_batch,
                                             const std::set<std::string>& batchedInputs,
                                             const std::set<std::string>& batchedOutputs)
    : IInferRequestInternal(networkInputs, networkOutputs),
      _myBatchedRequestWrapper(workerRequest),
      _batchId(batch_id),
      _batchSize(num_batch) {
    ShareBlobsWithBatchRequest(batchedInputs, batchedOutputs);
}

void AutoBatchInferRequest::ShareBlobsWithBatchRequest(const std::set<std::string>& batchedInputs,
                                                       const std::set<std::string>& batchedOutputs) {
    // Allocate all input blobs
    for (const auto& it : _networkInputs) {
        auto blob = _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first);
        Blob::Ptr res;
        switch (it.second->getTensorDesc().getPrecision()) {
        case InferenceEngine::Precision::FP32:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::FP32>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::I32:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I32>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::I8:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I8>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::I16:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I16>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::U16:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U16>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::U32:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U32>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::FP64:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::FP64>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::FP16:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::FP16>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::BF16:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::BF16>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::U64:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U64>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::I64:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I64>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::U8:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U8>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::BOOL:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::BOOL>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedInputs,
                _batchId,
                _batchSize);
            break;
        default:
            IE_THROW() << "Unsupported input precision " << it.second->getTensorDesc().getPrecision();
        }
        _inputs[it.first] = res;
    }
    // Allocate all output blobs
    for (const auto& it : _networkOutputs) {
        auto blob = _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first);
        Blob::Ptr res;
        switch (it.second->getTensorDesc().getPrecision()) {
        case InferenceEngine::Precision::FP32:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::FP32>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::I32:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I32>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::I8:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I8>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::I16:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I16>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::U16:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U16>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::U32:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U32>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::FP64:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::FP64>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::FP16:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::FP16>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::BF16:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::BF16>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::U64:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U64>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::I64:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I64>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::U8:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U8>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        case InferenceEngine::Precision::BOOL:
            res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::BOOL>(
                _myBatchedRequestWrapper._inferRequestBatched->GetBlob(it.first),
                it.first,
                batchedOutputs,
                _batchId,
                _batchSize);
            break;
        default:
            IE_THROW(NotImplemented) << "Unsupported input precision " << it.second->getTensorDesc().getPrecision();
        }
        _outputs[it.first] = res;
    }
}
void AutoBatchInferRequest::SetBlobsToAnotherRequest(SoIInferRequestInternal& req) {
    for (const auto& it : _networkInputs) {
        auto& name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        auto blob = GetBlob(name);
        if (req->GetBlob(name) != blob)
            req->SetBlob(name, blob);
    }
    for (const auto& it : _networkOutputs) {
        auto& name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        auto blob = GetBlob(name);
        if (req->GetBlob(name) != blob)
            req->SetBlob(name, blob);
    }
}

void AutoBatchInferRequest::CopyInputsIfNeeded() {
    for (const auto& it : _networkInputs) {
        auto& name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        CopyBlobIfNeeded(GetBlob(name), _myBatchedRequestWrapper._inferRequestBatched->GetBlob(name), true);
    }
}

void AutoBatchInferRequest::CopyBlobIfNeeded(InferenceEngine::Blob::CPtr src,
                                             InferenceEngine::Blob::Ptr dst,
                                             bool bInput) {
    auto bufferDst = dst->buffer();
    auto ptrDst = bufferDst.as<char*>();
    auto bufferSrc = src->cbuffer();
    auto ptrSrc = bufferSrc.as<const char*>();
    ptrdiff_t szDst = dst->byteSize();
    ptrdiff_t szSrc = src->byteSize();
    if (bInput) {
        ptrdiff_t offset = szSrc != szDst ? _batchId * szDst / _batchSize : 0;
        if ((ptrDst + offset) == ptrSrc)
            return;
        else
            memcpy(ptrDst + offset, ptrSrc, szSrc);
    } else {
        ptrdiff_t offset = szSrc != szDst ? _batchId * szSrc / _batchSize : 0;
        if ((ptrSrc + offset) == ptrDst)
            return;
        else
            memcpy(ptrDst, ptrSrc + offset, szDst);
    }
}

void AutoBatchInferRequest::CopyOutputsIfNeeded() {
    for (const auto& it : _networkOutputs) {
        auto& name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        CopyBlobIfNeeded(_myBatchedRequestWrapper._inferRequestBatched->GetBlob(name), GetBlob(name), false);
    }
}
}  // namespace autobatch_plugin
}  // namespace ov