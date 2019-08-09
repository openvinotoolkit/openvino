// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <string>
#include <map>
#include <functional>
#include <CPP/detection_output.hpp>  // todo: find a way to remove this
#include <description_buffer.hpp>
#include "cldnn_infer_request.h"
#include "cldnn_streams_task_executor.h"

using namespace InferenceEngine;

namespace CLDNNPlugin {

std::atomic<unsigned int> CLDNNInferRequest::runningCounter(0u);

const char CLDNNInferRequest::fp32_suffix[] = "_fp32";

Blob::Ptr CLDNNInferRequest::createInputBlob(const TensorDesc& desc, uint8_t* mem_ptr) {
    const Precision p = desc.getPrecision();

    switch (p) {
    case Precision::FP32:
        if (mem_ptr != nullptr)
            return make_shared_blob<float>(desc, reinterpret_cast<float*>(mem_ptr));
        else
            return make_shared_blob<float>(desc);
    case Precision::FP16:
        if (mem_ptr != nullptr)
            return make_shared_blob<uint16_t>(desc, reinterpret_cast<uint16_t*>(mem_ptr));
        else
            return make_shared_blob<uint16_t>(desc);
    case Precision::I16:
        if (mem_ptr != nullptr)
            return make_shared_blob<int16_t>(desc, reinterpret_cast<int16_t*>(mem_ptr));
        else
            return make_shared_blob<int16_t>(desc);
    case Precision::I32:
        if (mem_ptr != nullptr)
            return make_shared_blob<int32_t>(desc, reinterpret_cast<int32_t*>(mem_ptr));
        else
            return make_shared_blob<int32_t>(desc);
    case Precision::U8:
        if (mem_ptr != nullptr)
            return make_shared_blob<uint8_t>(desc, reinterpret_cast<uint8_t*>(mem_ptr));
        else
            return make_shared_blob<uint8_t>(desc);
    default:
        THROW_IE_EXCEPTION << "The plugin does not support input " << p.name() << " precision";
    }
}

Blob::Ptr CLDNNInferRequest::createOutputBlob(const TensorDesc& desc, uint8_t* mem_ptr) {
    const Precision p = desc.getPrecision();

    switch (p) {
    case Precision::FP32:
        if (mem_ptr != nullptr)
            return make_shared_blob<float>(desc, reinterpret_cast<float*>(mem_ptr));
        else
            return make_shared_blob<float>(desc);
    case Precision::FP16:
        if (mem_ptr != nullptr)
            return make_shared_blob<uint16_t>(desc, reinterpret_cast<uint16_t*>(mem_ptr));
        else
            return make_shared_blob<uint16_t>(desc);
    case Precision::I32:
        if (mem_ptr != nullptr)
            return make_shared_blob<int32_t>(desc, reinterpret_cast<int32_t*>(mem_ptr));
        else
            return make_shared_blob<int32_t>(desc);
    default:
        THROW_IE_EXCEPTION << "The plugin does not support output " << p.name() << " precision";
    }
}

void CLDNNInferRequest::copyOutputData(const cldnn::memory& outputMemory,
                                        Blob::Ptr bptr,
                                        buf_info* bi) {
    size_t n = (bi == nullptr) ? bptr->size() : bi->buf_size;
    size_t offset = (bi == nullptr) ? 0 : bi->buf_offset;

    auto layout = outputMemory.get_layout();
    auto size = layout.size;
    auto l_padd = layout.data_padding.lower_size();
    auto u_padd = layout.data_padding.upper_size();

    auto h_padding = u_padd.spatial[0] + l_padd.spatial[0];
    auto v_padding_l = (h_padding + size.spatial[0]) * u_padd.spatial[1];
    auto v_padding_u = (h_padding + size.spatial[0]) * l_padd.spatial[1];

    switch (bptr->getTensorDesc().getPrecision()) {
    case Precision::FP32: {
        TBlob<float>::Ptr out_f = std::dynamic_pointer_cast<TBlob<float>>(bptr);
        if (out_f == nullptr) {
            THROW_IE_EXCEPTION << "Invalid output blob";
        }
        auto resPtr = outputMemory.pointer<float>();
        float *resVec = out_f->data() + offset;

        if (h_padding || v_padding_l || v_padding_u) {
            size_t i = 0;
            for (size_t b = 0; b < size.batch[0]; b++) {
                for (size_t f = 0; f < size.feature[0]; f++) {
                    i += v_padding_l;
                    for (size_t y = 0; y < size.spatial[1]; y++) {
                        i += l_padd.spatial[0];
                        for (size_t x = 0; x < size.spatial[0]; x++, i++) {
                            *resVec++ = resPtr[i];
                        }
                        i += u_padd.spatial[0];
                    }
                    i += v_padding_u;
                }
            }
        } else {
            for (size_t i = 0; i < n; i++) {
                resVec[i] = resPtr[i];
            }
        }
    }
    break;
    case Precision::FP16: {
        auto* out_f = bptr->buffer().as<uint16_t*>();
        if (out_f == nullptr) {
            THROW_IE_EXCEPTION << "Invalid output blob";
        }
        auto resPtr = outputMemory.pointer<uint16_t>();
        uint16_t* resVec = out_f + offset;

        if (h_padding || v_padding_l || v_padding_u) {
            size_t i = 0;
            for (size_t b = 0; b < size.batch[0]; b++) {
                for (size_t f = 0; f < size.feature[0]; f++) {
                    i += v_padding_l;
                    for (size_t y = 0; y < size.spatial[1]; y++) {
                        i += l_padd.spatial[0];
                        for (size_t x = 0; x < size.spatial[0]; x++, i++) {
                            *resVec++ = resPtr[i];
                        }
                        i += u_padd.spatial[0];
                    }
                    i += v_padding_u;
                }
            }
        } else {
            for (size_t i = 0; i < n; i++) {
                resVec[i] = resPtr[i];
            }
        }
    }
    break;
    case Precision::I32: {
        TBlob<int32_t>::Ptr out_f = std::dynamic_pointer_cast<TBlob<int32_t>>(bptr);
        if (out_f == nullptr) {
            THROW_IE_EXCEPTION << "Invalid output blob";
        }
        auto resPtr = outputMemory.pointer<int32_t>();
        int32_t* resVec = out_f->data() + offset;

        if (h_padding || v_padding_l || v_padding_u) {
            size_t i = 0;
            for (size_t b = 0; b < size.batch[0]; b++) {
                for (size_t f = 0; f < size.feature[0]; f++) {
                    i += v_padding_l;
                    for (size_t y = 0; y < size.spatial[1]; y++) {
                        i += l_padd.spatial[0];
                        for (size_t x = 0; x < size.spatial[0]; x++, i++) {
                            *resVec++ = resPtr[i];
                        }
                        i += u_padd.spatial[0];
                    }
                    i += v_padding_u;
                }
            }
        } else {
            for (size_t i = 0; i < n; i++) {
                resVec[i] = resPtr[i];
            }
        }
    }
    break;
    default:
        THROW_IE_EXCEPTION << "The plugin does not support output " << bptr->getTensorDesc().getPrecision() << " precision";
    }
}

void CLDNNInferRequest::copyInputData(std::shared_ptr<cldnn::network> network,
                                    const cldnn::primitive_id &inputName,
                                    const cldnn::layout& inputLayout,
                                    const Blob &inputBlob, buf_info* bi) {
    size_t n = (bi == nullptr) ? inputBlob.size() : bi->buf_size;
    size_t offset = (bi == nullptr) ? 0 : bi->buf_offset;

    cldnn::primitive_id internalName = "input:" + inputName;
    switch (inputBlob.getTensorDesc().getPrecision()) {
    case Precision::FP32: {
        float* blob_ptr = const_cast<float*>(inputBlob.cbuffer().as<const float*>()) + offset;
        network->set_input_data(internalName, cldnn::memory::attach(inputLayout, blob_ptr, n));
        break;
    }
    case Precision::I32: {
        int32_t* blob_ptr = const_cast<int32_t*>(inputBlob.cbuffer().as<const int32_t*>()) + offset;
        network->set_input_data(internalName, cldnn::memory::attach(inputLayout, blob_ptr, n));
        break;
    }
    case Precision::FP16: {
        uint16_t* blob_ptr = const_cast<uint16_t*>(inputBlob.cbuffer().as<const uint16_t*>()) + offset;
        network->set_input_data(internalName, cldnn::memory::attach(inputLayout, blob_ptr, n));
        break;
    }
    case Precision::U8: {
        uint8_t* blob_ptr = const_cast<uint8_t*>(inputBlob.cbuffer().as<const uint8_t*>()) + offset;
        network->set_input_data(internalName, cldnn::memory::attach(inputLayout, blob_ptr, n));
        break;
    }
    default:
        THROW_IE_EXCEPTION << "The plugin does not support input " << inputBlob.getTensorDesc().getPrecision() << " precision";
    }
}

void CLDNNInferRequest::AllocateInputs() {
    // allocate inputs
    for (auto &input : m_graph->GetInputLayouts()) {
        std::string name = input.first;
        cldnn::layout layout = input.second;

        InputInfo::Ptr ni = _networkInputs.at(input.first);
        const TensorDesc& desc = ni->getTensorDesc();

        cldnn::memory inputMem = cldnn::memory::allocate(*(m_graph->GetEngine()), layout);
        cldnn::pointer<uint8_t> mem_ptr = inputMem.pointer<uint8_t>();

        inputsMemory.insert({ name, inputMem });
        _inputs[name] = createInputBlob(desc, mem_ptr.data());

        if (desc.getPrecision() == Precision::I16) {
            cldnn::layout layout_fp32 = layout;
            layout_fp32.data_type = cldnn::data_types::f32;
            cldnn::memory inputMem_fp32 = cldnn::memory::allocate(*(m_graph->GetEngine()), layout_fp32);
            inputsMemory.insert({ input.first + fp32_suffix, inputMem_fp32 });
        }
    }
}

void CLDNNInferRequest::AllocateInputsDyn() {
    // allocate inputs
    for (auto &input : m_graph->GetInputLayouts()) {
        InputInfo::Ptr ni = _networkInputs.at(input.first);
        TensorDesc desc = ni->getTensorDesc();
        SizeVector& dims = desc.getDims();

        if (!dims.empty()) {
            *dims.begin() = static_cast<size_t>(m_graph->GetMaxDynamicBatchSize());
        } else {
            THROW_IE_EXCEPTION << "Empty dimensions for input blob " << input.first;
        }

        Blob::Ptr inputBlob = createInputBlob(desc);
        if (desc.getPrecision() == Precision::I16) {
            desc.setPrecision(Precision::FP32);
            auto fp32inputBlob = InferenceEngine::make_shared_blob<float>(desc);
            fp32inputBlob->allocate();
            _inputs[input.first + fp32_suffix] = fp32inputBlob;
        }
        inputBlob->allocate();
        _inputs[input.first] = inputBlob;
    }
}

void CLDNNInferRequest::AllocateOutputs() {
    // allocate outputs
    bool can_reuse_internal_mem = !m_useStreams;
    for (auto& no : _networkOutputs) {
        std::string outputID = m_graph->MapOutputName(no.first);
        cldnn::memory output_mem = m_graph->GetNetwork()->get_output_memory(outputID);
        cldnn::pointer<uint8_t> output_mem_ptr = output_mem.pointer<uint8_t>();
        if (output_mem_ptr.data() == nullptr) {
            THROW_IE_EXCEPTION << "Empty output memory for primitive " << outputID;
        }

        DataPtr oi = no.second;
        const TensorDesc& desc = oi->getTensorDesc();

        if (can_reuse_internal_mem) {
            _outputs[no.first] = createOutputBlob(desc, output_mem_ptr.data());
        } else {
            Blob::Ptr outputBlob = createOutputBlob(desc);
            outputBlob->allocate();
            _outputs[no.first] = outputBlob;
        }
        outputsMap[no.first] = outputID;
    }
}

void CLDNNInferRequest::AllocateOutputsDyn() {
    // allocate outputs
    for (auto& no : _networkOutputs) {
        DataPtr oi = no.second;
        TensorDesc desc = oi->getTensorDesc();
        SizeVector& dims = desc.getDims();

        if (!dims.empty()) {
            *dims.begin() = static_cast<size_t>(m_graph->GetMaxDynamicBatchSize());
        } else {
            THROW_IE_EXCEPTION << "Empty dimensions for output blob " << no.first;
        }

        Blob::Ptr outputBlob = createOutputBlob(desc);
        outputBlob->allocate();
        _outputs[no.first] = outputBlob;
    }
}

void CLDNNInferRequest::SetGraph(std::shared_ptr<CLDNNPlugin::CLDNNGraph> graph) {
    m_graph = graph;

    if (m_graph == nullptr) {
        THROW_IE_EXCEPTION << NETWORK_NOT_LOADED_str;
    }

    if (m_graph->GetMaxDynamicBatchSize() > 1) {
        SetBatch(m_graph->GetMaxDynamicBatchSize());
        AllocateInputsDyn();
        AllocateOutputsDyn();
    } else {
        AllocateInputs();
        AllocateOutputs();
    }
}

void CLDNNInferRequest::SetBatch(int new_batch) {
    if (m_graph->GetMaxDynamicBatchSize() < 0)
        THROW_IE_EXCEPTION << "Dynamic batch is not enabled.";

    if (new_batch < 1 || new_batch > m_graph->GetMaxDynamicBatchSize()) {
        THROW_IE_EXCEPTION << "Invalid dynamic batch size " << new_batch <<
            " for this request.";
    }

    if (new_batch == m_curBatch)
        return;

    batchInputs.clear();
    batchOutputs.clear();

    // tune expected inputs
    for (auto &input : m_graph->GetInputLayouts()) {
        cldnn::tensor dims = input.second.size;
        const SizeVector sz = { 1, size_t(dims.feature[0]), size_t(dims.spatial[1]), size_t(dims.spatial[0]) };
        size_t single_batch = std::accumulate(std::begin(sz), std::end(sz), (size_t)1, std::multiplies<size_t>());
        std::vector<buf_info> in_buf;

        size_t offset = 0;
        size_t bsz = single_batch;
        int b = 0;

        // calculate metadata for input buffers
        for (unsigned nb = 0; nb < m_graph->GetNetworksCount(); nb++) {
            unsigned int mask = 1 << nb;

            buf_info ib = { offset, bsz };
            in_buf.push_back(ib);

            if (new_batch & mask)
                offset += bsz;
            bsz <<= 1;
        }

        batchInputs[input.first] = in_buf;
    }

    // tune expected outputs
    for (auto& no : _networkOutputs) {
        auto sz = m_graph->GetOutputSize(no.first);
        sz.front() = 1;
        size_t single_batch = std::accumulate(std::begin(sz), std::end(sz), (size_t)1, std::multiplies<size_t>());
        std::vector<buf_info> out_buf;

        size_t offset = 0;
        size_t bsz = single_batch;
        // calculate metadata for output buffers
        for (uint32_t nb = 0; nb < m_graph->GetNetworksCount(); nb++) {
            uint32_t mask = 1 << nb;

            buf_info ob = { offset, bsz };
            out_buf.push_back(ob);

            if (new_batch & mask)
                offset += bsz;

            bsz <<= 1;
        }

        batchOutputs[no.first] = out_buf;
    }

    m_curBatch = new_batch;
}

CLDNNInferRequest::CLDNNInferRequest(InputsDataMap networkInputs, OutputsDataMap networkOutputs)
        : InferRequestInternal(networkInputs, networkOutputs)
        , m_useProfiling(false)
        , m_useStreams(false) {
}

void CLDNNInferRequest::execAndParse() {
    runningCounter++;
    auto networkOutputs = m_graph->GetNetwork()->execute();

    // Collect outputs as requested by the model
    for (auto& no : _networkOutputs) {
        std::string outputID = outputsMap[no.first];
        auto outputMemory = networkOutputs.at(outputID).get_memory();
        Blob::Ptr bptr = _outputs[no.first];

        auto out_ptr = outputMemory.pointer<uint8_t>();
        auto blob_ptr = bptr->buffer().as<uint8_t*>();

        // If Async API is used, copy of output blobs is not needed, unless SetBlob function was called.
        // But in the case when old API is used we have to copy data to memory provided by user.
        if (blob_ptr != &out_ptr[0]) {
            copyOutputData(outputMemory, bptr);
        }
    }
    runningCounter--;

    // finally collect profiling info
    if (m_useProfiling) {
        m_graph->UpdatePerfStatistics();
    }
}

void CLDNNInferRequest::execAndParseDyn() {
    runningCounter++;
    std::vector<std::map<cldnn::primitive_id, cldnn::network_output>> networkOutputs(m_graph->GetNetworksCount());

    // set up exection and put all graphs into driver queue
    for (unsigned nb = 0; nb < m_graph->GetNetworksCount(); nb++) {
        unsigned int mask = 1 << nb;

        if (m_curBatch & mask) {
            networkOutputs[nb] = m_graph->GetNetwork(nb)->execute();
        }
    }

    // now try to get execution results
    for (unsigned nb = 0; nb < m_graph->GetNetworksCount(); nb++) {
        unsigned int mask = 1 << nb;

        if (m_curBatch & mask) {
            for (auto& no : _networkOutputs) {
                std::string outputID = m_graph->MapOutputName(no.first);
                auto outputMemory = networkOutputs[nb].at(outputID).get_memory();
                Blob::Ptr bptr = _outputs[no.first];

                copyOutputData(outputMemory, bptr, &batchOutputs[no.first][nb]);
            }
        }
    }
    runningCounter--;
}

void CLDNNInferRequest::InferImpl() {
    IE_PROFILING_AUTO_SCOPE(CLDNN_INFER)

    if (CLDNNPlugin::MultiWorkerTaskExecutor::ptrContext.ptrGraph != nullptr) {
        m_graph = CLDNNPlugin::MultiWorkerTaskExecutor::ptrContext.ptrGraph;
    }

    // execute input pre-processing.
    execDataPreprocessing(_inputs, true);  // "true" stands for serial preprocessing in case of OpenMP

    for (auto &item : _inputs) {
        if (m_graph->GetMaxDynamicBatchSize() > 1) {
            PrepareInputDyn(item.first, *item.second);
        } else {
            PrepareInput(item.first, *item.second);
        }
    }

    // The actual inference
    if (m_graph->GetMaxDynamicBatchSize() > 1) {
        execAndParseDyn();
    } else {
        execAndParse();
    }
}

void CLDNNInferRequest::GetPerformanceCounts(
        std::map<std::string, InferenceEngineProfileInfo> &perfMap) const {
    if (!m_useProfiling) {
        THROW_IE_EXCEPTION << "Performance counters were not enabled";
    } else {
        m_graph->GetPerformanceCounts(perfMap);
    }
}

void CLDNNInferRequest::PrepareInput(const cldnn::primitive_id &inputName, const Blob &inputBlob) {
    // Get input layout
    if (m_graph->GetInputLayouts().find(inputName) == m_graph->GetInputLayouts().end()) {
        THROW_IE_EXCEPTION << "Input name mismatch.";
    }
    auto inputLayout = m_graph->GetInputLayouts().at(inputName);
    auto is_same_buffer = [](const Blob& blob, const cldnn::memory& memory) -> bool {
        const std::string str_not_allocated("Input data was not allocated.");
        cldnn::pointer<const uint8_t> ptr = memory.pointer<const uint8_t>();
        const uint8_t* blob_ptr = blob.cbuffer().as<const uint8_t*>();
        const uint8_t* mem_ptr = ptr.data();
        if (blob_ptr == nullptr || mem_ptr == nullptr) {
            THROW_IE_EXCEPTION << str_not_allocated;
        }
        return (blob_ptr == mem_ptr) && (blob.byteSize() == memory.size());
    };

    cldnn::primitive_id internalName = "input:" + inputName;
    const cldnn::memory& memory = inputsMemory.at(inputName);
    if (inputBlob.getTensorDesc().getPrecision() == Precision::I16) {
        // clDNN doesn't support I16 input precision, so we always have to convert input data to fp32 precision
        const cldnn::memory& fp32_mem = inputsMemory.at(inputName+fp32_suffix);
        cldnn::pointer<float> ptr = fp32_mem.pointer<float>();
        InferenceEngine::copyToFloat<int16_t>(ptr.data(), &inputBlob);
        m_graph->GetNetwork()->set_input_data(internalName, fp32_mem);
    } else if (is_same_buffer(inputBlob, memory)) {
        // If input memory was allocated by cldnn engine and wasn't overwritten by user set_input_data method won't copy input data.
        switch (inputBlob.getTensorDesc().getPrecision()) {
            case Precision::FP32:
            case Precision::FP16:
            case Precision::U8:
            case Precision::I32: {
                m_graph->GetNetwork()->set_input_data(internalName, memory);
                break;
            }
            default:
                THROW_IE_EXCEPTION << "Unsupported input precision " << inputBlob.getTensorDesc().getPrecision();
        }
    } else {
        // Otherwise, we have to attach to user memory and then copy the data.
        copyInputData(m_graph->GetNetwork(), inputName, inputLayout, inputBlob);
    }
}

void CLDNNInferRequest::PrepareInputDyn(const cldnn::primitive_id &inputName, const Blob &inputBlob) {
    // now try to get execution results
    for (unsigned nb = 0; nb < m_graph->GetNetworksCount(); nb++) {
        unsigned int mask = 1 << nb;

        if (m_curBatch & mask) {
            auto inputLayout = m_graph->GetInputLayouts().at(inputName);
            inputLayout.size.batch[0] = mask;
            copyInputData(m_graph->GetNetwork(nb), inputName, inputLayout, inputBlob, &batchInputs[inputName][nb]);
        }
    }
}

};  // namespace CLDNNPlugin
