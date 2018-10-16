// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <string>
#include <map>
#include <functional>
#include <CPP/detection_output.hpp>  // todo: find a way to remove this
#include <description_buffer.hpp>
#include "cldnn_infer_request.h"

using namespace InferenceEngine;

namespace CLDNNPlugin {

const std::string CLDNNInferRequest::fp32_suffix = "_fp32";

Blob::Ptr CLDNNInferRequest::createInputBlob(const Precision& p, const Layout& l, const SizeVector& sz, uint8_t* mem_ptr) {
    switch (p) {
    case Precision::FP32:
        if (mem_ptr != nullptr)
            return make_shared_blob<float>(p, l, sz, reinterpret_cast<float*>(mem_ptr));
        else
            return make_shared_blob<float, const SizeVector>(p, l, sz);
    case Precision::FP16:
        if (mem_ptr != nullptr)
            return make_shared_blob<uint16_t>(p, l, sz, reinterpret_cast<uint16_t*>(mem_ptr));
        else
            return make_shared_blob<uint16_t, const SizeVector>(p, l, sz);
    case Precision::I16:
        if (mem_ptr != nullptr)
            return make_shared_blob<int16_t>(p, l, sz, reinterpret_cast<int16_t*>(mem_ptr));
        else
            return make_shared_blob<int16_t, const SizeVector>(p, l, sz);
    case Precision::U8:
        if (mem_ptr != nullptr)
            return make_shared_blob<uint8_t>(p, l, sz, reinterpret_cast<uint8_t*>(mem_ptr));
        else
            return make_shared_blob<uint8_t, const SizeVector>(Precision::U8, l, sz);
    default:
        THROW_IE_EXCEPTION << "The plugin does not support input " << p.name() << " precision";
    }
}

Blob::Ptr CLDNNInferRequest::createOutputBlob(const Precision& p, SizeVector& sz, uint8_t* mem_ptr) {
    Layout l = TensorDesc::getLayoutByDims(sz);

    switch (p) {
    case Precision::FP32:
        if (mem_ptr != nullptr)
            return make_shared_blob<float>(p, l, sz, reinterpret_cast<float*>(mem_ptr));
        else
            return make_shared_blob<float, const SizeVector>(p, l, sz);
    case Precision::FP16:
        if (mem_ptr != nullptr)
            return make_shared_blob<uint16_t>(p, l, sz, reinterpret_cast<uint16_t*>(mem_ptr));
        else
            return make_shared_blob<uint16_t, const SizeVector>(p, l, sz);
        break;
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

    switch (bptr->precision()) {
    case Precision::FP32: {
        TBlob<float>::Ptr out_f = std::dynamic_pointer_cast<TBlob<float>>(bptr);
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
        TBlob<uint16_t>::Ptr out_f = std::dynamic_pointer_cast<TBlob<uint16_t>>(bptr);
        auto resPtr = outputMemory.pointer<uint16_t>();
        uint16_t *resVec = out_f->data() + offset;

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
        THROW_IE_EXCEPTION << "The plugin does not support output " << bptr->precision() << " precision";
    }
}

void CLDNNInferRequest::copyInputData(std::shared_ptr<cldnn::network> network,
                                    const cldnn::primitive_id &inputName,
                                    const cldnn::layout& inputLayout,
                                    const Blob &inputBlob, buf_info* bi) {
    size_t n = (bi == nullptr) ? inputBlob.size() : bi->buf_size;
    size_t offset = (bi == nullptr) ? 0 : bi->buf_offset;

    switch (inputBlob.precision()) {
    case Precision::FP32: {
        float* blob_ptr = const_cast<float*>(inputBlob.cbuffer().as<const float*>()) + offset;
        network->set_input_data(inputName, cldnn::memory::attach(inputLayout, blob_ptr, n));
        break;
    }
    case Precision::FP16: {
        uint16_t* blob_ptr = const_cast<uint16_t*>(inputBlob.cbuffer().as<const uint16_t*>()) + offset;
        network->set_input_data(inputName, cldnn::memory::attach(inputLayout, blob_ptr, n));
        break;
    }
    case Precision::U8: {
        uint8_t* blob_ptr = const_cast<uint8_t*>(inputBlob.cbuffer().as<const uint8_t*>()) + offset;
        network->set_input_data(inputName, cldnn::memory::attach(inputLayout, blob_ptr, n));
        break;
    }
    default:
        THROW_IE_EXCEPTION << "The plugin does not support input " << inputBlob.precision() << " precision";
    }
}

void CLDNNInferRequest::AllocateInputs() {
    // allocate inputs
    for (auto &input : m_env.inputLayouts) {
        std::string name = input.first;
        cldnn::layout layout = input.second;
        cldnn::tensor dims = layout.size;

        InputInfo::Ptr ni = _networkInputs.at(input.first);

        const SizeVector sz = { size_t(dims.spatial[0]), size_t(dims.spatial[1]),
                                size_t(dims.feature[0]), size_t(dims.batch[0]) };
        Precision ip = ni->getInputPrecision();
        Layout l = TensorDesc::getLayoutByDims(sz);

        cldnn::memory inputMem = cldnn::memory::allocate(*(m_env.engine), layout);
        cldnn::pointer<uint8_t> mem_ptr = inputMem.pointer<uint8_t>();

        inputsMemory.insert({ name, inputMem });

        if (layout.format == cldnn::format::byxf) l = NHWC;

        _inputs[name] = createInputBlob(ip, l, sz, mem_ptr.data());
        if (ip == Precision::I16) {
            cldnn::layout layout_fp32 = layout;
            layout_fp32.data_type = cldnn::data_types::f32;
            cldnn::memory inputMem_fp32 = cldnn::memory::allocate(*(m_env.engine), layout_fp32);
            inputsMemory.insert({ input.first + fp32_suffix, inputMem_fp32 });
        }
    }
}

void CLDNNInferRequest::AllocateInputsDyn() {
    // allocate inputs
    for (auto &input : m_env.inputLayouts) {
        auto dims = input.second.size;
        InputInfo::Ptr ni = _networkInputs.at(input.first);
        const SizeVector sz = { size_t(dims.spatial[0]), size_t(dims.spatial[1]),
                                size_t(dims.feature[0]), size_t(m_env.m_max_batch) };
        Precision ip = ni->getInputPrecision();
        Layout l = TensorDesc::getLayoutByDims(sz);

        Blob::Ptr inputBlob = createInputBlob(ip, l, sz);
        if (ip == Precision::I16) {
            auto fp32inputBlob = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(
                Precision::FP32, NCHW,
                { size_t(dims.spatial[0]),
                size_t(dims.spatial[1]),
                size_t(dims.feature[0]),
                size_t(m_env.m_max_batch) });
            fp32inputBlob->allocate();
            _inputs[input.first + fp32_suffix] = fp32inputBlob;
        }
        inputBlob->allocate();
        _inputs[input.first] = inputBlob;
    }
}

void CLDNNInferRequest::AllocateOutputs() {
    auto networkOutputsIDs = m_env.network->get_output_ids();
    auto allPrimitiveIds = m_env.network->get_all_primitives();

    // allocate outputs
    for (auto& no : _networkOutputs) {
        // Find correct output ID. Start with name stored in IR.
        std::string outputID = m_env.primitiveIDs.at(no.first);
        while (std::find(networkOutputsIDs.begin(), networkOutputsIDs.end(), outputID) == networkOutputsIDs.end()) {
            // If current ID isn't found in cldnn network outputs, get previous primitive id and try again.
            auto prim = allPrimitiveIds.find(outputID);
            if (prim == allPrimitiveIds.end()) {
                THROW_IE_EXCEPTION << "Unknown primitive id " << outputID;
            }

            if (m_env.prevPrimitiveIDs.at(outputID).size() != 1 || prim->second != "_optimized_") {
                THROW_IE_EXCEPTION << "Unable to find parent for output primitive " << outputID;
            }
            outputID = m_env.prevPrimitiveIDs.at(outputID)[0];
        }

        outputsMap[no.first] = outputID;

        auto res_output = m_env.outputDims.find(no.first);

        InferenceEngine::SizeVector output;
        if (res_output != m_env.outputDims.end()) {
            // if output with this name exists, take it
            output = res_output->second;
        } else {
            // if doesn't try to locate fused layer
            output = m_env.outputDims.at(outputID);
        }

        cldnn::memory output_mem = m_env.network->get_output_memory(outputID);
        cldnn::pointer<uint8_t> output_mem_ptr = output_mem.pointer<uint8_t>();
        if (output_mem_ptr.data() == nullptr) {
            THROW_IE_EXCEPTION << "Empty output memory for primitive " << outputID;
        }

        DataPtr oi = no.second;
        Precision op = oi->getPrecision();

        _outputs[no.first] = createOutputBlob(op, output, output_mem_ptr.data());
    }
}

void CLDNNInferRequest::AllocateOutputsDyn() {
    // allocate outputs
    for (auto& no : _networkOutputs) {
        auto res_output = m_env.outputDims.find(no.first);

        InferenceEngine::SizeVector output;
        if (res_output != m_env.outputDims.end()) {
            // if output with this name exists, take it
            output = res_output->second;
        } else {
            // if doesn't try to locate fused layer
            std::string outputID = m_env.primitiveIDs.at(no.first);
            output = m_env.outputDims.at(outputID);
        }

        DataPtr oi = no.second;
        Precision op = oi->getPrecision();
        Blob::Ptr outputBlob = createOutputBlob(op, output);
        outputBlob->allocate();
        _outputs[no.first] = outputBlob;
    }
}

void CLDNNInferRequest::SetBatch(int new_batch) {
    if (m_env.m_max_batch < 0)
        THROW_IE_EXCEPTION << "Dynamic batch is not enabled.";

    if (new_batch < 1 || new_batch > m_env.m_max_batch) {
        THROW_IE_EXCEPTION << "Invalid dynamic batch size " << new_batch <<
            " for this request.";
    }

    if (new_batch == m_curBatch)
        return;

    batchInputs.clear();
    batchOutputs.clear();

    // tune expected inputs
    for (auto &input : m_env.inputLayouts) {
        cldnn::tensor dims = input.second.size;
        const SizeVector sz = { size_t(dims.spatial[0]), size_t(dims.spatial[1]), size_t(dims.feature[0]), 1 };
        size_t single_batch = std::accumulate(std::begin(sz), std::end(sz), (size_t)1, std::multiplies<size_t>());
        std::vector<buf_info> in_buf;

        size_t offset = 0;
        size_t bsz = single_batch;
        int b = 0;

        // calculate metadata for input buffers
        for (unsigned nb = 0; nb < m_env.m_bv_sz; nb++) {
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
        auto res_output = m_env.outputDims.find(no.first);

        InferenceEngine::SizeVector sz;
        if (res_output != m_env.outputDims.end())
            sz = res_output->second;
        else
            sz = m_env.outputDims.at(m_env.primitiveIDs.at(no.first));

        sz.back() = 1;
        size_t single_batch = std::accumulate(std::begin(sz), std::end(sz), (size_t)1, std::multiplies<size_t>());
        std::vector<buf_info> out_buf;

        size_t offset = 0;
        size_t bsz = single_batch;
        int b = 0;
        // calculate metadata for output buffers
        for (unsigned nb = 0; nb < m_env.m_bv_sz; nb++) {
            unsigned int mask = 1 << nb;

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

CLDNNInferRequest::CLDNNInferRequest(InferenceEnv env, bool useProfiling,
                                     InputsDataMap networkInputs, OutputsDataMap networkOutputs)
        : InferRequestInternal(networkInputs, networkOutputs),
          m_curBatch(-1),
          m_env(env),
          m_useProfiling(useProfiling) {
    if (m_env.m_max_batch > 1) {
        AllocateInputsDyn();
        AllocateOutputsDyn();
    } else {
        AllocateInputs();
        AllocateOutputs();
    }

    // Fill implementations map
    if (m_useProfiling) {
        auto extractImplementationFromInfo = [](const std::string& info) -> std::string {
            std::string def_implementation = "undef";
            std::string impl_section = "implementation :";
            std::string::size_type pos = info.find(impl_section);
            if (pos == std::string::npos) {
                return def_implementation;
            }

            std::string::size_type end_pos = info.find(',', pos);
            if (end_pos == std::string::npos) {
                return def_implementation;
            }

            std::string::size_type length = end_pos - pos - impl_section.size();

            auto trim = [](const std::string& str) {
                size_t first = str.find_first_not_of(' ');
                if (std::string::npos == first) {
                    return str;
                }
                size_t last = str.find_last_not_of(' ');
                return str.substr(first, (last - first + 1));
            };
            std::string tmp = trim(info.substr(pos + impl_section.size(), length));

            return tmp.length() > 1 ? tmp : def_implementation;
        };

        // Parse primitive info and extract implementation name.
        for (auto& id : m_env.profilingIDs) {
            std::string prim_info = "";
            try {
                prim_info = m_env.network->get_primitive_info(id);
            } catch (std::exception& e) { }

            implementationsMap.insert({id, extractImplementationFromInfo(prim_info)});
        }
    }
}

void CLDNNInferRequest::execAndParse() {
    auto networkOutputs = m_env.network->execute();

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

    // finally collect profiling info
    if (m_useProfiling) {
        std::map<cldnn::primitive_id, cldnn::event> executedPrimitives = m_env.network->get_executed_primitives();
        auto allPrimitives = m_env.network->get_all_primitives();

        // Get profiling info for all layers
        for (auto &profiledID : m_env.profilingIDs) {
            std::string impl = implementationsMap.at(profiledID);
            impl.copy(m_env.perfMap[profiledID].exec_type, impl.length());

            // Change status if layer wasn't executed by cldnn engine
            if (executedPrimitives.find(profiledID) == executedPrimitives.end()) {
                if (allPrimitives.find(profiledID) != allPrimitives.end() &&
                    allPrimitives.at(profiledID) == "_optimized_") {
                    // Layer was marked as optimized by cldnn
                    m_env.perfMap[profiledID].status = InferenceEngineProfileInfo::OPTIMIZED_OUT;
                } else {
                    // Layer wasn't run for some reason
                    m_env.perfMap[profiledID].status = InferenceEngineProfileInfo::NOT_RUN;
                }
                m_env.perfMap[profiledID].cpu_uSec = m_env.perfMap[profiledID].realTime_uSec = 0;
                continue;
            }

            auto event = executedPrimitives.at(profiledID);
            executedPrimitives.erase(profiledID);

            cldnn::instrumentation::profiling_info cldnnInfo{profiledID, event.get_profiling_info()};

            // Collect timings
            for (auto &interval : cldnnInfo.intervals) {
                using duration_t = std::chrono::duration<long long, std::chrono::microseconds::period>;
                auto count = std::chrono::duration_cast<duration_t>(interval.value->value()).count();

                if (interval.name == "submission") {
                    m_env.perfMap[profiledID].cpu_uSec = count;
                } else if (interval.name == "executing") {
                    m_env.perfMap[profiledID].realTime_uSec = count;
                } else if (interval.name == "duration") {  // "duration" is used for CPU layers
                    m_env.perfMap[profiledID].cpu_uSec = count;
                    static const std::string cpuExecType("CPU");
                    memset(m_env.perfMap[profiledID].exec_type, 0, sizeof(m_env.perfMap[profiledID].exec_type));
                    cpuExecType.copy(m_env.perfMap[profiledID].exec_type,
                        cpuExecType.length());  // Override execType as CPU
                }
            }
        }
    }
}

void CLDNNInferRequest::execAndParseDyn() {
    std::vector<std::map<cldnn::primitive_id, cldnn::network_output>> networkOutputs(m_env.m_bv_sz);

    // set up exection and put all graphs into driver queue
    for (unsigned nb = 0; nb < m_env.m_bv_sz; nb++) {
        unsigned int mask = 1 << nb;

        if (m_curBatch & mask) {
            networkOutputs[nb] = m_env.batchNetworks[nb]->execute();
        }
    }

    // now try to get execution results
    for (unsigned nb = 0; nb < m_env.m_bv_sz; nb++) {
        unsigned int mask = 1 << nb;

        if (m_curBatch & mask) {
            for (auto& no : _networkOutputs) {
                std::string outputID = no.first;
                while ((m_env.primitiveIDs.find(outputID) != m_env.primitiveIDs.end()) &&
                    (m_env.primitiveIDs.at(outputID) != outputID)) {
                    outputID = m_env.primitiveIDs.at(outputID);
                }

                auto outputMemory = networkOutputs[nb].at(outputID).get_memory();
                Blob::Ptr bptr = _outputs[no.first];

                copyOutputData(outputMemory, bptr, &batchOutputs[no.first][nb]);
            }
        }
    }
}

void CLDNNInferRequest::InferImpl() {
    IE_PROFILING_AUTO_SCOPE(CLDNN_INFER)

    // execute input pre-processing.
    execDataPreprocessing();

    for (auto &item : _inputs) {
        if (m_env.m_max_batch > 1) {
            PrepareInputDyn(item.first, *item.second);
        } else {
            PrepareInput(item.first, *item.second);
        }
    }

    // The actual inference
    if (m_env.m_max_batch > 1) {
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
        perfMap = m_env.perfMap;
    }
}

void CLDNNInferRequest::PrepareInput(const cldnn::primitive_id &inputName, const Blob &inputBlob) {
    // Get input layout
    if (m_env.inputLayouts.find(inputName) == m_env.inputLayouts.end()) {
        THROW_IE_EXCEPTION << "Input name mismatch.";
    }
    auto inputLayout = m_env.inputLayouts.at(inputName);
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

    const cldnn::memory& memory = inputsMemory.at(inputName);
    if (inputBlob.precision() == Precision::I16) {
        // clDNN doesn't support I16 input precision, so we always have to convert input data to fp32 precision
        const cldnn::memory& fp32_mem = inputsMemory.at(inputName+fp32_suffix);
        cldnn::pointer<float> ptr = fp32_mem.pointer<float>();
        InferenceEngine::copyToFloat<int16_t>(ptr.data(), &inputBlob);
        m_env.network->set_input_data(inputName, fp32_mem);
    } else if (is_same_buffer(inputBlob, memory)) {
        // If input memory was allocated by cldnn engine and wasn't overwritten by user set_input_data method won't copy input data.
        switch (inputBlob.precision()) {
            case Precision::FP32:
            case Precision::FP16:
            case Precision::U8: {
                m_env.network->set_input_data(inputName, memory);
                break;
            }
            default:
                THROW_IE_EXCEPTION << "Unsupported input precision " << inputBlob.precision();
        }
    } else {
        // Otherwise, we have to attach to user memory and then copy the data.
        copyInputData(m_env.network, inputName, inputLayout, inputBlob);
    }
}

void CLDNNInferRequest::PrepareInputDyn(const cldnn::primitive_id &inputName, const Blob &inputBlob) {
    // now try to get execution results
    for (unsigned nb = 0; nb < m_env.m_bv_sz; nb++) {
        unsigned int mask = 1 << nb;

        if (m_curBatch & mask) {
            auto inputLayout = m_env.inputLayouts.at(inputName);
            inputLayout.size.batch[0] = mask;
            copyInputData(m_env.batchNetworks[nb], inputName, inputLayout, inputBlob, &batchInputs[inputName][nb]);
        }
    }
}

};  // namespace CLDNNPlugin
