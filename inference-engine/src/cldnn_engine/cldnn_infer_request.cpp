// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <string>
#include <map>
#include <functional>
#include <utility>
#include <description_buffer.hpp>
#include "cldnn_infer_request.h"
#include "cldnn_remote_context.h"
#include "cldnn_executable_network.h"
#include "cldnn_itt.h"

using namespace InferenceEngine;

namespace CLDNNPlugin {

const char CLDNNInferRequest::fp32_suffix[] = "_fp32";
const char str_not_allocated[] = "Input data was not allocated.";
const char cannot_set_compound[] = "cannot set compound blob: supported only for input pre-processing";
const char wrong_nv12_blob[] = "NV12 input blob is expected for input with NV12 color format";

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
    case Precision::I64:
        if (mem_ptr != nullptr)
            return make_shared_blob<int64_t>(desc, reinterpret_cast<int64_t*>(mem_ptr));
        else
            return make_shared_blob<int64_t>(desc);
    case Precision::I8:
        if (mem_ptr != nullptr)
            return make_shared_blob<int8_t>(desc, reinterpret_cast<int8_t*>(mem_ptr));
        else
            return make_shared_blob<int8_t>(desc);
    case Precision::U8:
        if (mem_ptr != nullptr)
            return make_shared_blob<uint8_t>(desc, reinterpret_cast<uint8_t*>(mem_ptr));
        else
            return make_shared_blob<uint8_t>(desc);
    case Precision::BOOL:
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
     case Precision::I64:
        if (mem_ptr != nullptr)
            return make_shared_blob<int64_t>(desc, reinterpret_cast<int64_t*>(mem_ptr));
        else
            return make_shared_blob<int64_t>(desc);
    default:
        THROW_IE_EXCEPTION << "The plugin does not support output " << p.name() << " precision";
    }
}

void CLDNNInferRequest::input_attach(cldnn::primitive_id name, cldnn::memory& inputMem) {
    auto impl = getContextImpl(m_graph->GetContext());
    impl->acquire_lock();

    auto mem_itr = inputsMemory.find(name);

    if (mem_itr != inputsMemory.end())
        mem_itr->second = inputMem;
    else
        inputsMemory.insert({ name, inputMem });

    impl->release_lock();
}

void CLDNNInferRequest::input_alloc(cldnn::primitive_id name, const cldnn::layout& layout) {
    cldnn::memory input_mem = cldnn::memory::allocate(*(m_graph->GetEngine()), layout);
    input_attach(name, input_mem);
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

    auto locked = bptr->buffer();
    switch (bptr->getTensorDesc().getPrecision()) {
    case Precision::FP32: {
        auto out_f = locked.as<float*>();
        if (out_f == nullptr) {
            THROW_IE_EXCEPTION << "Invalid output blob";
        }
        auto resPtr = outputMemory.pointer<float>();
        float *resVec = out_f + offset;

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
        auto out_f = locked.as<uint16_t*>();
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
        auto out_f = locked.as<int32_t*>();
        if (out_f == nullptr) {
            THROW_IE_EXCEPTION << "Invalid output blob";
        }
        auto resPtr = outputMemory.pointer<int32_t>();
        int32_t* resVec = out_f + offset;

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
    case Precision::I64: {
        auto out_f = locked.as<int64_t*>();
        if (out_f == nullptr) {
            THROW_IE_EXCEPTION << "Invalid output blob";
        }
        auto resPtr = outputMemory.pointer<int64_t>();
        int64_t* resVec = out_f + offset;

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
    auto locked = inputBlob.cbuffer();
    switch (inputBlob.getTensorDesc().getPrecision()) {
    case Precision::FP32: {
        float* blob_ptr = const_cast<float*>(locked.as<const float*>()) + offset;
        network->set_input_data(internalName, cldnn::memory::attach(inputLayout, blob_ptr, n));
        break;
    }
    case Precision::I32: {
        int32_t* blob_ptr = const_cast<int32_t*>(locked.as<const int32_t*>()) + offset;
        network->set_input_data(internalName, cldnn::memory::attach(inputLayout, blob_ptr, n));
        break;
    }
    case Precision::I64: {
        int64_t* blob_ptr = const_cast<int64_t*>(locked.as<const int64_t*>()) + offset;
        network->set_input_data(internalName, cldnn::memory::attach(inputLayout, blob_ptr, n));
        break;
    }
    case Precision::FP16: {
        uint16_t* blob_ptr = const_cast<uint16_t*>(locked.as<const uint16_t*>()) + offset;
        network->set_input_data(internalName, cldnn::memory::attach(inputLayout, blob_ptr, n));
        break;
    }
    case Precision::I8: {
        int8_t* blob_ptr = const_cast<int8_t*>(locked.as<const int8_t*>()) + offset;
        network->set_input_data(internalName, cldnn::memory::attach(inputLayout, blob_ptr, n));
        break;
    }
    case Precision::U8: {
        uint8_t* blob_ptr = const_cast<uint8_t*>(locked.as<const uint8_t*>()) + offset;
        network->set_input_data(internalName, cldnn::memory::attach(inputLayout, blob_ptr, n));
        break;
    }
    case Precision::BOOL: {
        uint8_t* blob_ptr = const_cast<uint8_t*>(locked.as<const uint8_t*>()) + offset;
        network->set_input_data(internalName, cldnn::memory::attach(inputLayout, blob_ptr, n));
        break;
    }
    default:
        THROW_IE_EXCEPTION << "The plugin does not support input " << inputBlob.getTensorDesc().getPrecision() << " precision";
    }
}

void checkInputBlob(const Blob::Ptr &blob,
    const std::string &name,
    const InputInfo::Ptr foundInput,
    bool nv12_two_inputs = false) {
    const std::string strNotMatched("The input blob size is not equal to the network input size");

    if (!blob) {
        THROW_IE_EXCEPTION << str_not_allocated;
    }

    if (ColorFormat::NV12 == foundInput->getPreProcess().getColorFormat() &&
        nv12_two_inputs) {
        auto nv12_ptr = blob->as<NV12Blob>();

        if (nv12_ptr == nullptr) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << wrong_nv12_blob;
        }

        auto y_ptr = nv12_ptr->y()->as<gpu::ClBlob>();

        // if the blobs are not remote, check their size
        if (!y_ptr) {
            if (nv12_ptr->y()->buffer() == nullptr) THROW_IE_EXCEPTION << str_not_allocated;
        }

        auto uv_ptr = nv12_ptr->uv()->as<gpu::ClBlob>();
        if (!uv_ptr) {
            if (nv12_ptr->uv()->buffer() == nullptr) THROW_IE_EXCEPTION << str_not_allocated;
        }
    } else {
        SizeVector dims = foundInput->getTensorDesc().getDims();

        size_t refSize = foundInput->getTensorDesc().getLayout() != SCALAR
            ? details::product(dims)
            : 1;

        if (refSize != blob->size()) {
            THROW_IE_EXCEPTION << strNotMatched + ": got " << blob->size() << " expecting " << refSize;
        }

        if (!blob->is<gpu::ClBlob>()) {
            if (blob->buffer() == nullptr) THROW_IE_EXCEPTION << str_not_allocated;
        }
    }
}

void checkOutputBlob(const Blob::Ptr &blob,
    const std::string &name,
    const DataPtr foundOutput) {
    const std::string strNotAllocated("Output data was not allocated.");
    const std::string strNotMatched("The output blob size is not equal to the network output size");

    if (!blob) {
        THROW_IE_EXCEPTION << strNotAllocated;
    }
    SizeVector dims = foundOutput->getTensorDesc().getDims();
    size_t refSize = foundOutput->getTensorDesc().getLayout() != SCALAR
        ? details::product(dims)
        : 1;

    if (refSize != blob->size()) {
        THROW_IE_EXCEPTION << strNotMatched + ": got " << blob->size() << " expecting " << refSize;
    }

    if (!blob->is<gpu::ClBlob>()) {
        if (blob->buffer() == nullptr) THROW_IE_EXCEPTION << strNotAllocated;
    }
}

void CLDNNInferRequest::checkBlobs() {
    for (auto const &input : _inputs) {
        InputInfo::Ptr foundInput = nullptr;
        auto foundInputPair = std::find_if(std::begin(_networkInputs), std::end(_networkInputs),
            [&](const std::pair<std::string, InputInfo::Ptr> &pair) {
            return pair.first == input.first;
        });
        if (foundInputPair != std::end(_networkInputs)) {
            foundInput = foundInputPair->second;
        } else {
            THROW_IE_EXCEPTION << NOT_FOUND_str << "Failed to find input with name: \'" << input.first << "\'";
        }
        checkInputBlob(input.second, input.first, foundInput, m_graph->getConfig().nv12_two_inputs);
    }
    for (auto const &output : _outputs) {
        DataPtr foundOutput;
        auto foundOutputPair = std::find_if(std::begin(_networkOutputs), std::end(_networkOutputs),
            [&](const std::pair<std::string, DataPtr> &pair) {
            return pair.first == output.first;
        });
        if (foundOutputPair != std::end(_networkOutputs)) {
            foundOutput = foundOutputPair->second;
        } else {
            THROW_IE_EXCEPTION << NOT_FOUND_str << "Failed to find output with name: \'" << output.first << "\'";
        }
        checkOutputBlob(output.second, output.first, foundOutput);
    }
}

void CLDNNInferRequest::GetBlob(const char *name, Blob::Ptr &data) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "GetBlob");
    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    bool is_input = findInputAndOutputBlobByName(name, foundInput, foundOutput);

    if (is_input) {
        // ROI blob is returned only if it was set previously. Otherwise default blob is returned.
        auto it = _preProcData.find(name);
        if (it != _preProcData.end()) {
            data = it->second->getRoiBlob();
        } else {
            data = _inputs[name];
            checkInputBlob(data, name, foundInput);
        }
    } else {
        data = _outputs[name];
        checkOutputBlob(data, name, foundOutput);
    }
}

void CLDNNInferRequest::SetBlob(const char *name, const Blob::Ptr &data) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "SetBlob");

    // perform all common checks first
    if (name == nullptr) {
        THROW_IE_EXCEPTION << NOT_FOUND_str + "Failed to set blob with empty name";
    }
    if (!data)
        THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to set empty blob with name: \'" << name << "\'";

    size_t dataSize = data->size();
    if (0 == dataSize) {
        THROW_IE_EXCEPTION << "Input data is empty. Input name: \'" << name << "\'";
    }
    const bool compoundBlobPassed = data->is<CompoundBlob>();

    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    auto blobDesc = data->getTensorDesc();

    bool is_input = findInputAndOutputBlobByName(name, foundInput, foundOutput);
    const TensorDesc& desc = is_input
        ? foundInput->getTensorDesc()
        : foundOutput->getTensorDesc();

    if (desc.getPrecision() != blobDesc.getPrecision()) {
        THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
            << "Failed to set Blob with precision not corresponding to user "
            << (is_input ? "input" : "output") << " precision";
    }

    auto remote_ptr = data->as<gpu::ClBlob>();
    bool is_remote = remote_ptr != nullptr;
    if (is_remote) {
        auto impl = getBlobImpl(remote_ptr);
        impl->allocate_if_needed();
    }

    if (is_input) {
        cldnn::primitive_id internalName(name);

        if (is_remote) {
            auto inputMem = getBlobImpl(remote_ptr)->getMemory();
            input_attach(internalName, inputMem);
            _inputs[name] = data;
        } else if (compoundBlobPassed) {
            if (ColorFormat::NV12 == foundInput->getPreProcess().getColorFormat() &&
                m_graph->getConfig().nv12_two_inputs) {
                // try extracting Y and UV remote blobs from it
                // and put them into appropriate network inputs
                // that should then go into biplanar NV12 reorder
                auto nv12_ptr = data->as<NV12Blob>();

                if (nv12_ptr == nullptr) {
                    THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << wrong_nv12_blob;
                }

                auto y_ptr = nv12_ptr->y()->as<gpu::ClBlob>();
                if (y_ptr) {
                    auto y_impl = getBlobImpl(y_ptr);
                    y_impl->allocate_if_needed();
                    input_attach(internalName + "_Y", y_impl->getMemory());
                    is_remote = true;
                }

                auto uv_ptr = nv12_ptr->uv()->as<gpu::ClBlob>();
                if (uv_ptr) {
                    auto uv_impl = getBlobImpl(uv_ptr);
                    uv_impl->allocate_if_needed();
                    input_attach(internalName + "_UV", uv_impl->getMemory());
                    is_remote = true;
                }

                if (is_remote) _inputs[name] = data;
            }
        }

        if (!is_remote) {
            if (preProcessingRequired(foundInput, data)) {
                // Stores the given blob as ROI blob. It will be used to fill in network input
                // during pre-processing
                _preProcData[name] = CreatePreprocDataHelper();
                _preProcData[name]->isApplicable(data, _inputs[name]);
                _preProcData[name]->setRoiBlob(data);
            } else {
                if (compoundBlobPassed) {
                    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str << cannot_set_compound;
                }

                size_t blobSize = desc.getLayout() != SCALAR
                    ? details::product(desc.getDims())
                    : 1;
                if (dataSize != blobSize) {
                    THROW_IE_EXCEPTION << "Input blob size is not equal network input size ("
                        << dataSize << "!=" << blobSize << ").";
                }

                if (data->buffer() == nullptr)
                    THROW_IE_EXCEPTION << str_not_allocated << " Input name: \'" << name << "\'";
                _inputs[name] = data;
            }
        }
    } else {
        if (compoundBlobPassed) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str << cannot_set_compound;
        }

        if (is_remote) {
            std::string outputID = m_graph->MapOutputName(name);
            auto impl = getBlobImpl(remote_ptr);
            m_graph->GetNetwork()->set_output_memory(outputID, impl->getMemory());
        } else {
            size_t outputSize = desc.getLayout() != SCALAR
                ? details::product(desc.getDims())
                : 1;
            if (dataSize != outputSize) {
                THROW_IE_EXCEPTION << "Output blob size is not equal network output size (" << dataSize
                    << "!=" << outputSize << ").";
            }
            if (data->buffer() == nullptr)
                THROW_IE_EXCEPTION << str_not_allocated << " Input name: \'" << name << "\'";
        }
        _outputs[name] = data;
    }
}

void CLDNNInferRequest::AllocateInputs() {
    // allocate inputs
    for (auto& ni : _networkInputs) {
        std::string name = ni.first;
        const TensorDesc& desc = ni.second->getTensorDesc();

        if (ColorFormat::NV12 == ni.second->getPreProcess().getColorFormat() &&
            m_graph->getConfig().nv12_two_inputs) {
            cldnn::primitive_id YName(name + "_Y");
            cldnn::primitive_id UVName(name + "_UV");

            input_alloc(YName, m_graph->GetInputLayouts().at(YName));
            input_alloc(UVName, m_graph->GetInputLayouts().at(UVName));

            size_t height = desc.getDims()[2], width = desc.getDims()[3];
            cldnn::pointer<uint8_t> input_mem_ptr_Y = inputsMemory.at(YName).pointer<uint8_t>();
            TensorDesc ydesc(Precision::U8, { 1, 1, height, width }, Layout::NHWC);
            auto blobY = createInputBlob(ydesc, input_mem_ptr_Y.data());

            cldnn::pointer<uint8_t> input_mem_ptr_UV = inputsMemory.at(UVName).pointer<uint8_t>();
            TensorDesc uvdesc(Precision::U8, { 1, 2, height / 2, width / 2 }, Layout::NHWC);
            auto blobUV = createInputBlob(uvdesc, input_mem_ptr_UV.data());

            _inputs[name] = make_shared_blob<NV12Blob>(blobY, blobUV);
        } else {
            cldnn::layout layout = m_graph->GetInputLayouts().at(name);
            input_alloc(name, layout);
            cldnn::pointer<uint8_t> mem_ptr = inputsMemory.at(name).pointer<uint8_t>();
            _inputs[name] = createInputBlob(desc, mem_ptr.data());

            if (desc.getPrecision() == Precision::I16) {
                cldnn::layout layout_fp32 = layout;
                layout_fp32.data_type = cldnn::data_types::f32;
                input_alloc(name + fp32_suffix, layout_fp32);
            }
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
            " for this request. Got: " << new_batch << ". Expected value in range [1;" << m_graph->GetMaxDynamicBatchSize() << "]";
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

CLDNNInferRequest::CLDNNInferRequest(InputsDataMap networkInputs, OutputsDataMap networkOutputs,
                                     const CLDNNExecNetwork::Ptr& execNetwork)
        : InferRequestInternal(networkInputs, networkOutputs)
        , m_useProfiling(false)
        , m_useStreams(false) {
    IE_ASSERT(nullptr != execNetwork);
    streamExecutor = dynamic_cast<InferenceEngine::IStreamsExecutor*>(execNetwork->m_taskExecutor.get());
}

void CLDNNInferRequest::execAndParse() {
    auto networkOutputs = m_graph->GetNetwork()->execute();

    // Collect outputs as requested by the model
    for (auto& no : _networkOutputs) {
        Blob::Ptr bptr = _outputs[no.first];

        std::string outputID = outputsMap[no.first];
        auto outputMemory = networkOutputs.at(outputID).get_memory();

        // mapping remote blobs not needed -
        // let the user take care of them explicitly
        if (!bptr->is<gpu::ClBlob>()) {
            auto out_ptr = outputMemory.pointer<uint8_t>();
            auto blob_ptr = bptr->buffer().as<uint8_t*>();

            // If Async API is used, copy of output blobs is not needed, unless SetBlob function was called.
            // But in the case when old API is used we have to copy data to memory provided by user.
            if (blob_ptr != &out_ptr[0]) {
                copyOutputData(outputMemory, bptr);
            }
        }
    }

    // finally collect profiling info
    if (m_useProfiling) {
        m_graph->UpdatePerfStatistics();
    }
}

void CLDNNInferRequest::execAndParseDyn() {
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
}

void CLDNNInferRequest::InferImpl() {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "CLDNN_INFER");
    int streamID = 0;
    if (nullptr != streamExecutor) {
        streamID = streamExecutor->GetStreamId();
    }
    m_graph = static_cast<CLDNNExecNetwork*>(_exeNetwork.get())->m_graphs[streamID];
    // execute input pre-processing.
    execDataPreprocessing(_inputs, true);  // "true" stands for serial preprocessing in case of OpenMP

    for (auto &item : _inputs) {
        std::string name = item.first;
        Blob::Ptr inputBlob = item.second;

        if (m_graph->GetMaxDynamicBatchSize() > 1) {
            PrepareInputDyn(name, *inputBlob);
        } else {
            auto nv12_ptr = inputBlob->as<NV12Blob>();

            if (nv12_ptr == nullptr) {
                // regular blob
                PrepareInput(name, *inputBlob);
            } else {
                // special case for NV12 input blob
                PrepareInput(name + "_Y", *nv12_ptr->y());
                PrepareInput(name + "_UV", *nv12_ptr->uv());
            }
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

namespace {

template <typename T>
void copyToFloat(float* dst, const InferenceEngine::Blob* src) {
    if (!dst) {
        return;
    }
    const InferenceEngine::TBlob<T>* t_blob = dynamic_cast<const InferenceEngine::TBlob<T>*>(src);
    if (t_blob == nullptr) {
        THROW_IE_EXCEPTION << "input type is " << src->getTensorDesc().getPrecision() << " but input is not "
                           << typeid(T).name();
    }

    const T* srcPtr = t_blob->readOnly();
    if (srcPtr == nullptr) {
        THROW_IE_EXCEPTION << "Input data was not allocated.";
    }
    for (size_t i = 0; i < t_blob->size(); i++) dst[i] = srcPtr[i];
}

}  // namespace

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
    auto _nw_ptr = m_graph->GetNetwork();
    auto prec = inputBlob.getTensorDesc().getPrecision();

    if (inputBlob.is<gpu::ClBlob>()) {
        // no need to check for reuse
        _nw_ptr->set_input_data(internalName, memory);
    } else if (prec == Precision::I16) {
        // clDNN doesn't support I16 input precision, so we always have to convert input data to fp32 precision
        const cldnn::memory& fp32_mem = inputsMemory.at(inputName+fp32_suffix);
        cldnn::pointer<float> ptr = fp32_mem.pointer<float>();
        copyToFloat<int16_t>(ptr.data(), &inputBlob);
        _nw_ptr->set_input_data(internalName, fp32_mem);
    } else if (is_same_buffer(inputBlob, memory)) {
        // If input memory was allocated by cldnn engine and wasn't overwritten by user set_input_data method won't copy input data.
        switch (prec) {
            case Precision::FP32:
            case Precision::FP16:
            case Precision::I8:
            case Precision::U8:
            case Precision::BOOL:
            case Precision::I32:
            case Precision::I64: {
                _nw_ptr->set_input_data(internalName, memory);
                break;
            }
            default:
                THROW_IE_EXCEPTION << "Unsupported input precision " << prec;
        }
    } else {
        // Otherwise, we have to attach to user memory and then copy the data.
        copyInputData(_nw_ptr, inputName, inputLayout, inputBlob);
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
