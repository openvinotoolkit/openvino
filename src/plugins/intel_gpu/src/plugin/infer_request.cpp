// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <string>
#include <map>
#include <functional>
#include <utility>
#include <description_buffer.hpp>
#include "intel_gpu/plugin/infer_request.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/plugin/itt.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/preprocess/input_tensor_info.hpp"
#include <ie_algorithm.hpp>
#include <debug.h>

using namespace InferenceEngine;

namespace {

const char fp32_suffix[] = "_fp32";
const char cannot_set_compound[] = "cannot set compound blob: supported only for input pre-processing";
const char wrong_nv12_blob[] = "NV12 input blob is expected for input with NV12 color format";
const char unsupported_batched_blob[] = "Batched input blob is expected to contain NV12 blobs";
const char str_input_not_allocated[] = "Input data was not allocated.";
const char str_output_not_allocated[] = "Output data was not allocated.";

template <typename T>
void copyToFloat(float* dst, const InferenceEngine::Blob* src) {
    if (!dst) {
        return;
    }
    auto t_blob = dynamic_cast<const InferenceEngine::TBlob<T>*>(src);
    if (!t_blob) {
        IE_THROW() << "input type is " << src->getTensorDesc().getPrecision() << " but input is not "
                   << typeid(T).name();
    }

    const T* srcPtr = t_blob->readOnly();
    if (!srcPtr) {
        IE_THROW(NotAllocated) << str_input_not_allocated;
    }
    for (size_t i = 0; i < t_blob->size(); i++)
        dst[i] = srcPtr[i];
}

template<typename T>
void copyResultToOutputBlob(cldnn::memory::ptr src, Blob::Ptr dst, ov::runtime::intel_gpu::buf_info* bi, cldnn::stream& stream) {
    size_t n = (bi == nullptr) ? dst->size() : bi->buf_size;
    size_t offset = (bi == nullptr) ? 0 : bi->buf_offset;

    auto layout = src->get_layout();
    auto size = layout.size;

    auto locked_dst = dst->buffer();
    auto dst_ptr = locked_dst.as<T*>();
    if (dst_ptr == nullptr) {
        IE_THROW() << "Invalid output blob";
    }
    cldnn::mem_lock<T> src_lock{ src, stream };
    T* src_ptr = src_lock.data();
    dst_ptr += offset;

    if (layout.data_padding) {
        for (size_t b = 0; b < size.batch[0]; b++) {
            for (size_t f = 0; f < size.feature[0]; f++) {
                for (size_t w = 0; w < size.spatial[3]; w++) {
                    for (size_t z = 0; z < size.spatial[2]; z++) {
                        for (size_t y = 0; y < size.spatial[1]; y++) {
                            for (size_t x = 0; x < size.spatial[0]; x++) {
                                *dst_ptr++ = src_ptr[layout.get_linear_offset(cldnn::tensor(b, f, x, y, z, w))];
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            dst_ptr[i] = src_ptr[i];
        }
    }
}

inline void checkAlloc(const Blob::Ptr& blob, const std::string& err_str) {
    bool not_allocated = false;
    if (!blob->is<gpu::ClBlob>()) {
        not_allocated = (blob->buffer() == nullptr);
    } else {
        not_allocated = !ov::runtime::intel_gpu::getBlobImpl(blob->as<gpu::ClBlob>())->is_allocated();
    }
    if (not_allocated) {
        IE_THROW(NotAllocated) << err_str;
    }
}

NV12Blob *getNV12BlobOrException(BatchedBlob *batched_ptr, int idx) {
    auto nv12_ptr = batched_ptr->getBlob(idx)->as<NV12Blob>();
    if (nv12_ptr == nullptr)
        IE_THROW(NotImplemented) << unsupported_batched_blob;
    return nv12_ptr;
}

void checkInputBlob(const Blob::Ptr &blob,
    const std::string &name,
    const InputInfo::Ptr foundInput,
    bool nv12_two_inputs = false) {
    const std::string strNotMatched("The input blob size is not equal to the network input size");

    if (!blob) {
        IE_THROW(NotAllocated) << str_input_not_allocated;
    }

    if (ColorFormat::NV12 == foundInput->getPreProcess().getColorFormat() &&
        nv12_two_inputs) {
        if (auto nv12_ptr = blob->as<NV12Blob>()) {
            checkAlloc(nv12_ptr->y(), str_input_not_allocated);
            checkAlloc(nv12_ptr->uv(), str_input_not_allocated);
        } else if (auto batched_ptr = blob->as<BatchedBlob>()) {
            for (auto i = 0; i < batched_ptr->size(); i++) {
                auto nv12_ptr = getNV12BlobOrException(batched_ptr, i);
                checkAlloc(nv12_ptr->y(), str_input_not_allocated);
                checkAlloc(nv12_ptr->uv(), str_input_not_allocated);
            }
        } else {
            IE_THROW(ParameterMismatch) << wrong_nv12_blob;
        }

    } else {
        SizeVector dims = foundInput->getTensorDesc().getDims();

        size_t refSize = foundInput->getTensorDesc().getLayout() != SCALAR
            ? details::product(dims)
            : 1;

        if (refSize != blob->size()) {
            IE_THROW() << strNotMatched + ": got " << blob->size() << " expecting " << refSize;
        }

        checkAlloc(blob, str_input_not_allocated);
    }
}

void checkOutputBlob(const Blob::Ptr &blob,
    const std::string &name,
    const DataPtr foundOutput) {
    const std::string strNotMatched("The output blob size is not equal to the network output size");

    if (!blob) {
        IE_THROW(NotAllocated) << str_output_not_allocated;
    }
    SizeVector dims = foundOutput->getTensorDesc().getDims();
    size_t refSize = foundOutput->getTensorDesc().getLayout() != SCALAR
        ? details::product(dims)
        : 1;

    if (refSize != blob->size()) {
        IE_THROW() << strNotMatched + ": got " << blob->size() << " expecting " << refSize;
    }

    checkAlloc(blob, str_output_not_allocated);
}

bool same_host_mem(cldnn::memory::ptr memPtr, uint8_t* hostPtr) {
    uint8_t* bufferMem = nullptr;
    if (memPtr->get_allocation_type() == cldnn::allocation_type::usm_host) {
        bufferMem = reinterpret_cast<uint8_t*>(memPtr->get_internal_params().mem);
    }
    return bufferMem == hostPtr;
}

}  // namespace

namespace ov {
namespace runtime {
namespace intel_gpu {


// ----------------------------------------------------------------------------------------- //
// ---------------------------- IE API impl ------------------------------------------------ //
// ----------------------------------------------------------------------------------------- //
Blob::Ptr InferRequest::GetBlob(const std::string& name) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::GetBlob");
    Blob::Ptr data;
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
    return data;
}

void InferRequest::SetBlob(const std::string& name, const Blob::Ptr& data) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::SetBlob");

    // perform all common checks first
    if (name.empty()) {
        IE_THROW(NotFound) << "Failed to set blob with empty name";
    }
    if (!data)
        IE_THROW(NotAllocated) << "Failed to set empty blob with name: \'" << name << "\'";

    size_t dataSize = data->size();
    if (0 == dataSize) {
        IE_THROW() << "Input data is empty. Input name: \'" << name << "\'";
    }
    if (inputTensorsMap.find(name) != inputTensorsMap.end()) {
        inputTensorsMap.erase(name);
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
        IE_THROW(ParameterMismatch) << "Failed to set Blob with precision not corresponding to user "
                                    << (is_input ? "input" : "output") << " precision";
    }

    size_t dataBinSize = dataSize * data->element_size();
    size_t netReqBinSize = std::accumulate(desc.getDims().begin(), desc.getDims().end(),
                                           desc.getPrecision().size(),
                                           std::multiplies<size_t>());
    bool preProcResize = false;
    if (is_input) {
        preProcResize = foundInput->getPreProcess().getResizeAlgorithm() != ResizeAlgorithm::NO_RESIZE;
        const auto inputColorFormat = foundInput->getPreProcess().getColorFormat();
        preProcResize |= (inputColorFormat != ColorFormat::RAW) && (inputColorFormat != ColorFormat::BGR);
    }

    if (dataBinSize != netReqBinSize && !compoundBlobPassed && !preProcResize) {
        IE_THROW() << "Incorrect binary data size for " << (is_input ? "input" : "output") <<
                      " blob with name: \'" << name <<  "\' " <<
                      "Current: " << dataBinSize << " Required: " << netReqBinSize;
    }

    auto remote_ptr = data->as<gpu::ClBlob>();
    bool is_remote = remote_ptr != nullptr;
    if (is_remote) {
        auto impl = getBlobImpl(remote_ptr);
        if (!impl->is_allocated()) {
            impl->allocate();
        }
    }
    if (is_input) {
        if (is_remote) {
            _deviceInputs[name] = data;
            _inputs[name] = data;
        } else {
            auto nv12_ptr = data->as<NV12Blob>();
            auto batched_ptr = data->as<BatchedBlob>();
            bool is_batched = batched_ptr != nullptr;
            bool is_nv12 = nv12_ptr != nullptr;
            int expected_batch = is_batched ? desc.getDims()[0] : 1;
            if (ColorFormat::NV12 == foundInput->getPreProcess().getColorFormat() &&
                m_graph->getConfig().nv12_two_inputs) {
                // try extracting Y and UV remote blobs from it
                // and put them into appropriate network inputs
                // that should then go into biplanar NV12 reorder

                if (is_nv12 || is_batched) {
                    int num_blobs = is_batched ? batched_ptr->size() : 1;
                    for (auto i = 0; i < expected_batch; i++) {
                        std::string y_name = name + "_Y" + std::to_string(i);
                        std::string uv_name = name + "_UV" + std::to_string(i);
                        if (is_batched) {
                            int idx = i < num_blobs ? i : num_blobs-1;
                            nv12_ptr = getNV12BlobOrException(batched_ptr, idx);
                        }

                        auto y_ptr = nv12_ptr->y()->as<gpu::ClBlob>();
                        if (y_ptr) {
                            auto y_impl = getBlobImpl(y_ptr);
                            y_impl->allocate();
                            _deviceInputs[y_name] = nv12_ptr->y();
                            is_remote = true;
                        }

                        auto uv_ptr = nv12_ptr->uv()->as<gpu::ClBlob>();
                        if (uv_ptr) {
                            auto uv_impl = getBlobImpl(uv_ptr);
                            uv_impl->allocate();
                            _deviceInputs[uv_name] = nv12_ptr->uv();
                            is_remote = true;
                        }
                    }
                }
            }
            if (is_remote)
                _inputs[name] = data;
        }

        if (!is_remote) {
            if (preProcessingRequired(foundInput, data)) {
                // Stores the given blob as ROI blob. It will be used to fill in network input
                // during pre-processing
                if (_inputs[name]->is<gpu::ClBlob>()) {
                    Blob::Ptr inputHostBlob = create_host_blob(desc);
                    inputHostBlob->allocate();
                    _inputs[name] = inputHostBlob;
                }
                _preProcData[name] = CreatePreprocDataHelper();
                _preProcData[name]->isApplicable(data, _inputs[name]);
                _preProcData[name]->setRoiBlob(data);
            } else {
                if (compoundBlobPassed) {
                    IE_THROW(NotImplemented) << cannot_set_compound;
                }
                size_t blobSize = desc.getLayout() != SCALAR
                    ? details::product(desc.getDims())
                    : 1;
                if (dataSize != blobSize) {
                    IE_THROW() << "Input blob size is not equal network input size ("
                               << dataSize << "!=" << blobSize << ").";
                }

                if (data->buffer() == nullptr)
                    IE_THROW(NotAllocated) << str_input_not_allocated << " Input name: \'" << name << "\'";
                _inputs[name] = data;
            }
        }
    } else {
        if (compoundBlobPassed) {
            IE_THROW(NotImplemented) << cannot_set_compound;
        }

        if (is_remote) {
            _deviceOutputs[name] = data;
        } else {
            size_t outputSize = desc.getLayout() != SCALAR
                ? details::product(desc.getDims())
                : 1;
            if (dataSize != outputSize) {
                IE_THROW() << "Output blob size is not equal network output size (" << dataSize
                           << "!=" << outputSize << ").";
            }
            if (data->buffer() == nullptr)
                IE_THROW(NotAllocated) << str_input_not_allocated << " Input name: \'" << name << "\'";
        }
        _outputs[name] = data;
    }
}

void InferRequest::SetBlobs(const std::string& name, const std::vector<Blob::Ptr>& blobs) {
    if (blobs.size() == 1) {
        SetBlob(name, blobs[0]);
        return;
    }

    if (name.empty()) {
        IE_THROW(NotFound) << "Failed to set blobs with empty name";
    }
    if (blobs.empty()) {
        IE_THROW(NotAllocated) << "Failed to set empty blobs with name: \'" << name << "\'";
    }
    bool empty_data = std::any_of(blobs.begin(), blobs.end(), [](const Blob::Ptr& blob) {
        return blob->size() == 0;
    });
    if (empty_data) {
        IE_THROW() << "At least one of the input blobs is empty. Input name: \'" << name << "\'";
    }

    bool is_compound = std::any_of(blobs.begin(), blobs.end(), [](const Blob::Ptr& blob) {
        return blob->is<CompoundBlob>();
    });
    if (is_compound) {
        IE_THROW(NotImplemented) << cannot_set_compound;
    }

    bool is_buffer = std::all_of(blobs.begin(), blobs.end(), [](const Blob::Ptr& blob) {
        return blob->is<gpu::ClBufferBlob>();
    });
    bool is_surface = std::all_of(blobs.begin(), blobs.end(), [](const Blob::Ptr& blob) {
        return blob->is<gpu::ClImage2DBlob>();
    });
    bool is_remote = is_buffer || is_surface;

    bool is_host = std::all_of(blobs.begin(), blobs.end(), [](const Blob::Ptr& blob) {
        return blob->is<InferenceEngine::MemoryBlob>();
    });
    is_host &= !is_remote;

    if (!is_host && !is_remote) {
        IE_THROW() << "Incorrect input blobs. All blobs must be of the same type";
    }

    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    bool is_input = findInputAndOutputBlobByName(name, foundInput, foundOutput);

    if (!is_input) {
        IE_THROW() << "SetBlobs method doesn't support outputs";
    }

    if (is_buffer) {
        IE_THROW(NotImplemented) << "SetBlobs method doesn't support buffer blobs";
    }

    const TensorDesc& desc = foundInput->getTensorDesc();

    size_t dataBinSize = blobs.front()->size() * blobs.front()->element_size() * blobs.size();
    size_t netReqBinSize = std::accumulate(desc.getDims().begin(), desc.getDims().end(),
                                           desc.getPrecision().size(),
                                           std::multiplies<size_t>());
    bool preProcResize = false;
    if (is_input) {
        preProcResize = foundInput->getPreProcess().getResizeAlgorithm() != ResizeAlgorithm::NO_RESIZE;
        const auto inputColorFormat = foundInput->getPreProcess().getColorFormat();
        preProcResize |= (inputColorFormat != ColorFormat::RAW) && (inputColorFormat != ColorFormat::BGR);
    }
    if (dataBinSize != netReqBinSize && !preProcResize) {
        IE_THROW() << "Incorrect binary data size for input blobs with name: \'" << name <<  "\' " <<
                      "Current: " << dataBinSize << " Required: " << netReqBinSize;
    }

    if (_inputs.find(name) != _inputs.end()) {
        _inputs.erase(name);
    }

    if (is_remote) {
        for (auto& blob : blobs) {
            auto impl = getBlobImpl(blob->as<gpu::ClBlob>());
            if (!impl->is_allocated()) {
                impl->allocate();
            }
        }
    }

    inputTensorsMap.insert({ name, blobs });
}

void InferRequest::checkBlobs() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::checkBlobs");
    for (auto const &input : _inputs) {
        InputInfo::Ptr foundInput = nullptr;
        auto foundInputPair = std::find_if(std::begin(_networkInputs), std::end(_networkInputs),
            [&](const std::pair<std::string, InputInfo::Ptr> &pair) {
            return pair.first == input.first;
        });
        if (foundInputPair != std::end(_networkInputs)) {
            foundInput = foundInputPair->second;
        } else {
            IE_THROW(NotFound) << "Failed to find input with name: \'" << input.first << "\'";
        }
        checkInputBlob(input.second, input.first, foundInput, m_graph->getConfig().nv12_two_inputs);
    }
    for (auto const &output : _outputs) {
        DataPtr foundOutput = nullptr;
        auto foundOutputPair = std::find_if(std::begin(_networkOutputs), std::end(_networkOutputs),
            [&](const std::pair<std::string, DataPtr> &pair) {
            return pair.first == output.first;
        });
        if (foundOutputPair != std::end(_networkOutputs)) {
            foundOutput = foundOutputPair->second;
        } else {
            IE_THROW(NotFound) << "Failed to find output with name: \'" << output.first << "\'";
        }
        checkOutputBlob(output.second, output.first, foundOutput);
    }
}

void InferRequest::SetGraph(std::shared_ptr<Graph> graph) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::SetGraph");
    m_graph = graph;

    if (m_graph == nullptr) {
        IE_THROW(NetworkNotLoaded);
    }

    if (m_graph->GetMaxDynamicBatchSize() > 1) {
        SetBatch(m_graph->GetMaxDynamicBatchSize());
        allocate_inputs_dynamic();
        allocate_outputs_dynamic();
    } else {
        allocate_inputs();
        allocate_outputs();
    }
}

void InferRequest::SetBatch(int new_batch) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::SetBatch");
    if (m_graph->GetMaxDynamicBatchSize() < 0)
        IE_THROW() << "Dynamic batch is not enabled.";

    if (new_batch < 1 || new_batch > m_graph->GetMaxDynamicBatchSize()) {
        IE_THROW() << "Invalid dynamic batch size " << new_batch <<
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

InferRequest::InferRequest(InputsDataMap networkInputs, OutputsDataMap networkOutputs,
                                     const CompiledModel::Ptr& execNetwork)
        : IInferRequestInternal(networkInputs, networkOutputs) {
    IE_ASSERT(nullptr != execNetwork);
    streamExecutor = dynamic_cast<InferenceEngine::IStreamsExecutor*>(execNetwork->m_taskExecutor.get());
}

InferRequest::InferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                     const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                                     const CompiledModel::Ptr& execNetwork)
        : IInferRequestInternal(inputs, outputs) {
    IE_ASSERT(nullptr != execNetwork);
    streamExecutor = dynamic_cast<InferenceEngine::IStreamsExecutor*>(execNetwork->m_taskExecutor.get());
}

// ----------------------------------------------------------------------------------------- //
// ---------------------------- internal pipeline stages ----------------------------------- //
// ----------------------------------------------------------------------------------------- //
void InferRequest::preprocess_notify() {
    m_graph->wait(Graph::Stage::PREPROC);
    if (m_graph->GetMaxDynamicBatchSize() > 1) {
        preprocess_dynamic();
    } else {
        execDataPreprocessing(_inputs, true);  // "true" stands for serial preprocessing in case of OpenMP
    }
    m_graph->notify(Graph::Stage::PREPROC);
}

void InferRequest::preprocess() {
    if (m_graph->GetMaxDynamicBatchSize() > 1) {
        preprocess_dynamic();
    } else {
        execDataPreprocessing(_inputs, true);  // "true" stands for serial preprocessing in case of OpenMP
    }
}

void InferRequest::enqueue_notify() {
    m_graph->wait(Graph::Stage::EXECUTE);
    enqueue();
}

void InferRequest::enqueue() {
    if (m_graph->GetMaxDynamicBatchSize() > 1) {
        enqueue_dynamic();
        return;
    }

    // set input and output memory from request blob maps
    // into the network object primitives
    std::vector<cldnn::event::ptr> dependencies;

    for (const auto& inputTensor : inputTensorsMap) {
        const auto& blobs = inputTensor.second;
        auto blobsDesc = blobs.front()->getTensorDesc();

        bool is_surface = std::all_of(blobs.begin(), blobs.end(), [](const Blob::Ptr& blob) {
            return blob->is<gpu::ClImage2DBlob>();
        });

        if (is_surface) {
            for (size_t i = 0; i < blobs.size(); ++i) {
                std::string new_name = inputTensor.first + "_" + std::to_string(i);
                _inputs[new_name] = blobs[i];
                _deviceInputs[new_name] = blobs[i];
            }
        }
    }

    for (auto& item : _inputs) {
        std::string inputName = item.first;
        Blob::Ptr& inputBlob = item.second;

        auto nv12_ptr = inputBlob->as<NV12Blob>();
        auto batched_ptr = inputBlob->as<BatchedBlob>();
        bool is_batched = batched_ptr != nullptr;
        bool is_nv12 = nv12_ptr != nullptr;

        if (is_nv12 || is_batched) {
            int num_blobs = is_batched ? batched_ptr->size() : 1;
            int expected_batch = is_batched
                ? _networkInputs.at(inputName)->getTensorDesc().getDims()[0]
                : 1;
            for (auto i = 0; i < expected_batch; i++) {
                std::string y_name = inputName + "_Y" + std::to_string(i);
                std::string uv_name = inputName + "_UV" + std::to_string(i);
                if (is_batched) {
                    int idx = i < num_blobs ? i : num_blobs - 1;
                    nv12_ptr = getNV12BlobOrException(batched_ptr, idx);
                }
                prepare_input(y_name, nv12_ptr->y(), dependencies);
                prepare_input(uv_name, nv12_ptr->uv(), dependencies);
            }
        } else {
            // regular blob
            prepare_input(inputName, inputBlob, dependencies);
        }
    }

    for (auto& item : _outputs) {
        std::string outputName = item.first;
        Blob::Ptr& outputBlob = item.second;
        prepare_output(outputName, outputBlob);
    }

    internal_outputs.clear();
    internal_outputs = m_graph->GetNetwork()->execute(dependencies);

    // If dump layers path is set, only runs first inference.
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->dump_layers_path.length() > 0) {
        GPU_DEBUG_COUT << "Only run first inference to dump layers." << std::endl;
        exit(0);
    }
}

void InferRequest::wait_notify() {
    wait();
    m_graph->notify(Graph::Stage::EXECUTE);
}

void InferRequest::wait() {
    if (m_graph->GetMaxDynamicBatchSize() > 1) {
        wait_dynamic();
        return;
    }

    if (internal_outputs.empty()) {
        IE_THROW() << "Inference was not started!\n";
    }

    // wait for completion & collect outputs as requested by the model
    for (auto& no : _networkOutputs) {
        Blob::Ptr bptr = _outputs[no.first];
        std::string outputID = outputsMap.at(no.first);
        auto outputMemory = internal_outputs.at(outputID).get_memory();

        // mapping remote blobs not needed -
        // let the user take care of them explicitly
        if (!bptr->is<gpu::ClBlob>()) {
            bool same_mem = false;
            {
                auto dst_lock = bptr->cbuffer();
                auto dst_ptr = dst_lock.as<uint8_t*>();
                same_mem = same_host_mem(outputMemory, dst_ptr);
            }
            if (!same_mem) {
                copy_output_data(outputMemory, bptr);
            }
        }
    }

    // finally collect profiling info
    if (m_useProfiling) {
        m_graph->UpdatePerfStatistics();
    }
}

void InferRequest::preprocess_dynamic() {
    // execute input pre-processing.
    execDataPreprocessing(_inputs, true);  // "true" stands for serial preprocessing in case of OpenMP
}

void InferRequest::enqueue_dynamic() {
    internal_outputs_dynamic.clear();
    auto numNets = m_graph->GetNetworksCount();
    internal_outputs_dynamic.resize(numNets);

    // set up exection and put all graphs into driver queue
    for (unsigned nb = 0; nb < numNets; nb++) {
        unsigned int mask = 1 << nb;

        if (m_curBatch & mask) {
            for (auto& item : _inputs) {
                const cldnn::primitive_id& inputName = item.first;
                const Blob::Ptr inputBlob = item.second;

                auto inputLayout = m_graph->GetInputLayouts().at(inputName);
                inputLayout.size.batch[0] = mask;
                copy_input_data(m_graph->GetNetwork(nb), inputName, inputLayout, *inputBlob, &batchInputs[inputName][nb]);
            }
            internal_outputs_dynamic[nb] = m_graph->GetNetwork(nb)->execute();
        }
    }
}

void InferRequest::wait_dynamic() {
    if (internal_outputs_dynamic.empty()) {
        IE_THROW() << "Inference was not started!\n";
    }

    // now try to get execution results
    for (unsigned nb = 0; nb < m_graph->GetNetworksCount(); nb++) {
        unsigned int mask = 1 << nb;

        if (m_curBatch & mask) {
            for (auto& no : _networkOutputs) {
                std::string outputID = outputsMap.at(no.first);
                auto outputMemory = internal_outputs_dynamic[nb].at(outputID).get_memory();
                Blob::Ptr bptr = _outputs[no.first];

                copy_output_data(outputMemory, bptr, &batchOutputs[no.first][nb]);
            }
        }
    }
}

// ----------------------------------------------------------------------------------------- //
// ---------------------------- internal utils --------- ----------------------------------- //
// ----------------------------------------------------------------------------------------- //
void InferRequest::setup_stream_graph() {
    int streamID = 0;
    auto& streamGraphs = static_cast<CompiledModel*>(_exeNetwork.get())->m_graphs;
    if (nullptr != streamExecutor) {
        streamID = streamExecutor->GetStreamId();
        int numGraphs = streamGraphs.size();
        streamID = streamID % numGraphs;
    }
    m_graph = streamGraphs[streamID];
}

Blob::Ptr InferRequest::create_host_blob(const TensorDesc& desc, uint8_t* mem_ptr) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::create_host_blob");
    const Precision& p = desc.getPrecision();

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
    case Precision::U16:
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
        IE_THROW(NotImplemented) << "The plugin does not support " << p.name() << " blob precision";
    }
}

void InferRequest::copy_output_data(cldnn::memory::ptr src, Blob::Ptr dst, buf_info* bi) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::copy_output_data");
    auto& stream = m_graph->GetNetwork()->get_stream();
    switch (dst->getTensorDesc().getPrecision()) {
    case Precision::FP32: copyResultToOutputBlob<float>(src, dst, bi, stream);    break;
    case Precision::FP16: copyResultToOutputBlob<uint16_t>(src, dst, bi, stream); break;
    case Precision::I32:  copyResultToOutputBlob<int32_t>(src, dst, bi, stream);  break;
    case Precision::I64:  copyResultToOutputBlob<int64_t>(src, dst, bi, stream);  break;
    case Precision::U8:  copyResultToOutputBlob<uint8_t>(src, dst, bi, stream);  break;
    case Precision::I8:  copyResultToOutputBlob<int8_t>(src, dst, bi, stream);  break;
    default: IE_THROW(NotImplemented) << "The plugin does not support output " << dst->getTensorDesc().getPrecision() << " precision";
    }
}

void InferRequest::copy_input_data(std::shared_ptr<cldnn::network> network,
                                        const cldnn::primitive_id &inputName,
                                        const cldnn::layout& inputLayout,
                                        const Blob &inputBlob, buf_info* bi) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::copy_input_data");

    size_t offset = (bi == nullptr) ? 0 : bi->buf_offset;

    cldnn::primitive_id internalName = "parameter:" + inputName;
    auto locked = inputBlob.cbuffer();
    switch (inputBlob.getTensorDesc().getPrecision()) {
    case Precision::FP32: {
        float* blob_ptr = const_cast<float*>(locked.as<const float*>()) + offset;
        network->set_input_data(internalName, network->get_engine().attach_memory(inputLayout, blob_ptr));
        break;
    }
    case Precision::I32: {
        int32_t* blob_ptr = const_cast<int32_t*>(locked.as<const int32_t*>()) + offset;
        network->set_input_data(internalName, network->get_engine().attach_memory(inputLayout, blob_ptr));
        break;
    }
    case Precision::I64: {
        int64_t* blob_ptr = const_cast<int64_t*>(locked.as<const int64_t*>()) + offset;
        network->set_input_data(internalName, network->get_engine().attach_memory(inputLayout, blob_ptr));
        break;
    }
    case Precision::FP16: {
        uint16_t* blob_ptr = const_cast<uint16_t*>(locked.as<const uint16_t*>()) + offset;
        network->set_input_data(internalName, network->get_engine().attach_memory(inputLayout, blob_ptr));
        break;
    }
    case Precision::I8: {
        int8_t* blob_ptr = const_cast<int8_t*>(locked.as<const int8_t*>()) + offset;
        network->set_input_data(internalName, network->get_engine().attach_memory(inputLayout, blob_ptr));
        break;
    }
    case Precision::U8: {
        uint8_t* blob_ptr = const_cast<uint8_t*>(locked.as<const uint8_t*>()) + offset;
        network->set_input_data(internalName, network->get_engine().attach_memory(inputLayout, blob_ptr));
        break;
    }
    case Precision::BOOL: {
        uint8_t* blob_ptr = const_cast<uint8_t*>(locked.as<const uint8_t*>()) + offset;
        network->set_input_data(internalName, network->get_engine().attach_memory(inputLayout, blob_ptr));
        break;
    }
    default:
        IE_THROW() << "The plugin does not support input " << inputBlob.getTensorDesc().getPrecision() << " precision";
    }
}

Blob::Ptr InferRequest::host_blob_from_device_blob(Blob::Ptr blobPtr) {
    uint8_t* bufferMem = nullptr;
    auto clblobPtr = std::dynamic_pointer_cast<InferenceEngine::gpu::ClBlob>(blobPtr);
    if (clblobPtr) {
        const auto memPtr = getBlobImpl(clblobPtr.get())->getMemory();
        if (memPtr->get_allocation_type() == cldnn::allocation_type::usm_host) {
            bufferMem = reinterpret_cast<uint8_t*>(memPtr->get_internal_params().mem);
        }
    }
    Blob::Ptr hostBlob = create_host_blob(blobPtr->getTensorDesc(), bufferMem);
    hostBlob->allocate();

    return hostBlob;
}

void InferRequest::allocate_inputs() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::allocate_inputs");
    auto inputLayouts = m_graph->GetInputLayouts();

    // allocate inputs
    for (auto& ni : _networkInputs) {
        std::string name = ni.first;
        const TensorDesc& desc = ni.second->getTensorDesc();

        bool is_nv12_input = ColorFormat::NV12 == ni.second->getPreProcess().getColorFormat() &&
                             m_graph->getConfig().nv12_two_inputs;

        auto parameter = std::find_if(_parameters.begin(), _parameters.end(), [&](const std::shared_ptr<const ov::Node>& node) {
            return node->get_friendly_name() == name;
        });

        if (parameter != _parameters.end()) {
            if (parameter->get()->output(0).get_rt_info().count(ov::preprocess::TensorInfoMemoryType::get_type_info_static())) {
                std::string mem_type = parameter->get()->output(0).get_rt_info().at(ov::preprocess::TensorInfoMemoryType::get_type_info_static())
                                                                                .as<ov::preprocess::TensorInfoMemoryType>().value;
                if (mem_type.find(GPU_CONFIG_KEY(SURFACE)) != std::string::npos) {
                    is_nv12_input = true;
                }
            }
        }

        if (!is_nv12_input) {
            auto litr = inputLayouts.find(name);
            if (litr == inputLayouts.end()) {
                IE_THROW() << "Input layout for " << name << " is not found";
            }

            GPU_DEBUG_GET_INSTANCE(debug_config);
            GPU_DEBUG_IF(debug_config->verbose >= 2) {
                GPU_DEBUG_COUT << "[" << name << ": input blob]" << std::endl;
            }
            if (desc.getPrecision() == Precision::I16 || desc.getPrecision() == Precision::U16) {
                TensorDesc desc_fp32 = desc;
                desc_fp32.setPrecision(Precision::FP32);
                auto blobPtr = create_device_blob(desc_fp32, litr->second);
                _deviceInputs[name] = blobPtr;
                Blob::Ptr inputBlob = create_host_blob(desc);
                inputBlob->allocate();
                _inputs[name] = inputBlob;
            } else {
                auto blobPtr = create_device_blob(desc, litr->second);
                _deviceInputs[name] = blobPtr;
                _inputs[name] = host_blob_from_device_blob(blobPtr);
            }
        }
    }
}

void InferRequest::allocate_inputs_dynamic() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::allocate_inputs_dynamic");
    // allocate inputs
    for (auto &input : m_graph->GetInputLayouts()) {
        InputInfo::Ptr ni = _networkInputs.at(input.first);
        TensorDesc desc = ni->getTensorDesc();
        SizeVector& dims = desc.getDims();

        if (!dims.empty()) {
            *dims.begin() = static_cast<size_t>(m_graph->GetMaxDynamicBatchSize());
        } else {
            IE_THROW() << "Empty dimensions for input blob " << input.first;
        }

        Blob::Ptr inputBlob = create_host_blob(desc);
        if (desc.getPrecision() == Precision::I16 || desc.getPrecision() == Precision::U16) {
            desc.setPrecision(Precision::FP32);
            auto fp32inputBlob = InferenceEngine::make_shared_blob<float>(desc);
            fp32inputBlob->allocate();
            _inputs[input.first + fp32_suffix] = fp32inputBlob;
        }
        inputBlob->allocate();
        _inputs[input.first] = inputBlob;
    }
}

void InferRequest::allocate_outputs() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::allocate_outputs");
    // allocate outputs
    for (auto& no : _networkOutputs) {
        std::string outputID = m_graph->MapOutputName(no.first);
        const cldnn::layout output_layout = m_graph->GetNetwork()->get_output_memory(outputID)->get_layout();
        const TensorDesc& desc = no.second->getTensorDesc();

        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << "[" << no.first << ": output blob]" << std::endl;
        }
        auto blobPtr = create_device_blob(desc, output_layout);
        _deviceOutputs[no.first] = blobPtr;
        _outputs[no.first] = host_blob_from_device_blob(blobPtr);
        outputsMap[no.first] = outputID;
    }
}

void InferRequest::allocate_outputs_dynamic() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::allocate_outputs_dynamic");
    // allocate outputs
    for (auto& no : _networkOutputs) {
        std::string outputID = m_graph->MapOutputName(no.first);
        DataPtr oi = no.second;
        TensorDesc desc = oi->getTensorDesc();
        SizeVector& dims = desc.getDims();

        if (!dims.empty()) {
            *dims.begin() = static_cast<size_t>(m_graph->GetMaxDynamicBatchSize());
        } else {
            IE_THROW() << "Empty dimensions for output blob " << no.first;
        }

        Blob::Ptr outputBlob = create_host_blob(desc);
        outputBlob->allocate();
        _outputs[no.first] = outputBlob;
        outputsMap[no.first] = outputID;
    }
}

void InferRequest::InferImpl() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::InferImpl");
    setup_stream_graph();
    std::lock_guard<std::mutex> lk(m_graph->get_mutex());
    preprocess();
    enqueue();
    wait();
}

std::map<std::string, InferenceEngineProfileInfo> InferRequest::GetPerformanceCounts() const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::GetPerformanceCounts");
    if (!m_useProfiling) {
        IE_THROW() << "Performance counters were not enabled";
    } else {
        return m_graph->GetPerformanceCounts();
    }
}

void InferRequest::prepare_input(const cldnn::primitive_id& inputName, Blob::Ptr& inputBlob,
                                      std::vector<cldnn::event::ptr>& dependencies) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::prepare_input");
    auto inputLayoutItr = m_graph->GetInputLayouts().find(inputName);
    if (inputLayoutItr == m_graph->GetInputLayouts().end()) {
        IE_THROW() << "Input name mismatch.";
    }
    auto reqBlob = _deviceInputs.at(inputName)->as<gpu::ClBlob>();
    auto _nw_ptr = m_graph->GetNetwork();
    cldnn::primitive_id internalName = "parameter:" + inputName;
    const auto& prec = inputBlob->getTensorDesc().getPrecision();
    auto remote_ptr = inputBlob->as<gpu::ClBlob>();
    auto& stream = m_graph->GetNetwork()->get_stream();
    bool is_dev_input = remote_ptr != nullptr;

    switch (prec) {
        case Precision::FP32:
        case Precision::FP16:
        case Precision::I8:
        case Precision::U8:
        case Precision::BOOL:
        case Precision::I16:
        case Precision::U16:
        case Precision::I32:
        case Precision::I64: {
            auto impl = getBlobImpl(is_dev_input ?
                                    remote_ptr :
                                    reqBlob);
            if (!impl->is_allocated()) {
                IE_THROW() << str_input_not_allocated;
            }
            auto inputMem = impl->getMemory();

            auto input_layout = m_graph->GetInputLayouts().find(inputName);
            if (input_layout != m_graph->GetInputLayouts().end()) {
                if (input_layout->second.format != inputMem->get_layout().format) {
                    inputMem = m_graph->GetNetwork()->get_engine().reinterpret_buffer(*inputMem, input_layout->second);
                }
            }

            if (!is_dev_input) {
                if (prec == Precision::I16 || prec == Precision::U16) {
                    // GPU plugin doesn't support I16 input precision,
                    // so have to convert input data to fp32 precision
                    cldnn::mem_lock<float> ptr{ inputMem, stream };
                    if (prec == Precision::I16) {
                        copyToFloat<int16_t>(ptr.data(), inputBlob.get());
                    } else {
                        copyToFloat<uint16_t>(ptr.data(), inputBlob.get());
                    }
                } else {
                    auto src_lock = inputBlob->cbuffer();
                    auto src_ptr = src_lock.as<uint8_t*>();
                    if (!same_host_mem(inputMem, src_ptr)) {
                        auto ev = inputMem->copy_from(stream, src_ptr);
                        dependencies.push_back(ev);
                    }
                }
            }
            _nw_ptr->set_input_data(internalName, inputMem);
            break;
        }
        default:
            IE_THROW() << "Unsupported input precision " << prec;
    }
}

void InferRequest::prepare_output(const cldnn::primitive_id& outputName, Blob::Ptr& outputBlob) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::prepare_output");
    Blob::Ptr reqBlob = _deviceOutputs.at(outputName);
    cldnn::primitive_id internalName = outputsMap[outputName];
    auto _nw_ptr = m_graph->GetNetwork();
    auto remote_ptr = outputBlob->as<gpu::ClBlob>();
    auto output_blob_ptr = (reqBlob != outputBlob && remote_ptr != nullptr)
        ? remote_ptr
        : reqBlob->as<gpu::ClBlob>();
    auto impl = getBlobImpl(output_blob_ptr);
    if (!impl->is_allocated()) {
        IE_THROW(NotAllocated) << str_output_not_allocated;
    }
    auto outputMem = impl->getMemory();
    _nw_ptr->set_output_memory(internalName, outputMem);
}

InferenceEngine::Blob::Ptr InferRequest::create_device_blob(const InferenceEngine::TensorDesc& desc, const cldnn::layout& layout) {
    if (m_graph->GetEngine()->use_unified_shared_memory()) {
        auto blobPtr = std::make_shared<RemoteUSMbuffer>(m_graph->GetContext(),
                                                         m_graph->GetNetwork()->get_stream(),
                                                         desc,
                                                         layout,
                                                         nullptr,
                                                         0,
                                                         0,
                                                         RemoteBlobImpl::BlobType::BT_USM_HOST_INTERNAL);
        getBlobImpl(blobPtr.get())->allocate();
        return blobPtr;
    } else {
        auto blobPtr = std::make_shared<RemoteCLbuffer>(m_graph->GetContext(),
                                                        m_graph->GetNetwork()->get_stream(),
                                                        desc,
                                                        layout);
        getBlobImpl(blobPtr.get())->allocate();
        return blobPtr;
    }
}

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
