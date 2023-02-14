// Copyright (C) 2018-2023 Intel Corporation
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
#include "intel_gpu/plugin/remote_allocators.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/preprocess/input_tensor_info.hpp"
#include <ie_algorithm.hpp>
#include "ie_ngraph_utils.hpp"
#include <debug.h>

using namespace InferenceEngine;

namespace {

const char str_device_output_unsupported_blob[] = "Device output is of an unsupported blob type.";
const char str_input_not_allocated[] = "Input data was not allocated.";
const char str_output_not_allocated[] = "Output data was not allocated.";

template <typename src_t, typename dst_t>
void convertAndCopy(const InferenceEngine::Blob* src, dst_t* dst) {
    if (!dst) {
        return;
    }
    auto t_blob = dynamic_cast<const InferenceEngine::TBlob<src_t>*>(src);
    if (!t_blob) {
        IE_THROW() << "input type is " << src->getTensorDesc().getPrecision() << " but input is not "
                   << typeid(src_t).name();
    }

    const src_t* srcPtr = t_blob->readOnly();
    if (!srcPtr) {
        IE_THROW(NotAllocated) << str_input_not_allocated;
    }
    for (size_t i = 0; i < t_blob->size(); i++)
        dst[i] = srcPtr[i];
}

template<typename src_dt, typename dst_dt>
void copy_result_to_output_blob(InferenceEngine::Blob::Ptr src, InferenceEngine::Blob::Ptr dst, const cldnn::layout& src_layout) {
    auto locked_src = src->buffer();
    auto src_ptr = locked_src.as<src_dt*>();
    OPENVINO_ASSERT(src_ptr, "[GPU] Invalid source blob");

    auto locked_dst = dst->buffer();
    auto dst_ptr = locked_dst.as<dst_dt*>();
    OPENVINO_ASSERT(dst_ptr, "[GPU] Invalid output blob");

    if (src_layout.data_padding) {
        auto size = src_layout.get_tensor();
        for (int64_t b = 0; b < size.batch[0]; b++) {
            for (int64_t f = 0; f < size.feature[0]; f++) {
                for (int64_t w = 0; w < size.spatial[3]; w++) {
                    for (int64_t z = 0; z < size.spatial[2]; z++) {
                        for (int64_t y = 0; y < size.spatial[1]; y++) {
                            for (int64_t x = 0; x < size.spatial[0]; x++) {
                                *dst_ptr++ = src_ptr[src_layout.get_linear_offset(cldnn::tensor(b, f, x, y, z, w))];
                            }
                        }
                    }
                }
            }
        }
    } else {
        size_t n = dst->size();
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
        not_allocated = !ov::intel_gpu::getBlobImpl(blob->as<gpu::ClBlob>())->is_allocated();
    }
    if (not_allocated) {
        IE_THROW(NotAllocated) << err_str;
    }
}

void checkInputBlob(const Blob::Ptr &blob,
    const std::string &name,
    const InputInfo::Ptr foundInput) {
    const std::string strNotMatched("The input blob size is not equal to the network input size");

    if (!blob) {
        IE_THROW(NotAllocated) << str_input_not_allocated;
    }

    SizeVector dims = foundInput->getTensorDesc().getDims();
    size_t refSize = foundInput->getTensorDesc().getLayout() != SCALAR
        ? details::product(dims)
        : 1;

    if (refSize != blob->size()) {
        IE_THROW() << strNotMatched + ": got " << blob->size() << " expecting " << refSize;
    }

    checkAlloc(blob, str_input_not_allocated);
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
    auto node = is_input ? findInputByNodeName(name) : findOutputByNodeName(name);
    bool isDynamic = (node && node->get_output_partial_shape(0).is_dynamic());

    if (is_input) {
        data = _inputs[name];
        if (!isDynamic)
            checkInputBlob(data, name, foundInput);
    } else {
        data = _outputs[name];
        if (!isDynamic) {
            checkOutputBlob(data, name, foundOutput);
        }
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

    if (inputTensorsMap.find(name) != inputTensorsMap.end()) {
        inputTensorsMap.erase(name);
    }

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

    size_t netReqBinSize = std::accumulate(desc.getDims().begin(), desc.getDims().end(),
                                           desc.getPrecision().size(),
                                           std::multiplies<size_t>());
    auto node = is_input ? findInputByNodeName(name) : findOutputByNodeName(name);
    bool isDynamic = (node && node->get_output_partial_shape(0).is_dynamic());

    size_t dataSize = data->size();
    if (0 == dataSize && !isDynamic) {
        IE_THROW() << "Input data is empty. Input name: \'" << name << "\'";
    }

    size_t dataBinSize = dataSize * data->element_size();
    if (!isDynamic && dataBinSize != netReqBinSize) {
        IE_THROW() << "Incorrect binary data size for " << (is_input ? "input" : "output") <<
                      " blob with name: \'" << name <<  "\' " <<
                      "Current: " << dataBinSize << " Required: " << netReqBinSize;
    }

    if (is_input) {
        set_input(name, data);
    } else {
        set_output(name, data);
    }
}

void InferRequest::set_input(const std::string& name, const Blob::Ptr& data) {
    auto remote_ptr = data->as<gpu::ClBlob>();
    bool is_remote = remote_ptr != nullptr;

    auto node = findInputByNodeName(name);
    bool isDynamic = node && node->get_output_partial_shape(0).is_dynamic();

    if (is_remote) {
        _deviceInputs[name] = data;
        _inputs[name] = data;
    } else {
        if (data->buffer() == nullptr)
            IE_THROW(NotAllocated) << str_input_not_allocated << " Input name: \'" << name << "\'";
        _inputs[name] = data;
        if (isDynamic) {
            // We must create new input data if it has never been allocated or previously allocated
            // device blob is smaller than currently assigned user blob
            bool needs_realloc = _deviceInputs.find(name) == _deviceInputs.end() || _deviceInputs.at(name)->byteSize() < data->byteSize();
            if (needs_realloc) {
                _deviceInputs[name] = create_device_blob(data->getTensorDesc());
            } else {
                if (_deviceInputs.at(name)->getTensorDesc() != data->getTensorDesc())
                    _deviceInputs[name] = reinterpret_device_blob(_deviceInputs[name], data->getTensorDesc());
            }
        }
    }
}

void InferRequest::set_output(const std::string& name, const Blob::Ptr& data) {
    auto remote_ptr = data->as<gpu::ClBlob>();
    bool is_remote = remote_ptr != nullptr;

    auto node = findOutputByNodeName(name);
    bool isDynamic = node && node->get_output_partial_shape(0).is_dynamic();

    if (is_remote) {
        _deviceOutputs[name] = data;
    } else {
        if (!isDynamic) {
            if (data->buffer() == nullptr)
                IE_THROW(NotAllocated) << str_output_not_allocated << " Output name: \'" << name << "\'";
        }
    }
    _outputs[name] = data;
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

    const TensorDesc& desc = foundInput->getTensorDesc();

    size_t dataBinSize = blobs.front()->size() * blobs.front()->element_size() * blobs.size();
    size_t netReqBinSize = std::accumulate(desc.getDims().begin(), desc.getDims().end(),
                                           desc.getPrecision().size(),
                                           std::multiplies<size_t>());
    if (dataBinSize != netReqBinSize) {
        IE_THROW() << "Incorrect binary data size for input blobs with name: \'" << name <<  "\' " <<
                      "Current: " << dataBinSize << " Required: " << netReqBinSize;
    }

    if (is_surface) {
        for (size_t i = 0; i < blobs.size(); ++i) {
            std::string new_name = name + "_" + std::to_string(i);

            if (_inputs.find(new_name) != _inputs.end()) {
                _inputs.erase(new_name);
            }
        }
    } else {
        if (_inputs.find(name) != _inputs.end()) {
            _inputs.erase(name);
        }
    }

    inputTensorsMap[name] = blobs;
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
        auto node = findInputByNodeName(input.first);
        bool is_dynamic = (node && node->get_output_partial_shape(0).is_dynamic());
        if (!is_dynamic)
            checkInputBlob(input.second, input.first, foundInput);
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
        auto node = findOutputByNodeName(output.first);
        bool is_dynamic = (node && node->get_output_partial_shape(0).is_dynamic());
        if (!is_dynamic)
            checkOutputBlob(output.second, output.first, foundOutput);
    }
}

void InferRequest::SetGraph(std::shared_ptr<Graph> graph) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::SetGraph");
    m_graph = graph;

    if (m_graph == nullptr) {
        IE_THROW(NetworkNotLoaded);
    }

    allocate_inputs();
    allocate_outputs();
    variables_states_ = m_graph->AllocateVariablesMemories();
}

InferRequest::InferRequest(InputsDataMap networkInputs, OutputsDataMap networkOutputs,
                           const CompiledModel::Ptr& execNetwork)
        : IInferRequestInternal(networkInputs, networkOutputs) {
    IE_ASSERT(nullptr != execNetwork);
    streamExecutor = dynamic_cast<InferenceEngine::IStreamsExecutor*>(execNetwork->m_taskExecutor.get());
    m_context = std::dynamic_pointer_cast<InferenceEngine::gpu::ClContext>(execNetwork->GetContext());
    OPENVINO_ASSERT(m_context != nullptr, "[GPU] Can't initialize context of InferRequest: wrong context type");
}

InferRequest::InferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                           const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                           const CompiledModel::Ptr& execNetwork)
        : IInferRequestInternal(inputs, outputs) {
    IE_ASSERT(nullptr != execNetwork);
    streamExecutor = dynamic_cast<InferenceEngine::IStreamsExecutor*>(execNetwork->m_taskExecutor.get());
    m_context = std::dynamic_pointer_cast<InferenceEngine::gpu::ClContext>(execNetwork->GetContext());
    OPENVINO_ASSERT(m_context != nullptr, "[GPU] Can't initialize context of InferRequest: wrong context type");
}

// ----------------------------------------------------------------------------------------- //
// ---------------------------- internal pipeline stages ----------------------------------- //
// ----------------------------------------------------------------------------------------- //
void InferRequest::enqueue_notify() {
    m_graph->wait(Graph::Stage::EXECUTE);
    enqueue();
}

void InferRequest::enqueue() {
    // set input and output memory from request blob maps
    // into the network object primitives
    std::vector<cldnn::event::ptr> dependencies;

    for (const auto& inputTensor : inputTensorsMap) {
        const std::string name = inputTensor.first;
        const auto& blobs = inputTensor.second;

        auto blobsDesc = blobs.front()->getTensorDesc();
        blobsDesc.getDims().front() = blobs.size();

        bool is_surface = std::all_of(blobs.begin(), blobs.end(), [](const Blob::Ptr& blob) {
            return blob->is<gpu::ClImage2DBlob>();
        });
        bool is_buffer = std::all_of(blobs.begin(), blobs.end(), [](const Blob::Ptr& blob) {
            return blob->is<gpu::ClBufferBlob>();
        });
        bool is_remote = is_buffer || is_surface;

        if (is_surface) {
            for (size_t i = 0; i < blobs.size(); ++i) {
                std::string new_name = name + "_" + std::to_string(i);
                _inputs[new_name] = blobs[i];
                _deviceInputs[new_name] = blobs[i];
            }
        } else {
            uint8_t* dst = nullptr;
            if (_deviceInputs.find(name) != _deviceInputs.end()) {
                if (_deviceInputs[name]->getTensorDesc() == blobsDesc) {
                    dst = _deviceInputs[name]->buffer().as<uint8_t*>();
                }
            }
            if (dst == nullptr) {
                cldnn::layout layout(DataTypeFromPrecision(blobsDesc.getPrecision()),
                                     FormatFromTensorDesc(blobsDesc),
                                     tensor_from_dims(blobsDesc.getDims()));

                auto mergedBlobs = create_remote_blob<RemoteCLbuffer>(blobsDesc, layout, BlobType::BT_BUF_INTERNAL);
                dst = mergedBlobs->buffer().as<uint8_t*>();

                _inputs[name] = mergedBlobs;
                if (is_remote) {
                    _deviceInputs[name] = mergedBlobs;
                }
            }

            for (auto& blob : blobs) {
                const uint8_t* src = blob->cbuffer().as<const uint8_t*>();
                std::copy(src, src + blob->byteSize(), dst);
                dst += blob->byteSize();
            }
        }
    }

    for (auto& item : _inputs) {
        std::string inputName = item.first;
        Blob::Ptr& inputBlob = item.second;
        prepare_input(inputName, inputBlob, dependencies);
    }

    cldnn::network::variables_states_map variables_states;
    for (auto &variable_state_pair : variables_states_)
        variables_states.insert({ variable_state_pair.first, variable_state_pair.second[0] });

    auto networkPtr = m_graph->GetNetwork();

    networkPtr->assign_variables_memories(std::move(variables_states));

    for (auto& item : _outputs) {
        std::string outputName = item.first;
        Blob::Ptr& outputBlob = item.second;
        prepare_output(outputName, outputBlob);
    }

    internal_outputs.clear();
    internal_outputs = networkPtr->execute(dependencies);

    // If dump layers path is set, only runs first inference.
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->dump_layers_path.length() > 0) {
        GPU_DEBUG_INFO << "Only run first inference to dump layers." << std::endl;
        exit(0);
    }
}

void InferRequest::wait_notify() {
    wait();
    m_graph->notify(Graph::Stage::EXECUTE);
}

void InferRequest::wait() {
    if (internal_outputs.empty()) {
        IE_THROW() << "Inference was not started!\n";
    }
    // wait for completion & collect outputs as requested by the model
    for (auto& no : _networkOutputs) {
        // In dynamic case, graph API must be used to retrieve outputID
        // because it does not create outputsMap during SetGraph
        std::string outputID = outputsMap.empty() ? m_graph->MapOutputName(no.first) : outputsMap.at(no.first);
        auto outputMemory = internal_outputs.at(outputID).get_memory();
        auto outputLayout = internal_outputs.at(outputID).get_layout();
        if (outputMemory)
            outputMemory = m_graph->get_engine().reinterpret_buffer(*outputMemory, outputLayout);

        bool need_output_update = false;

        if (outputLayout.bytes_count() == 0) { // Output is empty tensor. Just update the output blob shape
            auto dims = SizeVector();
            for (size_t i = 0; i < outputLayout.get_shape().size(); ++i) {
                dims.push_back(outputLayout.get_shape()[i]);
            }
            _outputs[no.first]->setShape(dims);
        } else if (_outputs.find(no.first) == _outputs.end() || _outputs.at(no.first)->byteSize() != outputMemory->size()) {
            need_output_update = true;
        }

        if (need_output_update) {
            auto node = findOutputByNodeName(no.first);
            auto out_partial_shape = node->get_output_partial_shape(0);
            auto mem_dims = outputLayout.get_shape();
            size_t out_rank =  out_partial_shape.size();
            auto precision = InferenceEngine::Precision::FP32;
            auto dims = SizeVector(mem_dims.begin(), mem_dims.end());
            if (static_cast<int32_t>(out_rank) < static_cast<int32_t>(dims.size())) {
                for (size_t i = out_rank; i < dims.size(); i++) {
                    if (dims[i] != 1)
                        IE_THROW() << "[GPU] Unexpected out shape";
                }
                dims.resize(out_rank);
            }
            auto layout_by_rank = [](size_t rank) {
                switch (rank) {
                    case 6: return InferenceEngine::Layout::BLOCKED;
                    case 5: return InferenceEngine::Layout::NCDHW;
                    case 4: return InferenceEngine::Layout::NCHW;
                    case 3: return InferenceEngine::Layout::BLOCKED;
                    case 2: return InferenceEngine::Layout::NC;
                    case 1: return InferenceEngine::Layout::BLOCKED;
                    default: IE_THROW() << "[GPU] Unsupported out rank";
                }
            };
            auto layout = layout_by_rank(out_rank);
            auto tensorDesc = InferenceEngine::TensorDesc(precision, dims, layout);
            if (_outputs.find(no.first) == _outputs.end()) {
                _outputs[no.first] = create_host_blob(tensorDesc, false);
            } else {
                _outputs[no.first]->setShape(dims);
            }
        }
        Blob::Ptr bptr = _outputs[no.first];

        // mapping remote blobs not needed -
        // let the user take care of them explicitly
        if (!bptr->is<gpu::ClBlob>()) {
            bool same_mem = false;
            {
                auto dst_lock = bptr->cbuffer();
                auto dst_ptr = dst_lock.as<uint8_t*>();
                same_mem = outputMemory && same_host_mem(outputMemory, dst_ptr);
            }
            if (!same_mem && outputMemory && outputMemory->size()) {
                copy_output_data(outputMemory, bptr);
            }
        }
    }

    // finally collect profiling info
    if (m_useProfiling) {
        m_graph->UpdatePerfStatistics();
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

Blob::Ptr InferRequest::create_host_blob(const TensorDesc& desc, bool is_dynamic) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::create_host_blob");
    // Disable USM usage as USMHostAllocator may fail for attempt to allocate 0 bytes
    // If we add WA for such case to avoid driver call, then deallocate method will return false and Blob::setShape call will throw an exception
    bool use_usm = m_graph->get_engine().use_unified_shared_memory() && !is_dynamic;
    auto alloc = use_usm ? std::make_shared<USMHostAllocator>(m_context) : CreateDefaultAllocator();
    auto blob = make_blob_with_precision(desc, alloc);
    blob->allocate();
    return blob;
}

template<typename RemoteBlobType, typename>
InferenceEngine::Blob::Ptr InferRequest::create_remote_blob(const InferenceEngine::TensorDesc& desc, const cldnn::layout& layout,
                                                            const BlobType mem_type, void* mem_ptr) {
    auto blob = std::make_shared<RemoteBlobType>(m_context,
                                                 m_graph->GetNetwork()->get_stream(),
                                                 desc,
                                                 layout,
                                                 mem_ptr,
                                                 0,
                                                 0,
                                                 mem_type);
    OPENVINO_ASSERT(blob, "[GPU] Failed to allocate remote blob");
    blob->allocate();
    return blob;
}

template InferenceEngine::Blob::Ptr InferRequest::create_remote_blob<RemoteCLbuffer>(const InferenceEngine::TensorDesc&, const cldnn::layout&,
                                                                                     const BlobType, void*);
template InferenceEngine::Blob::Ptr InferRequest::create_remote_blob<RemoteUSMbuffer>(const InferenceEngine::TensorDesc&, const cldnn::layout&,
                                                                                      const BlobType, void*);

Blob::Ptr InferRequest::create_shared_device_blob(const InferenceEngine::TensorDesc& desc, const cldnn::layout& layout, void* usm_host_mem) {
    auto blob = create_remote_blob<RemoteUSMbuffer>(desc, layout, BlobType::BT_USM_SHARED, usm_host_mem);
    OPENVINO_ASSERT(blob, "[GPU] Failed to allocate shared host <-> device blob");
    return blob;
}

void InferRequest::copy_output_data(cldnn::memory::ptr src, Blob::Ptr dst) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::copy_output_data");
    auto is_convert_needed = [](const Precision& prc) {
        const std::vector<Precision> convert_needed = { Precision::I16, Precision::U16, Precision::FP64,
                                                        Precision::U32, Precision::U64 };
        const std::vector<Precision> convert_not_needed = { Precision::FP32, Precision::FP16, Precision::I64, Precision::I32,
                                                            Precision::I8, Precision::U8, Precision::BOOL };

        if (std::find(convert_needed.begin(), convert_needed.end(), prc) != convert_needed.end())
            return true;
        else if (std::find(convert_not_needed.begin(), convert_not_needed.end(), prc) != convert_not_needed.end())
            return false;
        else
            OPENVINO_ASSERT(false, "[GPU] Plugin does not support output ", prc, " precision");
    };

    const auto convert_needed = is_convert_needed(dst->getTensorDesc().getPrecision());
    const auto src_layout = src->get_layout();
    auto& stream = m_graph->GetNetwork()->get_stream();

    if (convert_needed || src_layout.data_padding) {
        if (!intermediate_output_blob || intermediate_output_blob->byteSize() < src_layout.bytes_count()) {
            auto desc = TensorDesc(Precision::U8,
                                   SizeVector{src_layout.bytes_count()},
                                   InferenceEngine::Layout::C);
            intermediate_output_blob = create_host_blob(desc, src_layout.is_dynamic());
        }

        OPENVINO_ASSERT(intermediate_output_blob, "[GPU] Intermediate blob for outputs precessing is not allocated");

        src->copy_to(stream, intermediate_output_blob->buffer());

        switch (dst->getTensorDesc().getPrecision()) {
        #define CASE(PRC, SRC_DT, DST_DT) \
            case PRC: copy_result_to_output_blob<SRC_DT, DST_DT> (intermediate_output_blob, dst, src_layout); break;
        CASE(Precision::FP64, float,    double)
        CASE(Precision::FP32, float,    float)
        CASE(Precision::FP16, uint16_t, uint16_t)
        CASE(Precision::I64,  int64_t,  int64_t)
        CASE(Precision::I32,  int32_t,  int32_t)
        CASE(Precision::I16,  float,    int16_t)
        CASE(Precision::I8,   int8_t,   int8_t)
        CASE(Precision::U16,  float,    uint16_t)
        CASE(Precision::U32,  int32_t,  uint32_t)
        CASE(Precision::U64,  int32_t,  uint64_t)
        CASE(Precision::U8,   uint8_t,  uint8_t)
        CASE(Precision::BOOL, int8_t,   int8_t)
        #undef CASE
        default: break;
        }
    } else {
        auto dst_ptr = dst->buffer().as<void*>();
        src->copy_to(stream, dst_ptr);
    }
}

void InferRequest::copy_input_data(std::shared_ptr<cldnn::network> network,
                                        const cldnn::primitive_id &inputName,
                                        const cldnn::layout& inputLayout,
                                        const Blob &inputBlob) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::copy_input_data");

    cldnn::primitive_id internalName = "parameter:" + inputName;
    auto locked = inputBlob.cbuffer();
    switch (inputBlob.getTensorDesc().getPrecision()) {
    case Precision::FP32: {
        float* blob_ptr = const_cast<float*>(locked.as<const float*>());
        network->set_input_data(internalName, network->get_engine().attach_memory(inputLayout, blob_ptr));
        break;
    }
    case Precision::I32: {
        int32_t* blob_ptr = const_cast<int32_t*>(locked.as<const int32_t*>());
        network->set_input_data(internalName, network->get_engine().attach_memory(inputLayout, blob_ptr));
        break;
    }
    case Precision::I64: {
        int64_t* blob_ptr = const_cast<int64_t*>(locked.as<const int64_t*>());
        network->set_input_data(internalName, network->get_engine().attach_memory(inputLayout, blob_ptr));
        break;
    }
    case Precision::FP16: {
        uint16_t* blob_ptr = const_cast<uint16_t*>(locked.as<const uint16_t*>());
        network->set_input_data(internalName, network->get_engine().attach_memory(inputLayout, blob_ptr));
        break;
    }
    case Precision::I8: {
        int8_t* blob_ptr = const_cast<int8_t*>(locked.as<const int8_t*>());
        network->set_input_data(internalName, network->get_engine().attach_memory(inputLayout, blob_ptr));
        break;
    }
    case Precision::U8: {
        uint8_t* blob_ptr = const_cast<uint8_t*>(locked.as<const uint8_t*>());
        network->set_input_data(internalName, network->get_engine().attach_memory(inputLayout, blob_ptr));
        break;
    }
    case Precision::BOOL: {
        uint8_t* blob_ptr = const_cast<uint8_t*>(locked.as<const uint8_t*>());
        network->set_input_data(internalName, network->get_engine().attach_memory(inputLayout, blob_ptr));
        break;
    }
    default:
        IE_THROW() << "The plugin does not support input " << inputBlob.getTensorDesc().getPrecision() << " precision";
    }
}

void InferRequest::allocate_inputs() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::allocate_inputs");
    auto inputLayouts = m_graph->GetInputLayouts();

    // allocate inputs
    for (auto& ni : _networkInputs) {
        std::string name = ni.first;
        const TensorDesc& desc = ni.second->getTensorDesc();

        bool is_nv12_input = false;

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
            OPENVINO_ASSERT(litr != inputLayouts.end(), "[GPU] Input layout for ", name, " is not found");
            const auto input_layout = litr->second;

            GPU_DEBUG_LOG << "[" << name << ": input blob]" << std::endl;
            if (desc.getPrecision() == Precision::I16 || desc.getPrecision() == Precision::U16) {
                TensorDesc desc_fp32 = desc;
                desc_fp32.setPrecision(Precision::FP32);
                _inputs[name] = create_host_blob(desc, input_layout.is_dynamic());
                if (input_layout.is_static())
                    _deviceInputs[name] = create_device_blob(desc_fp32);
            } else {
                _inputs[name] = create_host_blob(desc, input_layout.is_dynamic());
                // Pre-allocate device input only if USM is not supported; in other case it will be allocated
                // in prepare_input() function later
                if (input_layout.is_static() && !m_graph->get_engine().use_unified_shared_memory()) {
                    _deviceInputs[name] = create_device_blob(desc);
                }
            }
        }
    }
}

void InferRequest::allocate_outputs() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::allocate_outputs");

    // allocate outputs
    for (auto& no : _networkOutputs) {
        std::string outputID = m_graph->MapOutputName(no.first);
        const cldnn::layout output_layout = m_graph->GetNetwork()->get_node_output_layout(outputID);
        TensorDesc desc = no.second->getTensorDesc();
        // Due to some reason TensorDesc in InferRequest contains wrong dims
        // while ExecutableNetwork contains proper ones. Thus replace dims with once from exec network
        // Can be removed once 76176 is resolved.
        if (output_layout.is_static())
            desc.setDims(m_graph->GetOutputSize(no.first));

        GPU_DEBUG_LOG << "[" << no.first << ": output blob]" << std::endl;

        outputsMap[no.first] = outputID;
        if (desc.getPrecision() == Precision::I16 || desc.getPrecision() == Precision::U16 ||
            desc.getPrecision() == Precision::U32 || desc.getPrecision() == Precision::U64 ||
            desc.getPrecision() == Precision::FP64) {
            TensorDesc device_blob_desc = desc;

            if (desc.getPrecision() == Precision::U32 || desc.getPrecision() == Precision::U64)
                device_blob_desc.setPrecision(Precision::I32);
            else
                device_blob_desc.setPrecision(Precision::FP32);

            _outputs[no.first] = create_host_blob(desc, output_layout.is_dynamic());
            if (output_layout.is_static())
                _deviceOutputs[no.first] = create_device_blob(device_blob_desc);
        } else {
            _outputs[no.first] = create_host_blob(desc, output_layout.is_dynamic());
            // Pre-allocate device output only if USM is not supported; in other case it will be allocated
            // in prepare_output() function later
            if (output_layout.is_static() && !m_graph->get_engine().use_unified_shared_memory()) {
                _deviceOutputs[no.first] = create_device_blob(desc);
            }
        }
    }
}

void InferRequest::InferImpl() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::InferImpl");
    setup_stream_graph();
    std::lock_guard<std::mutex> lk(m_graph->get_mutex());
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

void InferRequest::allocate_dev_mem_if_needed(InferenceEngine::BlobMap& device_mems, InferenceEngine::Blob::Ptr& user_blob,
                                              const cldnn::primitive_id& blob_name, const cldnn::layout& layout, bool need_lockable_mem) {
    const auto input_ptr = static_cast<const void*>(user_blob->cbuffer());
    const auto alloc_type = m_graph->get_engine().detect_usm_allocation_type(input_ptr);
    const auto is_usm_host = alloc_type == cldnn::allocation_type::usm_host;
    const auto has_device_blob = device_mems.find(blob_name) != device_mems.end();
    bool can_skip_allocation = false;

    if (has_device_blob) {
        auto impl = getBlobImpl(device_mems[blob_name]->as<gpu::ClBlob>());

        OPENVINO_ASSERT(impl, str_device_output_unsupported_blob);
        OPENVINO_ASSERT(impl->is_allocated(), str_input_not_allocated);

        auto impl_mem = impl->get_memory();
        auto src_ptr = user_blob->cbuffer().as<uint8_t*>();
        // If device mem already exists, we can reuse blob if buffer has usm_host type and points to the same memory,
        // so we don't need to allocate new memory
        can_skip_allocation |= same_host_mem(impl_mem, src_ptr);
        // Or if blob has any type except usm_host - in that case explicit copy will be performed anyway
        // Or if blob has usm_host type and lockable memory is expected by impl
        can_skip_allocation |= need_lockable_mem ? impl_mem->get_allocation_type() == cldnn::allocation_type::usm_host
                                                 : impl_mem->get_allocation_type() != cldnn::allocation_type::usm_host;
        // In case of lockable memory we need to keep updated device's usm_host memory buffer with
        // user's blob to avoid incorrect behaviour if user will call set_blob() with
        // the following sequence (usm_host, system_host, usm_host, system_host...)
        if (need_lockable_mem)
            can_skip_allocation &= users_blobs_matching.find(blob_name) != users_blobs_matching.end()
                                && users_blobs_matching[blob_name] == user_blob;
    }

    if (!can_skip_allocation) {
        if (is_usm_host) {
            // For USM case we create host blob using custom USM host allocator
            // and then create shared device blob on top of this buffer
            device_mems[blob_name] = create_shared_device_blob(user_blob->getTensorDesc(), layout, user_blob->buffer().as<void*>());
        } else if (need_lockable_mem) {
            device_mems[blob_name] =
                create_remote_blob<RemoteUSMbuffer>(user_blob->getTensorDesc(), layout, BlobType::BT_USM_HOST_INTERNAL);
        } else {
            device_mems[blob_name] = create_device_blob(user_blob->getTensorDesc());
        }
        users_blobs_matching[blob_name] = user_blob;
    }
}

void InferRequest::prepare_input(const cldnn::primitive_id& inputName, Blob::Ptr& inputBlob,
                                      std::vector<cldnn::event::ptr>& dependencies) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "InferRequest::prepare_input");
    auto inputLayoutItr = m_graph->GetInputLayouts().find(inputName);
    OPENVINO_ASSERT(inputLayoutItr != m_graph->GetInputLayouts().end(), "[GPU] Input name mismatch");

    auto input_layout = inputLayoutItr->second;
    auto& prec = inputBlob->getTensorDesc().getPrecision();
    auto remote_ptr = inputBlob->as<gpu::ClBlob>();
    auto& stream = m_graph->GetNetwork()->get_stream();
    const bool is_dev_input = remote_ptr != nullptr;
    const bool can_use_usm = m_graph->get_engine().use_unified_shared_memory();

    auto conv_to_supported_prec = [](Precision::ePrecision prec) {
        switch (prec) {
            case Precision::I16:
            case Precision::U16:
            case Precision::FP64:
                return Precision::FP32;
            case Precision::U64:
            case Precision::U32:
                return Precision::I32;
            default: return prec;
        }
    };

    if (input_layout.is_dynamic()) {
        bool has_device_blob = _deviceInputs.find(inputName) != _deviceInputs.end();
        bool should_allocate_device_blob = !has_device_blob;
        if (has_device_blob) {
            auto device_blob = _deviceInputs.at(inputName);
            if (device_blob->byteSize() < inputBlob->byteSize()) {
                should_allocate_device_blob = true;
            }
        }

        if (should_allocate_device_blob) {
            _deviceInputs[inputName] = create_device_blob(inputBlob->getTensorDesc());
        } else {
            _deviceInputs[inputName] = reinterpret_device_blob(_deviceInputs[inputName], inputBlob->getTensorDesc());
        }
    } else if (input_layout.is_static() && !is_dev_input && can_use_usm) {
        allocate_dev_mem_if_needed(_deviceInputs, inputBlob, inputName, input_layout, (conv_to_supported_prec(prec) != prec));
    }
    OPENVINO_ASSERT(_deviceInputs.find(inputName) != _deviceInputs.end(), "[GPU] Couldn't find device blob allocated for ", inputName, " input");
    auto reqBlob = _deviceInputs.at(inputName)->as<gpu::ClBlob>();
    auto _nw_ptr = m_graph->GetNetwork();
    const cldnn::primitive_id internalName = "parameter:" + inputName;

    switch (prec) {
        case Precision::FP64:
        case Precision::FP32:
        case Precision::FP16:
        case Precision::I8:
        case Precision::U8:
        case Precision::BOOL:
        case Precision::I16:
        case Precision::U16:
        case Precision::I32:
        case Precision::U32:
        case Precision::U64:
        case Precision::I64: {
            auto impl = getBlobImpl(is_dev_input ?
                                    remote_ptr :
                                    reqBlob);
            if (!impl->is_allocated()) {
                IE_THROW() << str_input_not_allocated;
            }
            auto inputMem = impl->get_memory();

            auto input_layout = m_graph->GetInputLayouts().find(inputName);
            if (input_layout != m_graph->GetInputLayouts().end()) {
                if (input_layout->second.format != inputMem->get_layout().format && input_layout->second.is_static()) {
                    inputMem = m_graph->GetNetwork()->get_engine().reinterpret_buffer(*inputMem, input_layout->second);
                }
            }

            if (!is_dev_input) {
                Precision conv_prec = conv_to_supported_prec(prec);
                // TODO: Remove this checks once 95363 issue is solved
                if (conv_prec != prec && conv_prec == Precision::FP32) {
                    // GPU plugin doesn't support I16 input precision,
                    // so have to convert input data to fp32 precision
                    cldnn::mem_lock<float> ptr{ inputMem, stream };
                    if (prec == Precision::I16) {
                        convertAndCopy<int16_t, float>(inputBlob.get(), ptr.data());
                    } else if (prec == Precision::U16) {
                        convertAndCopy<uint16_t, float>(inputBlob.get(), ptr.data());
                    } else {
                        convertAndCopy<double, float>(inputBlob.get(), ptr.data());
                    }
                } else if (conv_prec != prec && conv_prec == Precision::I32) {
                    cldnn::mem_lock<int32_t> ptr{ inputMem, stream };
                    if (prec == Precision::U64) {
                        convertAndCopy<uint64_t, int32_t>(inputBlob.get(), ptr.data());
                    } else {
                        convertAndCopy<uint32_t, int32_t>(inputBlob.get(), ptr.data());
                    }
                } else {
                    auto src_lock = inputBlob->cbuffer();
                    auto src_ptr = src_lock.as<uint8_t*>();
                    if (!same_host_mem(inputMem, src_ptr)) {
                        auto ev = inputMem->copy_from(stream, src_ptr, false);
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
    const auto output_id = outputsMap.at(outputName);
    const auto output_layout = m_graph->GetNetwork()->get_node_output_layout(output_id);
    const bool is_static = output_layout.is_static();
    const bool can_use_usm = m_graph->get_engine().use_unified_shared_memory();
    auto remote_ptr = outputBlob->as<gpu::ClBlob>();
    const bool is_dev_input = remote_ptr != nullptr;

    if (is_static && can_use_usm && !is_dev_input) {
        auto is_cpu_impl = m_graph->GetNetwork()->is_cpu_impl(output_id);
        allocate_dev_mem_if_needed(_deviceOutputs, outputBlob, outputName, output_layout, is_cpu_impl);
    }

    OPENVINO_ASSERT(!is_static || _deviceOutputs.find(outputName) != _deviceOutputs.end(),
                    "[GPU] Couldn't find device blob allocated for ", outputName, " output");
    // Missing output in _deviceOutputs means that the network is dynamic and outputs couldn't be pre-allocated
    if (_deviceOutputs.find(outputName) == _deviceOutputs.end())
        return;
    Blob::Ptr reqBlob = _deviceOutputs.at(outputName);
    cldnn::primitive_id internalName = outputsMap[outputName];
    auto _nw_ptr = m_graph->GetNetwork();
    auto output_blob_ptr = (reqBlob != outputBlob && is_dev_input)
        ? remote_ptr
        : reqBlob->as<gpu::ClBlob>();
    auto impl = getBlobImpl(output_blob_ptr);
    if (!impl->is_allocated()) {
        IE_THROW(NotAllocated) << str_output_not_allocated;
    }
    auto outputMem = impl->get_memory();
    _nw_ptr->set_output_memory(internalName, outputMem);
}

InferenceEngine::Blob::Ptr InferRequest::create_device_blob(const InferenceEngine::TensorDesc& desc) {
    auto format = FormatFromLayout(desc.getLayout());
    auto dt = DataTypeFromPrecision(desc.getPrecision());
    ov::PartialShape shape(desc.getDims());

    // Currently, clDeviceMemAllocINTEL returns memory address allocated to other input blob if the current blob is empty
    // W/A for this issue:
    // Allocate with non-empty shape and then reinterprete with original shape
    for (auto &i : shape) {
        if (i == 0)
            i = 1;
    }

    auto l = cldnn::layout(shape, dt, format);

    if (m_graph->get_engine().use_unified_shared_memory()) {
        auto blob = create_remote_blob<RemoteUSMbuffer>(desc, l, BlobType::BT_USM_DEVICE_INTERNAL);
        return reinterpret_device_blob(blob, desc);
    } else {
        return create_remote_blob<RemoteCLbuffer>(desc, l, BlobType::BT_BUF_INTERNAL);
    }
}

std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> InferRequest::QueryState() {
    std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> ret{};
    ret.reserve(variables_states_.size());
    for (const auto& pair : variables_states_)
        ret.push_back(std::make_shared<VariableState>(pair.first, pair.second, m_graph->get_engine(), m_curBatch));
    return ret;
}

Blob::Ptr InferRequest::reinterpret_device_blob(Blob::Ptr data, const TensorDesc& new_desc) {
    auto format = FormatFromLayout(new_desc.getLayout());
    auto dt = DataTypeFromPrecision(new_desc.getPrecision());
    ov::PartialShape shape(new_desc.getDims());

    auto l = cldnn::layout(shape, dt, format);

    auto remote_blob = data->as<gpu::ClBlob>();
    if (!remote_blob)
        IE_THROW() << "Invalid blob used for reinterpretation";

    remote_blob->setShape(new_desc.getDims());

    auto impl = getBlobImpl(remote_blob);
    impl->reinterpret(l);

    return data;
}

}  // namespace intel_gpu
}  // namespace ov
