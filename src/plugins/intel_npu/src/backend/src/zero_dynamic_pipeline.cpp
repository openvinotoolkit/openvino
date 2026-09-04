// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_dynamic_pipeline.hpp"

#include <level_zero/ze_api.h>
#include <ze_graph_ext.h>

#include <sstream>

#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/vm/mem_ref_type.hpp"
#include "intel_npu/utils/vm/npu_vm_runtime_api.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_cmd_queue_pool.hpp"
#include "intel_npu/utils/zero/zero_remote_tensor.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"
#include "zero_infer_request.hpp"

namespace intel_npu {

struct MemRefTypeImpl {
    npu_vm_runtime_mem_ref_handle_t _memRef;

    MemRefTypeImpl() : _memRef(nullptr) {}

    ~MemRefTypeImpl() {
        destroyMemRef();
    }

    void alignWithHandle(MemRefType& memref) {
        if (_memRef == nullptr) {
            return;
        }

        if (npuVMRuntimeParseMemRef(_memRef,
                                    &memref._basePtr,
                                    &memref._data,
                                    &memref._offset,
                                    memref._sizes.data(),
                                    memref._strides.data(),
                                    &memref._dimsCount) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
            OPENVINO_THROW("Failed to parse MemRef handle");
        }
    }

    bool UpdateMemRefHandleStatus(MemRefType& memref, bool useV2Api = false) {
        // Update current MemRef handle to use latest metadata
        const bool handleCreated = _memRef == nullptr;
        if (_memRef == nullptr) {
            createMemRef(memref._dimsCount);
        }

        const uint32_t dirtyFlag = handleCreated ? MemRefType::ALL_DIRTY : memref.getDirtyFlag();
        if (dirtyFlag != 0) {
            auto result = npuVMRuntimeSetMemRef(
                _memRef,
                memref._basePtr,
                memref._data,
                memref._offset,
                useV2Api && !(dirtyFlag & MemRefType::SHAPE_DIRTY) ? nullptr : memref._sizes.data(),
                useV2Api && !(dirtyFlag & MemRefType::STRIDE_DIRTY) ? nullptr : memref._strides.data(),
                memref._dimsCount);
            if (result != NPU_VM_RUNTIME_RESULT_SUCCESS) {
                memref.markDirty(dirtyFlag);
                OPENVINO_THROW("Failed to update MemRef handle");
            }
            memref.clearDirty();
        }
        return dirtyFlag != 0;
    }

private:
    void createMemRef(int64_t dimsCount) {
        if (_memRef == nullptr) {
            auto result = npuVMRuntimeCreateMemRef(dimsCount, &_memRef);
            if (result != NPU_VM_RUNTIME_RESULT_SUCCESS) {
                OPENVINO_THROW("Failed to create MemRef handle");
            }
        }
    }

    void destroyMemRef() {
        if (_memRef != nullptr) {
            npuVMRuntimeDestroyMemRef(_memRef);
            _memRef = nullptr;
        }
    }
};

VMExecutionContext::~VMExecutionContext() {
    if (_handle != nullptr) {
        npuVMRuntimeDestroyExecutionContext(_handle);
        _handle = nullptr;
    }
}

npu_vm_runtime_execution_context_handle_t VMExecutionContext::ensure(npu_vm_runtime_handle_t vmRuntime) {
    if (_handle == nullptr) {
        const npu_vm_runtime_result_t result = npuVMRuntimeCreateExecutionContext(vmRuntime, &_handle);
        if (result != NPU_VM_RUNTIME_RESULT_SUCCESS) {
            OPENVINO_THROW("Failed to create a VM execution context");
        }
    }
    return _handle;
}

npu_vm_runtime_execution_context_handle_t VMExecutionContext::ensureV2(
    npu_vm_runtime_handle_t vmRuntime,
    ze_context_handle_t ctx,
    ze_device_handle_t device,
    ze_command_queue_handle_t commandQueue,
    ze_graph_dditable_ext_t* graphDdiTableExt,
    ze_command_queue_npu_dditable_ext_t* queueDdiTableExt) {
    if (_handle == nullptr) {
        npu_vm_runtime_create_execution_context_params_t params = {ctx,
                                                                   device,
                                                                   commandQueue,
                                                                   graphDdiTableExt,
                                                                   queueDdiTableExt};
        const npu_vm_runtime_result_t result = npuVMRuntimeCreateExecutionContext2(vmRuntime, &params, &_handle);
        if (result != NPU_VM_RUNTIME_RESULT_SUCCESS) {
            OPENVINO_THROW("Failed to create a VM execution context (v2)");
        }
    }
    return _handle;
}

// Init _inputsMemRef and _outputsMemRef vectors
void DynamicArguments::setArgumentProperties(uint32_t argi,
                                             const void* argv,
                                             const ov::Shape& sizes,
                                             const std::vector<size_t>& strides) {
    auto assign_slot = [&](MemRefType& slot) {
        if (strides.size() != sizes.size()) {
            OPENVINO_THROW("Updated shape and stride count mismatch: shape rank and stride count differ. Shape rank: ",
                           sizes.size(),
                           ", stride count: ",
                           strides.size());
        }
        slot.setArg(argv);
        slot.setSize(sizes);
        slot.setStrides(ov::Strides(strides));
    };

    if (argi < _inputsMemRef.size()) {
        assign_slot(_inputsMemRef[argi]);
    } else {
        auto idx = argi - _inputsMemRef.size();
        if (idx < _outputsMemRef.size()) {
            assign_slot(_outputsMemRef[idx]);
        }
    }
}

DynamicPipeline::DynamicPipeline(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                 const std::shared_ptr<IGraph>& graph,
                                 const Config& config,
                                 const std::vector<std::vector<std::shared_ptr<ZeroTensor>>>& input_tensors,
                                 const std::vector<std::shared_ptr<ZeroTensor>>& output_tensors)
    : IPipeline(init_structs, graph, utils::DEFAULT_BATCH_SIZE, config, "DynamicPipeline") {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Zero_infer_request::DynamicPipeline::DynamicPipeline");

    OPENVINO_ASSERT(!_run_inferences_sequentially, "In-order execution doesn't work for dynamic pipeline");

    _logger.debug("Initialization started");

    const auto versionResult = npuVMRuntimeGetAPIVersion(&_apiVersion);
    if (versionResult != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to get VM runtime version, error code: ", versionResult);
    }

    if (use_npu_vm_runtime_v2_api(_apiVersion)) {
        _logger.debug("DynamicPipeline: using v2.0 VM runtime API");
        const npu_vm_runtime_handle_t vmRuntime = static_cast<npu_vm_runtime_handle_t>(_graph->get_handle());
        const auto commandQueueDesc = _graph->get_command_queue_desc();
        _executionContext.ensureV2(vmRuntime,
                                   init_structs->getContext(),
                                   init_structs->getDevice(),
                                   commandQueueDesc.shared_common_queue() ? _command_queue->handle() : nullptr,
                                   _init_structs->getGraphDdiTable().getImpl(),
                                   _init_structs->getCommandQueueDdiTable().getImpl());
        _runtime_config_command_queue_desc = commandQueueDesc;
        _runtime_config_command_queue_desc_valid = true;
    } else {
        _logger.debug("DynamicPipeline: using v1.x VM runtime API");
        const npu_vm_runtime_handle_t vmRuntime = static_cast<npu_vm_runtime_handle_t>(_graph->get_handle());
        _executionContext.ensure(vmRuntime);
    }

    if (!use_npu_vm_runtime_v2_api(_apiVersion) && !_sync_output_with_fences) {
        _event_pool = std::make_shared<EventPool>(_init_structs, 1);
        _events.emplace_back(std::make_shared<Event>(_event_pool, 0));
    }
    _logger.debug("Event pool and command queue setup completed");

    const uint64_t num_of_subgraphs = _graph->get_metadata().numberOfSubgraphs;
    if (!use_npu_vm_runtime_v2_api(_apiVersion) && _sync_output_with_fences) {
        _fences.emplace_back(std::make_unique<Fence>(_command_queue));
    }

    _command_list_group = std::make_unique<PipelinedCommandLists>(num_of_subgraphs,
                                                                  _init_structs,
                                                                  use_npu_vm_runtime_v2_api(_apiVersion));

    auto& commandLists = _command_list_group;
    commandLists->initArguments(_graph->get_metadata());
    auto& dynamicArguments = commandLists->getArguments();

    size_t io_index = 0;
    for (const auto& desc : _graph->get_metadata().inputs) {
        // DynamicPipeline does not currently support weightless model, just thrown exception.
        OPENVINO_ASSERT(!desc.isMainInputWeights,
                        "DynamicPipeline does not support weightless graphs (input '",
                        desc.nameFromCompiler,
                        "' is a main-input weight)");
        OPENVINO_ASSERT(input_tensors.at(io_index).size() == 1,
                        "DynamicPipeline requires one full-batch tensor per input");

        _logger.debug("Update tensor property for input desc index: %u", desc.indexUsedByDriver);
        const auto& tensor = input_tensors.at(io_index).at(SINGLE_TENSOR);
        size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
        dynamicArguments.setArgumentProperties(desc.indexUsedByDriver,
                                               tensor->data(),
                                               tensor->get_shape(),
                                               get_strides(tensor->get_strides(), elementSize));
        ++io_index;
    }

    io_index = 0;
    for (const auto& desc : _graph->get_metadata().outputs) {
        _logger.debug("Update tensor property for output desc index: %u", desc.indexUsedByDriver);
        const auto& tensor = output_tensors.at(io_index);
        size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
        dynamicArguments.setArgumentProperties(desc.indexUsedByDriver,
                                               tensor->data(),
                                               tensor->get_shape(),
                                               get_strides(tensor->get_strides(), elementSize));
        ++io_index;
    }
    _logger.debug("Initialization completed");
}

void DynamicPipeline::push() {
    _logger.debug("push - started");

    const npu_vm_runtime_handle_t vmRuntime = static_cast<npu_vm_runtime_handle_t>(_graph->get_handle());
    OPENVINO_ASSERT(vmRuntime != nullptr, "DynamicPipeline requires a valid VM runtime engine");

    const auto useV2Api = use_npu_vm_runtime_v2_api(_apiVersion);
    const auto commandQueueDesc = _graph->get_command_queue_desc();
    const bool commandQueueVersionChanged = (commandQueueDesc.key() != _command_queue->desc().key());

    const npu_vm_runtime_config_desc_t* runtimeConfig = nullptr;
    if (useV2Api) {
        if (commandQueueVersionChanged && commandQueueDesc.shared_common_queue()) {
            _command_queue = ZeroCmdQueuePool::getInstance().getCommandQueue(_init_structs, commandQueueDesc);
        }

        if (_runtime_config_command_queue_desc_valid) {
            runtimeConfig = update_runtime_config(_runtime_config_command_queue_desc, commandQueueDesc);
        } else {
            _runtime_config_command_queue_desc = commandQueueDesc;
            _runtime_config_command_queue_desc_valid = true;
        }
    } else if (commandQueueVersionChanged) {
        _command_queue = ZeroCmdQueuePool::getInstance().getCommandQueue(_init_structs, commandQueueDesc);

        if (_sync_output_with_fences) {
            for (size_t i = 0; i < _fences.size(); i++) {
                _fences[i] = std::make_unique<Fence>(_command_queue);
            }
        }
    }

    const auto commandQueueHandle =
        useV2Api && !commandQueueDesc.shared_common_queue() ? nullptr : _command_queue->handle();
    OV_ITT_TASK_CHAIN(ZERO_PIPELINE_IP_PUSH, itt::domains::LevelZeroBackend, "Pipeline", "push");
    auto& commandLists = _command_list_group;
    auto& dynamicArguments = commandLists->getArguments();
    if (_logger.level() >= ov::log::Level::DEBUG) {
        _logger.debug("push - inputs info for dynamic graph:");
        for (auto& memType : dynamicArguments._inputsMemRef) {
            _logger.debug("push - input: %s", memType.toString().c_str());
        }
        _logger.debug("push - outputs info for dynamic graph:");
        for (auto& memType : dynamicArguments._outputsMemRef) {
            _logger.debug("push - output: %s", memType.toString().c_str());
        }
    }

    if (useV2Api) {
        execute_vm_runtime_v2(vmRuntime, dynamicArguments, commandQueueHandle, runtimeConfig);
        _runtime_config_command_queue_desc = commandQueueDesc;
        _runtime_config_command_queue_desc_valid = true;
    } else {
        const ze_fence_handle_t fence = _sync_output_with_fences ? _fences.front()->handle() : nullptr;
        execute_vm_runtime(vmRuntime, dynamicArguments, commandLists->getHandles(), commandQueueHandle, fence, nullptr);
    }

    _logger.debug("push - completed");
}

const npu_vm_runtime_config_desc_t* DynamicPipeline::update_runtime_config(
    const CommandQueueDesc& previousCommandQueueDesc,
    const CommandQueueDesc& currentCommandQueueDesc) {
    _runtimeConfigChain.clear();
    if (previousCommandQueueDesc.priority() != currentCommandQueueDesc.priority()) {
        _runtimeConfigChain.append(NPU_VM_RUNTIME_CONFIG_TYPE_QUEUE_PRIORITY,
                                   static_cast<npu_vm_runtime_config_value_t>(currentCommandQueueDesc.priority()));
    }
    if (previousCommandQueueDesc.workload() != currentCommandQueueDesc.workload() &&
        currentCommandQueueDesc.workload().has_value()) {
        _runtimeConfigChain.append(
            NPU_VM_RUNTIME_CONFIG_TYPE_WORKLOAD_TYPE,
            static_cast<npu_vm_runtime_config_value_t>(currentCommandQueueDesc.workload().value()));
    }
    if (previousCommandQueueDesc.options() != currentCommandQueueDesc.options()) {
        _runtimeConfigChain.append(NPU_VM_RUNTIME_CONFIG_TYPE_QUEUE_OPTIONS, currentCommandQueueDesc.options());
    }
    return _runtimeConfigChain.head();
}

void DynamicPipeline::execute_vm_runtime(npu_vm_runtime_handle_t vmRuntime,
                                         DynamicArguments& args,
                                         std::vector<ze_command_list_handle_t>& commandLists,
                                         ze_command_queue_handle_t commandQueue,
                                         ze_fence_handle_t fence,
                                         ze_event_handle_t event) {
    _logger.debug("Start to execute graph with runtime engine");

    const bool firstInference = !args._commandListsRecorded;
    bool noTensorChange = true;

    // These vectors only stage the memref handle pointers for the npuVMRuntimeExecute call below. The handle objects
    // themselves are owned by the persistent DynamicArguments (args._inputsMemRef[i]._impl), so keeping these vectors
    // local is safe even though device execution is async.
    std::vector<npu_vm_runtime_mem_ref_handle_t> inputMemRefHandles, outputMemRefHandles;

    auto processMemRefs = [&](auto& memRefs, auto& targetMemRefHandles) {
        targetMemRefHandles.clear();
        targetMemRefHandles.reserve(memRefs.size());

        for (auto& memref : memRefs) {
            std::shared_ptr<MemRefTypeImpl> impl = std::static_pointer_cast<MemRefTypeImpl>(memref._impl);
            if (impl == nullptr) {
                impl = std::make_shared<MemRefTypeImpl>();
                memref._impl = impl;
            }
            const bool memRefUpdated = impl->UpdateMemRefHandleStatus(memref);

            // VM runtime execute path always needs current memref handles.
            targetMemRefHandles.push_back(impl->_memRef);

            if (memRefUpdated) {
                noTensorChange = false;
            }
        }
    };

    processMemRefs(args._inputsMemRef, inputMemRefHandles);
    processMemRefs(args._outputsMemRef, outputMemRefHandles);

    if (!firstInference && noTensorChange) {
        _logger.debug("Reuse command list without update since no tensor change detected");
        auto result = zeCommandQueueExecuteCommandLists(commandQueue,
                                                        static_cast<uint32_t>(commandLists.size()),
                                                        commandLists.data(),
                                                        fence);
        if (result != ZE_RESULT_SUCCESS) {
            OPENVINO_THROW("Failed to submit command lists");
        }
        return;
    }

    _logger.debug("Reset command list to run with runtime");
    // Reset commandLists since there are tensor with new shapes or it is the first execution, can not reuse command
    // list with update
    for (auto& cmdList : commandLists) {
        const auto result = zeCommandListReset(cmdList);
        if (result != ZE_RESULT_SUCCESS) {
            OPENVINO_THROW("Failed to reset command list");
        }
    }

    npu_vm_runtime_execute_params_t params = {};
    params.executionContext = _executionContext.ensure(vmRuntime);
    params.pInputs = inputMemRefHandles.data();
    params.numOfInputs = static_cast<uint32_t>(inputMemRefHandles.size());
    params.pOutputs = outputMemRefHandles.data();
    params.numOfOutputs = static_cast<uint32_t>(outputMemRefHandles.size());
    params.ctx = _init_structs->getContext();
    params.device = _init_structs->getDevice();
    params.graphDdiTableExt = _init_structs->getGraphDdiTable().getImpl();
    params.commandLists = commandLists.data();
    params.numCommandLists = static_cast<uint64_t>(commandLists.size());
    params.commandQueue = commandQueue;
    params.inferenceFence = fence;
    params.event = event;

    _logger.debug("Execute graph with runtime engine");
    if (npuVMRuntimeExecute(vmRuntime, &params) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to execute VM runtime engine");
    }
    _logger.debug("Execution runtime engine is created successfully.");

    args._commandListsRecorded = true;
    _logger.debug("Completed to execute graph with runtime engine");
}

void DynamicPipeline::execute_vm_runtime_v2(npu_vm_runtime_handle_t vmRuntime,
                                            DynamicArguments& args,
                                            ze_command_queue_handle_t commandQueue,
                                            const npu_vm_runtime_config_desc_t* pConfig) {
    _logger.debug("execute_vm_runtime_v2 - started");

    auto processMemRefs = [&](auto& memRefs, auto& targetHandles) {
        targetHandles.clear();
        targetHandles.reserve(memRefs.size());
        for (auto& memref : memRefs) {
            auto impl = std::static_pointer_cast<MemRefTypeImpl>(memref._impl);
            if (impl == nullptr) {
                impl = std::make_shared<MemRefTypeImpl>();
                memref._impl = impl;
            }
            impl->UpdateMemRefHandleStatus(memref, true);
            targetHandles.push_back(impl->_memRef);
        }
    };

    processMemRefs(args._inputsMemRef, args._inputMemRefHandles);
    processMemRefs(args._outputsMemRef, args._outputMemRefHandles);

    auto& params = args._executeParams2;
    params.commandQueue = commandQueue;
    params.pConfig = pConfig;
    params.pInputs = args._inputMemRefHandles.data();
    params.numOfInputs = static_cast<uint32_t>(args._inputMemRefHandles.size());
    params.pOutputs = args._outputMemRefHandles.data();
    params.numOfOutputs = static_cast<uint32_t>(args._outputMemRefHandles.size());
    params.executionContext = _executionContext.handle();

    _logger.debug("execute_vm_runtime_v2 - calling npuVMRuntimeExecute2");
    const auto result = npuVMRuntimeExecute2(vmRuntime, &params);
    if (result != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to execute VM runtime engine (v2), error code: ", result);
    }

    _logger.debug("execute_vm_runtime_v2 - completed");
}

std::vector<ov::Shape> DynamicPipeline::predict_output_shapes(
    const std::vector<std::shared_ptr<ov::ITensor>>& inputTensors,
    const std::vector<std::shared_ptr<ov::ITensor>>& outputTensors) {
    _logger.debug("predict_output_shapes - started");

    const npu_vm_runtime_handle_t vmRuntime = static_cast<npu_vm_runtime_handle_t>(_graph->get_handle());
    OPENVINO_ASSERT(vmRuntime != nullptr, "predict_output_shapes requires a valid VM runtime engine");

    // Convert OV tensors/metadata into MemRef descriptors. A nullptr tensor falls back to the graph metadata
    // max shape. This keeps MemRef/DynamicArguments knowledge confined to the pipeline layer.
    const auto& metadata = _graph->get_metadata();
    auto buildMemRefs = [](const std::vector<std::shared_ptr<ov::ITensor>>& tensors,
                           const std::vector<IODescriptor>& descriptors) {
        OPENVINO_ASSERT(tensors.size() == descriptors.size(),
                        "Tensor count does not match descriptor count: ",
                        tensors.size(),
                        " vs ",
                        descriptors.size());
        std::vector<MemRefType> memRefs(descriptors.size());
        for (size_t i = 0; i < memRefs.size(); ++i) {
            const auto& tensor = tensors.at(i);
            if (tensor != nullptr) {
                // Use userTensor or levelzero Tensor to update memref handle
                memRefs[i].set(get_tensor_data_ptr(tensor), 0, tensor);
            } else {
                // If all tensors are not set, use metadata
                memRefs[i].setArg(nullptr);
                memRefs[i].setOffset(0);
                // TODO : BatchSize not checked here
                memRefs[i].setSize(descriptors.at(i).shapeFromCompiler.get_max_shape());
                memRefs[i].updateStride();
            }
        }
        return memRefs;
    };

    std::vector<MemRefType> inputsMemRefs = buildMemRefs(inputTensors, metadata.inputs);
    std::vector<MemRefType> outputsMemRefs = buildMemRefs(outputTensors, metadata.outputs);

    // Snapshot the pre-prediction (input-derived) output shapes to detect shape changes after prediction.
    const std::vector<MemRefType> originalOutputMemRefs(outputsMemRefs);

    std::vector<npu_vm_runtime_mem_ref_handle_t> inputMemRefHandles, outputMemRefHandles;

    auto processMemRefs = [&](auto& memRefs, auto& destMemRefHandles) {
        destMemRefHandles.clear();
        destMemRefHandles.reserve(memRefs.size());

        for (auto& memref : memRefs) {
            auto impl = std::static_pointer_cast<MemRefTypeImpl>(memref._impl);
            if (impl == nullptr) {
                impl = std::make_shared<MemRefTypeImpl>();
                memref._impl = impl;
            }
            impl->UpdateMemRefHandleStatus(memref, use_npu_vm_runtime_v2_api(_apiVersion));
            destMemRefHandles.push_back(impl->_memRef);
        }
    };

    processMemRefs(inputsMemRefs, inputMemRefHandles);
    processMemRefs(outputsMemRefs, outputMemRefHandles);

    npu_vm_runtime_result_t result = NPU_VM_RUNTIME_RESULT_SUCCESS;
    _logger.debug("VM runtime version: %u.%u", ZE_MAJOR_VERSION(_apiVersion), ZE_MINOR_VERSION(_apiVersion));

    if (_apiVersion == NPU_VM_RUNTIME_VERSION_1_0) {
        npu_vm_runtime_predict_output_shape_params_t params{};
        params.pInputs = inputMemRefHandles.data();
        params.numOfInputs = static_cast<uint32_t>(inputMemRefHandles.size());
        params.pOutputs = outputMemRefHandles.data();
        params.numOfOutputs = static_cast<uint32_t>(outputMemRefHandles.size());

        result = npuVMRuntimePredictOutputShape(vmRuntime, &params);
    } else {
        npu_vm_runtime_predict_output_shape_params_t2 params{};
        params.pInputs = inputMemRefHandles.data();
        params.numOfInputs = static_cast<uint32_t>(inputMemRefHandles.size());
        params.pOutputs = outputMemRefHandles.data();
        params.numOfOutputs = static_cast<uint32_t>(outputMemRefHandles.size());
        if (use_npu_vm_runtime_v2_api(_apiVersion)) {
            const auto commandQueueDesc = _graph->get_command_queue_desc();
            if (commandQueueDesc.shared_common_queue() && commandQueueDesc.key() != _command_queue->desc().key()) {
                _command_queue = ZeroCmdQueuePool::getInstance().getCommandQueue(_init_structs, commandQueueDesc);
            }
        }
        params.executionContext =
            use_npu_vm_runtime_v2_api(_apiVersion) ? _executionContext.handle() : _executionContext.ensure(vmRuntime);

        result = npuVMRuntimePredictOutputShape2(vmRuntime, &params);
    }

    if (result != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to predict output shape with VM runtime engine, error code: ", result);
    } else {
        for (size_t i = 0; i < outputsMemRefs.size(); ++i) {
            auto& out = outputsMemRefs[i];
            auto outImpl = std::static_pointer_cast<MemRefTypeImpl>(out._impl);

            if (outImpl == nullptr) {
                OPENVINO_THROW("MemRefType implementation is broken, unknown error happens in shape prediction.");
            }
            outImpl->alignWithHandle(out);
        }
        _logger.debug("Output shape prediction is done successfully.");
    }

    // Build predicted output shapes (OV shapes) and detect whether prediction changed any
    // output shape vs the pre-prediction (input-derived) shapes. MemRef stays internal to the pipeline layer.
    std::vector<ov::Shape> predictedShapes(outputsMemRefs.size());
    bool outputShapeChanged = false;
    for (size_t i = 0; i < outputsMemRefs.size(); ++i) {
        ov::Shape shape;
        shape.reserve(static_cast<size_t>(outputsMemRefs[i]._dimsCount));
        for (int64_t j = 0; j < outputsMemRefs[i]._dimsCount; ++j) {
            shape.push_back(static_cast<size_t>(outputsMemRefs[i]._sizes[j]));
        }

        predictedShapes[i] = std::move(shape);

        if (!outputShapeChanged && !outputsMemRefs[i].compare(originalOutputMemRefs[i])) {
            outputShapeChanged = true;
        }
    }
    if (outputShapeChanged) {
        _logger.debug("predict_output_shapes - output shape change detected");
    }

    if (_logger.level() >= ov::log::Level::DEBUG) {
        for (size_t i = 0; i < predictedShapes.size(); ++i) {
            _logger.debug("predict_output_shapes - output %zu predicted shape: %s",
                          i,
                          predictedShapes[i].to_string().c_str());
        }
    }
    return predictedShapes;
}

void DynamicPipeline::pull() {
    _logger.debug("pull - started");
    OV_ITT_TASK_CHAIN(ZERO_PIPELINE_IP_PULL, itt::domains::LevelZeroBackend, "DynamicPipeline", "pull");

    const npu_vm_runtime_handle_t vmRuntime =
        use_npu_vm_runtime_v2_api(_apiVersion) ? static_cast<npu_vm_runtime_handle_t>(_graph->get_handle()) : nullptr;

    if (use_npu_vm_runtime_v2_api(_apiVersion)) {
        auto& dynamicArguments = _command_list_group->getArguments();
        const auto result = npuVMRuntimeHostSync(vmRuntime, &dynamicArguments._executeParams2);
        if (result != NPU_VM_RUNTIME_RESULT_SUCCESS) {
            OPENVINO_THROW("npuVMRuntimeHostSync failed, error code: ", result);
        }
    } else if (_sync_output_with_fences) {
        _fences.front()->hostSynchronize();
    } else {
        _events.front()->hostSynchronize();
    }
    /// sample npu timestamps if feature was activated
    if (_npu_profiling != nullptr) {
        _npu_profiling->sampleNpuTimestamps();
    }

    _logger.debug("pull - completed");
}

void DynamicPipeline::reset() const {
    _logger.debug("reset - started");
    if (!use_npu_vm_runtime_v2_api(_apiVersion)) {
        if (_sync_output_with_fences) {
            _fences.front()->reset();
        } else {
            _events.front()->reset();
        }
    }
    _logger.debug("reset - completed");
}

void DynamicPipeline::update_graph_arguments(uint32_t index,
                                             const std::shared_ptr<ZeroTensor>& zeroTensor,
                                             const std::shared_ptr<ov::ITensor>& userTensor) {
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_UMCL, itt::domains::LevelZeroBackend, "DynamicPipeline", "updateCommandList");
    _logger.debug("update_graph_arguments - started");
    // This is the tensor with right shape and strides
    // The required check is already done in inferRequest
    const std::shared_ptr<ov::ITensor>& tensor = userTensor ? userTensor : zeroTensor;
    size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
    _command_list_group->updateMutableCommandList(index,
                                                  zeroTensor->data(),
                                                  get_strides(tensor->get_strides(), elementSize),
                                                  tensor->get_shape());
    _logger.debug("update_graph_arguments - completed");
}

void DynamicPipeline::update_graph_arguments(uint32_t index,
                                             const std::shared_ptr<ZeroTensor>& zeroTensor,
                                             size_t batch_index,
                                             const std::shared_ptr<ov::ITensor>& userTensor) {
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_UMCL,
                      itt::domains::LevelZeroBackend,
                      "DynamicPipeline",
                      "updateCommandListIndex");
    _logger.debug("update_graph_arguments - update command list by index");
    // This is the tensor with right shape and strides
    // The required check is already done in inferRequest
    const std::shared_ptr<ov::ITensor>& tensor = userTensor ? userTensor : zeroTensor;
    size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
    OPENVINO_ASSERT(batch_index == 0, "DynamicPipeline owns only one command-list group");

    _command_list_group->updateMutableCommandList(index,
                                                  zeroTensor->data(),
                                                  get_strides(tensor->get_strides(), elementSize),
                                                  tensor->get_shape());
}

}  // namespace intel_npu
