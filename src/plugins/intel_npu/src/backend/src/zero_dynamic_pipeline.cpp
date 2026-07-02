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
#include "intel_npu/utils/vm/dynamic_arguments.hpp"
#include "intel_npu/utils/vm/npu_vm_runtime_api.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_cmd_queue_pool.hpp"
#include "intel_npu/utils/zero/zero_remote_tensor.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"

namespace intel_npu {
struct MemRefTypeImpl {
    npu_vm_runtime_mem_ref_handle_t _memRef;
    bool _ptrUpdated = false;
    bool _shapeUpdated = false;
    bool _strideUpdated = false;

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
            throw std::runtime_error("Failed to parse MemRef handle");
        }
    }

    void UpdateMemRefHandleStatus(MemRefType& memref) {
        // Update current MemRef handle to use latest metadata
        if (_memRef == nullptr) {
            createMemRef(memref._dimsCount);
        } else {
            // Create a temporary MemRefType based on current handle and compare, use arg to create right size
            MemRefType tempMemRef(memref._basePtr,
                                  memref._data,
                                  memref._offset,
                                  memref._sizes,
                                  memref._strides,
                                  memref._dimsCount);
            alignWithHandle(tempMemRef);
            // Check ptr
            if (memref._basePtr != tempMemRef._basePtr || memref._data != tempMemRef._data ||
                memref._offset != tempMemRef._offset) {
                _ptrUpdated = true;
            } else {
                _ptrUpdated = false;
            }

            // Check shape
            if (memref._sizes != tempMemRef._sizes) {
                _shapeUpdated = true;
            } else {
                _shapeUpdated = false;
            }

            // Check strides
            if (memref._strides != tempMemRef._strides) {
                _strideUpdated = true;
            } else {
                _strideUpdated = false;
            }
        }
        auto result = npuVMRuntimeSetMemRef(_memRef,
                                            memref._basePtr,
                                            memref._data,
                                            memref._offset,
                                            memref._sizes.data(),
                                            memref._strides.data(),
                                            memref._dimsCount);
        if (result != NPU_VM_RUNTIME_RESULT_SUCCESS) {
            throw std::runtime_error("Failed to update MemRef handle");
        }
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

// init _inputsMemRef and _outputsMemRef vectors
void DynamicArguments::setArgumentProperties(uint32_t argi,
                                             const void* argv,
                                             const ov::Shape& sizes,
                                             const std::vector<size_t>& strides) {
    auto assign_slot = [&](MemRefType& slot) {
        slot._basePtr = slot._data = const_cast<void*>(argv);
        if (slot._dimsCount == 0) {
            slot._dimsCount = static_cast<int64_t>(sizes.size());
            slot._sizes.resize(sizes.size());
            slot._strides.resize(strides.size());
        } else if (slot._dimsCount != static_cast<int64_t>(sizes.size())) {
            OPENVINO_THROW("Dimension count mismatch. Current dimension count: ",
                           slot._dimsCount,
                           ", new dimension count: ",
                           sizes.size());
        } else if (strides.size() != static_cast<size_t>(sizes.size())) {
            OPENVINO_THROW("Updated shape and stride count mismatch: shape rank and stride count differ. Shape rank: ",
                           sizes.size(),
                           ", stride count: ",
                           strides.size());
        }
        for (int64_t i = 0; i < slot._dimsCount; i++) {
            slot._sizes[i] = static_cast<int64_t>(sizes[i]);
            slot._strides[i] = static_cast<int64_t>(strides[i]);
        }
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

DynamicArguments::~DynamicArguments() {
    if (_executionContext != nullptr) {
        npuVMRuntimeDestroyExecutionContext(_executionContext);
        _executionContext = nullptr;
    }
}

void DynamicArguments::ensureExecutionContext(npu_vm_runtime_handle_t vmRuntime) {
    if (_executionContext != nullptr) {
        return;
    }
    if (npuVMRuntimeCreateExecutionContext(vmRuntime, &_executionContext) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to create a VM execution context");
    }
}

DynamicPipeline::DynamicPipeline(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                 const std::shared_ptr<IGraph>& graph,
                                 const Config& config,
                                 const std::vector<std::vector<std::shared_ptr<ZeroTensor>>>& input_tensors,
                                 const std::vector<std::shared_ptr<ZeroTensor>>& output_tensors,
                                 std::shared_ptr<DynamicArguments> requestArguments,
                                 size_t batch_size)
    : IPipeline(init_structs, graph, batch_size, config, "DynamicPipeline") {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Zero_infer_request::DynamicPipeline::DynamicPipeline");

    OPENVINO_ASSERT(!_run_inferences_sequentially, "In-order execution doesn't work for dynamic pipeline");

    _logger.debug("Initialization started, batch size: %zu", _batch_size);

    if (!_sync_output_with_fences) {
        _event_pool = std::make_shared<EventPool>(_init_structs, _batch_size ? static_cast<uint32_t>(_batch_size) : 1);

        _events.reserve(_batch_size);
        for (size_t i = 0; i < _batch_size; i++) {
            _events.emplace_back(std::make_shared<Event>(_event_pool, static_cast<uint32_t>(i)));
        }
    }
    _logger.debug("Event pool and command queue setup completed");

    const uint64_t num_of_subgraphs = _graph->get_metadata().numberOfSubgraphs;

    _command_lists.reserve(_batch_size);
    if (batch_size > 1) {
        _logger.debug("Batch size %zu greater than 1, use new graph arguments for each batch", batch_size);
        for (size_t i = 0; i < _batch_size; i++) {
            _command_lists.emplace_back(
                std::make_unique<PipelinedCommandLists>(num_of_subgraphs, _init_structs, nullptr));
        }
    } else if (batch_size == 1) {
        _logger.debug("Batch size is 1, use the same graph arguments for all command lists");
        _command_lists.emplace_back(
            std::make_unique<PipelinedCommandLists>(num_of_subgraphs, _init_structs, requestArguments));
    } else {
        OPENVINO_THROW("Batch size must be greater than 0, but got ", batch_size);
    }

    if (_sync_output_with_fences) {
        _fences.reserve(_batch_size);
        for (size_t i = 0; i < _batch_size; i++) {
            _fences.emplace_back(std::make_unique<Fence>(_command_queue));
        }
    }

    for (size_t i = 0; i < _batch_size; i++) {
        _logger.debug("Set args for command list number: %zu", i);

        _command_lists.at(i)->initArguments(_graph->get_metadata());
        auto& dynamicArguments = _command_lists.at(i)->getArguments();

        size_t io_index = 0;
        for (const auto& desc : _graph->get_metadata().inputs) {
            // DynamicPipeline does not currently support weightless model, just thrown exception.
            OPENVINO_ASSERT(!desc.isMainInputWeights,
                            "DynamicPipeline does not support weightless graphs (input '",
                            desc.nameFromCompiler,
                            "' is a main-input weight)");

            if (input_tensors.at(io_index).size() > 1) {
                _logger.debug("Set args for input index: %zu", io_index);
                const auto& tensor = input_tensors.at(io_index).at(i);
                size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
                dynamicArguments.setArgumentProperties(desc.indexUsedByDriver,
                                                       tensor->data(),
                                                       tensor->get_shape(),
                                                       get_strides(tensor->get_strides(), elementSize));
                ++io_index;
                continue;
            }

            _logger.debug("Update tensor property for input desc index: %u", desc.indexUsedByDriver);
            const auto& tensor = input_tensors.at(io_index).at(0);
            size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
            if (tensor->get_element_type().bitwidth() < 8 || tensor->is_continuous() || tensor->get_strides().empty()) {
                dynamicArguments.setArgumentProperties(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_byte_size()) / _batch_size,
                    tensor->get_shape(),
                    get_strides(tensor->get_strides(), elementSize));
            } else {
                dynamicArguments.setArgumentProperties(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_strides()[0]),
                    tensor->get_shape(),
                    get_strides(tensor->get_strides(), elementSize));
            }
            ++io_index;
        }

        io_index = 0;
        for (const auto& desc : _graph->get_metadata().outputs) {
            _logger.debug("Update tensor property for output desc index: %u", desc.indexUsedByDriver);
            const auto& tensor = output_tensors.at(io_index);
            size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
            if (tensor->get_element_type().bitwidth() < 8 || tensor->is_continuous() || tensor->get_strides().empty()) {
                dynamicArguments.setArgumentProperties(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_byte_size()) / _batch_size,
                    tensor->get_shape(),
                    get_strides(tensor->get_strides(), elementSize));
            } else {
                dynamicArguments.setArgumentProperties(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_strides()[0]),
                    tensor->get_shape(),
                    get_strides(tensor->get_strides(), elementSize));
            }
            ++io_index;
        }
    }
    _logger.debug("Initialization completed");
}

void DynamicPipeline::push() {
    _logger.debug("push - started");

    const npu_vm_runtime_handle_t vmRuntime = static_cast<npu_vm_runtime_handle_t>(_graph->get_handle());
    OPENVINO_ASSERT(vmRuntime != nullptr, "DynamicPipeline requires a valid VM runtime engine");

    const auto command_queue_desc = _graph->get_command_queue_desc();
    const bool command_queue_version_changed = (command_queue_desc.key() != _command_queue->desc().key());
    if (command_queue_version_changed) {
        _command_queue = ZeroCmdQueuePool::getInstance().getCommandQueue(_init_structs, command_queue_desc);

        if (_sync_output_with_fences) {
            for (size_t i = 0; i < _fences.size(); i++) {
                _fences[i] = std::make_unique<Fence>(_command_queue);
            }
        }
    }

    auto commandQueueHandle = _command_queue->handle();
    for (size_t i = 0; i < _command_lists.size(); ++i) {
        OV_ITT_TASK_CHAIN(ZERO_PIPELINE_IP_PUSH, itt::domains::LevelZeroBackend, "Pipeline", "push");

        ze_fence_handle_t fence = nullptr;
        ze_event_handle_t event = nullptr;
        if (_sync_output_with_fences) {
            fence = _fences.at(i)->handle();
        }

        auto& command_lists = _command_lists.at(i);
        auto& dynamicArguments = command_lists->getArguments();
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

        execute_vm_runtime(vmRuntime, dynamicArguments, command_lists->getHandles(), commandQueueHandle, fence, event);
    }

    _logger.debug("push - completed");
}

void DynamicPipeline::execute_vm_runtime(npu_vm_runtime_handle_t vmRuntime,
                                         DynamicArguments& args,
                                         std::vector<ze_command_list_handle_t>& commandLists,
                                         ze_command_queue_handle_t commandQueue,
                                         ze_fence_handle_t fence,
                                         ze_event_handle_t event) {
    _logger.debug("Start to execute graph with runtime engine");
    bool noTensorChange = true;
    // _executedOnce is true only after a successful npuVMRuntimeExecute below
    const bool firstExecution = !args._executedOnce;

    auto processMemRefs = [&](auto& memRefs, auto& targetMemRefHandles) {
        targetMemRefHandles.clear();
        targetMemRefHandles.reserve(memRefs.size());
        for (auto& memref : memRefs) {
            auto impl = std::static_pointer_cast<MemRefTypeImpl>(memref._impl);
            if (impl == nullptr) {
                impl = std::make_shared<MemRefTypeImpl>();
                memref._impl = impl;
            }
            impl->UpdateMemRefHandleStatus(memref);
            targetMemRefHandles.push_back(impl->_memRef);

            if (impl->_ptrUpdated || impl->_shapeUpdated || impl->_strideUpdated) {
                noTensorChange = false;
            }
        }
    };

    processMemRefs(args._inputsMemRef, args._inputMemRefHandles);
    processMemRefs(args._outputsMemRef, args._outputMemRefHandles);

    if (!firstExecution && noTensorChange) {
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

    // Create the VM execution context (owned by args._impl, destroyed with it).
    args.ensureExecutionContext(vmRuntime);

    npu_vm_runtime_execute_params_t params{};
    params.executionContext = args._executionContext;
    params.pInputs = args._inputMemRefHandles.data();
    params.numOfInputs = static_cast<uint32_t>(args._inputMemRefHandles.size());
    params.pOutputs = args._outputMemRefHandles.data();
    params.numOfOutputs = static_cast<uint32_t>(args._outputMemRefHandles.size());
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
    } else {
        _logger.debug("Execution runtime engine is created successfully.");
    }

    args._executedOnce = true;

    _logger.debug("Completed to execute graph with runtime engine");
}

void DynamicPipeline::predict_output_shape(const IGraph& graph,
                                           DynamicArguments& args,
                                           std::vector<MemRefType>& inputsMemRefs,
                                           std::vector<MemRefType>& outputsMemRefs) {
    Logger logger("DynamicPipeline::predict_output_shape", Logger::global().level());
    logger.debug("predict_output_shape - started");

    const npu_vm_runtime_handle_t vmRuntime = static_cast<npu_vm_runtime_handle_t>(graph.get_handle());
    OPENVINO_ASSERT(vmRuntime != nullptr, "predict_output_shape requires a valid VM runtime engine");

    auto processMemRefs = [&](auto& memRefs, auto& targetMemRefHandles) {
        targetMemRefHandles.clear();
        targetMemRefHandles.reserve(memRefs.size());

        for (auto& memref : memRefs) {
            auto impl = std::static_pointer_cast<MemRefTypeImpl>(memref._impl);
            if (impl == nullptr) {
                impl = std::make_shared<MemRefTypeImpl>();
                memref._impl = impl;
            }
            impl->UpdateMemRefHandleStatus(memref);
            targetMemRefHandles.push_back(impl->_memRef);
        }
    };

    processMemRefs(inputsMemRefs, args._inputMemRefHandles);
    processMemRefs(outputsMemRefs, args._outputMemRefHandles);

    npu_vm_runtime_result_t result = NPU_VM_RUNTIME_RESULT_SUCCESS;
    npu_vm_runtime_version_t version{};
    if ((result = npuVMRuntimeGetAPIVersion(&version)) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to get VM runtime version, error code: ", result);
    }
    logger.debug("VM runtime version: %u.%u", ZE_MAJOR_VERSION(version), ZE_MINOR_VERSION(version));

    if (version == NPU_VM_RUNTIME_VERSION_1_0) {
        npu_vm_runtime_predict_output_shape_params_t params{};
        params.pInputs = args._inputMemRefHandles.data();
        params.numOfInputs = static_cast<uint32_t>(args._inputMemRefHandles.size());
        params.pOutputs = args._outputMemRefHandles.data();
        params.numOfOutputs = static_cast<uint32_t>(args._outputMemRefHandles.size());

        result = npuVMRuntimePredictOutputShape(vmRuntime, &params);
    } else {
        args.ensureExecutionContext(vmRuntime);

        npu_vm_runtime_predict_output_shape_params_t2 params{};
        params.pInputs = args._inputMemRefHandles.data();
        params.numOfInputs = static_cast<uint32_t>(args._inputMemRefHandles.size());
        params.pOutputs = args._outputMemRefHandles.data();
        params.numOfOutputs = static_cast<uint32_t>(args._outputMemRefHandles.size());
        params.executionContext = args._executionContext;

        result = npuVMRuntimePredictOutputShape2(vmRuntime, &params);
    }

    if (result != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to predict output shape with VM runtime engine, error code: ", result);
    } else {
        for (auto& out : outputsMemRefs) {
            std::shared_ptr<MemRefTypeImpl> outImpl = std::static_pointer_cast<MemRefTypeImpl>(out._impl);
            if (outImpl == nullptr) {
                OPENVINO_THROW("MemRefType implementation is broken, unknown error happens in shape prediction.");
            }
            outImpl->alignWithHandle(out);
        }
        logger.debug("Output shape prediction is done successfully.");
    }

    // Clear memref handles after shape prediction to avoid the next execution using wrong memref handles
    args._inputMemRefHandles.clear();
    args._outputMemRefHandles.clear();
}

void DynamicPipeline::pull() {
    _logger.debug("pull - started");
    OV_ITT_TASK_CHAIN(ZERO_PIPELINE_IP_PULL, itt::domains::LevelZeroBackend, "DynamicPipeline", "pull");

    for (size_t i = 0; i < _command_lists.size(); ++i) {
        if (_sync_output_with_fences) {
            _fences.at(i)->hostSynchronize();
        } else {
            _events.at(i)->hostSynchronize();
        }
        /// sample npu timestamps if feature was activated
        if (_npu_profiling != nullptr) {
            _npu_profiling->sampleNpuTimestamps();
        }
    }

    _logger.debug("pull - completed");
}

void DynamicPipeline::reset() const {
    _logger.debug("reset - started");
    for (size_t i = 0; i < _command_lists.size(); ++i) {
        if (_sync_output_with_fences) {
            _fences.at(i)->reset();
        } else {
            _events.at(i)->reset();
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
    // The required check is alredy done in inferRequest
    const std::shared_ptr<ov::ITensor>& tensor = userTensor ? userTensor : zeroTensor;
    size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
    const size_t number_of_command_lists = _command_lists.size();

    for (size_t i = 0; i < number_of_command_lists; i++) {
        if (tensor->get_element_type().bitwidth() < 8 || tensor->is_continuous() || tensor->get_strides().empty()) {
            _command_lists.at(i)->updateMutableCommandList(index,
                                                           static_cast<const unsigned char*>(zeroTensor->data()) +
                                                               (i * tensor->get_byte_size()) / number_of_command_lists,
                                                           get_strides(tensor->get_strides(), elementSize),
                                                           tensor->get_shape());
        } else {
            _command_lists.at(i)->updateMutableCommandList(
                index,
                static_cast<const unsigned char*>(zeroTensor->data()) + (i * tensor->get_strides()[0]),
                get_strides(tensor->get_strides(), elementSize),
                tensor->get_shape());
        }
    }
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
    // The required check is alredy done in inferRequest
    const std::shared_ptr<ov::ITensor>& tensor = userTensor ? userTensor : zeroTensor;
    size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
    const size_t number_of_command_lists = _command_lists.size();

    OPENVINO_ASSERT(batch_index < number_of_command_lists,
                    "Command list index is higher than the number of Command lists ",
                    batch_index);

    _command_lists.at(batch_index)
        ->updateMutableCommandList(index,
                                   zeroTensor->data(),
                                   get_strides(tensor->get_strides(), elementSize),
                                   tensor->get_shape());
}

}  // namespace intel_npu
