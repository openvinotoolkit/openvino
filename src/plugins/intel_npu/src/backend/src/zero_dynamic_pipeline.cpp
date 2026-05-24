// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_dynamic_pipeline.hpp"

#include <level_zero/ze_api.h>
#include <ze_graph_ext.h>

#include <sstream>

#include "intel_npu/common/dynamic_graph_vm_impl.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/vm/npu_vm_runtime_api.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_cmd_queue_pool.hpp"
#include "intel_npu/utils/zero/zero_remote_tensor.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"

namespace {
std::vector<size_t> get_strides(const std::vector<size_t>& strides_in_bytes, size_t element_size) {
    std::vector<size_t> element_strides(strides_in_bytes.size());
    std::transform(strides_in_bytes.begin(),
                   strides_in_bytes.end(),
                   element_strides.begin(),
                   [element_size](size_t byte_stride) {
                       OPENVINO_ASSERT(byte_stride % element_size == 0,
                                       "Stride ",
                                       byte_stride,
                                       " bytes is not aligned to element size ",
                                       element_size,
                                       " bytes. Strides must be multiples of element size.");

                       return byte_stride / element_size;
                   });

    return element_strides;
};

}  // namespace

namespace intel_npu {
DynamicPipeline::DynamicPipeline(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                 const std::shared_ptr<IGraph>& graph,
                                 const Config& config,
                                 const std::vector<std::vector<std::shared_ptr<ZeroTensor>>>& input_tensors,
                                 const std::vector<std::shared_ptr<ZeroTensor>>& output_tensors,
                                 size_t batch_size)
    : IPipeline(init_structs, graph, batch_size, config, "DynamicPipeline") {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Zero_infer_request::DynamicPipeline::DynamicPipeline");

    OPENVINO_ASSERT(!_run_inferences_sequentially, "In-order execution doesn't work for dynamic pipeline");

    _logger.debug("DynamicPipeline - initialization started, batch size: %i", _batch_size);

    if (!_sync_output_with_fences) {
        _event_pool = std::make_shared<EventPool>(_init_structs, _batch_size ? static_cast<uint32_t>(_batch_size) : 1);

        _events.reserve(_batch_size);
        for (size_t i = 0; i < _batch_size; i++) {
            _events.emplace_back(std::make_shared<Event>(_event_pool, static_cast<uint32_t>(i)));
        }
    }
    _logger.debug("DynamicPipeline - event pool and command queue setup completed");

    uint64_t num_of_subgraphs = graph->get_num_subgraphs();

    _command_lists.reserve(_batch_size);
    for (size_t i = 0; i < _batch_size; i++) {
        _command_lists.emplace_back(std::make_unique<PipelinedCommandLists>(num_of_subgraphs, _init_structs));
    }

    if (_sync_output_with_fences) {
        _fences.reserve(_batch_size);
        for (size_t i = 0; i < _batch_size; i++) {
            _fences.emplace_back(std::make_unique<Fence>(_command_queue));
        }
    }

    for (size_t i = 0; i < _batch_size; i++) {
        _logger.debug("DynamicPipeline - set args for command list number: %zu", i);

        _command_lists.at(i)->bind(_graph->get_metadata());
        auto& graphArguments = _command_lists.at(i)->getBinding();

        size_t io_index = 0;
        for (const auto& desc : _graph->get_metadata().inputs) {
            if (desc.isMainInputWeights) {
                // These values were set while running the "WeightlessGraph::init" method
                continue;
            }

            if (input_tensors.at(io_index).size() > 1) {
                _logger.debug("DynamicPipeline - set args for input index: %zu", io_index);
                const auto& tensor = input_tensors.at(io_index).at(i);
                size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
                graphArguments.setArgumentProperties(desc.indexUsedByDriver,
                                                     tensor->data(),
                                                     tensor->get_shape(),
                                                     get_strides(tensor->get_strides(), elementSize));
                ++io_index;
                continue;
            }

            _logger.debug("DynamicPipeline - update tensor property for input desc index: %d", desc.indexUsedByDriver);
            const auto& tensor = input_tensors.at(io_index).at(0);
            size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
            if (tensor->get_element_type().bitwidth() < 8 || tensor->is_continuous() || tensor->get_strides().empty()) {
                graphArguments.setArgumentProperties(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_byte_size()) / _batch_size,
                    tensor->get_shape(),
                    get_strides(tensor->get_strides(), elementSize));
            } else {
                graphArguments.setArgumentProperties(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_strides()[0]),
                    tensor->get_shape(),
                    get_strides(tensor->get_strides(), elementSize));
            }
            ++io_index;
        }

        io_index = 0;
        for (const auto& desc : _graph->get_metadata().outputs) {
            _logger.debug("DynamicPipeline - update tensor property for output desc index: %d", desc.indexUsedByDriver);
            const auto& tensor = output_tensors.at(io_index);
            size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
            if (tensor->get_element_type().bitwidth() < 8 || tensor->is_continuous() || tensor->get_strides().empty()) {
                graphArguments.setArgumentProperties(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_byte_size()) / _batch_size,
                    tensor->get_shape(),
                    get_strides(tensor->get_strides(), elementSize));
            } else {
                graphArguments.setArgumentProperties(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_strides()[0]),
                    tensor->get_shape(),
                    get_strides(tensor->get_strides(), elementSize));
            }
            ++io_index;
        }
    }
    _logger.debug("DynamicPipeline - initialization completed");
}

void DynamicPipeline::push() {
    _logger.debug("push - started");

    const npu_vm_runtime_handle_t vmRuntime = _graph->get_vm_runtime_handle();
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
        auto& graphArguments = command_lists->getBinding();
        if (_logger.level() >= ov::log::Level::DEBUG) {
            _logger.debug("push - inputs info for dynamic graph:");
            for (auto& memType : graphArguments._inputs) {
                _logger.debug("push - input: %s", memType.toString().c_str());
            }
            _logger.debug("push - outputs info for dynamic graph:");
            for (auto& memType : graphArguments._outputs) {
                _logger.debug("push - output: %s", memType.toString().c_str());
            }
        }

        execute_vm_runtime(vmRuntime, graphArguments, command_lists->getHandles(), commandQueueHandle, fence, event);
    }

    _logger.debug("push - completed");
}

void DynamicPipeline::execute_vm_runtime(npu_vm_runtime_handle_t vmRuntime,
                                         GraphArguments& args,
                                         std::vector<ze_command_list_handle_t>& commandLists,
                                         ze_command_queue_handle_t commandQueue,
                                         ze_fence_handle_t fence,
                                         ze_event_handle_t event) {
    _logger.debug("execute_vm_runtime - started");

    const bool firstExecution = (args._impl == nullptr);
    std::shared_ptr<DynamicGraphArgumentsImpl> argsImpl =
        firstExecution ? std::make_shared<DynamicGraphArgumentsImpl>()
                       : std::static_pointer_cast<DynamicGraphArgumentsImpl>(args._impl);

    bool noTensorChange = true;
    npu_vm_runtime_execute_params_t* params = &argsImpl->_executeParams;

    for (auto& in : args._inputs) {
        auto inImpl = std::static_pointer_cast<DynamicGraphMemRefImpl>(in._impl);
        if (inImpl == nullptr) {
            inImpl = std::make_shared<DynamicGraphMemRefImpl>();
            in._impl = inImpl;
        }
        inImpl->updateMemRefHandleStatus(in);
        if (firstExecution) {
            argsImpl->_inputMemRefs.push_back(inImpl->_memRef);
        } else if (inImpl->_ptrUpdated || inImpl->_shapeUpdated || inImpl->_strideUpdated) {
            noTensorChange = false;
        }
    }
    for (auto& out : args._outputs) {
        auto outImpl = std::static_pointer_cast<DynamicGraphMemRefImpl>(out._impl);
        if (outImpl == nullptr) {
            outImpl = std::make_shared<DynamicGraphMemRefImpl>();
            out._impl = outImpl;
        }
        outImpl->updateMemRefHandleStatus(out);
        if (firstExecution) {
            argsImpl->_outputMemRefs.push_back(outImpl->_memRef);
        } else if (outImpl->_ptrUpdated || outImpl->_shapeUpdated || outImpl->_strideUpdated) {
            noTensorChange = false;
        }
    }

    if (!firstExecution && noTensorChange) {
        _logger.debug("execute_vm_runtime - reuse command list (no tensor change)");
        auto result = zeCommandQueueExecuteCommandLists(commandQueue,
                                                        static_cast<uint32_t>(commandLists.size()),
                                                        commandLists.data(),
                                                        fence);
        if (result != ZE_RESULT_SUCCESS) {
            OPENVINO_THROW("Failed to submit command lists");
        }
        return;
    }

    _logger.debug("execute_vm_runtime - reset command lists");
    // Reset commandLists since there are tensors with new shapes (or first execution); can not reuse via update.
    for (auto& cmdList : commandLists) {
        zeCommandListReset(cmdList);
    }

    // Lazily create the VM execution context (owned by argsImpl, destroyed with it).
    if (params->executionContext == nullptr) {
        if (npuVMRuntimeCreateExecutionContext(vmRuntime, &params->executionContext) !=
            NPU_VM_RUNTIME_RESULT_SUCCESS) {
            OPENVINO_THROW("Failed to create a VM execution context");
        }
        _logger.debug("execute_vm_runtime - execution context created");
    }

    params->pInputs = argsImpl->_inputMemRefs.data();
    params->numOfInputs = static_cast<uint32_t>(argsImpl->_inputMemRefs.size());
    params->pOutputs = argsImpl->_outputMemRefs.data();
    params->numOfOutputs = static_cast<uint32_t>(argsImpl->_outputMemRefs.size());
    params->ctx = _init_structs->getContext();
    params->device = _init_structs->getDevice();
    params->graphDdiTableExt = _init_structs->getGraphDdiTable().getImpl();
    params->commandLists = commandLists.data();
    params->numCommandLists = static_cast<uint64_t>(commandLists.size());
    params->commandQueue = commandQueue;
    params->inferenceFence = fence;
    params->event = event;

    if (npuVMRuntimeExecute(vmRuntime, params) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to execute VM runtime engine");
    }

    if (firstExecution) {
        args._impl = argsImpl;
    }

    _logger.debug("execute_vm_runtime - completed");
}

void DynamicPipeline::predict_output_shape(std::vector<MemRefType>& inputDescriptors,
                                           std::vector<MemRefType>& outputDescriptors) {
    _logger.debug("predict_output_shape - started");

    const npu_vm_runtime_handle_t vmRuntime = _graph->get_vm_runtime_handle();
    OPENVINO_ASSERT(vmRuntime != nullptr, "predict_output_shape requires a valid VM runtime engine");

    std::vector<npu_vm_runtime_mem_ref_handle_t> inputs;
    inputs.reserve(inputDescriptors.size());
    for (auto& in : inputDescriptors) {
        std::shared_ptr<DynamicGraphMemRefImpl> inImpl =
            std::static_pointer_cast<DynamicGraphMemRefImpl>(in._impl);
        if (inImpl == nullptr) {
            inImpl = std::make_shared<DynamicGraphMemRefImpl>();
            in._impl = inImpl;
        }
        inImpl->updateMemRefHandleStatus(in);
        inputs.push_back(inImpl->_memRef);
    }

    std::vector<npu_vm_runtime_mem_ref_handle_t> outputs;
    outputs.reserve(outputDescriptors.size());
    for (auto& out : outputDescriptors) {
        std::shared_ptr<DynamicGraphMemRefImpl> outImpl =
            std::static_pointer_cast<DynamicGraphMemRefImpl>(out._impl);
        if (outImpl == nullptr) {
            outImpl = std::make_shared<DynamicGraphMemRefImpl>();
            out._impl = outImpl;
        }
        outImpl->updateMemRefHandleStatus(out);
        outputs.push_back(outImpl->_memRef);
    }

    npu_vm_runtime_predict_output_shape_params_t params{};
    params.pInputs = inputs.data();
    params.numOfInputs = static_cast<uint32_t>(inputs.size());
    params.pOutputs = outputs.data();
    params.numOfOutputs = static_cast<uint32_t>(outputs.size());

    if (npuVMRuntimePredictOutputShape(vmRuntime, &params) != NPU_VM_RUNTIME_RESULT_SUCCESS) {
        OPENVINO_THROW("Failed to predict output shapes via VM runtime engine");
    }

    for (auto& out : outputDescriptors) {
        std::shared_ptr<DynamicGraphMemRefImpl> outImpl =
            std::static_pointer_cast<DynamicGraphMemRefImpl>(out._impl);
        OPENVINO_ASSERT(outImpl != nullptr,
                        "MemRefType implementation is broken, unknown error happens in shape prediction.");
        outImpl->alignWithHandle(out);
    }

    _logger.debug("predict_output_shape - completed");
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
