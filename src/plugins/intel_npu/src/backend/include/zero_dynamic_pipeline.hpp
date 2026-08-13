// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/network_metadata.hpp"
#include "intel_npu/utils/vm/mem_ref_type.hpp"
#include "intel_npu/utils/vm/npu_vm_runtime_utils.hpp"
#include "zero_pipeline.hpp"

namespace intel_npu {

// VM runtime execution context owner. The handle is created lazily and shared between
// shape prediction (infer request) and execution (pipeline) so the same context is reused.
struct VMExecutionContext {
    npu_vm_runtime_execution_context_handle_t _handle = nullptr;

    VMExecutionContext() = default;
    VMExecutionContext(const VMExecutionContext&) = delete;
    VMExecutionContext& operator=(const VMExecutionContext&) = delete;
    VMExecutionContext(VMExecutionContext&&) = delete;
    VMExecutionContext& operator=(VMExecutionContext&&) = delete;
    ~VMExecutionContext();

    npu_vm_runtime_execution_context_handle_t ensure(npu_vm_runtime_handle_t vmRuntime);
    npu_vm_runtime_execution_context_handle_t ensureV2(npu_vm_runtime_handle_t vmRuntime,
                                                       ze_context_handle_t ctx,
                                                       ze_device_handle_t device,
                                                       ze_command_queue_handle_t commandQueue,
                                                       ze_graph_dditable_ext_t* graphDdiTableExt,
                                                       ze_command_queue_npu_dditable_ext_t* queueDdiTableExt);
    npu_vm_runtime_execution_context_handle_t handle() const {
        return _handle;
    }
};

struct DynamicArguments {
    std::vector<MemRefType> _inputsMemRef;
    std::vector<MemRefType> _outputsMemRef;
    std::vector<npu_vm_runtime_mem_ref_handle_t> _inputMemRefHandles;
    std::vector<npu_vm_runtime_mem_ref_handle_t> _outputMemRefHandles;
    npu_vm_runtime_execute_params2_t _executeParams2 = {};

    // True once the command lists have been recorded by a first npuVMRuntimeExecute call. Subsequent
    // executions can be replayed without re-recording when no tensor changed (see execute_vm_runtime).
    bool _commandListsRecorded = false;

    DynamicArguments() = default;
    DynamicArguments(const DynamicArguments&) = delete;
    DynamicArguments& operator=(const DynamicArguments&) = delete;
    DynamicArguments(DynamicArguments&&) = delete;
    DynamicArguments& operator=(DynamicArguments&&) = delete;
    ~DynamicArguments() = default;

    void setArgumentProperties(uint32_t argi,
                               const void* argv,
                               const ov::Shape& shapes,
                               const std::vector<size_t>& strides);
};

class DynamicPipeline final : public IPipeline {
    struct PipelinedCommandLists {
        std::shared_ptr<DynamicArguments> _arguments;

        std::vector<std::unique_ptr<CommandList>> _commandLists;
        // Store command list handles to pass it to ExecutionEngine
        std::vector<ze_command_list_handle_t> _commandListHandles;

        PipelinedCommandLists(size_t numCommandLists,
                              const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                              bool useV2 = false) {
            if (!useV2) {
                _commandLists.reserve(numCommandLists);
                for (size_t i = 0; i < numCommandLists; i++) {
                    _commandLists.emplace_back(std::make_unique<CommandList>(init_structs));
                }

                for (size_t i = 0; i < numCommandLists; i++) {
                    _commandListHandles.push_back(_commandLists[i]->handle());
                }
            }

            _arguments = std::make_shared<DynamicArguments>();
        }

        // Use metadata to initialize, which will later be updated again by setArgumentProperties
        void initArguments(const NetworkMetadata& metadata) {
            _arguments->_inputsMemRef.resize(metadata.inputs.size());
            auto& inputs = _arguments->_inputsMemRef;
            for (size_t i = 0; i < inputs.size(); ++i) {
                // Use size as placeholder of stride
                // For now, only considering the usage and subsequent comparison of shape, and strides
                const auto& shape = metadata.inputs[i].shapeFromCompiler.get_shape();
                inputs[i].setSize(shape);
                inputs[i].updateStride();
            }

            _arguments->_outputsMemRef.resize(metadata.outputs.size());
            auto& outputs = _arguments->_outputsMemRef;
            for (size_t i = 0; i < outputs.size(); ++i) {
                const auto& shape = metadata.outputs[i].shapeFromCompiler.get_shape();
                outputs[i].setSize(shape);
                outputs[i].updateStride();
            }
        }

        std::vector<ze_command_list_handle_t>& getHandles() {
            return _commandListHandles;
        }

        DynamicArguments& getArguments() {
            return *_arguments;
        }

        void updateMutableCommandList(uint32_t arg_index,
                                      const void* arg_value,
                                      const ov::Strides& strides,
                                      const ov::Shape& shapes) {
            // The strides are already divided by element size
            if (arg_index < _arguments->_inputsMemRef.size()) {
                _arguments->_inputsMemRef[arg_index].setArg(arg_value);
                _arguments->_inputsMemRef[arg_index].setSize(shapes);
                _arguments->_inputsMemRef[arg_index].setStrides(strides);
            } else {
                size_t output_index = static_cast<size_t>(arg_index) - _arguments->_inputsMemRef.size();
                if (output_index < _arguments->_outputsMemRef.size()) {
                    _arguments->_outputsMemRef[output_index].setArg(arg_value);
                    _arguments->_outputsMemRef[output_index].setSize(shapes);
                    _arguments->_outputsMemRef[output_index].setStrides(strides);
                }
            }
        }
    };

public:
    DynamicPipeline(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                    const std::shared_ptr<IGraph>& graph,
                    const Config& config,
                    const std::vector<std::vector<std::shared_ptr<ZeroTensor>>>& input_tensors,
                    const std::vector<std::shared_ptr<ZeroTensor>>& output_tensors);

    DynamicPipeline(const DynamicPipeline&) = delete;
    DynamicPipeline& operator=(const DynamicPipeline&) = delete;
    ~DynamicPipeline() override = default;

    void push() override;
    void pull() override;
    void reset() const override;
    void update_graph_arguments(uint32_t index,
                                const std::shared_ptr<ZeroTensor>& tensor,
                                const std::shared_ptr<ov::ITensor>& userTensor = nullptr) override;
    void update_graph_arguments(uint32_t index,
                                const std::shared_ptr<ZeroTensor>& tensor,
                                size_t batch_index,
                                const std::shared_ptr<ov::ITensor>& userTensor = nullptr) override;

    // Predicts VM runtime output shapes for the given input/output tensors. A nullptr tensor entry falls
    // back to the graph metadata max shape. Uses the pipeline's own graph handle and VM execution context,
    // which is the same context reused by execute_vm_runtime.
    std::vector<ov::Shape> predict_output_shapes(const std::vector<std::shared_ptr<ov::ITensor>>& inputTensors,
                                                 const std::vector<std::shared_ptr<ov::ITensor>>& outputTensors);

private:
    void execute_vm_runtime(npu_vm_runtime_handle_t vmRuntime,
                            DynamicArguments& args,
                            std::vector<ze_command_list_handle_t>& commandLists,
                            ze_command_queue_handle_t commandQueue,
                            ze_fence_handle_t fence,
                            ze_event_handle_t event);
    void execute_vm_runtime_v2(npu_vm_runtime_handle_t vmRuntime,
                               DynamicArguments& args,
                               ze_command_queue_handle_t commandQueue,
                               const npu_vm_runtime_config_desc_t* pConfig);
    const npu_vm_runtime_config_desc_t* update_runtime_config(const CommandQueueDesc& previousCommandQueueDesc,
                                                              const CommandQueueDesc& currentCommandQueueDesc);

    // VM execution context owned by this pipeline; shared between shape prediction and execution.
    VMExecutionContext _executionContext;
    npu_vm_runtime_version_t _apiVersion = NPU_VM_RUNTIME_VERSION_1_0;
    NpuVMRuntimeConfigChain _runtimeConfigChain;
    CommandQueueDesc _runtime_config_command_queue_desc;
    bool _runtime_config_command_queue_desc_valid = false;
    std::unique_ptr<PipelinedCommandLists> _command_list_group;
};

}  // namespace intel_npu
