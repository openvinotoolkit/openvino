// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/network_metadata.hpp"
#include "intel_npu/utils/vm/dynamic_arguments.hpp"
#include "zero_pipeline.hpp"
    ///add for PR35626, this is in compiler_adapter
// #include "ze_graph_ext_wrappers.hpp"

namespace intel_npu {

struct DynamicArguments {
    std::vector<MemRefType> _inputsMemRef;
    std::vector<MemRefType> _outputsMemRef;

    // Save the memref handle to a vector for VM execution and prediction to extend its lifetime.
    std::vector<npu_vm_runtime_mem_ref_handle_t> _inputMemRefHandles;
    std::vector<npu_vm_runtime_mem_ref_handle_t> _outputMemRefHandles;

    // // Share runtime_execution_context during VM execution and forecasting  和OVPR35626 二选一
    // npu_vm_runtime_execution_context_handle_t _executionContext = nullptr;

    ///// ov PR35626
    npu_vm_runtime_execute_params_t _executeParams = {};
    /////

    // ///这个应该放在public中吗？
    // //这行应该放在 dynamic_arguments.hpp 里吗？看上去是给ececute  executeGraph的使用的
    // bool _useInterpreter = true;
    // bool _optimizedDynamicStridesMode = false;
    // ov::intel_npu::CommandListMode _bindingCommandListMode;
    ////

    // // Set by the caller after the first successful @c npuVMRuntimeExecute.
    // bool _executedOnce = false;

    ///add for PR35626
    ov::intel_npu::CommandListMode _bindingCommandListMode;
    bool _useInterpreter = false;
    bool _optimizedDynamicStridesMode = false;

    DynamicArguments() = default;
    DynamicArguments(const DynamicArguments&) = delete;
    DynamicArguments& operator=(const DynamicArguments&) = delete;
    DynamicArguments(DynamicArguments&&) = delete;
    DynamicArguments& operator=(DynamicArguments&&) = delete;
    ~DynamicArguments();

    // Create the VM execution context for vmRuntime. No-op if already created.
    void ensureExecutionContext(npu_vm_runtime_handle_t vmRuntime);

    void setArgumentProperties(uint32_t argi,
                               const void* argv,
                               const ov::Shape& shapes,
                               const std::vector<size_t>& strides);
    ///add for PR35626
    void setOptimizedDynamicStridesMode(bool enabled) {
        _optimizedDynamicStridesMode = enabled;
    }
};

class DynamicPipeline final : public IPipeline {
    struct PipelinedCommandLists {
        std::shared_ptr<DynamicArguments> _arguments;

        std::vector<std::unique_ptr<CommandList>> _commandLists;
        // Store command list handles to pass it to ExecutionEngine
        std::vector<ze_command_list_handle_t> _commandListHandles;

        PipelinedCommandLists(size_t numCommandLists,
                              const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                              std::shared_ptr<DynamicArguments> requestArguments) {
            _commandLists.reserve(numCommandLists);
            for (size_t i = 0; i < numCommandLists; i++) {
                _commandLists.emplace_back(std::make_unique<CommandList>(init_structs));
            }

            for (size_t i = 0; i < numCommandLists; i++) {
                _commandListHandles.push_back(_commandLists[i]->handle());
            }

            if (requestArguments != nullptr) {
                _arguments = requestArguments;
            } else {
                _arguments = std::make_shared<DynamicArguments>();
            }
        }

        size_t size() const {
            return _commandListHandles.size();
        }

        ze_command_list_handle_t* data() {
            return _commandListHandles.data();
        }

        // Use metadata to initialize, which will later be updated again by setArgumentProperties
        void initArguments(const NetworkMetadata& metadata) {
            _arguments->_inputsMemRef.resize(metadata.inputs.size());
            auto& inputs = _arguments->_inputsMemRef;
            for (size_t i = 0; i < inputs.size(); ++i) {
                // Use size as placeholder of stride
                // For now, only considering the usage and subsequent comparison of shape, and strides
                const auto& shape = metadata.inputs[i].shapeFromCompiler.get_shape();
                inputs[i]._dimsCount = static_cast<int64_t>(shape.size());
                inputs[i]._sizes.assign(shape.begin(), shape.end());
                inputs[i]._strides.resize(shape.size());
                inputs[i].updateStride();
            }

            _arguments->_outputsMemRef.resize(metadata.outputs.size());
            auto& outputs = _arguments->_outputsMemRef;
            for (size_t i = 0; i < outputs.size(); ++i) {
                const auto& shape = metadata.outputs[i].shapeFromCompiler.get_shape();
                outputs[i]._dimsCount = static_cast<int64_t>(shape.size());
                outputs[i]._sizes.assign(shape.begin(), shape.end());
                outputs[i]._strides.resize(shape.size());
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

        void resetCommandList() {
            for (auto& cmd_list : _commandLists) {
                cmd_list->reset();
            }
        }
    };

public:
    DynamicPipeline(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                    const std::shared_ptr<IGraph>& graph,
                    const Config& config,
                    const std::vector<std::vector<std::shared_ptr<ZeroTensor>>>& input_tensors,
                    const std::vector<std::shared_ptr<ZeroTensor>>& output_tensors,
                    std::shared_ptr<DynamicArguments> requestArguments,
                    size_t batch_size = 1);

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

    // Predicts VM runtime output shape. Independent of pipeline instance state, this depends only on the VM runtime
    // handle and argument-provided context, making it a static method.
    static void predict_output_shape(const IGraph& graph,
                                     DynamicArguments& args,
                                     std::vector<MemRefType>& inputsMemRef,
                                     std::vector<MemRefType>& outputsMemRef);

private:
    void execute_vm_runtime(npu_vm_runtime_handle_t vmRuntime,
                            DynamicArguments& args,
                            std::vector<ze_command_list_handle_t>& commandLists,
                            ze_command_queue_handle_t commandQueue,
                            ze_fence_handle_t fence,
                            ze_event_handle_t event);

    std::vector<std::unique_ptr<PipelinedCommandLists>> _command_lists;
};

}  // namespace intel_npu
