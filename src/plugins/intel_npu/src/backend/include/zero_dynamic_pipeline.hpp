// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef NPU_LLVM_BACKEND

#pragma once

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/MemRefUtils.h>

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "zero_memory.hpp"
#include "zero_pipeline.hpp"
#include "zero_profiling.hpp"
#include "zero_tensor.hpp"
#include "irgraph.hpp"

namespace intel_npu {

struct DynamicPipeline : public Pipeline {
    struct PipelinedCommandLists {    
        IRGraph::GraphArguments _binding;

        std::vector<std::unique_ptr<CommandList>> _commandLists;
        // to store command list handles to pass it to ExecutionEngine
        std::vector<ze_command_list_handle_t> _commandListHandles;
        
        PipelinedCommandLists(size_t numCommandLists, const std::shared_ptr<ZeroInitStructsHolder>& init_structs, const uint32_t& group_ordinal) {
            _commandLists.reserve(numCommandLists);
            for (size_t i = 0; i < numCommandLists; i++) {
                _commandLists.emplace_back(
                    std::make_unique<CommandList>(init_structs, group_ordinal));
            }

            for (size_t i = 0; i < numCommandLists; i++) {
                _commandListHandles.push_back(_commandLists[i]->handle());
            }
        }

        size_t size() const { return _commandListHandles.size(); }

        ze_command_list_handle_t* data() { return _commandListHandles.data(); }

        void bind(IRGraph* graph);

        std::vector<ze_command_list_handle_t>& getHandles() {
            return _commandListHandles;
        }

        IRGraph::GraphArguments& getBinding() {
            return _binding;
        }

        void appendBarrier() const {
            // TODO
        }

        void appendNpuTimestamp(uint64_t* timestamp_buff) const {
            // TODO
        }

        void updateMutableCommandList(uint32_t arg_index, const void* arg_value) const {
            // TODO
        }

        void appendWaitOnEvent(const std::shared_ptr<Event>& event) {
            event->AppendWaitOnEvent(**_commandLists.rbegin());
        }

        void appendReset(const std::shared_ptr<Event>& event) {
            event->AppendEventReset(**_commandLists.rbegin());
        }

        void appendSignalEvent(std::shared_ptr<Event>& event) {
            event->AppendSignalEvent(**_commandLists.rbegin());
        }
    };

public:
    DynamicPipeline(const Config& config,
                    const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                    const std::shared_ptr<IGraph>& graph,
                    const std::vector<std::vector<std::shared_ptr<ov::ITensor>>>& input_tensors,
                    const std::vector<std::shared_ptr<ov::ITensor>>& output_tensors);

    DynamicPipeline(const DynamicPipeline&) = delete;
    DynamicPipeline& operator=(const DynamicPipeline&) = delete;
    virtual ~DynamicPipeline() = default;

    virtual void push() override;
    virtual void pull() override;
    virtual void reset() const override;

    virtual void update_graph_arguments(uint32_t arg_index, const void* arg_data, size_t byte_size);
    virtual void update_graph_arguments_batching(uint32_t arg_index, const void* arg_data, size_t batch_index);
protected:
    const std::vector<std::vector<std::shared_ptr<ov::ITensor>>> _levelZeroInputTensors;
    const std::vector<std::shared_ptr<ov::ITensor>> _levelZeroOutputTensors;
    std::vector<std::unique_ptr<PipelinedCommandLists>> _command_lists;
};

}  // namespace intel_npu

#endif // NPU_LLVM_BACKEND