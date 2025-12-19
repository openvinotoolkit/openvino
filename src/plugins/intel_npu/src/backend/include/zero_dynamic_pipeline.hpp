// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "irgraph.hpp"
#include "zero_profiling.hpp"
#include "zero_tensor.hpp"

namespace intel_npu {

struct DynamicPipeline {
    struct PipelinedCommandLists {
        mutable IRGraph::GraphArguments _binding;

        std::vector<std::unique_ptr<CommandList>> _commandLists;
        // to store command list handles to pass it to ExecutionEngine
        std::vector<ze_command_list_handle_t> _commandListHandles;

        PipelinedCommandLists(size_t numCommandLists,
                              const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                              const uint32_t& group_ordinal) {
            _commandLists.reserve(numCommandLists);
            for (size_t i = 0; i < numCommandLists; i++) {
                _commandLists.emplace_back(std::make_unique<CommandList>(init_structs, group_ordinal));
            }

            for (size_t i = 0; i < numCommandLists; i++) {
                _commandListHandles.push_back(_commandLists[i]->handle());
            }
        }

        size_t size() const {
            return _commandListHandles.size();
        }

        ze_command_list_handle_t* data() {
            return _commandListHandles.data();
        }

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

        void updateMutableCommandList(uint32_t arg_index,
                                      const void* arg_value,
                                      const ov::Strides& strides,
                                      const ov::Shape& shapes) {
            if (arg_index < _binding._inputs.size()) {
                std::ostringstream oss;
                oss << _binding._inputs[arg_index];
                Logger::global().debug("Orig tensor MemType: %s", oss.str().c_str());
                _binding._inputs[arg_index].setArg(arg_value);
                // size_t shapesSize = shapes.size();
                for (int64_t i = 0; i < _binding._inputs[arg_index].dimsCount; i++) {
                    _binding._inputs[arg_index].sizes[i] = shapes[i];
                }

                // size_t stridesSize = strides.size();
                for (int64_t i = 0; i < _binding._inputs[arg_index].dimsCount; i++) {
                    _binding._inputs[arg_index].strides[i] = strides[i];
                }

                // Need stride based on element but not byte
                _binding._inputs[arg_index].updateStride();

                oss.clear();
                oss.str("");
                oss << _binding._inputs[arg_index];
                Logger::global().debug("Updated to MemRefType: %s", oss.str().c_str());

            } else {
                size_t output_index = static_cast<size_t>(arg_index) - _binding._inputs.size();
                if (output_index < _binding._outputs.size()) {
                    std::ostringstream oss;
                    oss << _binding._outputs[output_index];
                    Logger::global().debug("Orig output tensor MemType: %s", oss.str().c_str());
                    _binding._outputs[output_index].setArg(arg_value);

                    // size_t shapesSize = shapes.size();
                    for (int64_t i = 0; i < _binding._outputs[output_index].dimsCount; i++) {
                        _binding._outputs[output_index].sizes[i] = shapes[i];
                    }

                    // size_t stridesSize = strides.size();
                    for (int64_t i = 0; i < _binding._outputs[output_index].dimsCount; i++) {
                        _binding._outputs[output_index].strides[i] = strides[i];
                    }

                    // Need stride based on element but not byte
                    _binding._outputs[output_index].updateStride();

                    oss.clear();
                    oss.str("");
                    oss << _binding._outputs[output_index];
                    Logger::global().debug("Updated to MemRefType: %s", oss.str().c_str());
                }
            }
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
                    const std::vector<std::vector<std::shared_ptr<ZeroTensor>>>& input_tensors,
                    const std::vector<std::shared_ptr<ZeroTensor>>& output_tensors,
                    size_t batch_size = 1);

    DynamicPipeline(const DynamicPipeline&) = delete;
    DynamicPipeline& operator=(const DynamicPipeline&) = delete;
    ~DynamicPipeline() = default;

    void push();
    void pull();
    void reset() const;
    void update_graph_arguments(uint32_t index,
                                const void* arg_data,
                                size_t byte_size,
                                [[maybe_unused]] const ov::Strides& strides,
                                [[maybe_unused]] const ov::Shape& shapes);
    void update_graph_arguments_batching(uint32_t arg_index,
                                         const void* arg_data,
                                         [[maybe_unused]] const ov::Strides& strides,
                                         [[maybe_unused]] const ov::Shape& shapes,
                                         size_t batch_index);

    virtual std::vector<ov::ProfilingInfo> get_profiling_info() const;

protected:
    std::vector<size_t> get_strides(const std::vector<size_t>& strides_in_bytes, size_t element_size) const;

    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    std::shared_ptr<IGraph> _graph;
    const Config _config;
    const uint32_t _id;

    std::unique_ptr<zeroProfiling::ProfilingQuery> _profiling_query;
    std::shared_ptr<zeroProfiling::NpuInferProfiling> _npu_profiling;

    /**
     * @brief Indicates how many command lists will be used inside the pipeline.
     * @details Leveraging multiple command lists implies distributing the input/output buffers accross the batch axis
     * between these lists.
     *
     * If batching is handled on compiler's side then a single command list shall be used, we don't do any
     * specific operation inside the plugin in this case.
     */
    size_t _number_of_command_lists;

    std::shared_ptr<CommandQueue> _command_queue;
    std::vector<std::unique_ptr<PipelinedCommandLists>> _command_lists;
    std::vector<std::unique_ptr<Fence>> _fences;
    std::shared_ptr<EventPool> _event_pool;
    std::vector<std::shared_ptr<Event>> _events;
    bool _sync_output_with_fences = true;
    Logger _logger;

    // const std::vector<std::vector<std::shared_ptr<ZeroTensor>>> _levelZeroInputTensors;
    // const std::vector<std::shared_ptr<ZeroTensor>> _levelZeroOutputTensors;
};

}  // namespace intel_npu