// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_pipeline.hpp"

#include <ze_api.h>
#include <ze_graph_ext.h>

#include "intel_npu/al/itt.hpp"
#include "intel_npu/al/prefix.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"

namespace intel_npu {

struct DiscretePipeline final : public Pipeline {
public:
    DiscretePipeline(const Config& config,
                     const ze_device_handle_t& device_handle,
                     ze_context_handle_t& context,
                     ze_graph_dditable_ext_curr_t* graph_ddi_table_ext,
                     const std::shared_ptr<const IExecutor>& executorPtr,
                     ze_graph_profiling_query_handle_t profiling_handle,
                     const std::array<std::shared_ptr<CommandQueue>, stage::COUNT>& command_queues,
                     const uint32_t& group_ordinal,
                     const std::vector<std::shared_ptr<ov::ITensor>>& inputTensors,
                     const std::vector<std::shared_ptr<ov::ITensor>>& outputTensors)
        : _config(config),
          _command_queues{command_queues},
          _command_list{{{device_handle, context, graph_ddi_table_ext, _config, group_ordinal},
                         {device_handle, context, graph_ddi_table_ext, _config, group_ordinal},
                         {device_handle, context, graph_ddi_table_ext, _config, group_ordinal}}},
          _fence{{{*_command_queues[stage::UPLOAD], _config},
                  {*_command_queues[stage::EXECUTE], _config},
                  {*_command_queues[stage::READBACK], _config}}},
          _event_pool(device_handle, context, stage::COUNT, _config),
          _event{{{_event_pool.handle(), stage::UPLOAD, _config},
                  {_event_pool.handle(), stage::EXECUTE, _config},
                  {_event_pool.handle(), stage::READBACK, _config}}},
          _logger("DiscretePipeline", _config.get<LOG_LEVEL>()) {
        _logger.debug("DiscretePipeline - initialize started");
        const ZeroExecutor* executor = static_cast<const ZeroExecutor*>(executorPtr.get());
        static const std::size_t alignment = STANDARD_PAGE_SIZE;

        OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Zero_infer_request::DiscretePipeline::DiscretePipeline");
        for (const auto& desc : executor->get_input_descriptors()) {
            _deviceInputs.appendArgument(zeroUtils::getSizeIOBytes(desc.info));
        }
        _deviceInputs.allocate(device_handle, context);

        _logger.debug("DiscretePipeline - appending memory copy and set argument value for input");

        size_t inputIndex = 0;
        for (const auto& desc : executor->get_input_descriptors()) {
            const void* tensorBuffer = reinterpret_cast<const void*>(inputTensors.at(inputIndex)->data());

            const std::size_t argSize = zeroUtils::getSizeIOBytes(desc.info);
            std::size_t size = argSize + alignment - (argSize % alignment);

            _command_list[stage::UPLOAD].appendMemoryCopy(_deviceInputs.getDevicePtr(inputIndex), tensorBuffer, size);

            executor->setArgumentValue(desc.idx, _deviceInputs.getDevicePtr(inputIndex));
            ++inputIndex;
        }

        _logger.debug("DiscretePipeline - append signal event");

        _command_list[stage::UPLOAD].appendBarrier();
        _event[stage::UPLOAD].AppendSignalEvent(_command_list[stage::UPLOAD]);

        for (const auto& desc : executor->get_output_descriptors()) {
            _deviceOutputs.appendArgument(zeroUtils::getSizeIOBytes(desc.info));
        }
        _deviceOutputs.allocate(device_handle, context);

        _logger.debug("DiscretePipeline - appending memory copy and set argument value for output");

        size_t outputIndex = 0;
        for (const auto& desc : executor->get_output_descriptors()) {
            void* tensorBuffer = reinterpret_cast<void*>(outputTensors.at(outputIndex)->data());

            const std::size_t argSize = zeroUtils::getSizeIOBytes(desc.info);
            std::size_t size = argSize + alignment - (argSize % alignment);

            _command_list[stage::READBACK].appendMemoryCopy(tensorBuffer,
                                                            _deviceOutputs.getDevicePtr(outputIndex),
                                                            size);

            executor->setArgumentValue(desc.idx, _deviceOutputs.getDevicePtr(outputIndex));
            ++outputIndex;
        }

        _event[stage::UPLOAD].AppendWaitOnEvent(_command_list[stage::EXECUTE]);
        _logger.debug("DiscretePipeline - appendGraphExecute");
        _command_list[stage::EXECUTE].appendGraphExecute(executor->graph(), profiling_handle);
        _logger.debug("DiscretePipeline - appendEventReset");
        _event[stage::UPLOAD].AppendEventReset(_command_list[stage::READBACK]);

        for (auto& commandList : _command_list) {
            commandList.close();
        }
        _logger.debug("DiscretePipeline - initialize completed");
    };

    DiscretePipeline(const DiscretePipeline&) = delete;
    DiscretePipeline& operator=(const DiscretePipeline&) = delete;
    virtual ~DiscretePipeline() = default;

    void push(size_t) override {
        _logger.debug("DiscretePipeline - push() started");
        OV_ITT_TASK_CHAIN(ZERO_INFER_REQUEST_DP_PUSH,
                          itt::domains::LevelZeroBackend,
                          "DiscretePipeline::push",
                          "UPLOAD");
        // Dispatch command to copy input data from upload heap to default heap
        _command_queues[stage::UPLOAD]->executeCommandList(_command_list[stage::UPLOAD]);

        OV_ITT_TASK_NEXT(ZERO_INFER_REQUEST_DP_PUSH, "EXECUTE");
        // Submit the command list for execute
        _command_queues[stage::EXECUTE]->executeCommandList(_command_list[stage::EXECUTE], _fence[stage::EXECUTE]);
        _logger.debug("DiscretePipeline - push() completed");
    };

    void pull(size_t) override {
        _logger.debug("DiscretePipeline - pull() started");
        OV_ITT_TASK_CHAIN(ZERO_INFER_REQUEST_DP_PULL,
                          itt::domains::LevelZeroBackend,
                          "DiscretePipeline::pull",
                          "EXECUTE");
        // Wait for execute to finish
        _fence[stage::EXECUTE].hostSynchronize();
        OV_ITT_TASK_NEXT(ZERO_INFER_REQUEST_DP_PULL, "READBACK");
        // Schedule the copy of outputs from zeDriverAllocDeviceMem to zeDriverAllocHostMem
        _command_queues[stage::READBACK]->executeCommandList(_command_list[stage::READBACK], _fence[stage::READBACK]);
        // Wait for output copy to finish execution for _fence from the host, to make sure that data
        // is available in the hostMem buffer of the output
        _fence[stage::READBACK].hostSynchronize();
        _logger.debug("DiscretePipeline - pull() completed");
    };

    void reset(size_t) const override {
        // Reset the fence objects
        for (auto& fence : _fence) {
            fence.reset();
        }
    };

private:
    const Config _config;
    const std::array<std::shared_ptr<CommandQueue>, stage::COUNT>& _command_queues;
    std::array<CommandList, stage::COUNT> _command_list;
    std::array<Fence, stage::COUNT> _fence;
    EventPool _event_pool;
    std::array<Event, stage::COUNT> _event;
    Logger _logger;
};

struct IntegratedPipeline final : public Pipeline {
public:
    IntegratedPipeline(const Config& config,
                       const ze_device_handle_t& device_handle,
                       ze_context_handle_t& context,
                       ze_graph_dditable_ext_curr_t* graph_ddi_table_ext,
                       const std::shared_ptr<const IExecutor>& executorPtr,
                       ze_graph_profiling_query_handle_t profiling_handle,
                       std::shared_ptr<zeroProfiling::NpuInferProfiling> npu_profiling,
                       CommandQueue& command_queue,
                       const uint32_t& group_ordinal,
                       const std::vector<std::shared_ptr<ov::ITensor>>& inputTensors,
                       const std::vector<std::shared_ptr<ov::ITensor>>& outputTensors,
                       const size_t batch_size)
        : _config(config),
          _command_queue{command_queue},
          _event_pool{device_handle, context, batch_size ? static_cast<uint32_t>(batch_size) : 1, _config},
          _npu_profiling(std::move(npu_profiling)),
          _logger("IntegratedPipeline", _config.get<LOG_LEVEL>()) {
        const ZeroExecutor* executor = static_cast<const ZeroExecutor*>(executorPtr.get());

        OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend,
                           "Zero_infer_request::IntegratedPipeline::IntegratedPipeline");
        _logger.debug("IntegratedPipeline - initialize started");

        _command_lists.reserve(batch_size);
        _events.reserve(batch_size);
        _fences.reserve(batch_size);
        _logger.debug("IntegratedPipeline - emplace_back _event_pool and _command_queue");
        for (size_t i = 0; i < batch_size; i++) {
            _command_lists.emplace_back(
                std::make_unique<CommandList>(device_handle, context, graph_ddi_table_ext, _config, group_ordinal));
            _events.emplace_back(std::make_unique<Event>(_event_pool.handle(), static_cast<uint32_t>(i), _config));
            _fences.emplace_back(std::make_unique<Fence>(_command_queue, _config));
        }

        for (size_t i = 0; i < batch_size; i++) {
            size_t inputIndex = 0;
            for (const auto& desc : executor->get_input_descriptors()) {
                const std::shared_ptr<ov::ITensor>& inputTensor = inputTensors.at(inputIndex);
                size_t inputTensorByteSize = inputTensor->get_byte_size();
                executor->setArgumentValue(
                    desc.idx,
                    static_cast<unsigned char*>(inputTensor->data()) + (i * inputTensorByteSize) / batch_size);
                ++inputIndex;
            }

            size_t outputIndex = 0;
            for (const auto& desc : executor->get_output_descriptors()) {
                const std::shared_ptr<ov::ITensor>& outputTensor = outputTensors.at(outputIndex);
                size_t outputTensorByteSize = outputTensor->get_byte_size();
                executor->setArgumentValue(
                    desc.idx,
                    static_cast<unsigned char*>(outputTensor->data()) + (i * outputTensorByteSize) / batch_size);
                ++outputIndex;
            }

            /// append timestamp command if feature was activated
            if (_npu_profiling != nullptr) {
                _command_lists.at(i)->appendBarrier();
                _command_lists.at(i)->appendNpuTimestamp(
                    reinterpret_cast<uint64_t*>(_npu_profiling->npu_ts_infer_start));
            }

            _command_lists.at(i)->appendGraphExecute(executor->graph(), profiling_handle);

            /// append timestamp command if feature was activated
            if (_npu_profiling != nullptr) {
                _command_lists.at(i)->appendBarrier();
                _command_lists.at(i)->appendNpuTimestamp(reinterpret_cast<uint64_t*>(_npu_profiling->npu_ts_infer_end));
            }

            // appendBarrier used in L0 as well
            if (!sync_output_with_fences_) {
                _command_lists.at(i)->appendBarrier();
                _events.at(i)->AppendSignalEvent(*_command_lists.at(i));
            }
            _command_lists.at(i)->close();
        }
        _logger.debug("IntegratedPipeline - initialize completed");
    }

    IntegratedPipeline(const IntegratedPipeline&) = delete;
    IntegratedPipeline& operator=(const IntegratedPipeline&) = delete;
    virtual ~IntegratedPipeline() = default;

    void push(size_t batch_index) override {
        _logger.debug("IntegratedPipeline - push() started");
        OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_PUSH, itt::domains::LevelZeroBackend, "IntegratedPipeline", "push");
        if (sync_output_with_fences_) {
            _command_queue.executeCommandList(*_command_lists.at(batch_index), *_fences.at(batch_index));
        } else {
            _command_queue.executeCommandList(*_command_lists.at(batch_index));
        }
        _logger.debug("IntegratedPipeline - push() completed");
    };

    void pull(size_t batch_index) override {
        _logger.debug("IntegratedPipeline - pull() started");
        OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_PULL, itt::domains::LevelZeroBackend, "IntegratedPipeline", "pull");
        if (sync_output_with_fences_) {
            _fences.at(batch_index)->hostSynchronize();
        } else {
            _events.at(batch_index)->hostSynchronize();
        }
        /// sample npu timestamps if feature was activated
        if (_npu_profiling != nullptr) {
            _npu_profiling->sampleNpuTimestamps();
        }
        _logger.debug("IntegratedPipeline - pull() completed");
    };

    void reset(size_t batch_index) const override {
        _logger.debug("IntegratedPipeline - rest() started");
        if (sync_output_with_fences_) {
            _fences.at(batch_index)->reset();
        } else {
            _events.at(batch_index)->reset();
        }
        _logger.debug("IntegratedPipeline - rest() completed");
    };

private:
    const Config _config;
    CommandQueue& _command_queue;
    std::vector<std::unique_ptr<CommandList>> _command_lists;
    std::vector<std::unique_ptr<Fence>> _fences;
    EventPool _event_pool;
    std::vector<std::unique_ptr<Event>> _events;
    bool sync_output_with_fences_ = true;
    std::shared_ptr<zeroProfiling::NpuInferProfiling> _npu_profiling;
    Logger _logger;
};

std::unique_ptr<Pipeline> makePipeline(const std::shared_ptr<const IExecutor>& executorPtr,
                                       const Config& config,
                                       zeroProfiling::ProfilingPool& profiling_pool,
                                       zeroProfiling::ProfilingQuery& profiling_query,
                                       std::shared_ptr<zeroProfiling::NpuInferProfiling> npu_profiling,
                                       const std::vector<std::shared_ptr<ov::ITensor>>& inputTensors,
                                       const std::vector<std::shared_ptr<ov::ITensor>>& outputTensors,
                                       const size_t batch_size) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Infer_request::makePipeline");
    if (profiling_pool.create())
        profiling_query.create(profiling_pool._handle);

    const ZeroExecutor* executor = static_cast<const ZeroExecutor*>(executorPtr.get());

    const ze_device_handle_t device_handle = executor->getInitStructs()->getDevice();
    ze_context_handle_t context = executor->getInitStructs()->getContext();
    ze_graph_dditable_ext_curr_t* graph_ddi_table_ext = executor->getInitStructs()->getGraphDdiTable();
    auto& command_queues = executor->getCommandQueue();
    uint32_t group_ordinal = executor->get_group_ordinal();

    ze_device_properties_t properties = {};
    properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    zeroUtils::throwOnFail("zeDeviceGetProperties", zeDeviceGetProperties(device_handle, &properties));

    if (properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) {
        return std::make_unique<IntegratedPipeline>(config,
                                                    device_handle,
                                                    context,
                                                    graph_ddi_table_ext,
                                                    executorPtr,
                                                    profiling_query.getHandle(),
                                                    npu_profiling,
                                                    *command_queues[stage::EXECUTE],
                                                    group_ordinal,
                                                    inputTensors,
                                                    outputTensors,
                                                    batch_size);
    }

    return std::make_unique<DiscretePipeline>(config,
                                              device_handle,
                                              context,
                                              graph_ddi_table_ext,
                                              executorPtr,
                                              profiling_query.getHandle(),
                                              command_queues,
                                              group_ordinal,
                                              inputTensors,
                                              outputTensors);
}

}  // namespace intel_npu
