// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_pipeline.hpp"

#include <ze_api.h>
#include <ze_graph_ext.h>

#include "intel_npu/common/itt.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"

namespace intel_npu {

Pipeline::Pipeline(const Config& config,
                   const std::shared_ptr<ZeroInitStructsHolder>& initStructs,
                   const std::shared_ptr<IGraph>& graph,
                   zeroProfiling::ProfilingPool& profiling_pool,
                   zeroProfiling::ProfilingQuery& profiling_query,
                   std::shared_ptr<zeroProfiling::NpuInferProfiling> npu_profiling,
                   const std::vector<std::vector<std::optional<TensorData>>>& inputTensorsData,
                   const std::vector<std::optional<TensorData>>& outputTensorsData,
                   size_t numberOfCommandLists,
                   uint32_t group_ordinal)
    : _config(config),
      _command_queue(graph->get_command_queue()),
      _event_pool{initStructs->getDevice(),
                  initStructs->getContext(),
                  numberOfCommandLists ? static_cast<uint32_t>(numberOfCommandLists) : 1},
      _npu_profiling(std::move(npu_profiling)),
      _logger("Pipeline", _config.get<LOG_LEVEL>()) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Zero_infer_request::Pipeline::Pipeline");
    _logger.debug("Pipeline - initialize started");

    if (profiling_pool.create()) {
        profiling_query.create(profiling_pool._handle);
    }

    _command_lists.reserve(numberOfCommandLists);
    _events.reserve(numberOfCommandLists);
    _fences.reserve(numberOfCommandLists);
    _logger.debug("Pipeline - emplace_back _event_pool and _command_queue");
    for (size_t i = 0; i < numberOfCommandLists; i++) {
        _command_lists.emplace_back(
            std::make_unique<CommandList>(initStructs->getDevice(),
                                          initStructs->getContext(),
                                          initStructs->getGraphDdiTable(),
                                          group_ordinal,
                                          initStructs->getMutableCommandListVersion() ? true : false));
        _events.emplace_back(std::make_unique<Event>(_event_pool.handle(), static_cast<uint32_t>(i)));
        _fences.emplace_back(std::make_unique<Fence>(*_command_queue));
    }

    for (size_t i = 0; i < numberOfCommandLists; i++) {
        size_t ioIndex = 0;
        for (const auto& desc : graph->get_input_descriptors()) {
            if (inputTensorsData.at(ioIndex).size() > 1) {
                graph->set_argument_value(desc.idx, inputTensorsData.at(ioIndex).at(i)->mem);

                ++ioIndex;
                continue;
            }

            graph->set_argument_value(desc.idx,
                                      static_cast<unsigned char*>(inputTensorsData.at(ioIndex).at(0)->mem) +
                                          (i * inputTensorsData.at(ioIndex).at(0)->size) / numberOfCommandLists);

            ++ioIndex;
        }

        ioIndex = 0;
        for (const auto& desc : graph->get_output_descriptors()) {
            graph->set_argument_value(desc.idx,
                                      static_cast<unsigned char*>(outputTensorsData.at(ioIndex)->mem) +
                                          (i * outputTensorsData.at(ioIndex)->size) / numberOfCommandLists);
            ++ioIndex;
        }

        /// append timestamp command if feature was activated
        if (_npu_profiling != nullptr) {
            _command_lists.at(i)->appendBarrier();
            _command_lists.at(i)->appendNpuTimestamp(reinterpret_cast<uint64_t*>(_npu_profiling->npu_ts_infer_start));
        }

        _command_lists.at(i)->appendGraphExecute(static_cast<ze_graph_handle_t>(graph->get_handle()),
                                                 profiling_query.getHandle());

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
    _logger.debug("Pipeline - initialize completed");
}

void Pipeline::push() {
    _logger.debug("Pipeline - push() started");

    for (size_t i = 0; i < _command_lists.size(); ++i) {
        OV_ITT_TASK_CHAIN(ZERO_PIPELINE_IP_PUSH, itt::domains::LevelZeroBackend, "Pipeline", "push");
        if (sync_output_with_fences_) {
            _command_queue->executeCommandList(*_command_lists.at(i), *_fences.at(i));
        } else {
            _command_queue->executeCommandList(*_command_lists.at(i));
        }
    }

    _logger.debug("Pipeline - push() completed");
};

void Pipeline::pull() {
    _logger.debug("Pipeline - pull() started");
    OV_ITT_TASK_CHAIN(ZERO_PIPELINE_IP_PULL, itt::domains::LevelZeroBackend, "Pipeline", "pull");

    for (size_t i = 0; i < _command_lists.size(); ++i) {
        if (sync_output_with_fences_) {
            _fences.at(i)->hostSynchronize();
        } else {
            _events.at(i)->hostSynchronize();
        }
        /// sample npu timestamps if feature was activated
        if (_npu_profiling != nullptr) {
            _npu_profiling->sampleNpuTimestamps();
        }
    }

    _logger.debug("Pipeline - pull() completed");
};

void Pipeline::reset() const {
    _logger.debug("Pipeline - rest() started");

    for (size_t i = 0; i < _command_lists.size(); ++i) {
        if (sync_output_with_fences_) {
            _fences.at(i)->reset();
        } else {
            _events.at(i)->reset();
        }
    }

    _logger.debug("Pipeline - rest() completed");
};

void Pipeline::updateCommandList(const TensorData& tensorsData, uint32_t index) {
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_UMCL, itt::domains::LevelZeroBackend, "Pipeline", "updateCommandList");
    _logger.debug("Pipeline - updateCommandList");

    const size_t numberOfCommandLists = _command_lists.size();

    for (size_t i = 0; i < numberOfCommandLists; i++) {
        _command_lists.at(i)->updateMutableCommandList(
            index,
            static_cast<unsigned char*>(tensorsData.mem) + (i * tensorsData.size) / numberOfCommandLists);
        _command_lists.at(i)->close();
    }
};

void Pipeline::updateCommandList(const TensorData& tensorsData, uint32_t index, size_t commandListIndex) {
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_UMCL, itt::domains::LevelZeroBackend, "Pipeline", "updateCommandList");
    _logger.debug("Pipeline - updateCommandList");

    const size_t numberOfCommandLists = _command_lists.size();

    OPENVINO_ASSERT(commandListIndex < numberOfCommandLists,
                    "Command list index is higgher than the number of Command lists ",
                    commandListIndex);

    _command_lists.at(commandListIndex)->updateMutableCommandList(index, tensorsData.mem);
    _command_lists.at(commandListIndex)->close();
};

}  // namespace intel_npu
