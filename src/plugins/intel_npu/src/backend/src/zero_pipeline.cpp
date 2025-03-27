// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_pipeline.hpp"

#include <ze_api.h>
#include <ze_graph_ext.h>

#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/runtime.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"
#include "zero_remote_tensor.hpp"

namespace intel_npu {

Pipeline::Pipeline(const Config& config,
                   const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                   const std::shared_ptr<IGraph>& graph,
                   const std::vector<std::vector<std::shared_ptr<ov::ITensor>>>& input_tensors,
                   const std::vector<std::shared_ptr<ov::ITensor>>& output_tensors)
    : _graph(graph),
      _config(config),
      _id(_graph->get_unique_id()),
      _number_of_command_lists(_graph->get_batch_size().has_value() ? *_graph->get_batch_size() : 1),
      _logger("Pipeline", _config.get<LOG_LEVEL>()) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Zero_infer_request::Pipeline::Pipeline");
    _logger.debug("Pipeline - initialize started");

    OPENVINO_ASSERT(_sync_output_with_fences || !_config.get<RUN_INFERENCES_SEQUENTIALLY>(),
                    "In-order execution doesn't work in case synchronization of the inferences is done using events");

    if (_config.has<PERF_COUNT>() && _config.get<PERF_COUNT>()) {
        auto profiling_pool =
            std::make_shared<zeroProfiling::ProfilingPool>(init_structs, _graph, zeroProfiling::POOL_SIZE);
        _profiling_query = std::make_unique<zeroProfiling::ProfilingQuery>(init_structs, 0);

        if (profiling_pool->create()) {
            _profiling_query->create(profiling_pool);
        }

        if (_config.get<PROFILING_TYPE>() == ov::intel_npu::ProfilingType::INFER) {
            _logger.debug("ZeroInferRequest::ZeroInferRequest - profiling type == ov::intel_npu::ProfilingType::INFER");
            _npu_profiling = std::make_shared<zeroProfiling::NpuInferProfiling>(init_structs, _config.get<LOG_LEVEL>());
        }
    }

    if (!_sync_output_with_fences || _config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        _event_pool =
            std::make_shared<EventPool>(init_structs->getDevice(),
                                        init_structs->getContext(),
                                        _number_of_command_lists ? static_cast<uint32_t>(_number_of_command_lists) : 1);

        _events.reserve(_number_of_command_lists);
        for (size_t i = 0; i < _number_of_command_lists; i++) {
            _events.emplace_back(std::make_shared<Event>(_event_pool, static_cast<uint32_t>(i)));
        }
    }

    _command_lists.reserve(_number_of_command_lists);
    for (size_t i = 0; i < _number_of_command_lists; i++) {
        _command_lists.emplace_back(
            std::make_unique<CommandList>(init_structs, _graph->get_command_queue_group_ordinal()));
    }

    if (_sync_output_with_fences) {
        _fences.reserve(_number_of_command_lists);

        for (size_t i = 0; i < _number_of_command_lists; i++) {
            _fences.emplace_back(std::make_unique<Fence>(_graph->get_command_queue()));
        }
    }

    for (size_t i = 0; i < _number_of_command_lists; i++) {
        size_t io_index = 0;
        for (const auto& desc : graph->get_input_descriptors()) {
            if (input_tensors.at(io_index).size() > 1) {
                void* data = nullptr;
                auto remote_tensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(input_tensors.at(io_index).at(i));
                if (remote_tensor == nullptr) {
                    data = input_tensors.at(io_index).at(i)->data();
                } else {
                    data = remote_tensor->get_original_memory();
                }

                graph->set_argument_value(desc.idx, data);

                ++io_index;
                continue;
            }

            void* data = nullptr;
            auto remote_tensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(input_tensors.at(io_index).at(0));
            if (remote_tensor == nullptr) {
                data = input_tensors.at(io_index).at(0)->data();
            } else {
                data = remote_tensor->get_original_memory();
            }

            graph->set_argument_value(
                desc.idx,
                static_cast<unsigned char*>(data) +
                    (i * input_tensors.at(io_index).at(0)->get_byte_size()) / _number_of_command_lists);

            ++io_index;
        }

        io_index = 0;
        for (const auto& desc : graph->get_output_descriptors()) {
            void* data = nullptr;
            auto remote_tensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(output_tensors.at(io_index));
            if (remote_tensor == nullptr) {
                data = output_tensors.at(io_index)->data();
            } else {
                data = remote_tensor->get_original_memory();
            }

            graph->set_argument_value(
                desc.idx,
                static_cast<unsigned char*>(data) +
                    (i * output_tensors.at(io_index)->get_byte_size()) / _number_of_command_lists);
            ++io_index;
        }

        if (_config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
            if (_graph->get_last_submitted_event(i)) {
                _graph->get_last_submitted_event(i)->AppendWaitOnEvent(*_command_lists.at(i));
            }
        }

        /// append timestamp command if feature was activated
        if (_npu_profiling != nullptr) {
            _command_lists.at(i)->appendBarrier();
            _command_lists.at(i)->appendNpuTimestamp(reinterpret_cast<uint64_t*>(_npu_profiling->npu_ts_infer_start));
        }

        _command_lists.at(i)->appendGraphExecute(static_cast<ze_graph_handle_t>(graph->get_handle()),
                                                 _profiling_query ? _profiling_query->getHandle() : nullptr);

        /// append timestamp command if feature was activated
        if (_npu_profiling != nullptr) {
            _command_lists.at(i)->appendBarrier();
            _command_lists.at(i)->appendNpuTimestamp(reinterpret_cast<uint64_t*>(_npu_profiling->npu_ts_infer_end));
        }

        if (_config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
            if (_graph->get_last_submitted_event(i)) {
                _graph->get_last_submitted_event(i)->AppendEventReset(*_command_lists.at(i));
            }

            _events.at(i)->AppendSignalEvent(*_command_lists.at(i));
            _graph->set_last_submitted_event(_events.at(i), i);
        }

        // appendBarrier used in L0 as well
        if (!_sync_output_with_fences) {
            _command_lists.at(i)->appendBarrier();
            _events.at(i)->AppendSignalEvent(*_command_lists.at(i));
        }
    }
    _logger.debug("Pipeline - initialize completed");
}

void Pipeline::push() {
    _logger.debug("Pipeline - push() started");

    if (_config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        if (_id) {
            auto previousIndex = _graph->get_last_submitted_id();

            if (_id != ++previousIndex) {
                OPENVINO_THROW("Inferences should be called in the same order they were called the first time!");
            }
        }

        _graph->set_last_submitted_id(_id);
    }

    for (size_t i = 0; i < _command_lists.size(); ++i) {
        _command_lists.at(i)->close();

        OV_ITT_TASK_CHAIN(ZERO_PIPELINE_IP_PUSH, itt::domains::LevelZeroBackend, "Pipeline", "push");
        if (_sync_output_with_fences) {
            _graph->get_command_queue()->executeCommandList(*_command_lists.at(i), *_fences.at(i));
        } else {
            _graph->get_command_queue()->executeCommandList(*_command_lists.at(i));
        }
    }

    _logger.debug("Pipeline - push() completed");
};

void Pipeline::pull() {
    _logger.debug("Pipeline - pull() started");
    OV_ITT_TASK_CHAIN(ZERO_PIPELINE_IP_PULL, itt::domains::LevelZeroBackend, "Pipeline", "pull");

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

    _logger.debug("Pipeline - pull() completed");
};

void Pipeline::reset() const {
    _logger.debug("Pipeline - rest() started");

    for (size_t i = 0; i < _command_lists.size(); ++i) {
        if (_sync_output_with_fences) {
            _fences.at(i)->reset();
        } else {
            _events.at(i)->reset();
        }
    }

    _logger.debug("Pipeline - rest() completed");
};

void Pipeline::update_graph_arguments(uint32_t arg_index, const void* arg_data, size_t byte_size) {
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_UMCL, itt::domains::LevelZeroBackend, "Pipeline", "updateCommandList");
    _logger.debug("Pipeline - updateCommandList");

    const size_t number_of_command_lists = _command_lists.size();

    for (size_t i = 0; i < number_of_command_lists; i++) {
        _command_lists.at(i)->updateMutableCommandList(
            arg_index,
            static_cast<const unsigned char*>(arg_data) + (i * byte_size) / number_of_command_lists);
    }
};

void Pipeline::update_graph_arguments_batching(uint32_t arg_index, const void* arg_data, size_t command_list_index) {
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_UMCL, itt::domains::LevelZeroBackend, "Pipeline", "updateCommandListIndex");
    _logger.debug("Pipeline - updateCommandListIndex");

    const size_t number_of_command_lists = _command_lists.size();

    OPENVINO_ASSERT(command_list_index < number_of_command_lists,
                    "Command list index is higher than the number of Command lists ",
                    command_list_index);

    _command_lists.at(command_list_index)->updateMutableCommandList(arg_index, arg_data);
};

std::vector<ov::ProfilingInfo> Pipeline::get_profiling_info() const {
    _logger.debug("InferRequest::get_profiling_info started");
    if (!_config.has<PERF_COUNT>() || !_config.get<PERF_COUNT>()) {
        _logger.warning("InferRequest::get_profiling_info complete with empty {}.");
        return {};
    }

    if (_config.get<PROFILING_TYPE>() == ov::intel_npu::ProfilingType::INFER) {
        _logger.debug("InferRequest::get_profiling_info complete with _npu_profiling->getNpuInferStatistics().");
        return _npu_profiling->getNpuInferStatistics();
    }
    /// PROFILING_TYPE = MODEL or undefined = fallback to model profiling
    if (_config.get<COMPILER_TYPE>() == ov::intel_npu::CompilerType::MLIR) {
        // For plugin compiler retreive raw profiling data from backend and delegate
        // processing to the compiler
        _logger.debug("InferRequest::get_profiling_info complete with compiler->process_profiling_output().");
        return _graph->process_profiling_output(_profiling_query->getData<uint8_t>(), _config);
    } else {
        _logger.debug("InferRequest::get_profiling_info complete with _profiling_query.getLayerStatistics().");
        return _profiling_query->getLayerStatistics();
    }
}

}  // namespace intel_npu
