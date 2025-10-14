// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#    ifdef _MSC_VER
#        pragma warning(push)
#        pragma warning(disable : 4146 4267 4244 4996)
#    endif

#    include "zero_dynamic_pipeline.hpp"

#    include <ze_api.h>
#    include <ze_graph_ext.h>

#    include <sstream>

#    include "intel_npu/common/itt.hpp"
#    include "intel_npu/config/options.hpp"
#    include "intel_npu/prefix.hpp"
#    include "intel_npu/utils/logger/logger.hpp"
#    include "intel_npu/utils/zero/zero_api.hpp"
#    include "intel_npu/utils/zero/zero_remote_tensor.hpp"
#    include "intel_npu/utils/zero/zero_types.hpp"

namespace intel_npu {

DynamicPipeline::DynamicPipeline(const Config& config,
                                 const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                 const std::shared_ptr<IGraph>& graph,
                                 const std::vector<std::vector<std::shared_ptr<ZeroTensor>>>& input_tensors,
                                 const std::vector<std::shared_ptr<ZeroTensor>>& output_tensors,
                                 size_t batch_size)
    : _init_structs(init_structs),
      _graph(graph),
      _config(config),
      _id(_graph->get_unique_id()),
      _number_of_command_lists(batch_size),
      _logger("DynamicPipeline", _config.get<LOG_LEVEL>())
/*_levelZeroInputTensors(input_tensors),
_levelZeroOutputTensors(output_tensors)*/
{
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Zero_infer_request::DynamicPipeline::DynamicPipeline");

    _logger.debug("DynamicPipeline - initialize started, number_of_command_lists %i", _number_of_command_lists);

    // if (_init_structs->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 1) &&
    //     _config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
    //     _graph->resize_last_submitted_event(_number_of_command_lists);
    // }

    OPENVINO_ASSERT(_sync_output_with_fences || !_config.get<RUN_INFERENCES_SEQUENTIALLY>() ||
                        _init_structs->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 1),
                    "In-order execution doesn't work in case synchronization of the inferences is done using events");

    if (_config.has<PERF_COUNT>() && _config.get<PERF_COUNT>()) {
        auto profiling_pool =
            std::make_shared<zeroProfiling::ProfilingPool>(_init_structs, _graph, zeroProfiling::POOL_SIZE);
        _profiling_query = std::make_unique<zeroProfiling::ProfilingQuery>(_init_structs, 0);

        if (profiling_pool->create()) {
            _profiling_query->create(profiling_pool);
        }

        if (_config.get<PROFILING_TYPE>() == ov::intel_npu::ProfilingType::INFER) {
            _logger.debug("ZeroInferRequest::ZeroInferRequest - profiling type == ov::intel_npu::ProfilingType::INFER");
            _npu_profiling =
                std::make_shared<zeroProfiling::NpuInferProfiling>(_init_structs, _config.get<LOG_LEVEL>());
        }
    }

    if (!_sync_output_with_fences || (_init_structs->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 1) &&
                                      _config.get<RUN_INFERENCES_SEQUENTIALLY>())) {
        _event_pool =
            std::make_shared<EventPool>(_init_structs->getDevice(),
                                        _init_structs->getContext(),
                                        _number_of_command_lists ? static_cast<uint32_t>(_number_of_command_lists) : 1);

        _events.reserve(_number_of_command_lists);
        for (size_t i = 0; i < _number_of_command_lists; i++) {
            _events.emplace_back(std::make_shared<Event>(_event_pool, static_cast<uint32_t>(i)));
        }
    }
    _logger.debug("DynamicPipeline - emplace_back _event_pool and _command_queue completed");

    intel_npu::IRGraph* irGraph = dynamic_cast<intel_npu::IRGraph*>(graph.get());

    uint64_t num_of_subgraphs = irGraph->get_num_subgraphs();

    _command_lists.reserve(_number_of_command_lists);
    for (size_t i = 0; i < _number_of_command_lists; i++) {
        _command_lists.emplace_back(std::make_unique<PipelinedCommandLists>(num_of_subgraphs,
                                                                            _init_structs,
                                                                            _graph->get_command_queue_group_ordinal()));
    }

    if (_sync_output_with_fences) {
        _fences.reserve(_number_of_command_lists);

        for (size_t i = 0; i < _number_of_command_lists; i++) {
            _fences.emplace_back(std::make_unique<Fence>(_graph->get_command_queue()));
        }
    }

    for (size_t i = 0; i < _number_of_command_lists; i++) {
        _logger.debug("DynamicPipeline - set args for command list number: %zu", i);
        size_t io_index = 0;
        for (const auto& desc : graph->get_input_descriptors()) {
            // if (isMainInputWeightsName(desc.info.name)) {
            //     // These values were set while running the "WeightlessGraph::init" method
            //     continue;
            // }

            if (input_tensors.at(io_index).size() > 1) {
                _logger.debug("DynamicPipeline - set args for input index: %zu", io_index);

                irGraph->set_argument_property(desc.idx,
                                               input_tensors.at(io_index).at(i)->data(),
                                               input_tensors.at(io_index).at(i)->get_strides(),
                                               input_tensors.at(io_index).at(i)->get_shape());

                ++io_index;
                continue;
            }

            _logger.debug(" update tensor property for input desc index: %d", desc.idx);
            irGraph->set_argument_property(
                desc.idx,
                static_cast<unsigned char*>(input_tensors.at(io_index).at(0)->data()) +
                    (i * input_tensors.at(io_index).at(0)->get_byte_size()) / _number_of_command_lists,
                input_tensors.at(io_index).at(0)->get_strides(),
                input_tensors.at(io_index).at(0)->get_shape());

            ++io_index;
        }

        io_index = 0;
        for (const auto& desc : graph->get_output_descriptors()) {
            _logger.debug("DynamicPipeline - update tensor property for output desc index: %d", desc.idx);
            irGraph->set_argument_property(
                desc.idx,
                static_cast<unsigned char*>(output_tensors.at(io_index)->data()) +
                    (i * output_tensors.at(io_index)->get_byte_size()) / _number_of_command_lists,
                output_tensors.at(io_index)->get_strides(),
                output_tensors.at(io_index)->get_shape());

            ++io_index;
        }

        if (_init_structs->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 1) &&
            _config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
            if (_graph->get_last_submitted_event(i)) {
                _command_lists.at(i)->appendWaitOnEvent(_graph->get_last_submitted_event(i));
            }
        }

        /// TODO, profiling needs to add timestamp before and after graph execute, but the execute is added inside blob
        /// now
        // /// append timestamp command if feature was activated
        // if (_npu_profiling != nullptr) {
        //     _command_lists.at(i)->appendBarrier();
        //     _command_lists.at(i)->appendNpuTimestamp(reinterpret_cast<uint64_t*>(_npu_profiling->npu_ts_infer_start));
        // }

        _command_lists.at(i)->bind(dynamic_cast<intel_npu::IRGraph*>(graph.get()));

        // /// Old graph execute called here

        // /// append timestamp command if feature was activated
        // if (_npu_profiling != nullptr) {
        //     _command_lists.at(i)->appendBarrier();
        //     _command_lists.at(i)->appendNpuTimestamp(reinterpret_cast<uint64_t*>(_npu_profiling->npu_ts_infer_end));
        // }

        // if (_init_structs->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 1) &&
        //     _config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        //     if (_graph->get_last_submitted_event(i)) {
        //         _command_lists.at(i)->appendReset(_graph->get_last_submitted_event(i));
        //     }

        //     _command_lists.at(i)->appendSignalEvent(_events.at(i));
        //     _graph->set_last_submitted_event(_events.at(i), i);
        // }

        // // appendBarrier used in L0 as well
        // if (!_sync_output_with_fences) {
        //     _command_lists.at(i)->appendBarrier();
        //     _command_lists.at(i)->appendSignalEvent(_events.at(i));
        // }
    }
    _logger.debug("DynamicPipeline - initialize completed");
}

void DynamicPipeline::PipelinedCommandLists::bind(IRGraph* graph) {
    graph->getBinding(_binding);
}

void DynamicPipeline::push() {
    _logger.debug("DynamicPipeline - push() started");
    //_logger.debug("inputs.size = %d, outputs.size=%d", _levelZeroInputTensors.size(), _levelZeroOutputTensors.size());

    if (_init_structs->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 1) &&
        _config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        if (_id) {
            auto previousIndex = _graph->get_last_submitted_id();

            if (_id != ++previousIndex) {
                OPENVINO_THROW("Inferences should be called in the same order they were called the first time!");
            }
        }

        _graph->set_last_submitted_id(_id);
    }

    auto commandQueueHandle = _graph->get_command_queue()->handle();
    for (size_t i = 0; i < _command_lists.size(); ++i) {
        OV_ITT_TASK_CHAIN(ZERO_PIPELINE_IP_PUSH, itt::domains::LevelZeroBackend, "Pipeline", "push");

        ze_fence_handle_t fence = nullptr;
        ze_event_handle_t event = nullptr;
        if (_sync_output_with_fences) {
            fence = _fences.at(i)->handle();
        } else {
            // TODO
        }

        auto& command_lists = _command_lists.at(i);
        auto graphArguments = command_lists->getBinding();
        _logger.debug("Inputs info for IRGraph:");
        for (auto& memType : graphArguments._inputs) {
            _logger.debug(" sizes: %d*%d*%d*%d",
                          memType->sizes[0],
                          memType->sizes[1],
                          memType->sizes[2],
                          memType->sizes[3]);
            _logger.debug(" strides: %d*%d*%d*%d",
                          memType->strides[0],
                          memType->strides[1],
                          memType->strides[2],
                          memType->strides[3]);
            _logger.debug(" basePtr: %p data: %p offset: %d", memType->basePtr, memType->data, memType->offset);
            _logger.debug("");
        }
        _logger.debug("Outputs info for IRGraph:");
        for (auto& memType : graphArguments._outputs) {
            _logger.debug(" sizes: %d*%d*%d*%d",
                          memType->sizes[0],
                          memType->sizes[1],
                          memType->sizes[2],
                          memType->sizes[3]);
            _logger.debug(" strides: %d*%d*%d*%d",
                          memType->strides[0],
                          memType->strides[1],
                          memType->strides[2],
                          memType->strides[3]);
            _logger.debug(" basePtr: %p data: %p offset: %d", memType->basePtr, memType->data, memType->offset);
            _logger.debug("");
        }

        dynamic_cast<IRGraph*>(_graph.get())
            ->execute(_init_structs,
                      command_lists->getBinding(),
                      command_lists->getHandles(),
                      commandQueueHandle,
                      fence,
                      event,
                      nullptr);
    }

    _logger.debug("DynamicPipeline - push() completed");
}

void DynamicPipeline::pull() {
    _logger.debug("DynamicPipeline - pull() started");
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

    _logger.debug("DynamicPipeline - pull() completed");
};

void DynamicPipeline::reset() const {
    _logger.debug("DynamicPipeline - reset() started");
    for (size_t i = 0; i < _command_lists.size(); ++i) {
        if (_sync_output_with_fences) {
            _fences.at(i)->reset();
        } else {
            _events.at(i)->reset();
        }
    }

    _logger.debug("Pipeline - rest() completed");
};

void DynamicPipeline::update_graph_arguments(uint32_t arg_index,
                                             const void* arg_data,
                                             size_t byte_size,
                                             [[maybe_unused]] const ov::Strides& strides,
                                             [[maybe_unused]] const ov::Shape& shapes) {
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_UMCL, itt::domains::LevelZeroBackend, "DynamicPipeline", "updateCommandList");
    _logger.debug("DynamicPipeline - updateCommandList");

    const size_t number_of_command_lists = _command_lists.size();

    for (size_t i = 0; i < number_of_command_lists; i++) {
        _command_lists.at(i)->updateMutableCommandList(
            arg_index,
            static_cast<const unsigned char*>(arg_data) + (i * byte_size) / number_of_command_lists,
            strides,
            shapes);
    }
};

void DynamicPipeline::update_graph_arguments_batching(uint32_t arg_index,
                                                      const void* arg_data,
                                                      [[maybe_unused]] const ov::Strides& strides,
                                                      [[maybe_unused]] const ov::Shape& shapes,
                                                      size_t batch_index) {
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_UMCL,
                      itt::domains::LevelZeroBackend,
                      "DynamicPipeline",
                      "updateCommandListIndex");
    _logger.debug("DynamicPipeline - updateCommandListIndex");

    const size_t number_of_command_lists = _command_lists.size();

    OPENVINO_ASSERT(batch_index < number_of_command_lists,
                    "batch_index is higher than the number of Command lists ",
                    batch_index);

    _command_lists.at(batch_index)->updateMutableCommandList(arg_index, arg_data, strides, shapes);
};

std::vector<ov::ProfilingInfo> DynamicPipeline::get_profiling_info() const {
    // TODO: Need a way to get profiling info
    _logger.debug("InferRequest::get_profiling_info started");
    if (!_config.has<PERF_COUNT>() || !_config.get<PERF_COUNT>()) {
        _logger.warning("InferRequest::get_profiling_info complete with empty {}.");
        return {};
    }

    if (_config.get<PROFILING_TYPE>() == ov::intel_npu::ProfilingType::INFER) {
        _logger.debug("InferRequest::get_profiling_info complete with _npu_profiling->getNpuInferStatistics().");
        return _npu_profiling->getNpuInferStatistics();
    }
    // /// PROFILING_TYPE = MODEL or undefined = fallback to model profiling
    // if (_config.get<COMPILER_TYPE>() == ov::intel_npu::CompilerType::MLIR) {
    // For plugin compiler retreive raw profiling data from backend and delegate
    // processing to the compiler
    _logger.debug("InferRequest::get_profiling_info complete with compiler->process_profiling_output().");
    return _graph->process_profiling_output(_profiling_query->getData<uint8_t>(), _config);
    // } else {
    //     _logger.debug("InferRequest::get_profiling_info complete with _profiling_query.getLayerStatistics().");
    //     return _profiling_query->getLayerStatistics();
    // }
}

}  // namespace intel_npu

#    ifdef _MSC_VER
#        pragma warning(pop)
#endif
