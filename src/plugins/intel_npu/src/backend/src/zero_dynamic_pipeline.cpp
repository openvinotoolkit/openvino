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

namespace {
std::vector<size_t> get_strides(const std::vector<size_t>& strides_in_bytes, size_t element_size) {
    std::vector<size_t> element_strides(strides_in_bytes.size());
    std::transform(strides_in_bytes.rbegin(),
                   strides_in_bytes.rend(),
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
    // TODO: not support now
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

        _command_lists.at(i)->bind(dynamic_cast<intel_npu::IRGraph*>(graph.get()));
        auto& graphArguments = _command_lists.at(i)->getBinding();

        size_t io_index = 0;
        for (const auto& desc : _graph->get_metadata().inputs) {
            // TODO: Not support now
            // if (desc.isMainInputWeights) {
            //     // These values were set while running the "WeightlessGraph::init" method
            //     continue;
            // }

            if (input_tensors.at(io_index).size() > 1) {
                _logger.debug("DynamicPipeline - set args for input index: %zu", io_index);
                const auto& tensor = input_tensors.at(io_index).at(i);

                if (tensor->get_element_type().bitwidth() < 8 || tensor->is_continuous() ||
                    tensor->get_strides().empty()) {
                    irGraph->set_argument_value(desc.indexUsedByDriver, tensor->data());
                    graphArguments.setArgumentValue(desc.indexUsedByDriver, tensor->data());
                } else {
                    irGraph->set_argument_value_with_strides(
                        desc.indexUsedByDriver,
                        tensor->data(),
                        get_strides(tensor->get_strides(), tensor->get_element_type().size()));
                    graphArguments.setArgumentValueWithStrides(
                        desc.indexUsedByDriver,
                        tensor->data(),
                        get_strides(tensor->get_strides(), tensor->get_element_type().size()));
                }
                ++io_index;
                continue;
            }

            _logger.debug(" update tensor property for input desc index: %d", desc.indexUsedByDriver);
            const auto& tensor = input_tensors.at(io_index).at(0);
            if (tensor->get_element_type().bitwidth() < 8 || tensor->is_continuous() || tensor->get_strides().empty()) {
                irGraph->set_argument_value(desc.indexUsedByDriver,
                                            static_cast<unsigned char*>(tensor->data()) +
                                                (i * tensor->get_byte_size()) / _number_of_command_lists);
                graphArguments.setArgumentValue(desc.indexUsedByDriver,
                                                static_cast<unsigned char*>(tensor->data()) +
                                                    (i * tensor->get_byte_size()) / _number_of_command_lists);
            } else {
                irGraph->set_argument_value_with_strides(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_strides()[0]),
                    get_strides(tensor->get_strides(), tensor->get_element_type().size()));
                graphArguments.setArgumentValueWithStrides(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_strides()[0]),
                    get_strides(tensor->get_strides(), tensor->get_element_type().size()));
            }

            ++io_index;
        }

        io_index = 0;
        for (const auto& desc : _graph->get_metadata().outputs) {
            _logger.debug("DynamicPipeline - update tensor property for output desc index: %d", desc.indexUsedByDriver);
            const auto& tensor = output_tensors.at(io_index);
            if (tensor->get_element_type().bitwidth() < 8 || tensor->is_continuous() || tensor->get_strides().empty()) {
                irGraph->set_argument_value(desc.indexUsedByDriver,
                                            static_cast<unsigned char*>(tensor->data()) +
                                                (i * tensor->get_byte_size()) / _number_of_command_lists);
                graphArguments.setArgumentValue(desc.indexUsedByDriver,
                                                static_cast<unsigned char*>(tensor->data()) +
                                                    (i * tensor->get_byte_size()) / _number_of_command_lists);
            } else {
                irGraph->set_argument_value_with_strides(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_strides()[0]),
                    tensor->get_strides());

                graphArguments.setArgumentValueWithStrides(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_strides()[0]),
                    get_strides(tensor->get_strides(), tensor->get_element_type().size()));
            }
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

        //_command_lists.at(i)->bind(dynamic_cast<intel_npu::IRGraph*>(graph.get()));

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
        auto& graphArguments = command_lists->getBinding();
        _logger.debug("Inputs info for IRGraph:");
        for (auto& memType : graphArguments._inputs) {
            _logger.debug("input: %s", memType.toString().c_str());
        }
        _logger.debug("Outputs info for IRGraph:");
        for (auto& memType : graphArguments._outputs) {
            _logger.debug("output: %s", memType.toString().c_str());
        }
        // Before call execute, shall clear memref containers in graphArguments to avoid dangling ptrs
        graphArguments._inputMemRefs.clear();
        graphArguments._outputMemRefs.clear();

        // Need change to use arguments in pipeline
        std::vector<IRGraph::MemRefType> inputPros = graphArguments._inputs;
        std::vector<IRGraph::MemRefType> outputPros = graphArguments._outputs;

        // TENTATIVE CODE TO ALLOCATE a memref handle
        for (size_t i = 0; i < inputPros.size(); ++i) {
            inputPros[i].UpdateMemRefHandleStatus();
        }

        for (size_t i = 0; i < outputPros.size(); ++i) {
            outputPros[i].UpdateMemRefHandleStatus();
        }

        // TODO: Skip predict output shape to avoid segment fault
        dynamic_cast<IRGraph*>(_graph.get())->predict_output_shape(inputPros, outputPros);

        bool shapeChanged = false;
        for (size_t i = 0; i < outputPros.size(); i++) {
            for (int64_t j = 0; j < outputPros[i].dimsCount; j++) {
                if (graphArguments._outputs[i].sizes[j] != outputPros[i].sizes[j]) {
                    _logger.info(
                        "Output tensor %d shape and predicted shape mimsmatch at dim %zu, changed from %zu to %zu",
                        i,
                        j,
                        graphArguments._outputs[i].sizes[j],
                        outputPros[i].sizes[j]);
                    shapeChanged = true;
                    graphArguments._outputs[i].sizes[j] = outputPros[i].sizes[j];
                }
            }
            if (shapeChanged) {
                graphArguments._outputs[i].updateStride();
            }
        }
        if (!shapeChanged) {
            _logger.debug("No output shape changed detected");
        } else {
            _logger.warning("Use predicted shape to replace the real tensor shape and update its strides");
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
}

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
}

void DynamicPipeline::update_graph_arguments(uint32_t index,
                                             const std::shared_ptr<ZeroTensor>& zeroTensor,
                                             std::shared_ptr<ov::ITensor> userTensor) {
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_UMCL, itt::domains::LevelZeroBackend, "DynamicPipeline", "updateCommandList");
    _logger.debug("DynamicPipeline - updateCommandList");
    // This is the tensor with right shape and strides
    // The required check is alredy done in inferRequest
    const std::shared_ptr<ov::ITensor>& tensor = userTensor ? userTensor : zeroTensor;

    const size_t number_of_command_lists = _command_lists.size();

    for (size_t i = 0; i < number_of_command_lists; i++) {
        if (tensor->get_element_type().bitwidth() < 8 || tensor->is_continuous() || tensor->get_strides().empty()) {
            _command_lists.at(i)->updateMutableCommandList(index,
                                                           static_cast<const unsigned char*>(zeroTensor->data()) +
                                                               (i * tensor->get_byte_size()) / number_of_command_lists,
                                                           tensor->get_strides(),
                                                           tensor->get_shape());
        } else {
            _command_lists.at(i)->updateMutableCommandList(
                index,
                static_cast<const unsigned char*>(zeroTensor->data()) + (i * tensor->get_strides()[0]),
                get_strides(tensor->get_strides(), tensor->get_element_type().size()),
                tensor->get_shape());
        }
    }
}

void DynamicPipeline::update_graph_arguments(uint32_t index,
                                             const std::shared_ptr<ZeroTensor>& zeroTensor,
                                             std::shared_ptr<ov::ITensor> userTensor,
                                             size_t batch_index) {
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_UMCL,
                      itt::domains::LevelZeroBackend,
                      "DynamicPipeline",
                      "updateCommandListIndex");
    _logger.debug("DynamicPipeline - updateCommandListIndex");
    // This is the tensor with right shape and strides
    // The required check is alredy done in inferRequest
    const std::shared_ptr<ov::ITensor>& tensor = userTensor ? userTensor : zeroTensor;

    const size_t number_of_command_lists = _command_lists.size();

    OPENVINO_ASSERT(batch_index < number_of_command_lists,
                    "Command list index is higher than the number of Command lists ",
                    batch_index);

    if (tensor->get_element_type().bitwidth() < 8 || tensor->is_continuous() || tensor->get_strides().empty()) {
        _command_lists.at(batch_index)
            ->updateMutableCommandList(index, zeroTensor->data(), tensor->get_strides(), tensor->get_shape());
    } else {
        _command_lists.at(batch_index)
            ->updateMutableCommandList(index,
                                       zeroTensor->data(),
                                       get_strides(tensor->get_strides(), tensor->get_element_type().size()),
                                       tensor->get_shape());
    }
}

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
    /// PROFILING_TYPE = MODEL or undefined = fallback to model profiling
    // if (_config.get<COMPILER_TYPE>() == ov::intel_npu::CompilerType::PLUGIN) {
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
