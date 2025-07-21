// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef NPU_LLVM_BACKEND
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4146 4267 4244 4996)
#endif

#include "zero_dynamic_pipeline.hpp"

#include <llvm/Support/Error.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/MemRefUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <ze_api.h>
#include <ze_graph_ext.h>

#include <sstream>

#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_remote_tensor.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"

namespace intel_npu {

DynamicPipeline::DynamicPipeline(const Config& config,
                                 const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                 const std::shared_ptr<IGraph>& graph,
                                 const std::vector<std::vector<std::shared_ptr<ov::ITensor>>>& input_tensors,
                                 const std::vector<std::shared_ptr<ov::ITensor>>& output_tensors)
    : Pipeline(config, init_structs, graph, "DynamicPipeline"),
      _levelZeroInputTensors(input_tensors),
      _levelZeroOutputTensors(output_tensors) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Zero_infer_request::DynamicPipeline::DynamicPipeline");
    _logger.debug("DynamicPipeline - initialize started");

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

    // TODO: We have multiple command list to support batch, do we still need this for IR blob?
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

    // TODO: How many command list shall we create here? one for input to update tensor, one for output to update
    // tensor, then how to deal with batch
    uint64_t num_of_subgraphs = _graph->get_num_subgraphs();

    _command_lists.reserve(_number_of_command_lists);
    for (size_t i = 0; i < _number_of_command_lists; i++) {
        _command_lists.emplace_back(
            std::make_unique<PipelinedCommandLists>(num_of_subgraphs, _init_structs, _graph->get_command_queue_group_ordinal()));
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

            // TODO: we do not have graph handle, so can not call setGraphArgumentValue(), then IR call this? but how
            // can IR know new tensor come?
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

            // TODO: we do not have graph handle, so can not call setGraphArgumentValue(), then IR call this? but how
            // can IR know new tensor come?
            graph->set_argument_value(
                desc.idx,
                static_cast<unsigned char*>(data) +
                    (i * output_tensors.at(io_index)->get_byte_size()) / _number_of_command_lists);
            ++io_index;
        }

        if (_init_structs->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 1) &&
            _config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
            if (_graph->get_last_submitted_event(i)) {
                // TODO: this wait shall for the final execution, but if multiple graph inside IR, how to wait? And
                // still no graph handle, IR shall maintain this
                _command_lists.at(i)->appendWaitOnEvent(_graph->get_last_submitted_event(i));
            }
        }

        /// append timestamp command if feature was activated
        if (_npu_profiling != nullptr) {
            _command_lists.at(i)->appendBarrier();
            // TODO: we can only change the first command list, all subgraph inside IR shall also add this?
            _command_lists.at(i)->appendNpuTimestamp(reinterpret_cast<uint64_t*>(_npu_profiling->npu_ts_infer_start));
        }

        _command_lists.at(i)->bind(dynamic_cast<intel_npu::IRGraph*>(graph.get()));

        // FIXME(askrebko): commands will added on the fly
        //_command_lists.at(i)->appendGraphExecute(static_cast<ze_graph_handle_t>(graph->get_handle()),
        //                                         _profiling_query ? _profiling_query->getHandle() : nullptr);
        
        /// append timestamp command if feature was activated
        if (_npu_profiling != nullptr) {
            _command_lists.at(i)->appendBarrier();
            // TODO: we can only change the first command list, all subgraph inside IR shall also add this?
            _command_lists.at(i)->appendNpuTimestamp(reinterpret_cast<uint64_t*>(_npu_profiling->npu_ts_infer_end));
        }

        if (_init_structs->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 1) &&
            _config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
            if (_graph->get_last_submitted_event(i)) {
                _command_lists.at(i)->appendReset(_graph->get_last_submitted_event(i));
            }

            _command_lists.at(i)->appendSignalEvent(_events.at(i));
            _graph->set_last_submitted_event(_events.at(i), i);
        }

        // appendBarrier used in L0 as well
        if (!_sync_output_with_fences) {
            // TODO: if we have multiple commandlist inside IR, then the barrier seems useless here.
            _command_lists.at(i)->appendBarrier();
            _command_lists.at(i)->appendSignalEvent(_events.at(i));
        }
    }
}

void DynamicPipeline::PipelinedCommandLists::bind( IRGraph* graph ) {
    graph->getBinding(_binding);
}

void DynamicPipeline::push() {
    _logger.debug("DynamicPipeline - push() started");
    _logger.debug("inputs.size = %d, outputs.size=%d", _levelZeroInputTensors.size(), _levelZeroOutputTensors.size());

    void* contextHandlePtr = _init_structs->getContext();
    void* deviceHandlePtr = _init_structs->getDevice();
    void* ddiTableHandlePtr = _init_structs->getGraphDdiTable().getImpl();
   

    // TODO: if we support batch, need close more. If we have multiple graph inside IR, need know which to close
    //_command_lists.at(0)->close();

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

        auto& commandLists = _command_lists.at(i);

        ze_fence_handle_t fence = nullptr;
        ze_event_handle_t event = nullptr;
        if (_sync_output_with_fences) {
            fence = _fences.at(i)->handle();
        }
        else {
            // TODO
        }

        auto& command_lists = _command_lists.at(i);
        dynamic_cast<IRGraph*>(_graph.get())->execute(_init_structs, command_lists->getBinding(), command_lists->getHandles(), commandQueueHandle, fence, event, nullptr);
    }

    _logger.debug("DynamicPipeline - push() completed");
}

void DynamicPipeline::pull() {
    Pipeline::pull(_command_lists.size());
};

void DynamicPipeline::reset() const {
    Pipeline::reset(_command_lists);
};

void DynamicPipeline::update_graph_arguments(uint32_t arg_index, const void* arg_data, size_t byte_size) {
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_UMCL, itt::domains::LevelZeroBackend, "Pipeline", "updateCommandList");
    _logger.debug("Pipeline - updateCommandList");

    Pipeline::update_graph_arguments(arg_index, arg_data, byte_size, _command_lists);
};

void DynamicPipeline::update_graph_arguments_batching(uint32_t arg_index,
                                                      const void* arg_data,
                                                      size_t command_list_index) {
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_UMCL,
                      itt::domains::LevelZeroBackend,
                      "DynamicPipeline",
                      "updateCommandListIndex");
    _logger.debug("DynamicPipeline - updateCommandListIndex");

    update_graph_arguments_batching(arg_index, arg_data, command_list_index);
};

}  // namespace intel_npu


#ifdef _MSC_VER
#pragma warning(pop)
#endif
#endif