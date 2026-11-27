// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_execution_list.hpp"

#include <level_zero/ze_api.h>

#include <chrono>

#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/zero/zero_cmd_queue_pool.hpp"
#include "zero_pipeline.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_us(const Clock::time_point& start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

unsigned long long to_ull(uint64_t value) {
    return static_cast<unsigned long long>(value);
}

}  // namespace

namespace intel_npu {

ZeroExecutionList::ZeroExecutionList(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                     const void* queue_owner)
    : _init_structs(init_structs),
      _queue_owner(queue_owner != nullptr ? queue_owner : this),
      _logger("ZeroExecutionList", Logger::global().level()) {
    OPENVINO_ASSERT(_init_structs != nullptr, "Execution list requires an initialized NPU device");
}

void ZeroExecutionList::append(const std::shared_ptr<ZeroInferRequest>& request) {
    OPENVINO_ASSERT(request != nullptr, "Cannot append an empty request to the execution list");
    OPENVINO_ASSERT(!_running, "Cannot append to an execution list that is running");
    OPENVINO_ASSERT(_command_list == nullptr, "Cannot append to an execution list after it has been started");

    // Follows the LOG_LEVEL of the compiled models, like the pipelines of the requests do
    const auto request_log_level = request->_config.get<LOG_LEVEL>();
    if (request_log_level > _logger.level()) {
        _logger.setLevel(request_log_level);
    }

    const auto& graph = request->_graph;
    OPENVINO_ASSERT(graph->get_kind() != GraphKind::Dynamic, "Execution list doesn't support dynamic models");
    OPENVINO_ASSERT(!graph->get_batch_size().has_value() || *graph->get_batch_size() == 1,
                    "Execution list doesn't support batching handled by the plugin");

    // The list submits on a queue of its own (or of its queue owner), so its members may come from different graphs
    // (each graph has its own queue unless NPU_SHARED_COMMON_QUEUE is set). The barriers in the list order the
    // executions, so the in-order option of a member's queue is not needed.
    const auto member_desc = graph->get_command_queue_desc();
    const auto list_options = member_desc.options() & ~static_cast<uint32_t>(ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC);
    if (_command_queue == nullptr) {
        const CommandQueueDesc list_desc(member_desc.priority(),
                                         member_desc.workload(),
                                         list_options,
                                         _queue_owner,
                                         false);
        _command_queue = ZeroCmdQueuePool::getInstance().getCommandQueue(_init_structs, list_desc);
        _logger.info("append - command queue %p (owner %p), options %#x, mutable command list ext version %u.%u",
                     static_cast<void*>(_command_queue->handle()),
                     _queue_owner,
                     list_options,
                     ZE_MAJOR_VERSION(_init_structs->getMutableCommandListExtVersion()),
                     ZE_MINOR_VERSION(_init_structs->getMutableCommandListExtVersion()));
    } else {
        const auto& list_desc = _command_queue->desc();
        OPENVINO_ASSERT(member_desc.priority() == list_desc.priority() && member_desc.workload() == list_desc.workload() &&
                            list_options == list_desc.options(),
                        "All requests of an execution list must use the same command queue settings (priority, "
                        "turbo, workload type)");
    }

    // From now on the request doesn't build a pipeline of its own. If it already ran on its own, the existing pipeline
    // stays unused.
    request->_executedByExecutionList = true;

    _members.push_back({request, 0, {}});
    _logger.debug("append - request %zu, graph %p, %zu inputs, %zu outputs",
                  _members.size() - 1,
                  graph->get_handle(),
                  request->_metadata.inputs.size(),
                  request->_metadata.outputs.size());
}

ZeroExecutionList::Arguments ZeroExecutionList::collect_arguments(const ZeroInferRequest& request) const {
    // Same rule as the pipelines of the requests: a strided view (e.g. a tile of a larger tensor) is passed with its
    // strides, which requires the port to be compiled with NPU_ENABLE_STRIDES_FOR
    const auto make_argument = [](uint32_t index, const std::shared_ptr<ZeroTensor>& tensor) {
        Argument argument{index, tensor->data(), {}};
        if (!(tensor->get_element_type().bitwidth() < 8 || tensor->is_continuous() || tensor->get_strides().empty())) {
            argument.strides = IPipeline::get_strides(*tensor, true);
        }
        return argument;
    };

    Arguments arguments;

    const auto& inputs = request._metadata.inputs;
    for (size_t index = 0; index < inputs.size(); ++index) {
        if (inputs.at(index).isMainInputWeights) {
            // Set while running the "WeightlessGraph::init" method
            continue;
        }

        const auto& tensors = request.get_level_zero_inputs(index);
        OPENVINO_ASSERT(tensors.size() == 1, "Execution list doesn't support batched inputs");
        const auto& tensor = tensors.at(SINGLE_TENSOR);
        OPENVINO_ASSERT(tensor != nullptr && tensor->data() != nullptr, "Input zero tensor is not allocated");

        arguments.push_back(make_argument(inputs.at(index).indexUsedByDriver, tensor));
    }

    const auto& outputs = request._metadata.outputs;
    for (size_t index = 0; index < outputs.size(); ++index) {
        const auto& tensor = request._levelZeroOutputTensors.at(index);
        OPENVINO_ASSERT(tensor != nullptr && tensor->data() != nullptr, "Output zero tensor is not allocated");

        arguments.push_back(make_argument(outputs.at(index).indexUsedByDriver, tensor));
    }

    return arguments;
}

void ZeroExecutionList::record() {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "ZeroExecutionList::record");
    const auto start = Clock::now();

    _command_list = std::make_unique<CommandList>(_init_structs);
    if (_fence == nullptr) {
        _fence = std::make_unique<Fence>(_command_queue);
    }
    ++_recordings;
    // A list that is reset before every submission records every time, only the first recording is reported at INFO
    const bool first = _recordings == 1;
    if (first) {
        _logger.info("record - started, %zu requests, command list %p, fence %p",
                     _members.size(),
                     static_cast<void*>(_command_list->handle()),
                     static_cast<void*>(_fence->handle()));
    } else {
        _logger.debug("record - started (recording %llu), %zu requests, command list %p",
                      to_ull(_recordings),
                      _members.size(),
                      static_cast<void*>(_command_list->handle()));
    }

    for (size_t index = 0; index < _members.size(); ++index) {
        auto& member = _members.at(index);
        const auto& graph = member.request->_graph;

        // The graph arguments are shared by all requests of the graph and captured when the execution is appended
        std::lock_guard<std::mutex> lock(graph->get_mutex());

        member.arguments = collect_arguments(*member.request);
        for (const auto& argument : member.arguments) {
            _logger.debug("record - request %zu, argument %u = %p%s",
                          index,
                          argument.index,
                          argument.data,
                          argument.strides.empty() ? "" : " (strided)");
            if (argument.strides.empty()) {
                graph->set_argument_value(argument.index, argument.data);
            } else {
                graph->set_argument_value_with_strides(argument.index, argument.data, argument.strides);
            }
        }

        if (index > 0) {
            // Orders the executions, a regular command list doesn't guarantee it without barriers
            _command_list->appendBarrier();
            _logger.debug("record - request %zu, barrier appended", index);
        }

        // The id fetched when the command list was created belongs to the first mutable command
        member.command_id = index == 0 ? _command_list->commandId() : _command_list->getNextCommandId();
        _command_list->appendGraphExecute(static_cast<ze_graph_handle_t>(graph->get_handle()), nullptr);
        _logger.debug("record - request %zu, graph execute appended, graph %p, command id %llu",
                      index,
                      graph->get_handle(),
                      to_ull(member.command_id));
    }

    if (first) {
        _logger.info("record - completed, %zu graph executes in %.1f us", _members.size(), elapsed_us(start));
    } else {
        _logger.debug("record - completed, %zu graph executes in %.1f us", _members.size(), elapsed_us(start));
    }
}

bool ZeroExecutionList::update_arguments() {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "ZeroExecutionList::update_arguments");

    std::vector<Arguments> current;
    current.reserve(_members.size());
    for (size_t member_index = 0; member_index < _members.size(); ++member_index) {
        const auto& member = _members.at(member_index);
        current.push_back(collect_arguments(*member.request));
        for (size_t index = 0; index < current.back().size(); ++index) {
            if (current.back().at(index).strides != member.arguments.at(index).strides) {
                // A plain tensor became a strided view (or the other way around): recorded again, so that no update
                // has to reset the strides of a recorded argument
                _logger.debug("update_arguments - request %zu, argument %u changed its strides, recording again",
                              member_index,
                              current.back().at(index).index);
                return false;
            }
        }
    }

    size_t updated = 0;
    for (size_t member_index = 0; member_index < _members.size(); ++member_index) {
        auto& member = _members.at(member_index);
        auto& arguments = current.at(member_index);
        for (size_t index = 0; index < arguments.size(); ++index) {
            const auto& argument = arguments.at(index);
            const auto& recorded = member.arguments.at(index);
            if (argument == recorded) {
                continue;
            }

            OPENVINO_ASSERT(_init_structs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0),
                            "Changing the tensors of an execution list is not supported with this driver");
            _logger.debug("update_arguments - request %zu, command id %llu, argument %u: %p -> %p%s",
                          member_index,
                          to_ull(member.command_id),
                          argument.index,
                          recorded.data,
                          argument.data,
                          argument.strides.empty() ? "" : " (strided)");
            if (argument.strides.empty()) {
                _command_list->updateMutableCommandList(argument.index, argument.data, member.command_id);
            } else {
                _command_list->updateMutableCommandListWithStrides(argument.index,
                                                                   argument.data,
                                                                   argument.strides,
                                                                   member.command_id);
            }
            ++updated;
        }
        member.arguments = std::move(arguments);
    }

    if (updated > 0) {
        // INFO (not DEBUG): a submission that patches the recorded commands is the prime suspect for device losses
        _logger.info("update_arguments - before submission %llu: %zu arguments updated in the recorded command list",
                     to_ull(_submissions + 1),
                     updated);
    }
    return true;
}

void ZeroExecutionList::prepare() {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "ZeroExecutionList::prepare");
    OPENVINO_ASSERT(!_members.empty(), "Cannot prepare an empty execution list");
    OPENVINO_ASSERT(!_running, "Cannot prepare an execution list that is running");

    // Allocates the tensors and copies inputs that are not in Level Zero memory
    for (auto& member : _members) {
        member.request->prepare_inputs();
        member.request->prepare_outputs();
    }

    if (_command_list == nullptr || !update_arguments()) {
        record();
    }

    _command_list->close();
}

void ZeroExecutionList::start_async() {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "ZeroExecutionList::start_async");
    OPENVINO_ASSERT(!_members.empty(), "Cannot start an empty execution list");
    OPENVINO_ASSERT(!_running, "The execution list is already running");

    const auto start = Clock::now();
    prepare();
    const auto prepared_us = elapsed_us(start);

    const auto submit_start = Clock::now();
    _command_queue->executeCommandList(*_command_list, *_fence);
    const auto submit_us = elapsed_us(submit_start);
    _running = true;
    _submit_time = Clock::now();
    ++_submissions;

    if (_submissions == 1) {
        _logger.info("start_async - first submission: %zu graph executes submitted as one command list",
                     _members.size());
    }
    _logger.debug("start_async - submission %llu: command list %p with %zu graph executes submitted to queue %p, "
                  "prepare %.1f us, zeCommandQueueExecuteCommandLists %.1f us, total %.1f us",
                  to_ull(_submissions),
                  static_cast<void*>(_command_list->handle()),
                  _members.size(),
                  static_cast<void*>(_command_queue->handle()),
                  prepared_us,
                  submit_us,
                  elapsed_us(start));
}

void ZeroExecutionList::wait() {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "ZeroExecutionList::wait");
    OPENVINO_ASSERT(_running, "The execution list was not started");

    const auto start = Clock::now();
    _fence->hostSynchronize();
    const auto sync_us = elapsed_us(start);
    const auto since_submit_us = elapsed_us(_submit_time);
    _running = false;

    for (auto& member : _members) {
        member.request->update_user_outputs();
    }

    _fence->reset();

    if (_submissions == 1) {
        _logger.info("wait - first submission completed %.1f us after it was submitted", since_submit_us);
    }
    _logger.debug("wait - submission %llu: fence %p signaled, host wait %.1f us, %.1f us after submit, "
                  "outputs %.1f us",
                  to_ull(_submissions),
                  static_cast<void*>(_fence->handle()),
                  sync_us,
                  since_submit_us,
                  elapsed_us(start) - sync_us);
}

void ZeroExecutionList::reset() {
    OPENVINO_ASSERT(!_running, "Cannot reset an execution list that is running");

    // The requests stay marked as executed by an execution list: they never built pipelines of their own
    _members.clear();
    _command_list.reset();
    _logger.debug("reset - the list is empty, the next submission records a new command list");
}

}  // namespace intel_npu
