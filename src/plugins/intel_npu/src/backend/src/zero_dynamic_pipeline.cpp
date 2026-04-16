// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_dynamic_pipeline.hpp"

#include <level_zero/ze_api.h>
#include <ze_graph_ext.h>

#include <sstream>

#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_cmd_queue_pool.hpp"
#include "intel_npu/utils/zero/zero_remote_tensor.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"

namespace {
std::vector<size_t> get_strides(const std::vector<size_t>& strides_in_bytes, size_t element_size) {
    std::vector<size_t> element_strides(strides_in_bytes.size());
    std::transform(strides_in_bytes.begin(),
                   strides_in_bytes.end(),
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
DynamicPipeline::DynamicPipeline(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                 const std::shared_ptr<IGraph>& graph,
                                 const Config& config,
                                 const std::vector<std::vector<std::shared_ptr<ZeroTensor>>>& input_tensors,
                                 const std::vector<std::shared_ptr<ZeroTensor>>& output_tensors,
                                 size_t batch_size)
    : IPipeline(init_structs, graph, batch_size, config, "DynamicPipeline") {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Zero_infer_request::DynamicPipeline::DynamicPipeline");

    OPENVINO_ASSERT(!_run_inferences_sequentially, "In-order execution doesn't work for dynamic pipeline");

    _logger.debug("DynamicPipeline - initialization started, batch size: %i", _batch_size);

    intel_npu::IDynamicGraph* dynamicGraph = dynamic_cast<intel_npu::IDynamicGraph*>(graph.get());
    OPENVINO_ASSERT(dynamicGraph != nullptr, "Failed to cast graph to IDynamicGraph");

    if (!_sync_output_with_fences) {
        _event_pool = std::make_shared<EventPool>(_init_structs->getDevice(),
                                                  _init_structs->getContext(),
                                                  _batch_size ? static_cast<uint32_t>(_batch_size) : 1);

        _events.reserve(_batch_size);
        for (size_t i = 0; i < _batch_size; i++) {
            _events.emplace_back(std::make_shared<Event>(_event_pool, static_cast<uint32_t>(i)));
        }
    }
    _logger.debug("DynamicPipeline - event pool and command queue setup completed");

    uint64_t num_of_subgraphs = dynamicGraph->get_num_subgraphs();

    _command_lists.reserve(_batch_size);
    for (size_t i = 0; i < _batch_size; i++) {
        _command_lists.emplace_back(std::make_unique<PipelinedCommandLists>(num_of_subgraphs, _init_structs));
    }

    if (_sync_output_with_fences) {
        _fences.reserve(_batch_size);
        for (size_t i = 0; i < _batch_size; i++) {
            _fences.emplace_back(std::make_unique<Fence>(_command_queue));
        }
    }

    for (size_t i = 0; i < _batch_size; i++) {
        _logger.debug("DynamicPipeline - set args for command list number: %zu", i);

        _command_lists.at(i)->bind(dynamicGraph);
        auto& graphArguments = _command_lists.at(i)->getBinding();

        size_t io_index = 0;
        for (const auto& desc : _graph->get_metadata().inputs) {
            if (desc.isMainInputWeights) {
                // These values were set while running the "WeightlessGraph::init" method
                continue;
            }

            if (input_tensors.at(io_index).size() > 1) {
                _logger.debug("DynamicPipeline - set args for input index: %zu", io_index);
                const auto& tensor = input_tensors.at(io_index).at(i);
                if (tensor->get_element_type().bitwidth() < 8 || tensor->is_continuous() ||
                    tensor->get_strides().empty()) {
                    dynamicGraph->set_argument_value(desc.indexUsedByDriver, tensor->data());
                } else {
                    dynamicGraph->set_argument_value_with_strides(
                        desc.indexUsedByDriver,
                        tensor->data(),
                        get_strides(tensor->get_strides(), tensor->get_element_type().size()));
                }
                size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
                graphArguments.setArgumentProperties(desc.indexUsedByDriver,
                                                     tensor->data(),
                                                     tensor->get_shape(),
                                                     get_strides(tensor->get_strides(), elementSize));
                ++io_index;
                continue;
            }

            _logger.debug("DynamicPipeline - update tensor property for input desc index: %d", desc.indexUsedByDriver);
            const auto& tensor = input_tensors.at(io_index).at(0);
            size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
            if (tensor->get_element_type().bitwidth() < 8 || tensor->is_continuous() || tensor->get_strides().empty()) {
                dynamicGraph->set_argument_value(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_byte_size()) / _batch_size);
                graphArguments.setArgumentProperties(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_byte_size()) / _batch_size,
                    tensor->get_shape(),
                    get_strides(tensor->get_strides(), elementSize));
            } else {
                dynamicGraph->set_argument_value_with_strides(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_strides()[0]),
                    get_strides(tensor->get_strides(), tensor->get_element_type().size()));
                graphArguments.setArgumentProperties(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_strides()[0]),
                    tensor->get_shape(),
                    get_strides(tensor->get_strides(), elementSize));
            }
            ++io_index;
        }

        io_index = 0;
        for (const auto& desc : _graph->get_metadata().outputs) {
            _logger.debug("DynamicPipeline - update tensor property for output desc index: %d", desc.indexUsedByDriver);
            const auto& tensor = output_tensors.at(io_index);
            size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
            if (tensor->get_element_type().bitwidth() < 8 || tensor->is_continuous() || tensor->get_strides().empty()) {
                dynamicGraph->set_argument_value(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_byte_size()) / _batch_size);
                graphArguments.setArgumentProperties(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_byte_size()) / _batch_size,
                    tensor->get_shape(),
                    get_strides(tensor->get_strides(), elementSize));
            } else {
                dynamicGraph->set_argument_value_with_strides(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_strides()[0]),
                    get_strides(tensor->get_strides(), elementSize));

                graphArguments.setArgumentProperties(
                    desc.indexUsedByDriver,
                    static_cast<unsigned char*>(tensor->data()) + (i * tensor->get_strides()[0]),
                    tensor->get_shape(),
                    get_strides(tensor->get_strides(), elementSize));
            }
            ++io_index;
        }
    }
    _logger.debug("DynamicPipeline - initialization completed");
}

void DynamicPipeline::push() {
    _logger.debug("push - started");

    auto* dynamicGraph = dynamic_cast<IDynamicGraph*>(_graph.get());
    OPENVINO_ASSERT(dynamicGraph != nullptr, "Failed to cast graph to IDynamicGraph");

    const auto command_queue_desc = _graph->get_command_queue_desc();
    const bool command_queue_version_changed = (command_queue_desc.key() != _command_queue->desc().key());
    if (command_queue_version_changed) {
        _command_queue = ZeroCmdQueuePool::getInstance().getCommandQueue(_init_structs, command_queue_desc);

        if (_sync_output_with_fences) {
            for (size_t i = 0; i < _fences.size(); i++) {
                _fences[i] = std::make_unique<Fence>(_command_queue);
            }
        }
    }

    auto commandQueueHandle = _command_queue->handle();
    for (size_t i = 0; i < _command_lists.size(); ++i) {
        OV_ITT_TASK_CHAIN(ZERO_PIPELINE_IP_PUSH, itt::domains::LevelZeroBackend, "Pipeline", "push");

        ze_fence_handle_t fence = nullptr;
        ze_event_handle_t event = nullptr;
        if (_sync_output_with_fences) {
            fence = _fences.at(i)->handle();
        }

        auto& command_lists = _command_lists.at(i);
        auto& graphArguments = command_lists->getBinding();
        if (_logger.level() >= ov::log::Level::DEBUG) {
            _logger.debug("push - inputs info for dynamic graph:");
            for (auto& memType : graphArguments._inputs) {
                _logger.debug("push - input: %s", memType.toString().c_str());
            }
            _logger.debug("push - outputs info for dynamic graph:");
            for (auto& memType : graphArguments._outputs) {
                _logger.debug("push - output: %s", memType.toString().c_str());
            }
        }

        // L0 wrapper handle closed command list
        command_lists->resetCommandList();

        dynamicGraph->execute(_init_structs,
                              graphArguments,
                              command_lists->getHandles(),
                              commandQueueHandle,
                              fence,
                              event,
                              nullptr);
    }

    _logger.debug("push - completed");
}

void DynamicPipeline::pull() {
    _logger.debug("pull - started");
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

    _logger.debug("pull - completed");
}

void DynamicPipeline::reset() const {
    _logger.debug("reset - started");
    for (size_t i = 0; i < _command_lists.size(); ++i) {
        if (_sync_output_with_fences) {
            _fences.at(i)->reset();
        } else {
            _events.at(i)->reset();
        }
    }

    _logger.debug("reset - completed");
}

void DynamicPipeline::update_graph_arguments(uint32_t index,
                                             const std::shared_ptr<ZeroTensor>& zeroTensor,
                                             const std::shared_ptr<ov::ITensor>& userTensor) {
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_UMCL, itt::domains::LevelZeroBackend, "DynamicPipeline", "updateCommandList");
    _logger.debug("update_graph_arguments - update command list");
    // This is the tensor with right shape and strides
    // The required check is alredy done in inferRequest
    const std::shared_ptr<ov::ITensor>& tensor = userTensor ? userTensor : zeroTensor;
    size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
    const size_t number_of_command_lists = _command_lists.size();

    for (size_t i = 0; i < number_of_command_lists; i++) {
        if (tensor->get_element_type().bitwidth() < 8 || tensor->is_continuous() || tensor->get_strides().empty()) {
            _command_lists.at(i)->updateMutableCommandList(index,
                                                           static_cast<const unsigned char*>(zeroTensor->data()) +
                                                               (i * tensor->get_byte_size()) / number_of_command_lists,
                                                           get_strides(tensor->get_strides(), elementSize),
                                                           tensor->get_shape());
        } else {
            _command_lists.at(i)->updateMutableCommandList(
                index,
                static_cast<const unsigned char*>(zeroTensor->data()) + (i * tensor->get_strides()[0]),
                get_strides(tensor->get_strides(), elementSize),
                tensor->get_shape());
        }
    }
}

void DynamicPipeline::update_graph_arguments(uint32_t index,
                                             const std::shared_ptr<ZeroTensor>& zeroTensor,
                                             size_t batch_index,
                                             const std::shared_ptr<ov::ITensor>& userTensor) {
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_UMCL,
                      itt::domains::LevelZeroBackend,
                      "DynamicPipeline",
                      "updateCommandListIndex");
    _logger.debug("update_graph_arguments - update command list by index");
    // This is the tensor with right shape and strides
    // The required check is alredy done in inferRequest
    const std::shared_ptr<ov::ITensor>& tensor = userTensor ? userTensor : zeroTensor;
    size_t elementSize = tensor->get_element_type().bitwidth() < 8 ? 1 : tensor->get_element_type().size();
    const size_t number_of_command_lists = _command_lists.size();

    OPENVINO_ASSERT(batch_index < number_of_command_lists,
                    "Command list index is higher than the number of Command lists ",
                    batch_index);

    if (tensor->get_element_type().bitwidth() < 8 || tensor->is_continuous() || tensor->get_strides().empty()) {
        _command_lists.at(batch_index)
            ->updateMutableCommandList(index,
                                       zeroTensor->data(),
                                       get_strides(tensor->get_strides(), elementSize),
                                       tensor->get_shape());
    } else {
        _command_lists.at(batch_index)
            ->updateMutableCommandList(index,
                                       zeroTensor->data(),
                                       get_strides(tensor->get_strides(), elementSize),
                                       tensor->get_shape());
    }
}

}  // namespace intel_npu
