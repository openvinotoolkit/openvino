// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/igraph.hpp"

#include "intel_npu/config/compiler.hpp"
#include "intel_npu/config/runtime.hpp"

namespace {
constexpr std::size_t BATCH_AXIS = 0;
constexpr std::size_t DEFAULT_BATCH_SIZE = 1;
}  // namespace

namespace intel_npu {

IGraph::IGraph(ze_graph_handle_t handle,
               NetworkMetadata metadata,
               const Config& config,
               std::optional<std::vector<uint8_t>> blob)
    : _handle(handle),
      _metadata(std::move(metadata)),
      _logger("IGraph", config.get<LOG_LEVEL>()) {
    if (blob.has_value()) {
        _blob = std::move(*blob);
    }
}

const NetworkMetadata& IGraph::get_metadata() const {
    return _metadata;
}

ze_graph_handle_t IGraph::get_handle() const {
    return _handle;
}

void IGraph::update_network_name(std::string_view name) {
    _metadata.name = name;
}

const std::vector<ArgumentDescriptor>& IGraph::get_input_descriptors() const {
    return _input_descriptors;
}

const std::vector<ArgumentDescriptor>& IGraph::get_output_descriptors() const {
    return _output_descriptors;
}

const std::shared_ptr<CommandQueue>& IGraph::get_command_queue() const {
    return _command_queue;
}

void IGraph::set_workload_type(const ov::WorkloadType workloadType) const {
    if (_command_queue == nullptr) {
        return;
    }

    ze_command_queue_workload_type_t zeWorkloadType;
    switch (workloadType) {
    case ov::WorkloadType::DEFAULT:
        zeWorkloadType = ze_command_queue_workload_type_t::ZE_WORKLOAD_TYPE_DEFAULT;
        break;
    case ov::WorkloadType::EFFICIENT:
        zeWorkloadType = ze_command_queue_workload_type_t::ZE_WORKLOAD_TYPE_BACKGROUND;
        break;
    default:
        OPENVINO_THROW("Unknown value for WorkloadType!");
    }

    _command_queue->setWorkloadType(zeWorkloadType);
}

std::mutex& IGraph::get_mutex() {
    return _mutex;
}

void IGraph::set_last_submitted_event(const std::shared_ptr<Event>& event, size_t indexOfCommandList) {
    _last_submitted_event[indexOfCommandList] = event;
}

const std::shared_ptr<Event>& IGraph::get_last_submitted_event(size_t indexOfCommandList) const {
    return _last_submitted_event[indexOfCommandList];
}

uint32_t IGraph::get_unique_id() {
    return _unique_id++;
}

void IGraph::set_last_submitted_id(uint32_t id_index) {
    _last_submitted_id = id_index;
}

const uint32_t IGraph::get_last_submitted_id() const {
    return _last_submitted_id;
}

std::optional<size_t> IGraph::get_batch_size(const NetworkMetadata& metadata) {
    if (!metadata.outputs.at(0).shapeFromIRModel.has_value()) {
        _logger.debug("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    const ov::PartialShape& firstOutputShape = *metadata.outputs.at(0).shapeFromIRModel;
    if (firstOutputShape.is_dynamic()) {
        _logger.warning("Networks using dynamic shapes are not supported when batching is handled by the plugin");
        return std::nullopt;
    }
    if (firstOutputShape.rank().get_length() == 0) {
        _logger.warning("Networks using rank 0 shapes for inputs/outputs are not supported when batching is "
                        "handled by the plugin");
        return std::nullopt;
    }

    const size_t candidateBatchSize = firstOutputShape[BATCH_AXIS].get_length();
    if (candidateBatchSize == 0 || candidateBatchSize == DEFAULT_BATCH_SIZE) {
        _logger.debug("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    auto checkDescriptorsUseCandidateBatchSize = [candidateBatchSize](const std::vector<IODescriptor>& descriptors) {
        for (const IODescriptor& descriptor : descriptors) {
            OPENVINO_ASSERT(descriptor.shapeFromIRModel.has_value(),
                            "Missing value for the \"shapeFromIRModel\" attribute, I/O descriptor");

            const ov::PartialShape& shapeFromCompiler = descriptor.shapeFromCompiler;
            const ov::PartialShape& shapeFromIRModel = *descriptor.shapeFromIRModel;

            if (shapeFromCompiler.is_dynamic() || shapeFromCompiler.rank().get_length() == 0 ||
                *shapeFromCompiler.begin() != DEFAULT_BATCH_SIZE) {
                return false;
            }

            if (!descriptor.isStateInput && !descriptor.isStateOutput && !descriptor.isShapeTensor) {
                if (shapeFromIRModel.is_dynamic() || shapeFromIRModel.rank().get_length() == 0 ||
                    *shapeFromIRModel.begin() != candidateBatchSize) {
                    return false;
                }
            }
        }

        return true;
    };

    if (!checkDescriptorsUseCandidateBatchSize(metadata.inputs) ||
        !checkDescriptorsUseCandidateBatchSize(metadata.outputs)) {
        _logger.debug("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    _logger.debug("Batching is handled by the plugin");

    return candidateBatchSize;
}

const std::optional<std::size_t> IGraph::get_batch_size() const {
    return _batch_size;
}

}  // namespace intel_npu
