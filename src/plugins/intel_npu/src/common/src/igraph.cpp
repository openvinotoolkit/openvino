// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/igraph.hpp"

#include "intel_npu/config/options.hpp"

namespace {
constexpr std::size_t BATCH_AXIS = 0;
constexpr std::size_t DEFAULT_BATCH_SIZE = 1;
}  // namespace

namespace intel_npu {

IONodeMetadata::IONodeMetadata(const ArgumentDescriptor& d) : descriptor(d) {}

std::optional<size_t> IONodeMetadata::extract_batch_impl(const ov::Shape& shape,
                                                         std::optional<ov::PartialShape> partial_shape,
                                                         size_t index) const {
    if (partial_shape.has_value()) {
        if (partial_shape.value()[index].is_dynamic()) {
            return shape[index];
        }
        return std::nullopt;
    }
    return shape[index];
}

std::optional<size_t> IONodeMetadata::extract_batch(const ov::Shape& shape) const {
    return extract_batch(shape, std::nullopt);
}

std::optional<size_t> IONodeMetadata::extract_batch(const ov::Shape& shape,
                                                    std::optional<ov::PartialShape> partial_shape) const {
    if (descriptor.has_value()) {
        auto nLayout = descriptor.value().info.networkLayout;
        if (nLayout == ZE_GRAPH_ARGUMENT_LAYOUT_NCHW || nLayout == ZE_GRAPH_ARGUMENT_LAYOUT_NHWC ||
            nLayout == ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW || nLayout == ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC ||
            nLayout == ZE_GRAPH_ARGUMENT_LAYOUT_NC) {
            return extract_batch_impl(shape, partial_shape.has_value() ? partial_shape : std::nullopt, 0);
        } else if (nLayout == ZE_GRAPH_ARGUMENT_LAYOUT_CN) {
            return extract_batch_impl(shape, partial_shape.has_value() ? partial_shape : std::nullopt, 1);
        }
    }
    return std::nullopt;  // TODO get layout from shape
}

IGraph::IGraph(NetworkMetadata metadata, const Config& config, std::optional<ov::Tensor> blob)
    : _metadata(std::move(metadata)),
      _blob(std::move(blob)),
      _logger("IGraph", config.get<LOG_LEVEL>()) {}

const NetworkMetadata& IGraph::get_metadata() const {
    return _metadata;
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

uint32_t IGraph::get_command_queue_group_ordinal() const {
    return _command_queue_group_ordinal;
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

void IGraph::resize_last_submitted_event(size_t batch) {
    _last_submitted_event.resize(batch);
}

void IGraph::set_batch_size(std::size_t batch) {
    _batch_size = batch;
}

void IGraph::reset_last_batch_size() {
    _batch_size.reset();
}

uint32_t IGraph::get_unique_id() {
    return _unique_id++;
}

void IGraph::set_last_submitted_id(uint32_t id_index) {
    _last_submitted_id = id_index;
}

uint32_t IGraph::get_last_submitted_id() const {
    return _last_submitted_id;
}

std::optional<size_t> IGraph::determine_batch_size(const NetworkMetadata& metadata,
                                                   const std::vector<ov::SoPtr<ov::ITensor>>& tensors,
                                                   const IONodeMetadata& input_output_info) const {
    if (get_batch_size() > 1) {
        return get_batch_size();
    }

    if (tensors.empty()) {
        return std::nullopt;  // Return std::nullopt if no input tensors are set
    }

    const auto& first_tensor = tensors.at(0);
    if (!first_tensor) {
        return std::nullopt;  // Return std::nullopt if the first tensor is null
    }

    const auto& first_shape = first_tensor->get_shape();
    if (first_shape.empty()) {
        return std::nullopt;  // Return std::nullopt if the shape is empty
    }

    if (!metadata.inputs.at(0).shapeFromIRModel.has_value()) {
        _logger.debug("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    // A first dimensionin shape  may be appeared 'C' as well.
    // We need to get batch Idx and determine a true batch value here.
    // Let's use input_output_info as a helper.
    const ov::PartialShape& firstPartialShape = *metadata.inputs.at(0).shapeFromIRModel;
    auto candidateBatchSizeIfExist = input_output_info.extract_batch(first_shape, firstPartialShape);
    if (!candidateBatchSizeIfExist.has_value()) {
        return std::nullopt;  // Return std::nullopt if there is no batch dimension
    }
    const size_t candidateBatchSize = candidateBatchSizeIfExist.value();
    _logger.debug("Candidate batch size: %zu, shape: %s", candidateBatchSize, first_shape.to_string().c_str());
    auto checkBatchSizeConsistency = [candidateBatchSize,
                                      &input_output_info](const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
        for (const auto& tensor : tensors) {
            if (!tensor) {
                return false;  // Tensor is null
            }

            const auto& shape = tensor->get_shape();
            if (shape.empty()) {
                return false;  // Inconsistent batch size
            }

            auto batchIfExist = input_output_info.extract_batch(shape);
            if (!batchIfExist.has_value()) {
                return false;  // no batch size to check
            }

            if (batchIfExist.value() != candidateBatchSize) {
                return false;
            }
        }
        return true;
    };

    if (!checkBatchSizeConsistency(tensors)) {
        _logger.warning("Inconsistent batch sizes in input tensors");
        return std::nullopt;  // Return std::nullopt if batch sizes are inconsistent
    }

    _logger.debug("Determined batch size: %zu", candidateBatchSize);

    return candidateBatchSize;
}

std::optional<size_t> IGraph::get_batch_size(const NetworkMetadata& metadata,
                                             const std::vector<ov::SoPtr<ov::ITensor>>& tensors,
                                             const IONodeMetadata& input_output_info) {
    if (!metadata.outputs.at(0).shapeFromIRModel.has_value()) {
        _logger.debug("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    const ov::PartialShape& firstOutputShape = *metadata.outputs.at(0).shapeFromIRModel;
    if (firstOutputShape.is_dynamic()) {
        _logger.debug(
            "Networks using dynamic batch are handled by the plugin. Let's determine batch size over tensors: %zu",
            tensors.size());
        return !tensors.empty() ? determine_batch_size(metadata, tensors, input_output_info) : std::nullopt;
    }
    if (firstOutputShape.rank().get_length() == 0) {
        _logger.warning("Networks using rank 0 shapes for inputs/outputs are not supported when batching is "
                        "handled by the plugin");
        return std::nullopt;
    }

    const size_t candidateBatchSize = firstOutputShape[BATCH_AXIS].get_max_length();
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
