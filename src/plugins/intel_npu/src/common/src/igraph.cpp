// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/igraph.hpp"

#include "openvino/core/except.hpp"

namespace intel_npu {

std::pair<uint64_t, std::optional<std::vector<uint64_t>>> IGraph::export_blob(std::ostream&) const {
    OPENVINO_THROW("export_blob not implemented");
}

std::vector<ov::ProfilingInfo> IGraph::process_profiling_output(const std::vector<uint8_t>&) const {
    OPENVINO_THROW("process_profiling_output not implemented");
}

void IGraph::set_argument_value(uint32_t, const void*) const {
    OPENVINO_THROW("set_argument_value not implemented");
}

void IGraph::set_argument_value_with_strides(uint32_t, const void*, const std::vector<size_t>&) const {
    OPENVINO_THROW("set_argument_value_with_strides not implemented");
}

std::vector<size_t> IGraph::get_init_sizes() const {
    OPENVINO_THROW("get_init_sizes called on a weightful IGraph object");
}

std::optional<std::string> IGraph::get_compiler_compatibility_descriptor() const {
    OPENVINO_THROW("get_compiler_compatibility_descriptor not implemented");
}

void IGraph::initialize(const FilteredConfig& config) {
    std::lock_guard<std::mutex> lock(_initialize_mutex);

    if (_init_completed.load(std::memory_order_acquire)) {
        return;
    }

    initialize_impl(config);
}

void IGraph::initialize_impl(const FilteredConfig&) {
    OPENVINO_THROW("initialize_impl not implemented");
}

const NetworkMetadata& IGraph::get_metadata() const {
    OPENVINO_THROW("get_metadata not implemented");
}

ze_graph_handle_t IGraph::get_handle() const {
    OPENVINO_THROW("get_handle not implemented");
}

void IGraph::update_network_name(std::string_view) {
    OPENVINO_THROW("update_network_name not implemented");
}

CommandQueueDesc IGraph::get_command_queue_desc() const {
    OPENVINO_THROW("get_command_queue_desc not implemented");
}

void IGraph::set_workload_type(const ov::WorkloadType) {
    OPENVINO_THROW("set_workload_type not implemented");
}

void IGraph::set_model_priority(const ov::hint::Priority) {
    OPENVINO_THROW("set_model_priority not implemented");
}

void IGraph::set_last_submitted_event(const std::shared_ptr<Event>&, size_t) {
    OPENVINO_THROW("set_last_submitted_event not implemented");
}

const std::shared_ptr<Event>& IGraph::get_last_submitted_event(size_t) const {
    OPENVINO_THROW("get_last_submitted_event not implemented");
}

void IGraph::resize_last_submitted_event(size_t) {
    OPENVINO_THROW("resize_last_submitted_event not implemented");
}

void IGraph::set_batch_size(std::size_t) {
    OPENVINO_THROW("set_batch_size not implemented");
}

const std::optional<std::size_t> IGraph::get_batch_size() const {
    OPENVINO_THROW("get_batch_size not implemented");
}

uint32_t IGraph::get_unique_id() {
    OPENVINO_THROW("get_unique_id not implemented");
}

void IGraph::set_last_submitted_id(uint32_t) {
    OPENVINO_THROW("set_last_submitted_id not implemented");
}

uint32_t IGraph::get_last_submitted_id() const {
    OPENVINO_THROW("get_last_submitted_id not implemented");
}

void IGraph::evict_memory() {}

}  // namespace intel_npu
