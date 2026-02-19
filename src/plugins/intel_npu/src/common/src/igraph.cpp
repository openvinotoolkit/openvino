// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/igraph.hpp"

#include "openvino/core/except.hpp"

namespace intel_npu {

std::pair<uint64_t, std::optional<std::vector<uint64_t>>> IGraph::export_blob(std::ostream&) const {
    OPENVINO_THROW("export_blob not implemented");
}

std::vector<ov::ProfilingInfo> IGraph::process_profiling_output(const std::vector<uint8_t>&, const Config&) const {
    OPENVINO_THROW("process_profiling_output not implemented");
}

void IGraph::set_argument_value(uint32_t, const void*) const {
    OPENVINO_THROW("set_argument_value not implemented");
}

void IGraph::set_argument_value_with_strides(uint32_t, const void*, const std::vector<size_t>&) const {
    OPENVINO_THROW("set_argument_value_with_strides not implemented");
}

void IGraph::initialize(const Config&) {
    OPENVINO_THROW("initialize not implemented");
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

const std::shared_ptr<CommandQueue>& IGraph::get_command_queue() const {
    OPENVINO_THROW("get_command_queue not implemented");
}

uint32_t IGraph::get_command_queue_group_ordinal() const {
    OPENVINO_THROW("get_command_queue_group_ordinal not implemented");
}

void IGraph::set_workload_type(const ov::WorkloadType) const {
    OPENVINO_THROW("set_workload_type not implemented");
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

}  // namespace intel_npu
