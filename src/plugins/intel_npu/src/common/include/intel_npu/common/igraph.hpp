// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/network_metadata.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {

class IGraph : public std::enable_shared_from_this<IGraph> {
public:
    IGraph() = default;

    /**
     * @brief Writes the compiled model along with some metadata to the provided stream. The content of the stream can
     * later be used for importing the model.
     *
     * @param stream Where the content is placed
     * @return A pair made of the size of the main binary object and an optional variable. The optional variable
     * constitues the size of each init binary object if weights separation is enabled.
     */
    virtual std::pair<uint64_t, std::optional<std::vector<uint64_t>>> export_blob(std::ostream& stream) const;

    virtual std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData) const;

    virtual void set_argument_value(uint32_t id, const void* data) const;
    virtual void set_argument_value_with_strides(uint32_t id,
                                                 const void* data,
                                                 const std::vector<size_t>& strides) const;

    void initialize(const FilteredConfig& config);

    virtual ~IGraph() = default;

    virtual const NetworkMetadata& get_metadata() const;
    virtual ze_graph_handle_t get_handle() const;

    virtual void update_network_name(std::string_view name);

    virtual CommandQueueDesc get_command_queue_desc() const;
    virtual void set_workload_type(const ov::WorkloadType workloadType);
    virtual void set_model_priority(const ov::hint::Priority modelPriority);

    std::mutex& get_mutex() {
        return _initialize_mutex;
    }

    bool init_completed() const {
        return _init_completed.load(std::memory_order_acquire);
    }

    virtual void set_last_submitted_event(const std::shared_ptr<Event>& event, size_t indexOfCommandList);
    virtual const std::shared_ptr<Event>& get_last_submitted_event(size_t indexOfCommandList) const;
    virtual void resize_last_submitted_event(size_t batch);
    virtual void set_batch_size(std::size_t batch);

    virtual const std::optional<std::size_t> get_batch_size() const;

    virtual uint32_t get_unique_id();
    virtual void set_last_submitted_id(uint32_t id_index);
    virtual uint32_t get_last_submitted_id() const;

    virtual void evict_memory();

    virtual std::optional<bool> is_profiling_blob() const = 0;

protected:
    virtual void initialize_impl(const FilteredConfig& config);

    // Used to protect graph initialization (including zero pipeline creation) in the graph. Initialization should
    // happen only once per graph, typically when the graph is first used (e.g. when the first inference starts)
    std::mutex _initialize_mutex;
    std::atomic<bool> _init_completed{false};
};

}  // namespace intel_npu
