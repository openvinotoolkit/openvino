// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "intel_npu/common/network_metadata.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {

/// Backend-owned submission-ordering state. Deliberately only forward declared: graphs carry it
/// without needing its definition, so this header stays free of the submission machinery.
class SubmissionOrder;

enum class BlobType : uint8_t { ELF, LLVM, BYTECODE };

enum class GraphKind : uint8_t { Weightful, Weightless, Dynamic };

constexpr const char* to_string(GraphKind kind) noexcept {
    switch (kind) {
    case GraphKind::Weightful:
        return "weightful";
    case GraphKind::Weightless:
        return "weightless";
    case GraphKind::Dynamic:
        return "dynamic";
    }
    return "unknown";
}

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

    void initialize(const Config& config);

    virtual ~IGraph() = default;

    virtual const NetworkMetadata& get_metadata() const;
    // Returns the underlying native handle. Concrete graphs return different handle types:
    //   Graph        -> ze_graph_handle_t
    //   DynamicGraph -> npu_vm_runtime_handle_t
    // Callers must static_cast the result to the type matching the concrete graph implementation.
    virtual void* get_handle() const;

    // Returns the concrete kind of this graph. Derived classes override to identify themselves.
    virtual GraphKind get_kind() const;

    virtual BlobType get_blob_type() const;

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

    virtual void set_batch_size(std::size_t batch);

    virtual const std::optional<std::size_t> get_batch_size() const;

    /**
     * @brief Installs `candidate` as this graph's submission-ordering state if none is installed
     * yet, and returns whichever instance is now in effect.
     *
     * The graph carries this state because it is precisely the scope across which inferences have
     * to be ordered: every pipeline built on one graph must share one instance. The state itself is
     * created and used by the backend and is opaque here, which keeps the submission machinery out
     * of this interface.
     *
     * Idempotent, so every pipeline can call it unconditionally without coordinating; the losers of
     * the race simply get the instance that was installed first and drop their candidate.
     */
    std::shared_ptr<SubmissionOrder> install_submission_order(std::shared_ptr<SubmissionOrder> candidate);

    virtual void evict_memory();

    virtual std::optional<bool> is_profiling_blob() const = 0;

    /**
     * @brief Returns the compatibility descriptor of this graph, if any.
     * @details The descriptor is determined when the graph is created (imported from blob metadata,
     *          returned by the VCL/plugin compiler, or fetched from the driver on the
     *          compiler-in-driver path when L0 API version >= 1.16) and is immutable thereafter.
     *          The descriptor format is defined by the compiler and is opaque to the plugin.
     * @return A view of the descriptor string if available, or std::nullopt if:
     *         - The graph was compiled without generating a descriptor
     *         - The driver does not support zeDeviceGetRuntimeRequirements (L0 < 1.16)
     *         - This is a WeightlessGraph (not supported)
     */
    virtual std::optional<std::string_view> get_compatibility_descriptor() const;

protected:
    virtual void initialize_impl(const Config& config);

    // Used to protect graph initialization (including zero pipeline creation) in the graph. Initialization should
    // happen only once per graph, typically when the graph is first used (e.g. when the first inference starts)
    std::mutex _initialize_mutex;
    std::atomic<bool> _init_completed{false};

    // Guards the submission-order slot only. Kept separate from _initialize_mutex so installing the
    // state cannot interact with graph initialization.
    mutable std::mutex _submission_order_mutex;
    std::shared_ptr<SubmissionOrder> _submission_order;
};

}  // namespace intel_npu
