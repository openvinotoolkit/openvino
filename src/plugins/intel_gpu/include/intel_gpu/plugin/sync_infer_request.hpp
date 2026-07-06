// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/plugin/output_memory_block.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "intel_gpu/plugin/graph.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <shared_mutex>

namespace ov::intel_gpu {

class CompiledModel;

enum class TensorOwner : uint8_t {
    USER = 0,
    PLUGIN = 1
};

struct TensorWrapper {
    TensorWrapper(const std::shared_ptr<ov::ITensor>& _ptr, TensorOwner _owner)
        : ptr(_ptr)
        , owner(_owner)
        , actual_size(_ptr ? _ptr->get_byte_size() : 0) {}

    TensorWrapper(const TensorWrapper& other) = default;
    TensorWrapper() = default;

    std::shared_ptr<ov::ITensor> ptr;
    TensorOwner owner;
    size_t actual_size;
};

// Couples the lazily-allocated user-input map with its mutex so the map is reachable only
// through a held lock: read() takes a shared lock (read-only view), write() an exclusive one.
class GuardedUserInputs {
public:
    using map_t = std::unordered_map<size_t, TensorWrapper>;

    GuardedUserInputs() = default;
    GuardedUserInputs(const GuardedUserInputs&) = delete;

    template <typename LockType, typename MapRef>
    class Accessor {
    public:
        Accessor(LockType lock, MapRef map) : m_lock(std::move(lock)), m_map(map) {}
        auto operator->() const { return &m_map; }
        MapRef operator*() const { return m_map; }
    private:
        LockType m_lock;
        MapRef m_map;
    };

    Accessor<std::shared_lock<std::shared_mutex>, const map_t&> read() const {
        return { std::shared_lock<std::shared_mutex>(m_mutex), m_map};
    }
    Accessor<std::unique_lock<std::shared_mutex>, map_t&> write() const {
        return {std::unique_lock<std::shared_mutex>(m_mutex), m_map};
    }

private:
    mutable std::shared_mutex m_mutex;
    mutable map_t m_map;
};

class SyncInferRequest : public ov::ISyncInferRequest {
public:
    using Ptr = std::shared_ptr<SyncInferRequest>;

    explicit SyncInferRequest(const std::shared_ptr<const CompiledModel>& compiled_model);
    SyncInferRequest(const SyncInferRequest &) = delete;
    ~SyncInferRequest() override;

    void infer() override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    void set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

    void set_task_executor(const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor);
    void setup_stream_graph();
    void enqueue_notify();
    void wait_notify();

    void enqueue();
    void wait();

    bool use_external_queue() const { return m_use_external_queue; }

private:
    void check_tensors() const override;

    // Materializes a deferred (lazy) input slot on first access. Const-safe; takes
    // an exclusive lock on m_user_inputs.
    void ensure_input_allocated(size_t input_idx) const;

    // Self-guarding: reachable only via read()/write(), which lock internally.
    GuardedUserInputs m_user_inputs;
    std::unordered_map<size_t, TensorWrapper> m_user_outputs;

    std::unordered_map<size_t, TensorWrapper> m_plugin_inputs;
    std::unordered_map<size_t, TensorWrapper> m_plugin_outputs;

    std::unordered_map<size_t, ov::Output<const ov::Node>> m_input_ports_map;
    std::unordered_map<size_t, ov::Output<const ov::Node>> m_output_ports_map;

    std::unordered_map<size_t, std::string> m_output_names_map;

    std::map<cldnn::primitive_id, cldnn::network_output> m_internal_outputs;
    VariablesMap m_variables;

    std::shared_ptr<Graph> m_graph;
    RemoteContextImpl::Ptr m_context = nullptr;
    std::shared_ptr<ov::threading::IStreamsExecutor> m_stream_executor = nullptr;
    std::shared_ptr<cldnn::ShapePredictor> m_shape_predictor = nullptr;
    bool m_enable_profiling = false;
    bool m_use_external_queue = false;

    void prepare_state(const std::string& name, const std::shared_ptr<VariableStateBase>& variable);
    std::vector<cldnn::event::ptr> prepare_input(const std::string& internal_name,
                                                 size_t input_idx,
                                                 const ov::Output<const ov::Node>& port,
                                                 const TensorWrapper& user_tensor_wrapper);
    std::vector<cldnn::event::ptr> prepare_output(size_t output_idx, const ov::Output<const ov::Node>& port, const TensorWrapper& user_tensor_wrapper);
    std::vector<cldnn::event::ptr> prepare_batched_input(size_t input_idx,
                                                         const ov::Output<const ov::Node>& port,
                                                         const std::vector<ov::SoPtr<ov::ITensor>>& user_tensors);

    TensorWrapper create_or_share_device_tensor(const TensorWrapper& user_tensor_wrapper,
                                                const std::string& name,
                                                const ov::PartialShape& pshape,
                                                ov::element::Type element_type,
                                                bool need_lockable_mem) const;
    std::shared_ptr<ov::ITensor> reinterpret_device_tensor(std::shared_ptr<RemoteTensorImpl> tensor, const ov::Shape new_shape) const;
    std::shared_ptr<ov::ITensor> create_host_tensor(const ov::PartialShape& port_shape, const ov::element::Type& port_element_type) const;
    std::shared_ptr<ov::ITensor> create_device_tensor(const ov::PartialShape& pshape, ov::element::Type element_type, bool need_lockable_memory = false) const;

    void allocate_inputs();
    void allocate_outputs();
    void allocate_states();
    void allocate_input(size_t input_idx, GuardedUserInputs::map_t& user_inputs);
    void allocate_output(const ov::Output<const ov::Node>& port, size_t output_idx);
    cldnn::event::ptr copy_output_data(cldnn::memory::ptr src, ov::ITensor& dst) const;

    void init_mappings();
    bool is_batched_input(const ov::Output<const ov::Node>& port) const;
    uint64_t total_output_bytes = 0;

    // Per-output-port OutputMemoryBlock for zero-copy dynamic output.
    // Keyed by output port index. Only populated when USM host memory is available.
    std::unordered_map<size_t, std::unique_ptr<OutputMemoryBlock>> m_output_memory_blocks;

    // Variable to hold the inference request string with compiled model name
    // to prevent this string being constructed for each inference call
    std::string m_itt_infer_request_str;
};

}  // namespace ov::intel_gpu
