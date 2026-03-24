// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "compiled_model.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"

namespace ov::npuw::failsafe {

class InferRequest;

class CompiledModel final : public ov::npuw::ICompiledModel {
public:
    using Factory = std::function<std::shared_ptr<ov::ICompiledModel>(const std::string& device)>;

    static std::shared_ptr<CompiledModel> create(const std::shared_ptr<ov::Model>& model,
                                                 const std::shared_ptr<const ov::IPlugin>& plugin,
                                                 std::vector<std::string> devices,
                                                 Factory factory);

    CompiledModel(const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  std::vector<std::string> devices,
                  Factory factory);

    void export_model(std::ostream& model) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

    std::size_t active_device_index() const;
    std::string active_device_name() const;

private:
    friend class InferRequest;

    struct ActiveState {
        std::size_t device_index = 0;
        std::size_t generation = 0;
        std::shared_ptr<ov::ICompiledModel> compiled_model;
    };

    ActiveState ensure_compiled_locked() const;
    ActiveState failover_from_locked(std::size_t generation, const char* stage, std::exception_ptr failure) const;
    std::shared_ptr<ov::IAsyncInferRequest> create_request(std::size_t& generation) const;
    bool is_generation_current(std::size_t generation) const;

    std::vector<std::string> m_devices;
    Factory m_factory;
    mutable std::mutex m_mutex;
    mutable std::optional<ActiveState> m_active_state;
};

class InferRequest final : public ov::ISyncInferRequest {
public:
    explicit InferRequest(std::shared_ptr<const CompiledModel> compiled_model);

    void infer() override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;

    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& port) const override;
    void set_tensors(const ov::Output<const ov::Node>& port,
                     const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

    void check_tensors() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    friend class CompiledModel;

    struct PortKey {
        const ov::Node* node = nullptr;
        std::size_t index = 0;
        bool is_output = false;

        bool operator<(const PortKey& other) const {
            return std::tie(node, index, is_output) < std::tie(other.node, other.index, other.is_output);
        }
    };

    struct PortTensors {
        ov::Output<const ov::Node> port;
        std::vector<ov::SoPtr<ov::ITensor>> tensors;
    };

    void materialize() const;
    void ensure_request_locked() const;
    PortKey port_key_locked(const ov::Output<const ov::Node>& port) const;
    bool is_output_port_locked(const ov::Output<const ov::Node>& port) const;
    PortTensors& get_or_create_port_tensors_locked(const ov::Output<const ov::Node>& port) const;
    void bind_input_tensors_locked(const ov::Output<const ov::Node>& port,
                                   const std::vector<ov::SoPtr<ov::ITensor>>& tensors) const;
    void sync_output_tensors_locked() const;
    void rebind_user_tensors_locked() const;

    std::shared_ptr<const CompiledModel> m_failsafe_compiled_model;
    mutable std::mutex m_mutex;
    mutable std::shared_ptr<ov::IAsyncInferRequest> m_request;
    mutable std::size_t m_generation = std::numeric_limits<std::size_t>::max();
    mutable std::map<PortKey, PortTensors> m_public_tensors;
};

}  // namespace ov::npuw::failsafe
