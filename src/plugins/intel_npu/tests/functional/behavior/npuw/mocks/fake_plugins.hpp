// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <memory>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/util/shared_object.hpp"

namespace ov {
namespace npuw {
namespace tests {
class FakeCompiledModel : public ov::ICompiledModel {
public:
    FakeCompiledModel(const std::shared_ptr<const ov::Model>& model, const std::shared_ptr<const ov::IPlugin>& plugin,
                      const ov::AnyMap& config);

    // Methods from a base class ov::ICompiledModel
    void export_model(std::ostream& model) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;
    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
    const std::shared_ptr<const ov::Model>& get_model() const;
    ov::SoPtr<ov::IRemoteContext> get_context() const;

private:
    ov::AnyMap m_config;
    std::shared_ptr<const ov::Model> m_model;
};

class FakeInferRequest : public ov::ISyncInferRequest {
public:
    FakeInferRequest(const std::shared_ptr<const FakeCompiledModel>& compiled_model);
    ~FakeInferRequest() = default;

    void infer() override;
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    void allocate_tensor_impl(ov::SoPtr<ov::ITensor>& tensor, const ov::element::Type& element_type,
                              const ov::Shape& shape);
    std::shared_ptr<const ov::Model> m_model;
};

class FakePluginBase : public ov::IPlugin {
public:
    FakePluginBase(const std::string& name, const std::unordered_set<std::string>& supported_ops,
                   bool dynamism_supported = false);

    virtual const ov::Version& get_const_version() = 0;
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::string& model_path,
                                                      const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override;
    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

protected:
    std::string m_default_device_id = "0";
    std::unordered_set<std::string> m_supported_ops;
    bool m_dynamism_supported = false;
    bool m_profiling = false;
    bool m_loaded_from_cache{false};
};

class FakeNpuPlugin : public FakePluginBase {
public:
    FakeNpuPlugin(const std::string& name);
    const ov::Version& get_const_version() override;
    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

private:
    int32_t num_streams{0};
    bool exclusive_async_requests = false;
};

class FakeCpuPlugin : public FakePluginBase {
public:
    FakeCpuPlugin(const std::string& name);
    const ov::Version& get_const_version() override;
    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;
};

}  // namespace tests
}  // namespace npuw
}  // namespace ov
