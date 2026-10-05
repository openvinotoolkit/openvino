// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Unit tests for ov::npuw::FuncMemMgr::assign_memory() covering the fix from
// commit 28c64c461d85a9ffe7dbf2da76b1ee2fa4ddf787 (EISW-236774):
// a funcall output that is BOTH a global Result AND internally consumed by a
// downstream subgraph must be pre-allocated (not skipped), otherwise
// connect_subrequests() throws map::at. A pure global output (no internal
// consumers) must still be skipped and allocated on-demand.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/make_tensor.hpp"

#define private public
#include "compiled_model.hpp"
#undef private

#include "just_sync_infer_request.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"

namespace {

class TestPlugin final : public ov::IPlugin {
public:
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::compile_model call");
    }
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&,
                                                      const ov::SoPtr<ov::IRemoteContext>&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::compile_model(context) call");
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::import_model(stream) call");
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::import_model(stream, context) call");
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::import_model(blob) call");
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::import_model(blob, context) call");
    }
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::query_model call");
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Test plugin does not expose properties");
    }
    bool is_property_supported(const std::string&, const ov::AnyMap&) const override {
        return false;
    }
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::create_context call");
    }
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::get_default_context call");
    }
};

class FakeSubCompiledModel final : public ov::ICompiledModel {
public:
    FakeSubCompiledModel(const std::shared_ptr<ov::Model>& model, const std::shared_ptr<const ov::IPlugin>& plugin)
        : ov::ICompiledModel(model, plugin, nullptr, nullptr),
          m_model(model) {}

    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_model;
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string& name) const override {
        // submodel_device() queries execution_devices and expects a vector<string>.
        if (name == ov::execution_devices.name()) {
            return std::vector<std::string>{};
        }
        return {};
    }
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        return {};
    }
    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override {
        return {};
    }

private:
    std::shared_ptr<ov::Model> m_model;
};

std::shared_ptr<TestPlugin> make_test_plugin() {
    auto plugin = std::make_shared<TestPlugin>();
    auto core = std::make_shared<testing::NiceMock<ov::MockICore>>();

    ON_CALL(*core, get_supported_property(testing::_, testing::_, testing::_))
        .WillByDefault([](const std::string&, const ov::AnyMap& properties, const bool) {
            return properties;
        });
    ON_CALL(*core, get_property(testing::_, testing::_, testing::_))
        .WillByDefault([](const std::string&, const std::string& name, const ov::AnyMap&) -> ov::Any {
            if (name == ov::available_devices.name()) {
                return std::vector<std::string>{};
            }
            if (name == ov::intel_npu::compiler_version.name()) {
                return int64_t{0};
            }
            if (name == ov::device::architecture.name()) {
                return std::string{};
            }
            if (name == ov::supported_properties.name() || name == ov::internal::supported_properties.name()) {
                return std::vector<ov::PropertyName>{};
            }
            return {};
        });
    ON_CALL(*core, get_property(testing::_, testing::_))
        .WillByDefault([](const std::string&, const std::string& name) -> ov::Any {
            if (name == ov::available_devices.name()) {
                return std::vector<std::string>{};
            }
            if (name == ov::supported_properties.name()) {
                return std::vector<ov::PropertyName>{};
            }
            if (name == ov::intel_npu::compiler_version.name()) {
                return static_cast<int64_t>(0);
            }
            if (name == ov::device::architecture.name()) {
                return std::string{};
            }
            return {};
        });

    plugin->set_core(core);
    return plugin;
}

std::shared_ptr<ov::Model> make_simple_model(const std::string& name) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto res = std::make_shared<ov::op::v0::Result>(param);
    return std::make_shared<ov::Model>(ov::OutputVector{res->output(0)}, ov::ParameterVector{param}, name);
}

// Builds a minimal CompiledModel with an empty routing state, ready to be
// populated with submodels by the individual tests.
std::shared_ptr<ov::npuw::CompiledModel> make_base_compiled_model(const std::shared_ptr<TestPlugin>& plugin) {
    auto model = make_simple_model("top_model");
    auto compiled = std::make_shared<ov::npuw::CompiledModel>(model, plugin, true);

    compiled->m_inputs_to_submodels_inputs.clear();
    compiled->m_outputs_to_submodels_outputs.clear();
    compiled->m_param_subscribers.clear();
    compiled->m_submodels_input_to_prev_output.clear();
    compiled->m_dev_list.clear();
    compiled->m_non_npuw_props.clear();
    compiled->m_compiled_submodels.clear();
    return compiled;
}

// Appends a function prototype submodel with a single f32[1] output. Returns its index.
std::size_t add_prototype_submodel(const std::shared_ptr<ov::npuw::CompiledModel>& compiled,
                                   const std::shared_ptr<TestPlugin>& plugin) {
    ov::npuw::CompiledModel::CompiledModelDesc desc;
    desc.compiled_model =
        ov::SoPtr<ov::ICompiledModel>{std::make_shared<FakeSubCompiledModel>(make_simple_model("proto"), plugin)};
    compiled->m_compiled_submodels.push_back(std::move(desc));
    return compiled->m_compiled_submodels.size() - 1;
}

// Appends a funcall submodel that is replaced by the given prototype index. Returns its index.
std::size_t add_funcall_submodel(const std::shared_ptr<ov::npuw::CompiledModel>& compiled, std::size_t proto_idx) {
    ov::npuw::CompiledModel::CompiledModelDesc desc;
    desc.replaced_by = proto_idx;
    compiled->m_compiled_submodels.push_back(std::move(desc));
    return compiled->m_compiled_submodels.size() - 1;
}

struct AllocRecorder {
    int calls = 0;
    ov::npuw::FuncMemMgr::AllocFcn fcn() {
        return [this](const ov::element::Type& type, const ov::Shape& shape, const std::string&) -> ov::npuw::TensorPtr {
            ++calls;
            return ov::get_tensor_impl(ov::Tensor(type, shape));
        };
    }
};

}  // namespace

// A funcall output that is both a global Result and internally consumed by a
// downstream subgraph must be pre-allocated (the fixed behavior).
TEST(FuncMemMgrDualRoleOutputTest, GlobalOutputThatIsInternallyConsumedIsAssigned) {
    auto plugin = make_test_plugin();
    auto compiled = make_base_compiled_model(plugin);

    const auto proto_idx = add_prototype_submodel(compiled, plugin);
    const auto funcall_idx = add_funcall_submodel(compiled, proto_idx);
    compiled->m_compiled_submodels.emplace_back();  // tail consumer subgraph

    const ov::npuw::LinkFrom funcall_out{funcall_idx, 0u};

    // The funcall output is a global Result...
    compiled->m_outputs_to_submodels_outputs = {funcall_out};
    // ...and is also internally consumed by the tail consumer subgraph.
    const auto consumer_idx = compiled->m_compiled_submodels.size() - 1;
    compiled->m_submodels_input_to_prev_output[{consumer_idx, 0u}] = funcall_out;

    ov::npuw::FuncMemMgr mgr(compiled);
    AllocRecorder recorder;
    mgr.set_alloc(recorder.fcn());

    mgr.assign_memory();

    EXPECT_EQ(recorder.calls, 1) << "Dual-role funcall output must be pre-allocated, not skipped";
    EXPECT_NE(mgr.get_tensor(funcall_out)._ptr, nullptr) << "A tensor must be available for the dual-role output";
}

// A pure global output (no internal consumers) must still be skipped and left
// for on-demand allocation.
TEST(FuncMemMgrDualRoleOutputTest, PureGlobalOutputIsSkipped) {
    auto plugin = make_test_plugin();
    auto compiled = make_base_compiled_model(plugin);

    const auto proto_idx = add_prototype_submodel(compiled, plugin);
    const auto funcall_idx = add_funcall_submodel(compiled, proto_idx);

    const ov::npuw::LinkFrom funcall_out{funcall_idx, 0u};

    // The funcall output is a global Result and has no internal consumers.
    compiled->m_outputs_to_submodels_outputs = {funcall_out};

    ov::npuw::FuncMemMgr mgr(compiled);
    AllocRecorder recorder;
    mgr.set_alloc(recorder.fcn());

    mgr.assign_memory();

    EXPECT_EQ(recorder.calls, 0) << "Pure global output must not be pre-allocated";
    EXPECT_EQ(mgr.get_tensor(funcall_out)._ptr, nullptr) << "No tensor should be pre-assigned for a pure global output";
}
