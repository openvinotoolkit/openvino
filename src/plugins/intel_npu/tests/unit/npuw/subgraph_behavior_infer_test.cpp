// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <any>
#include <atomic>
#include <cstring>
#include <functional>
#include <map>
#include <numeric>
#include <mutex>
#include <vector>

#define private public
#include "compiled_model.hpp"
#undef private
#include "just_sync_infer_request.hpp"
#include "llm_test_helpers.hpp"
#include "model_builder.hpp"
#include "unfold_sync_infer_request.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"

namespace {

using ov::test::npuw::build_llm_test_model;

constexpr std::size_t kSeqLen = 4u;
constexpr std::size_t kPastKvLen = 4u;
constexpr std::size_t kKVCacheSize = kSeqLen + kPastKvLen;
struct BehaviorHits {
    std::mutex mutex;
    std::vector<std::pair<std::size_t, std::size_t>> values;
};

class TestPlugin final : public ov::IPlugin {
public:
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::compile_model call in subgraph behavior test");
    }
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&,
                                                      const ov::SoPtr<ov::IRemoteContext>&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::compile_model(context) call in subgraph behavior test");
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::import_model(stream) call in subgraph behavior test");
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::import_model(stream, context) call in subgraph behavior test");
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::import_model(blob) call in subgraph behavior test");
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::import_model(blob, context) call in subgraph behavior test");
    }
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::query_model call in subgraph behavior test");
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Test plugin does not expose properties");
    }
    bool is_property_supported(const std::string&, const ov::AnyMap&) const override {
        return false;
    }
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::create_context call in subgraph behavior test");
    }
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap&) const override {
        OPENVINO_THROW("Unexpected TestPlugin::get_default_context call in subgraph behavior test");
    }
};

std::shared_ptr<ov::Model> build_static_llm_model() {
    auto model = build_llm_test_model();
    ov::pass::StatefulToStateless().run_on_model(model);
    model = model->clone();

    std::map<std::string, ov::PartialShape> new_shapes;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        const auto& pshape = input.get_partial_shape();

        if (name.find("input_ids") != std::string::npos || name.find("token_type_ids") != std::string::npos) {
            new_shapes[name] = ov::PartialShape{1, kSeqLen};
        } else if (name.find("attention_mask") != std::string::npos) {
            new_shapes[name] = ov::PartialShape{1, kKVCacheSize};
        } else if (name.find("position_ids") != std::string::npos) {
            new_shapes[name] = ov::PartialShape{1, kSeqLen};
        } else {
            auto static_shape = pshape;
            static_shape[0] = 1;
            static_shape[2] = kPastKvLen;
            new_shapes[name] = static_shape;
        }
    }

    model->reshape(new_shapes);
    model->validate_nodes_and_infer_types();
    return model;
}

std::size_t count_sdpa_nodes(const std::shared_ptr<ov::Model>& model) {
    const auto& ordered_ops = model->get_ordered_ops();
    return std::count_if(ordered_ops.begin(), ordered_ops.end(), [](const std::shared_ptr<ov::Node>& op) {
        return ov::is_type<ov::op::v13::ScaledDotProductAttention>(op);
    });
}

std::size_t count_runtime_behaviors(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model) {
    return std::count_if(compiled_model->m_compiled_submodels.begin(),
                         compiled_model->m_compiled_submodels.end(),
                         [](const auto& desc) {
                             return desc.pipeline.runtime_behavior.has_value();
                         });
}

class FakeSubCompiledModel;

class FakeSubInferRequest final : public ov::ISyncInferRequest {
public:
    explicit FakeSubInferRequest(std::shared_ptr<const FakeSubCompiledModel> compiled_model);

    void infer() override;
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override {
        return ov::ISyncInferRequest::get_tensor(port);
    }
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override {
        ov::ISyncInferRequest::set_tensor(port, tensor);
    }
    void check_tensors() const override {}
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }
};

class FakeSubAsyncInferRequest final : public ov::IAsyncInferRequest {
public:
    explicit FakeSubAsyncInferRequest(const std::shared_ptr<ov::ISyncInferRequest>& request)
        : ov::IAsyncInferRequest(nullptr, nullptr, nullptr),
          m_request(request) {}

    void start_async() override {
        try {
            m_request->infer();
            if (m_callback) {
                m_callback(nullptr);
            }
        } catch (...) {
            if (m_callback) {
                m_callback(std::current_exception());
                return;
            }
            throw;
        }
    }

    void wait() override {}

    bool wait_for(const std::chrono::milliseconds&) override {
        return true;
    }

    void cancel() override {}

    void set_callback(std::function<void(std::exception_ptr)> callback) override {
        m_callback = std::move(callback);
    }

    void infer() override {
        m_request->infer();
    }

    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return m_request->get_profiling_info();
    }

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override {
        return m_request->get_tensor(port);
    }

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override {
        m_request->set_tensor(port, tensor);
    }

    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& port) const override {
        return m_request->get_tensors(port);
    }

    void set_tensors(const ov::Output<const ov::Node>& port,
                     const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override {
        m_request->set_tensors(port, tensors);
    }

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return m_request->query_state();
    }

    const std::shared_ptr<const ov::ICompiledModel>& get_compiled_model() const override {
        return m_request->get_compiled_model();
    }

    const std::vector<ov::Output<const ov::Node>>& get_inputs() const override {
        return m_request->get_inputs();
    }

    const std::vector<ov::Output<const ov::Node>>& get_outputs() const override {
        return m_request->get_outputs();
    }

private:
    std::shared_ptr<ov::ISyncInferRequest> m_request;
    std::function<void(std::exception_ptr)> m_callback;
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
        if (name == ov::execution_devices.name()) {
            return std::vector<std::string>{"CPU"};
        }
        OPENVINO_THROW("Unsupported property: ", name);
    }
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        auto self = std::static_pointer_cast<const FakeSubCompiledModel>(shared_from_this());
        return std::make_shared<FakeSubInferRequest>(std::move(self));
    }
    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override {
        return std::make_shared<FakeSubAsyncInferRequest>(create_sync_infer_request());
    }

private:
    std::shared_ptr<ov::Model> m_model;
};

FakeSubInferRequest::FakeSubInferRequest(std::shared_ptr<const FakeSubCompiledModel> compiled_model)
    : ov::ISyncInferRequest(std::move(compiled_model)) {
    for (const auto& input : get_compiled_model()->inputs()) {
        ov::ISyncInferRequest::set_tensor(input,
                                          ov::get_tensor_impl(ov::Tensor(input.get_element_type(), input.get_shape())));
    }
    for (const auto& output : get_compiled_model()->outputs()) {
        ov::ISyncInferRequest::set_tensor(output,
                                          ov::get_tensor_impl(ov::Tensor(output.get_element_type(), output.get_shape())));
    }
}

void FakeSubInferRequest::infer() {
    for (const auto& output : get_compiled_model()->outputs()) {
        auto tensor = ov::ISyncInferRequest::get_tensor(output);
        std::memset(tensor->data(), 0, tensor->get_byte_size());
    }
}

class SubgraphBehaviorInferTest : public ::testing::Test {
protected:
    ov::AnyMap base_props() const {
        return {{"NPU_USE_NPUW", "YES"},
                {"NPUW_DEVICES", "CPU"},
                {"NPUW_UNFOLD_IREQS", "NO"},
                {"NPUW_ATTN", "DYNAMIC"},
                {"NPUW_ONLINE_PIPELINE", "REP"},
                {"NPUW_ONLINE_ISOLATE", "ATTN"}};
    }

    ov::AnyMap unfold_props() const {
        auto props = base_props();
        props["NPUW_UNFOLD_IREQS"] = "YES";
        return props;
    }
    std::shared_ptr<testing::NiceMock<ov::MockICore>> make_core(const std::shared_ptr<const ov::IPlugin>& plugin) const {
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
        ON_CALL(*core,
                compile_model(testing::Matcher<const std::shared_ptr<const ov::Model>&>(testing::_),
                              testing::Matcher<const std::string&>(testing::StrEq("CPU")),
                              testing::Matcher<const ov::AnyMap&>(testing::_)))
            .WillByDefault([plugin](const std::shared_ptr<const ov::Model>& submodel, const std::string&, const ov::AnyMap&) {
                return ov::SoPtr<ov::ICompiledModel>{std::make_shared<FakeSubCompiledModel>(
                    std::const_pointer_cast<ov::Model>(submodel), plugin)};
            });

        return core;
    }
};

TEST_F(SubgraphBehaviorInferTest, SdpaBehaviorCanOverrideStaticLlmSubgraphExecution) {
    auto baseline_model = build_static_llm_model();
    ASSERT_GT(count_sdpa_nodes(baseline_model), 0u) << "The synthesized LLM model must contain SDPA nodes";
    auto hits = std::make_shared<BehaviorHits>();

    auto plugin = std::make_shared<TestPlugin>();
    auto core = make_core(plugin);
    plugin->set_core(core);

    auto baseline_compiled = std::make_shared<ov::npuw::CompiledModel>(baseline_model, plugin, base_props());
    EXPECT_EQ(count_runtime_behaviors(baseline_compiled), 0u);
    auto baseline_request = baseline_compiled->create_infer_request();
    ASSERT_NE(baseline_request, nullptr);
    baseline_request->infer();
    EXPECT_TRUE(hits->values.empty());

    auto behavior_model = build_static_llm_model();
    ASSERT_GT(count_sdpa_nodes(behavior_model), 0u);
    ov::npuw::v1::subgraphs::PatternRegistry behavior_registry;
    auto behavior_compiled = std::make_shared<ov::npuw::CompiledModel>(behavior_model, plugin, base_props(), &behavior_registry);
    bool attached_behavior = false;
    for (auto& desc : behavior_compiled->m_compiled_submodels) {
        if (!desc.compiled_model) {
            continue;
        }

        ov::npuw::v1::subgraphs::RuntimeBehaviorSpec spec;
        spec.registration.group = "test";
        spec.registration.name = "record-hit";
        spec.context.put<std::shared_ptr<BehaviorHits>>(hits);
        spec.factory = [](const ov::npuw::v1::subgraphs::Context& ctx) -> ov::npuw::v1::subgraphs::ISubgraphBehavior::Ptr {
            const auto recorder = ctx.get<std::shared_ptr<BehaviorHits>>();
            return std::make_unique<ov::npuw::v1::subgraphs::DirectBehavior>(
                [recorder](ov::npuw::v1::subgraphs::InferContext& infer_ctx) {
                    infer_ctx.legacy_infer();
                    std::lock_guard<std::mutex> lock(recorder->mutex);
                    recorder->values.emplace_back(infer_ctx.subgraph_idx, infer_ctx.real_subgraph_idx);
                });
        };
        desc.pipeline.runtime_behavior = std::move(spec);
        attached_behavior = true;
    }
    ASSERT_TRUE(attached_behavior) << "No compiled subgraph was available for runtime behavior injection";
    auto behavior_request = behavior_compiled->create_infer_request();
    ASSERT_NE(behavior_request, nullptr);
    behavior_request->infer();

    ASSERT_FALSE(hits->values.empty()) << "The SDPA stub behavior was not invoked during inference";
}

TEST_F(SubgraphBehaviorInferTest, RuntimeBehaviorForcesJustInferRequestWhenUnfoldIsEnabled) {
    auto plugin = std::make_shared<TestPlugin>();
    auto core = make_core(plugin);
    plugin->set_core(core);

    auto baseline_model = build_static_llm_model();
    auto baseline_compiled = std::make_shared<ov::npuw::CompiledModel>(baseline_model, plugin, unfold_props());
    auto baseline_request = baseline_compiled->create_sync_infer_request();
    ASSERT_NE(baseline_request, nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<ov::npuw::UnfoldInferRequest>(baseline_request), nullptr);

    auto behavior_model = build_static_llm_model();
    auto hits = std::make_shared<BehaviorHits>();
    ov::npuw::v1::subgraphs::PatternRegistry behavior_registry;
    auto behavior_compiled = std::make_shared<ov::npuw::CompiledModel>(behavior_model, plugin, unfold_props(), &behavior_registry);

    bool attached_behavior = false;
    for (auto& desc : behavior_compiled->m_compiled_submodels) {
        if (!desc.compiled_model) {
            continue;
        }

        ov::npuw::v1::subgraphs::RuntimeBehaviorSpec spec;
        spec.registration.group = "test";
        spec.registration.name = "record-hit";
        spec.context.put<std::shared_ptr<BehaviorHits>>(hits);
        spec.factory = [](const ov::npuw::v1::subgraphs::Context& ctx) -> ov::npuw::v1::subgraphs::ISubgraphBehavior::Ptr {
            const auto recorder = ctx.get<std::shared_ptr<BehaviorHits>>();
            return std::make_unique<ov::npuw::v1::subgraphs::DirectBehavior>(
                [recorder](ov::npuw::v1::subgraphs::InferContext& infer_ctx) {
                    infer_ctx.legacy_infer();
                    std::lock_guard<std::mutex> lock(recorder->mutex);
                    recorder->values.emplace_back(infer_ctx.subgraph_idx, infer_ctx.real_subgraph_idx);
                });
        };
        desc.pipeline.runtime_behavior = std::move(spec);
        attached_behavior = true;
    }
    ASSERT_TRUE(attached_behavior);

    auto behavior_request = behavior_compiled->create_sync_infer_request();
    ASSERT_NE(behavior_request, nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<ov::npuw::JustInferRequest>(behavior_request), nullptr);
    EXPECT_EQ(std::dynamic_pointer_cast<ov::npuw::UnfoldInferRequest>(behavior_request), nullptr);
}

}  // namespace
