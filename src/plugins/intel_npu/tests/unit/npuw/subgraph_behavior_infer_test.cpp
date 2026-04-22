// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <any>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <unistd.h>
#include <vector>

#define private public
#include "compiled_model.hpp"
#undef private
#include "llm_test_helpers.hpp"
#include "model_builder.hpp"
#include "plugin.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"

namespace intel_npu {

std::atomic<int> Plugin::_compiledModelLoadCounter{0};

Plugin::Plugin() : _logger("test_npu_plugin", ov::log::Level::NO) {}

void Plugin::set_property(const ov::AnyMap&) {}

ov::Any Plugin::get_property(const std::string&, const ov::AnyMap&) const {
    OPENVINO_THROW("Test plugin does not expose properties");
}

bool Plugin::is_property_supported(const std::string&, const ov::AnyMap&) const {
    return false;
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const {
    OPENVINO_THROW("Unexpected Plugin::compile_model call in subgraph behavior test");
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>&,
                                                          const ov::AnyMap&,
                                                          const ov::SoPtr<ov::IRemoteContext>&) const {
    OPENVINO_THROW("Unexpected Plugin::compile_model(context) call in subgraph behavior test");
}

ov::SoPtr<ov::IRemoteContext> Plugin::create_context(const ov::AnyMap&) const {
    OPENVINO_THROW("Unexpected Plugin::create_context call in subgraph behavior test");
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const ov::AnyMap&) const {
    OPENVINO_THROW("Unexpected Plugin::get_default_context call in subgraph behavior test");
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream&, const ov::AnyMap&) const {
    OPENVINO_THROW("Unexpected Plugin::import_model(stream) call in subgraph behavior test");
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream&,
                                                         const ov::SoPtr<ov::IRemoteContext>&,
                                                         const ov::AnyMap&) const {
    OPENVINO_THROW("Unexpected Plugin::import_model(stream, context) call in subgraph behavior test");
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor&, const ov::AnyMap&) const {
    OPENVINO_THROW("Unexpected Plugin::import_model(blob) call in subgraph behavior test");
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor&,
                                                         const ov::SoPtr<ov::IRemoteContext>&,
                                                         const ov::AnyMap&) const {
    OPENVINO_THROW("Unexpected Plugin::import_model(blob, context) call in subgraph behavior test");
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const {
    OPENVINO_THROW("Unexpected Plugin::query_model call in subgraph behavior test");
}

}  // namespace intel_npu

namespace {

using ov::test::npuw::build_llm_test_model;

constexpr std::size_t kSeqLen = 4u;
constexpr std::size_t kPastKvLen = 4u;
constexpr std::size_t kKVCacheSize = kSeqLen + kPastKvLen;
struct MarkerPath {
    std::string value;
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

class FakeSubCompiledModel final : public ov::ICompiledModel {
public:
    FakeSubCompiledModel(const std::shared_ptr<ov::Model>& model, const std::shared_ptr<const ov::IPlugin>& plugin)
        : ov::ICompiledModel(model, plugin),
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
        return std::make_shared<ov::IAsyncInferRequest>(create_sync_infer_request(), get_task_executor(), get_callback_executor());
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

    static std::filesystem::path marker_path() {
        return std::filesystem::temp_directory_path() /
               ("npuw-subgraph-behavior-" + std::to_string(::getpid()) + ".log");
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
    const auto marker = marker_path();
    std::filesystem::remove(marker);

    auto plugin = std::make_shared<intel_npu::Plugin>();
    auto core = make_core(plugin);
    plugin->set_core(core);

    auto baseline_compiled = std::make_shared<ov::npuw::CompiledModel>(baseline_model, plugin, base_props());
    EXPECT_EQ(count_runtime_behaviors(baseline_compiled), 0u);
    auto baseline_request = baseline_compiled->create_infer_request();
    ASSERT_NE(baseline_request, nullptr);
    baseline_request->infer();
    EXPECT_FALSE(std::filesystem::exists(marker));

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
        spec.registration.name = "write-marker";
        spec.context.put<MarkerPath>({marker.string()});
        spec.factory = [](const ov::npuw::v1::subgraphs::Context& ctx) -> ov::npuw::v1::subgraphs::ISubgraphBehavior::Ptr {
            const auto path = ctx.get<MarkerPath>().value;
            return std::make_unique<ov::npuw::v1::subgraphs::DirectBehavior>(
                [path](ov::npuw::v1::subgraphs::InferContext& infer_ctx) {
                    infer_ctx.legacy_infer();
                    std::ofstream out(path, std::ios::app);
                    out << "hit:" << infer_ctx.subgraph_idx << ':' << infer_ctx.real_subgraph_idx << '\n';
                });
        };
        desc.pipeline.runtime_behavior = std::move(spec);
        attached_behavior = true;
    }
    ASSERT_TRUE(attached_behavior) << "No compiled subgraph was available for runtime behavior injection";
    auto behavior_request = behavior_compiled->create_infer_request();
    ASSERT_NE(behavior_request, nullptr);
    behavior_request->infer();

    ASSERT_TRUE(std::filesystem::exists(marker)) << "The SDPA stub behavior was not invoked during inference";
    std::ifstream input(marker);
    const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    EXPECT_NE(contents.find("hit:"), std::string::npos);
    std::filesystem::remove(marker);
}

}  // namespace
