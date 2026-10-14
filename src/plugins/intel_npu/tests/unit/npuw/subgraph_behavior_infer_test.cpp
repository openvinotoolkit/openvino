// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <any>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

#define private   public
#define protected public
#include "base_sync_infer_request.hpp"
#include "compiled_model.hpp"
#undef protected
#undef private
#include "attn/attn_subgraph.hpp"
#include "embedding/redirect_new_kv_to_output.hpp"
#include "just_sync_infer_request.hpp"
#include "llm_test_helpers.hpp"
#include "model_builder.hpp"
#include "npuw_transformations/optimize_value_tensors.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "partitioning/patterns/sdpa.hpp"
#include "perf_tags.hpp"
#include "pyramid_attention.hpp"
#include "unfold_sync_infer_request.hpp"
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

// Publishes a perf attribution handle from prepare() - the same lifecycle point the
// pyramid attention behavior uses - and records the calls that actually reach run().
// run() only fires for subgraphs dispatched through JustInferRequest::unsafe_infer(),
// which is exactly why timing must NOT live there.
class AttributingBehavior final : public ov::npuw::v1::subgraphs::ISubgraphBehavior {
public:
    using Switch = std::shared_ptr<std::atomic<bool>>;

    AttributingBehavior(std::shared_ptr<BehaviorHits> hits, std::string level, Switch publish)
        : m_hits(std::move(hits)),
          m_level(std::move(level)),
          m_publish(std::move(publish)) {}

    void prepare(ov::npuw::v1::subgraphs::InferContext& ctx) override {
        auto& request = ctx.infer_request;
        if (!m_publish->load()) {
            return;
        }
        ov::npuw::perf::metric_ptr handle = nullptr;
        request.set_perf_attribution(ctx.real_subgraph_idx, request.perf().handle(handle, [&]() {
            return request.perf_submodel_tag(ctx.subgraph_idx) + "/attn/" + m_level;
        }));
    }

    void run(ov::npuw::v1::subgraphs::InferContext& ctx) override {
        ctx.legacy_infer();
        std::lock_guard<std::mutex> lock(m_hits->mutex);
        m_hits->values.emplace_back(ctx.subgraph_idx, ctx.real_subgraph_idx);
    }

private:
    std::shared_ptr<BehaviorHits> m_hits;
    std::string m_level;
    Switch m_publish;
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

// Count subgraphs where the attention runtime behavior was attached.  The test keys off the
// behavior's registered attn identity rather than only handles_function_prologue, since that
// callback flag is not unique to attention behaviors.
std::size_t count_dyn_attn_behaviors(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model) {
    return std::count_if(compiled_model->m_compiled_submodels.begin(),
                         compiled_model->m_compiled_submodels.end(),
                         [](const auto& desc) {
                             if (!desc.pipeline.runtime_behavior.has_value()) {
                                 return false;
                             }
                             const auto& spec = *desc.pipeline.runtime_behavior;
                             return spec.handles_function_prologue &&
                                    spec.registration.group == ov::npuw::patterns::attn::SDPA::group_name() &&
                                    spec.registration.name == ov::npuw::patterns::attn::SDPA::pattern_name();
                         });
}

class FakeSubCompiledModel;

class FakeSubInferRequest final : public ov::ISyncInferRequest {
public:
    explicit FakeSubInferRequest(std::shared_ptr<const FakeSubCompiledModel> compiled_model);

    void set_fail_on_infer(bool fail) {
        m_fail_on_infer = fail;
    }

    void infer() override;
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override {
        check_port(port);
        return ov::ISyncInferRequest::get_tensor(port);
    }
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override {
        check_port(port);
        ov::ISyncInferRequest::set_tensor(port, tensor);
    }
    void check_tensors() const override {}
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }

private:
    void check_port(const ov::Output<const ov::Node>& port) const;
    bool m_fail_on_infer = false;
};

class FakeSubAsyncInferRequest final : public ov::IAsyncInferRequest {
public:
    explicit FakeSubAsyncInferRequest(const std::shared_ptr<ov::ISyncInferRequest>& request)
        : ov::IAsyncInferRequest(nullptr, nullptr, nullptr),
          m_request(request) {}

    ~FakeSubAsyncInferRequest() override {
        if (m_worker.joinable()) {
            m_worker.join();
        }
    }

    void set_async_delay(std::chrono::milliseconds delay) {
        m_delay = delay;
    }

    void set_invoke_callback(bool invoke) {
        m_invoke_callback = invoke;
    }

    void set_fail_on_infer(bool fail) {
        std::static_pointer_cast<FakeSubInferRequest>(m_request)->set_fail_on_infer(fail);
    }

    bool has_callback() const {
        return static_cast<bool>(m_callback);
    }

    bool callback_reported_error() const {
        return m_callback_reported_error;
    }

    void start_async() override {
        // A new run has its own result, like a new promise in ov::IAsyncInferRequest
        m_error = nullptr;
        m_callback_reported_error = false;
        if (m_delay.count() > 0) {
            m_worker = std::thread([this] {
                std::this_thread::sleep_for(m_delay);
                run();
            });
            return;
        }
        run();
    }

    void wait() override {
        if (m_worker.joinable()) {
            m_worker.join();
        }
        if (m_error) {
            std::rethrow_exception(m_error);
        }
    }

    void run() {
        try {
            m_request->infer();
            if (m_invoke_callback && m_callback) {
                m_callback(nullptr);
            }
        } catch (...) {
            auto error = std::current_exception();
            if (m_invoke_callback && m_callback) {
                m_callback_reported_error = true;
                try {
                    m_callback(error);
                } catch (...) {
                    error = std::current_exception();
                }
            }
            m_error = error;
        }
    }

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
    std::chrono::milliseconds m_delay{0};
    bool m_invoke_callback = true;
    bool m_callback_reported_error = false;
    std::thread m_worker;
    std::exception_ptr m_error;
};

class FakeSubCompiledModel final : public ov::ICompiledModel {
public:
    FakeSubCompiledModel(const std::shared_ptr<ov::Model>& model,
                         const std::shared_ptr<const ov::IPlugin>& plugin,
                         bool own_ports_only = false)
        : ov::ICompiledModel(model, plugin, nullptr, nullptr),
          m_model(model),
          m_own_ports_only(own_ports_only) {}

    bool own_ports_only() const {
        return m_own_ports_only;
    }

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
    bool m_own_ports_only = false;
};

// When the compiled submodels are imported from a cache, each blob gets its own I/O nodes
// rebuilt from the blob metadata, so a port of one compiled model is not guaranteed to be
// resolved by an infer request created from another one. own_ports_only models exactly that.
void FakeSubInferRequest::check_port(const ov::Output<const ov::Node>& port) const {
    const auto& compiled_model = get_compiled_model();
    if (!std::static_pointer_cast<const FakeSubCompiledModel>(compiled_model)->own_ports_only()) {
        return;
    }
    const auto& inputs = compiled_model->inputs();
    const auto& outputs = compiled_model->outputs();
    OPENVINO_ASSERT(std::find(inputs.begin(), inputs.end(), port) != inputs.end() ||
                        std::find(outputs.begin(), outputs.end(), port) != outputs.end(),
                    "Cannot find tensor for port ",
                    port);
}

FakeSubInferRequest::FakeSubInferRequest(std::shared_ptr<const FakeSubCompiledModel> compiled_model)
    : ov::ISyncInferRequest(std::move(compiled_model)) {
    for (const auto& input : get_compiled_model()->inputs()) {
        ov::ISyncInferRequest::set_tensor(input,
                                          ov::get_tensor_impl(ov::Tensor(input.get_element_type(), input.get_shape())));
    }
    for (const auto& output : get_compiled_model()->outputs()) {
        ov::ISyncInferRequest::set_tensor(
            output,
            ov::get_tensor_impl(ov::Tensor(output.get_element_type(), output.get_shape())));
    }
}

void FakeSubInferRequest::infer() {
    if (m_fail_on_infer) {
        OPENVINO_THROW("Fake subrequest failed");
    }
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
                {"NPUW_FOLD", "YES"},
                {"NPUW_ONLINE_PIPELINE", "REP"},
                {"NPUW_ONLINE_ISOLATE", "ATTN"},
                // The test model has only 2 layers so repeated blocks appear only twice.
                // Lower the thresholds so they survive cleanUpUniquesImpl and ens.repeated
                // stays non-empty (required for the FOLD pass to run at all).
                {"NPUW_ONLINE_KEEP_BLOCKS", "2"},
                {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "1"}};
    }

    ov::AnyMap unfold_props() const {
        auto props = base_props();
        props["NPUW_UNFOLD_IREQS"] = "YES";
        return props;
    }
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    void check_unfold_profile(bool async) {
        auto plugin = std::make_shared<TestPlugin>();
        auto core = make_core(plugin);
        plugin->set_core(core);

        auto props = unfold_props();
        props["NPUW_FUNCALL_ASYNC"] = async ? "YES" : "NO";
        auto compiled = std::make_shared<ov::npuw::CompiledModel>(build_static_llm_model(), plugin, props);
        auto request = std::dynamic_pointer_cast<ov::npuw::UnfoldInferRequest>(compiled->create_sync_infer_request());
        ASSERT_NE(request, nullptr);
        request->m_perf.execution.enable(true);
        std::static_pointer_cast<ov::ISyncInferRequest>(request)->infer();

        std::map<std::string, std::size_t> expected;
        for (std::size_t idx = 0; idx < request->total_subrequests(); ++idx) {
            if (!request->valid_subrequest(idx)) {
                continue;
            }
            const auto real_idx = compiled->m_compiled_submodels[idx].replaced_by.value_or(idx);
            ++expected[ov::npuw::perf::tags::device(compiled->submodel_device(real_idx))];
            ++expected[request->perf_submodel_tag(real_idx)];
        }
        ASSERT_FALSE(expected.empty());
        const auto& metrics = request->m_perf.execution.metrics;
        ASSERT_EQ(expected.size(), metrics.size()) << "Unfold must not create attribution entries";
        for (const auto& entry : expected) {
            ASSERT_EQ(1u, metrics.count(entry.first)) << entry.first;
            EXPECT_EQ(entry.second, metrics.at(entry.first).count()) << entry.first;
        }
        request->m_perf.execution.enable(false);
    }
#endif
    std::shared_ptr<testing::NiceMock<ov::MockICore>> make_core(const std::shared_ptr<const ov::IPlugin>& plugin,
                                                                bool own_ports_only = false) const {
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
            .WillByDefault([plugin, own_ports_only](const std::shared_ptr<const ov::Model>& submodel,
                                                    const std::string&,
                                                    const ov::AnyMap&) {
                return ov::SoPtr<ov::ICompiledModel>{
                    std::make_shared<FakeSubCompiledModel>(std::const_pointer_cast<ov::Model>(submodel),
                                                           plugin,
                                                           own_ports_only)};
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
    auto behavior_compiled =
        std::make_shared<ov::npuw::CompiledModel>(behavior_model, plugin, base_props(), &behavior_registry);
    bool attached_behavior = false;
    for (auto& desc : behavior_compiled->m_compiled_submodels) {
        if (!desc.compiled_model) {
            continue;
        }

        ov::npuw::v1::subgraphs::RuntimeBehaviorSpec spec;
        spec.registration.group = "test";
        spec.registration.name = "record-hit";
        spec.context.put<std::shared_ptr<BehaviorHits>>(hits);
        spec.factory =
            [](const ov::npuw::v1::subgraphs::Context& ctx) -> ov::npuw::v1::subgraphs::ISubgraphBehavior::Ptr {
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
    auto behavior_compiled =
        std::make_shared<ov::npuw::CompiledModel>(behavior_model, plugin, unfold_props(), &behavior_registry);

    bool attached_behavior = false;
    for (auto& desc : behavior_compiled->m_compiled_submodels) {
        if (!desc.compiled_model) {
            continue;
        }

        ov::npuw::v1::subgraphs::RuntimeBehaviorSpec spec;
        spec.registration.group = "test";
        spec.registration.name = "record-hit";
        spec.context.put<std::shared_ptr<BehaviorHits>>(hits);
        spec.factory =
            [](const ov::npuw::v1::subgraphs::Context& ctx) -> ov::npuw::v1::subgraphs::ISubgraphBehavior::Ptr {
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

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
TEST_F(SubgraphBehaviorInferTest, UnfoldSyncProfilesDeviceAndSubmodelOnly) {
    check_unfold_profile(false);
}

TEST_F(SubgraphBehaviorInferTest, UnfoldAsyncProfilesDeviceAndSubmodelOnly) {
    check_unfold_profile(true);
}

TEST_F(SubgraphBehaviorInferTest, UnfoldAsyncRecordsEachSiblingCompletionTime) {
    auto plugin = std::make_shared<TestPlugin>();
    auto core = make_core(plugin);
    plugin->set_core(core);

    auto props = unfold_props();
    props["NPUW_FUNCALL_ASYNC"] = "YES";
    auto compiled = std::make_shared<ov::npuw::CompiledModel>(build_static_llm_model(), plugin, props);
    auto request = std::dynamic_pointer_cast<ov::npuw::UnfoldInferRequest>(compiled->create_sync_infer_request());
    ASSERT_NE(request, nullptr);
    request->m_perf.execution.enable(true);

    std::size_t first = request->total_subrequests();
    for (std::size_t idx = 0; idx + 1 < request->total_subrequests(); ++idx) {
        if (!request->valid_subrequest(idx) || !request->valid_subrequest(idx + 1)) {
            continue;
        }
        const auto real = compiled->m_compiled_submodels[idx].replaced_by.value_or(idx);
        if (real == compiled->m_compiled_submodels[idx + 1].replaced_by.value_or(idx + 1)) {
            first = idx;
            break;
        }
    }
    ASSERT_LT(first + 1, request->total_subrequests()) << "The test needs two overlapped function calls";
    auto slow = std::dynamic_pointer_cast<FakeSubAsyncInferRequest>(request->m_subrequests[first]._ptr);
    auto fast = std::dynamic_pointer_cast<FakeSubAsyncInferRequest>(request->m_subrequests[first + 1]._ptr);
    ASSERT_NE(slow, nullptr);
    ASSERT_NE(fast, nullptr);
    slow->set_async_delay(std::chrono::milliseconds{140});
    fast->set_async_delay(std::chrono::milliseconds{5});

    std::static_pointer_cast<ov::ISyncInferRequest>(request)->infer();

    const auto real = compiled->m_compiled_submodels[first].replaced_by.value_or(first);
    const auto& samples = request->m_perf.execution.metrics.at(request->perf_submodel_tag(real)).samples();
    std::size_t prior = 0;
    for (std::size_t idx = 0; idx < first; ++idx) {
        if (request->valid_subrequest(idx) && compiled->m_compiled_submodels[idx].replaced_by.value_or(idx) == real) {
            ++prior;
        }
    }
    ASSERT_LT(prior + 1, samples.size());
    EXPECT_GT(samples[prior], 100.f);
    EXPECT_LT(samples[prior + 1], samples[prior] * 0.5f);
    request->m_perf.execution.enable(false);
}

TEST_F(SubgraphBehaviorInferTest, UnfoldAsyncWithoutCallbackUsesWaitTime) {
    auto plugin = std::make_shared<TestPlugin>();
    auto core = make_core(plugin);
    plugin->set_core(core);

    auto props = unfold_props();
    props["NPUW_FUNCALL_ASYNC"] = "YES";
    auto compiled = std::make_shared<ov::npuw::CompiledModel>(build_static_llm_model(), plugin, props);
    auto request = std::dynamic_pointer_cast<ov::npuw::UnfoldInferRequest>(compiled->create_sync_infer_request());
    ASSERT_NE(request, nullptr);
    request->m_perf.execution.enable(true);

    std::size_t idx = 0;
    while (idx < request->total_subrequests() && !request->valid_subrequest(idx)) {
        ++idx;
    }
    ASSERT_LT(idx, request->total_subrequests());
    auto subrequest = std::dynamic_pointer_cast<FakeSubAsyncInferRequest>(request->m_subrequests[idx]._ptr);
    ASSERT_NE(subrequest, nullptr);
    subrequest->set_invoke_callback(false);
    subrequest->set_async_delay(std::chrono::milliseconds{5});

    std::static_pointer_cast<ov::ISyncInferRequest>(request)->infer();

    const auto real = compiled->m_compiled_submodels[idx].replaced_by.value_or(idx);
    const auto& samples = request->m_perf.execution.metrics.at(request->perf_submodel_tag(real)).samples();
    ASSERT_FALSE(samples.empty());
    EXPECT_GE(samples.front(), 0.f);
    EXPECT_LT(samples.front(), 1000.f);
    request->m_perf.execution.enable(false);
}

TEST_F(SubgraphBehaviorInferTest, UnfoldAsyncFailedChildClearsPendingCallbacksAndDoesNotRecordFailure) {
    auto plugin = std::make_shared<TestPlugin>();
    auto core = make_core(plugin);
    plugin->set_core(core);

    auto props = unfold_props();
    props["NPUW_FUNCALL_ASYNC"] = "YES";
    auto compiled = std::make_shared<ov::npuw::CompiledModel>(build_static_llm_model(), plugin, props);
    auto request = std::dynamic_pointer_cast<ov::npuw::UnfoldInferRequest>(compiled->create_sync_infer_request());
    ASSERT_NE(request, nullptr);
    request->m_perf.execution.enable(true);

    std::size_t first = request->total_subrequests();
    for (std::size_t idx = 0; idx + 1 < request->total_subrequests(); ++idx) {
        if (!request->valid_subrequest(idx) || !request->valid_subrequest(idx + 1)) {
            continue;
        }
        const auto real = compiled->m_compiled_submodels[idx].replaced_by.value_or(idx);
        if (real == compiled->m_compiled_submodels[idx + 1].replaced_by.value_or(idx + 1)) {
            first = idx;
            break;
        }
    }
    ASSERT_LT(first + 1, request->total_subrequests()) << "The test needs two pending function calls";
    auto failed = std::dynamic_pointer_cast<FakeSubAsyncInferRequest>(request->m_subrequests[first]._ptr);
    auto sibling = std::dynamic_pointer_cast<FakeSubAsyncInferRequest>(request->m_subrequests[first + 1]._ptr);
    ASSERT_NE(failed, nullptr);
    ASSERT_NE(sibling, nullptr);
    failed->set_fail_on_infer(true);
    failed->set_async_delay(std::chrono::milliseconds{5});
    sibling->set_async_delay(std::chrono::milliseconds{20});

    EXPECT_THROW(std::static_pointer_cast<ov::ISyncInferRequest>(request)->infer(), ov::Exception);
    EXPECT_TRUE(failed->callback_reported_error());
    const auto real = compiled->m_compiled_submodels[first].replaced_by.value_or(first);
    for (std::size_t idx = first;
         idx < request->total_subrequests() && compiled->m_compiled_submodels[idx].replaced_by.value_or(idx) == real;
         ++idx) {
        if (!request->valid_subrequest(idx)) {
            continue;
        }
        auto pending = std::dynamic_pointer_cast<FakeSubAsyncInferRequest>(request->m_subrequests[idx]._ptr);
        ASSERT_NE(pending, nullptr);
        EXPECT_FALSE(pending->has_callback()) << "Callback remains on pending subrequest " << idx;
    }

    const auto tag = request->perf_submodel_tag(real);
    const auto& metrics = request->m_perf.execution.metrics;
    std::size_t prior = 0;
    for (std::size_t idx = 0; idx < first; ++idx) {
        if (request->valid_subrequest(idx) && compiled->m_compiled_submodels[idx].replaced_by.value_or(idx) == real) {
            ++prior;
        }
    }
    const auto recorded = metrics.count(tag) == 0u ? 0u : metrics.at(tag).count();
    EXPECT_EQ(prior, recorded) << "The failed child must not add a sample";

    // A later run of the same request recovers: the old child error is not rethrown
    failed->set_fail_on_infer(false);
    EXPECT_NO_THROW(std::static_pointer_cast<ov::ISyncInferRequest>(request)->infer());
    EXPECT_FALSE(failed->callback_reported_error());
    request->m_perf.execution.enable(false);
}
#endif

// --- Dynamic-attention behavior gating tests ---
//
// These tests verify two properties:
//
//  1. When NPUW_ATTN=DYNAMIC + NPUW_ONLINE_ISOLATE=ATTN are set and the model has SDPA nodes
//     with dynamic KV-cache dimensions, the attn runtime behavior IS attached to the attention
//     subgraphs.
//
//  2. When either condition is absent (STATIC mode or no isolation), NO DynAttnBehavior is
//     attached.  This proves the gate is working correctly.
//
// build_dynamic_llm_model() produces a stateful→stateless-converted LLM whose KV-cache
// parameters retain dynamic dimensions — the exact shape that function::Attention::from()
// requires to succeed in DYNAMIC mode.

TEST_F(SubgraphBehaviorInferTest, DynAttnBehaviorAttachedWhenDynamicAttentionRequested) {
    auto model = ov::test::npuw::build_dynamic_attention_llm_model();
    ASSERT_GT(count_sdpa_nodes(model), 0u) << "Dynamic LLM model must contain SDPA nodes";

    auto plugin = std::make_shared<TestPlugin>();
    auto core = make_core(plugin);
    plugin->set_core(core);

    // base_props() has NPUW_ATTN=DYNAMIC and NPUW_ONLINE_ISOLATE=ATTN
    auto compiled = std::make_shared<ov::npuw::CompiledModel>(model, plugin, base_props());
    EXPECT_GT(count_dyn_attn_behaviors(compiled), 0u)
        << "DynAttnBehavior must be attached when NPUW_ATTN=DYNAMIC and NPUW_ONLINE_ISOLATE=ATTN";
}

TEST_F(SubgraphBehaviorInferTest, DynAttnBehaviorNotAttachedWithStaticAttentionMode) {
    auto model = ov::test::npuw::build_dynamic_attention_llm_model();

    auto plugin = std::make_shared<TestPlugin>();
    auto core = make_core(plugin);
    plugin->set_core(core);

    auto props = base_props();
    props["NPUW_ATTN"] = std::string("STATIC");
    auto compiled = std::make_shared<ov::npuw::CompiledModel>(model, plugin, props);
    EXPECT_EQ(count_dyn_attn_behaviors(compiled), 0u) << "DynAttnBehavior must NOT be attached when NPUW_ATTN=STATIC";
}

TEST_F(SubgraphBehaviorInferTest, DynAttnBehaviorNotAttachedWithoutAttnIsolation) {
    auto model = ov::test::npuw::build_dynamic_attention_llm_model();

    auto plugin = std::make_shared<TestPlugin>();
    auto core = make_core(plugin);
    plugin->set_core(core);

    // Remove NPUW_ONLINE_ISOLATE so no "attn" functions are created by the online partitioner.
    ov::AnyMap props = {{"NPU_USE_NPUW", "YES"},
                        {"NPUW_DEVICES", "CPU"},
                        {"NPUW_UNFOLD_IREQS", "NO"},
                        {"NPUW_ATTN", std::string("DYNAMIC")},
                        {"NPUW_FOLD", "YES"},
                        {"NPUW_ONLINE_PIPELINE", "REP"},
                        {"NPUW_ONLINE_KEEP_BLOCKS", "2"},
                        {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "1"}};
    auto compiled = std::make_shared<ov::npuw::CompiledModel>(model, plugin, props);
    EXPECT_EQ(count_dyn_attn_behaviors(compiled), 0u)
        << "DynAttnBehavior must NOT be attached when NPUW_ONLINE_ISOLATE=ATTN is not set";
}

// --- Perf attribution tests ---
//
// These pin the bug class the per-submodel breakdown exists to prevent: an attention
// behavior publishes "attribute my next run to this metric" from prepare(), and the
// generic subgraph timer in IBaseInferRequest::infer() must pick it up for EVERY logical
// call of the prototype - regardless of which dispatch path unsafe_during() selects.

#ifdef NPU_PLUGIN_DEVELOPER_BUILD

namespace {

using AttrSwitch = AttributingBehavior::Switch;

// Attaches AttributingBehavior to every compiled (prototype) subgraph and returns the
// number of _logical_ submodels that map onto each of them.
std::map<std::size_t, std::size_t> attach_attributing_behavior(const std::shared_ptr<ov::npuw::CompiledModel>& compiled,
                                                               const std::shared_ptr<BehaviorHits>& hits,
                                                               const std::string& level,
                                                               const AttrSwitch& publish) {
    for (auto& desc : compiled->m_compiled_submodels) {
        if (!desc.compiled_model) {
            continue;
        }
        ov::npuw::v1::subgraphs::RuntimeBehaviorSpec spec;
        spec.registration.group = "test";
        spec.registration.name = "attribute";
        spec.context.put<std::shared_ptr<BehaviorHits>>(hits);
        spec.context.put<std::string>(level);
        spec.context.put<AttrSwitch>(publish);
        spec.factory =
            [](const ov::npuw::v1::subgraphs::Context& ctx) -> ov::npuw::v1::subgraphs::ISubgraphBehavior::Ptr {
            return std::make_unique<AttributingBehavior>(ctx.get<std::shared_ptr<BehaviorHits>>(),
                                                         ctx.get<std::string>(),
                                                         ctx.get<AttrSwitch>());
        };
        desc.pipeline.runtime_behavior = std::move(spec);
    }

    std::map<std::size_t, std::size_t> logical_calls;
    for (std::size_t idx = 0; idx < compiled->m_compiled_submodels.size(); idx++) {
        const auto& desc = compiled->m_compiled_submodels[idx];
        const auto real_idx = desc.replaced_by.value_or(idx);
        if (!compiled->m_compiled_submodels[real_idx].compiled_model) {
            continue;
        }
        logical_calls[real_idx]++;
    }
    return logical_calls;
}

ov::npuw::IBaseInferRequest* enable_profiling(const std::shared_ptr<ov::ISyncInferRequest>& request) {
    auto* base = dynamic_cast<ov::npuw::IBaseInferRequest*>(request.get());
    if (base == nullptr) {
        return nullptr;
    }
    base->m_perf.execution.enable(true);
    return base;
}

}  // namespace

TEST_F(SubgraphBehaviorInferTest, PerfAttributionCoversEveryLogicalCall) {
    auto plugin = std::make_shared<TestPlugin>();
    auto core = make_core(plugin);
    plugin->set_core(core);

    auto model = build_static_llm_model();
    auto hits = std::make_shared<BehaviorHits>();
    auto publish = std::make_shared<std::atomic<bool>>(true);
    ov::npuw::v1::subgraphs::PatternRegistry registry;
    auto compiled = std::make_shared<ov::npuw::CompiledModel>(model, plugin, base_props(), &registry);
    const auto logical_calls = attach_attributing_behavior(compiled, hits, "level[00]", publish);
    ASSERT_FALSE(logical_calls.empty());

    auto request = compiled->create_sync_infer_request();
    ASSERT_NE(request, nullptr);
    auto* base = enable_profiling(request);
    ASSERT_NE(base, nullptr);

    request->infer();

    for (const auto& entry : logical_calls) {
        const auto real_idx = entry.first;
        const auto calls = entry.second;
        const auto submodel_tag = base->perf_submodel_tag(real_idx);
        const auto attn_tag = submodel_tag + "/attn/level[00]";

        const auto attributed = base->m_perf.execution.metrics.find(attn_tag);
        ASSERT_NE(attributed, base->m_perf.execution.metrics.end()) << "No attribution recorded for " << attn_tag;
        EXPECT_EQ(calls, attributed->second.count()) << "Attribution must cover every logical call of " << attn_tag;

        const auto submodel = base->m_perf.execution.metrics.find(submodel_tag);
        ASSERT_NE(submodel, base->m_perf.execution.metrics.end());
        EXPECT_EQ(calls, submodel->second.count());
    }

    base->m_perf.execution.enable(false);  // keep the test output clean
}

TEST_F(SubgraphBehaviorInferTest, PerfAttributionIsClearedBetweenRuns) {
    auto plugin = std::make_shared<TestPlugin>();
    auto core = make_core(plugin);
    plugin->set_core(core);

    auto model = build_static_llm_model();
    auto hits = std::make_shared<BehaviorHits>();
    auto publish = std::make_shared<std::atomic<bool>>(true);
    ov::npuw::v1::subgraphs::PatternRegistry registry;
    auto compiled = std::make_shared<ov::npuw::CompiledModel>(model, plugin, base_props(), &registry);
    const auto logical_calls = attach_attributing_behavior(compiled, hits, "level[00]", publish);
    ASSERT_FALSE(logical_calls.empty());

    auto request = compiled->create_sync_infer_request();
    auto* base = enable_profiling(request);
    ASSERT_NE(base, nullptr);

    request->infer();

    std::map<std::size_t, std::size_t> after_first;
    for (const auto& entry : logical_calls) {
        after_first[entry.first] =
            base->m_perf.execution.metrics.at(base->perf_submodel_tag(entry.first) + "/attn/level[00]").count();
    }

    // A level that stops being selected must not leak attribution into the next run
    publish->store(false);
    request->infer();

    for (const auto& entry : logical_calls) {
        const auto tag = base->perf_submodel_tag(entry.first) + "/attn/level[00]";
        EXPECT_EQ(after_first.at(entry.first), base->m_perf.execution.metrics.at(tag).count())
            << "Stale attribution leaked into the next run for " << tag;
        // the submodel entry itself must have doubled - the run really happened
        EXPECT_EQ(2u * entry.second, base->m_perf.execution.metrics.at(base->perf_submodel_tag(entry.first)).count());
    }

    base->m_perf.execution.enable(false);
}

TEST_F(SubgraphBehaviorInferTest, DisabledProfilingCollectsNothingAndKeepsTheDispatchPath) {
    auto plugin = std::make_shared<TestPlugin>();
    auto core = make_core(plugin);
    plugin->set_core(core);

    auto make_run = [&](bool profiling) {
        auto model = build_static_llm_model();
        auto hits = std::make_shared<BehaviorHits>();
        auto publish = std::make_shared<std::atomic<bool>>(true);
        ov::npuw::v1::subgraphs::PatternRegistry registry;
        auto compiled = std::make_shared<ov::npuw::CompiledModel>(model, plugin, base_props(), &registry);
        attach_attributing_behavior(compiled, hits, "level[00]", publish);

        auto request = compiled->create_sync_infer_request();
        auto* base = dynamic_cast<ov::npuw::IBaseInferRequest*>(request.get());
        EXPECT_NE(base, nullptr);
        if (profiling) {
            enable_profiling(request);
        }
        request->infer();
        const bool collected = base != nullptr && base->m_perf.execution.has_samples();
        if (base != nullptr) {
            base->m_perf.execution.enable(false);
        }
        return std::make_pair(hits->values, collected);
    };

    const auto off = make_run(false);
    const auto on = make_run(true);

    EXPECT_FALSE(off.second) << "Nothing must be collected when profiling is off";
    EXPECT_TRUE(on.second);
    // Same dispatch decisions either way - profiling must not change execution topology
    EXPECT_EQ(off.first, on.first);
}

#endif  // NPU_PLUGIN_DEVELOPER_BUILD

// Regression test for "Cannot find tensor for port" on the first inference of an LLM imported
// from CACHE_DIR with chunked prefill (PYRAMID prefill attention). The pyramid variants are
// separate compiled models with their own infer requests; every port bound on a variant request
// must come from that variant's compiled model, not from the main (last) one.
TEST_F(SubgraphBehaviorInferTest, PyramidBehaviorBindsOnlyActiveVariantPorts) {
    auto model = ov::test::npuw::build_dynamic_attention_llm_model();
    // Same transformations as the chunked LLM prefill pipeline: decompose SDPA so the PYRAMID
    // pattern can match, and output the new chunk KV only.
    ov::npuw::util::OptimizeValueTensors(true).run_on_model(model);
    ov::npuw::RedirectNewKvToOutput().run_on_model(model);

    auto plugin = std::make_shared<TestPlugin>();
    auto core = make_core(plugin, /*own_ports_only=*/true);
    plugin->set_core(core);

    auto props = base_props();
    props["NPUW_ATTN"] = std::string("PYRAMID");
    // No Failsafe wrappers - they are not created for imported blobs either.
    props["NPUW_FALLBACK_EXEC"] = std::string("NO");
    auto compiled = std::make_shared<ov::npuw::CompiledModel>(model, plugin, props);

    std::size_t num_pyramid_models = 0u;
    for (const auto& desc : compiled->m_compiled_submodels) {
        if (const auto* pyramid = ov::npuw::attn::get_compiled_pyramid(desc.pipeline.context)) {
            num_pyramid_models = std::max(num_pyramid_models, pyramid->_compiled_models.size());
        }
    }
    ASSERT_GT(num_pyramid_models, 1u) << "The test model must produce more than one pyramid attention variant";

    auto request = compiled->create_infer_request();
    ASSERT_NE(request, nullptr);
    for (const auto& input : compiled->inputs()) {
        ov::Tensor tensor(input.get_element_type(), input.get_shape());
        std::memset(tensor.data(), 0, tensor.get_byte_size());
        if (input.get_names().count("position_ids")) {
            // A first chunk without history selects the smallest (non-main) pyramid variant
            std::iota(tensor.data<int64_t>(), tensor.data<int64_t>() + tensor.get_size(), int64_t{0});
        }
        request->set_tensor(input, ov::get_tensor_impl(tensor));
    }
    EXPECT_NO_THROW(request->infer());
}

}  // namespace
