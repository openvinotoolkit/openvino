// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <any>
#include <array>
#include <atomic>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <vector>

// Parse OpenVINO's exported classes before the private-access test shim;
// changing Constant's access specifiers changes imported symbol names on MSVC.
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"

#define private public
#include "compiled_model.hpp"
#undef private
#include "just_sync_infer_request.hpp"
#include "llm_test_helpers.hpp"
#include "model_builder.hpp"
#include "moe/moe_subgraph.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "partitioning/patterns/sdpa.hpp"
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

struct EvaluationLog {
    std::mutex mutex;
    std::map<const ov::ICompiledModel*, std::vector<const ov::ISyncInferRequest*>> executions;
    std::atomic<size_t> pending{0};
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
    explicit FakeSubAsyncInferRequest(const std::shared_ptr<ov::ISyncInferRequest>& request,
                                      std::shared_ptr<EvaluationLog> evaluation = {})
        : ov::IAsyncInferRequest(nullptr, nullptr, nullptr),
          m_request(request),
          m_evaluation(std::move(evaluation)) {}

    void start_async() override {
        if (m_evaluation) {
            OPENVINO_ASSERT(!m_pending, "Test request is still pending");
            m_pending = true;
            ++m_evaluation->pending;
            return;
        }
        complete();
    }

    void complete() {
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

    void wait() override {
        if (m_pending) {
            m_pending = false;
            --m_evaluation->pending;
            complete();
        }
    }

    bool wait_for(const std::chrono::milliseconds&) override {
        wait();
        return true;
    }

    void cancel() override {}

    void set_callback(std::function<void(std::exception_ptr)> callback) override {
        m_callback = std::move(callback);
    }

    void infer() override {
        OPENVINO_ASSERT(!m_pending, "Test request is still pending");
        m_request->infer();
    }

    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return m_request->get_profiling_info();
    }

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override {
        return m_request->get_tensor(port);
    }

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override {
        OPENVINO_ASSERT(!m_pending, "Cannot rebind a pending test request");
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
    std::shared_ptr<EvaluationLog> m_evaluation;
    bool m_pending = false;
};

    class FakeSubCompiledModel final : public ov::ICompiledModel {
public:
    FakeSubCompiledModel(const std::shared_ptr<ov::Model>& model,
                         const std::shared_ptr<const ov::IPlugin>& plugin,
                         std::shared_ptr<EvaluationLog> evaluation = {})
        : ov::ICompiledModel(model, plugin, nullptr, nullptr),
          m_model(model),
          m_evaluation(std::move(evaluation)) {}

    const std::shared_ptr<EvaluationLog>& evaluation_log() const {
        return m_evaluation;
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
        return std::make_shared<FakeSubAsyncInferRequest>(create_sync_infer_request(), m_evaluation);
    }

private:
    std::shared_ptr<ov::Model> m_model;
    std::shared_ptr<EvaluationLog> m_evaluation;
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
    // Sparse request pools run a warmup before bindings are populated.
    for (const auto& input : get_compiled_model()->inputs()) {
        auto tensor = ov::ISyncInferRequest::get_tensor(input);
        std::memset(tensor->data(), 0, tensor->get_byte_size());
    }
}

void FakeSubInferRequest::infer() {
    const auto compiled = std::static_pointer_cast<const FakeSubCompiledModel>(get_compiled_model());
    if (const auto& log = compiled->evaluation_log()) {
        ov::TensorVector inputs, outputs;
        for (const auto& port : compiled->inputs())
            inputs.push_back(ov::make_tensor(ov::ISyncInferRequest::get_tensor(port)));
        for (const auto& port : compiled->outputs())
            outputs.push_back(ov::make_tensor(ov::ISyncInferRequest::get_tensor(port)));
        OPENVINO_ASSERT(compiled->get_runtime_model()->evaluate(outputs, inputs), "Test model evaluation failed");
        std::lock_guard<std::mutex> lock(log->mutex);
        log->executions[compiled.get()].push_back(this);
        return;
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
    std::shared_ptr<testing::NiceMock<ov::MockICore>> make_core(const std::shared_ptr<const ov::IPlugin>& plugin,
                                                                std::shared_ptr<EvaluationLog> evaluation = {}) const {
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
            .WillByDefault([plugin, evaluation](const std::shared_ptr<const ov::Model>& submodel,
                                                const std::string&,
                                                const ov::AnyMap&) {
                return ov::SoPtr<ov::ICompiledModel>{
                    std::make_shared<FakeSubCompiledModel>(std::const_pointer_cast<ov::Model>(submodel),
                                                           plugin,
                                                           evaluation)};
            });

        return core;
    }
};

std::shared_ptr<ov::Model> build_moe_dispatch_test_model(size_t tokens, bool weight_offsets = false) {
    constexpr size_t experts = 4, hidden = 8, intermediate = 16, k = 2;
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{tokens, hidden});
    auto router = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{tokens, experts});
    auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{tokens, k});
    x->output(0).set_names({"hidden"});
    router->output(0).set_names({"logits"});
    scores->output(0).set_names({"scores"});
    auto topk = std::make_shared<ov::op::v11::TopK>(router,
                                                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {k}),
                                                    1,
                                                    ov::op::v11::TopK::Mode::MAX,
                                                    ov::op::v11::TopK::SortType::SORT_VALUES);
    ov::OutputVector outputs;
    for (size_t layer = 0; layer < 2; ++layer) {
        auto weight = [layer, experts, weight_offsets](size_t out, size_t in, size_t seed) -> ov::Output<ov::Node> {
            std::vector<float> values(experts * out * in);
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = (static_cast<float>((i + seed + layer + i / (out * in)) % 11) - 5.0f) / 16.0f;
            if (!weight_offsets)
                return ov::op::v0::Constant::create(ov::element::f32, ov::Shape{experts, out, in}, values);

            const ov::Shape grouped_shape{experts, out, in / 4, 4};
            std::vector<int8_t> packed_values(values.size());
            for (size_t i = 0; i < packed_values.size(); ++i)
                packed_values[i] = static_cast<int8_t>((i + seed + layer + i / (out * in)) % 8) - 4;
            auto packed = ov::op::v0::Constant::create(ov::element::i4, grouped_shape, packed_values);
            const ov::Shape offset_shape{1, out, in / 4, 1};
            const ov::Shape scale_shape{experts, out, in / 4, 1};
            std::vector<float> offsets(ov::shape_size(offset_shape)), scales(ov::shape_size(scale_shape));
            for (size_t i = 0; i < offsets.size(); ++i)
                offsets[i] = static_cast<float>(1 + (i + layer) % 5) / 8.0f;
            for (size_t i = 0; i < scales.size(); ++i)
                scales[i] = static_cast<float>(1 + (i + seed) % 7) / 32.0f;
            auto zero_point = ov::op::v0::Constant::create(ov::element::f32, offset_shape, offsets);
            auto shifted = std::make_shared<ov::op::v1::Subtract>(
                std::make_shared<ov::op::v0::Convert>(packed, ov::element::f32), zero_point);
            auto scaled = std::make_shared<ov::op::v1::Multiply>(
                shifted, ov::op::v0::Constant::create(ov::element::f32, scale_shape, scales));
            auto biased = std::make_shared<ov::op::v1::Add>(scaled, zero_point);
            return std::make_shared<ov::op::v1::Reshape>(
                biased,
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {experts, out, in}),
                false);
        };
        auto tile = std::make_shared<ov::op::v0::Tile>(
            x,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {experts, size_t{1}}));
        auto expanded = std::make_shared<ov::op::v1::Reshape>(
            tile,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {experts, tokens, hidden}),
            false);
        auto gate = std::make_shared<ov::op::v0::MatMul>(expanded, weight(intermediate, hidden, 1), false, true);
        auto up = std::make_shared<ov::op::v0::MatMul>(expanded, weight(intermediate, hidden, 2), false, true);
        auto activated = std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v4::Swish>(gate), up);
        auto down = std::make_shared<ov::op::v0::MatMul>(activated, weight(hidden, intermediate, 3), false, true);
        auto expert = std::make_shared<ov::op::v1::Reshape>(
            down,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {experts, tokens, hidden}),
            false);
        auto scatter = std::make_shared<ov::op::v12::ScatterElementsUpdate>(
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{tokens, experts}, {0.0f}),
            topk->output(1),
            scores,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
        auto transposed = std::make_shared<ov::op::v1::Transpose>(
            scatter,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0}));
        auto mixing = std::make_shared<ov::op::v1::Reshape>(
            transposed,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {experts, tokens, size_t{1}}),
            false);
        outputs.push_back(
            std::make_shared<ov::op::v1::ReduceSum>(std::make_shared<ov::op::v1::Multiply>(expert, mixing),
                                                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
                                                    false));
    }
    return std::make_shared<ov::Model>(outputs, ov::ParameterVector{x, router, scores}, "moe_dispatch_regression");
}

class MoEWeightChainDispatchTest : public SubgraphBehaviorInferTest, public ::testing::WithParamInterface<bool> {};

TEST_P(MoEWeightChainDispatchTest, MoESparseDispatchPreservesScoresGlobalInputsAndCachedRequests) {
    for (const size_t tokens : {size_t{1}, size_t{7}}) {
        SCOPED_TRACE(tokens);
        auto model = build_moe_dispatch_test_model(tokens, GetParam());
        auto evaluation = std::make_shared<EvaluationLog>();
        auto plugin = std::make_shared<TestPlugin>();
        auto core = make_core(plugin, evaluation);
        plugin->set_core(core);
        ov::AnyMap props{{"NPUW_DEVICES", "CPU"},
                         {"NPUW_FALLBACK_EXEC", "NO"},
                         {"NPUW_FOLD", "YES"},
                         {"NPUW_FUNCALL_FOR_ALL", "YES"},
                         {"NPUW_UNFOLD_IREQS", "NO"},
                         {"NPUW_ONLINE_PIPELINE", "REP"},
                         {"NPUW_ONLINE_ISOLATE", "MOE"},
                         {"NPUW_ONLINE_KEEP_BLOCKS_TAGGED", "expert"},
                         {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "4"},
                         {"NPUW_F16IC", "NO"},
                         {"NPUW_DQ", "NO"},
                         {"NPUW_MOE_POOL_SIZE", "2"},
                         {"NPUW_MOE_TOKEN_CHUNK_SIZE", "3"}};
        auto compiled = std::make_shared<ov::npuw::CompiledModel>(model->clone(), plugin, props);
        std::set<const ov::ICompiledModel*> expert_models;
        size_t expert_calls = 0;
        for (const auto& desc : compiled->m_compiled_submodels) {
            const auto real = desc.replaced_by ? &compiled->m_compiled_submodels.at(*desc.replaced_by) : &desc;
            if (const auto* state = ov::npuw::moe::get_compiled_experts(real->pipeline.context)) {
                ++expert_calls;
                EXPECT_EQ(state->num_active_experts, 2u);
                EXPECT_EQ(state->input_token_count, tokens);
                for (const auto& [chunk, submodel] : state->_compiled_models)
                    expert_models.insert(submodel._ptr.get());
            }
        }
        ASSERT_EQ(expert_calls, 2u) << "Must test the actual sparse executor, not a dense fallback";
        auto request = compiled->create_infer_request();
        evaluation->executions.clear();  // Exclude request-pool warmups.
        const auto history = [&]() {
            std::vector<const ov::ISyncInferRequest*> result;
            std::lock_guard<std::mutex> lock(evaluation->mutex);
            for (const auto* submodel : expert_models) {
                const auto& calls = evaluation->executions[submodel];
                result.insert(result.end(), calls.begin(), calls.end());
            }
            return result;
        };
        ov::TensorVector inputs;
        for (const auto& port : model->inputs())
            inputs.emplace_back(port.get_element_type(), port.get_shape());
        const auto check = [&]() {
            ov::TensorVector expected;
            for (const auto& port : model->outputs())
                expected.emplace_back(port.get_element_type(), port.get_shape());
            ASSERT_TRUE(model->evaluate(expected, inputs));
            for (size_t i = 0; i < inputs.size(); ++i)
                request->set_tensor(compiled->inputs()[i], ov::get_tensor_impl(inputs[i]));
            request->infer();
            EXPECT_EQ(evaluation->pending.load(), 0u);
            for (size_t i = 0; i < expected.size(); ++i) {
                const auto actual = request->get_tensor(compiled->outputs()[i]);
                ASSERT_EQ(actual->get_shape(), expected[i].get_shape());
                for (size_t element = 0; element < expected[i].get_size(); ++element)
                    EXPECT_NEAR(actual->data<float>()[element], expected[i].data<float>()[element], 1e-5f);
            }
        };
        const std::vector<std::array<float, 2>> scores{{0.7f, 0.3f},
                                                       {0.0f, 0.9f},
                                                       {-0.4f, 0.8f},
                                                       {0.0f, 0.0f},
                                                       {0.6f, 0.0f},
                                                       {0.7f, 0.3f},
                                                       {0.0f, 0.9f},
                                                       {1e-7f, -2e-8f}};
        for (size_t iteration = 0; iteration < scores.size(); ++iteration) {
            for (size_t i = 0; i < inputs[0].get_size(); ++i)
                inputs[0].data<float>()[i] = (static_cast<float>((i + iteration) % 9) - 4.0f) / 8.0f;
            std::fill_n(inputs[1].data<float>(), inputs[1].get_size(), -2.0f);
            for (size_t token = 0; token < tokens; ++token) {
                const size_t first = (iteration % 2 == 0 ? 0 : 1) + token;
                const size_t second = (iteration % 2 == 0 ? 1 : 3) + token;
                inputs[1].data<float>()[4 * token + first % 4] = 2.0f;
                inputs[1].data<float>()[4 * token + second % 4] = 1.0f;
                inputs[2].data<float>()[2 * token] = scores[iteration][0];
                inputs[2].data<float>()[2 * token + 1] = scores[iteration][1];
            }
            const auto before = history().size();
            check();
            if (scores[iteration][0] == 0.0f && scores[iteration][1] == 0.0f) {
                EXPECT_EQ(history().size(), before);
                for (const auto& port : compiled->outputs()) {
                    const auto tensor = request->get_tensor(port);
                    for (size_t i = 0; i < ov::shape_size(tensor->get_shape()); ++i)
                        EXPECT_FLOAT_EQ(tensor->data<float>()[i], 0.0f);
                }
            } else if (tokens == 1) {
                EXPECT_EQ(history().size() - before, 2u);  // One K-expert inference per layer.
            } else {
                EXPECT_GT(history().size(), before);
            }
        }
        if (tokens == 1) {
            const auto calls = history();
            ASSERT_EQ(calls.size(), 14u);
            // The first and third iterations select the same expert set with
            // different scores, separated by a different cached selection.
            EXPECT_EQ(calls[0], calls[4]);
            EXPECT_EQ(calls[1], calls[5]);
            EXPECT_NE(calls[0], calls[2]);
        } else {
            // Expert 0 starts before parse-ahead sees expert 1's invalid score.
            std::fill_n(inputs[1].data<float>(), inputs[1].get_size(), -2.0f);
            for (size_t token = 0; token < tokens; ++token) {
                inputs[1].data<float>()[4 * token] = 2.0f;
                inputs[1].data<float>()[4 * token + 1] = 1.0f;
                inputs[2].data<float>()[2 * token] = 0.7f;
                inputs[2].data<float>()[2 * token + 1] = std::numeric_limits<float>::quiet_NaN();
            }
            EXPECT_THROW(request->infer(), ov::Exception);
            EXPECT_EQ(evaluation->pending.load(), 0u) << "Exceptional prefill must drain outstanding chunks";
            for (size_t token = 0; token < tokens; ++token)
                inputs[2].data<float>()[2 * token + 1] = 0.3f;
            check();  // The same request remains usable after validation failure.
        }
    }
}

INSTANTIATE_TEST_SUITE_P(PlainAndOffsetWeights, MoEWeightChainDispatchTest, ::testing::Bool());

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
    EXPECT_EQ(count_dyn_attn_behaviors(compiled), 0u)
        << "DynAttnBehavior must NOT be attached when NPUW_ATTN=STATIC";
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

}  // namespace
