// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe/moe_executor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "compiled_model.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace {

struct ExecutionLog {
    std::mutex mutex;
    std::vector<const ov::IAsyncInferRequest*> calls;
    std::atomic<size_t> pending{0};
    bool fail_wait = false;
};

// Deliberately prepared expert executables: these tests exercise the existing
// executor contract without requiring topology matching, unrolling or an LLM.
// All request operations used by MoEExecutor are overridden; no base pipeline
// or synchronous request is needed for this deferred, evaluatable fixture.
class EvaluatedRequest final : public ov::IAsyncInferRequest {
public:
    EvaluatedRequest(std::shared_ptr<const ov::ICompiledModel> compiled,
                     std::shared_ptr<ov::Model> model,
                     std::shared_ptr<ExecutionLog> log)
        : ov::IAsyncInferRequest(nullptr, nullptr, nullptr),
          m_compiled(std::move(compiled)),
          m_model(std::move(model)),
          m_log(std::move(log)) {
        for (const auto& port : m_compiled->inputs())
            allocate(port);
        for (const auto& port : m_compiled->outputs())
            allocate(port);
    }

    void infer() override {
        OPENVINO_ASSERT(!m_pending, "Cannot infer a pending request");
        ov::TensorVector inputs, outputs;
        for (const auto& port : get_inputs())
            inputs.push_back(ov::make_tensor(get_tensor(port)));
        for (const auto& port : get_outputs())
            outputs.push_back(ov::make_tensor(get_tensor(port)));
        OPENVINO_ASSERT(m_model->evaluate(outputs, inputs), "Expert evaluation failed");
        std::lock_guard<std::mutex> lock(m_log->mutex);
        m_log->calls.push_back(this);
    }
    void start_async() override {
        OPENVINO_ASSERT(!m_pending, "Request already pending");
        m_pending = true;
        ++m_log->pending;
    }
    void wait() override {
        if (!m_pending)
            return;
        m_pending = false;
        --m_log->pending;
        if (m_log->fail_wait) {
            m_log->fail_wait = false;
            OPENVINO_THROW("Deferred expert failure");
        }
        infer();
    }
    bool wait_for(const std::chrono::milliseconds&) override {
        wait();
        return true;
    }
    void cancel() override {}
    void set_callback(std::function<void(std::exception_ptr)>) override {}
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override {
        return m_tensors.at(port);
    }
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override {
        OPENVINO_ASSERT(!m_pending, "Cannot rebind a pending request");
        m_tensors.at(port) = tensor;
    }
    const std::vector<ov::Output<const ov::Node>>& get_inputs() const override {
        return m_compiled->inputs();
    }
    const std::vector<ov::Output<const ov::Node>>& get_outputs() const override {
        return m_compiled->outputs();
    }
    const std::shared_ptr<const ov::ICompiledModel>& get_compiled_model() const override {
        return m_compiled;
    }

private:
    void allocate(const ov::Output<const ov::Node>& port) {
        ov::Tensor tensor(port.get_element_type(), port.get_shape());
        std::memset(tensor.data(), 0, tensor.get_byte_size());
        m_tensors.emplace(port, ov::get_tensor_impl(tensor));
    }
    std::shared_ptr<const ov::ICompiledModel> m_compiled;
    std::shared_ptr<ov::Model> m_model;
    std::shared_ptr<ExecutionLog> m_log;
    std::map<ov::Output<const ov::Node>, ov::SoPtr<ov::ITensor>> m_tensors;
    bool m_pending = false;
};

class EvaluatedModel final : public ov::ICompiledModel {
public:
    EvaluatedModel(const std::shared_ptr<ov::Model>& model, std::shared_ptr<ExecutionLog> log)
        : ov::ICompiledModel(model, std::make_shared<ov::test::npuw::NullPlugin>()),
          m_model(model),
          m_log(std::move(log)) {}
    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override {
        return std::make_shared<EvaluatedRequest>(shared_from_this(), m_model, m_log);
    }
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        OPENVINO_THROW("Only evaluated async requests are used by this fixture");
    }
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_model;
    }
    void export_model(std::ostream&) const override {}
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&) const override {
        return {};
    }

private:
    std::shared_ptr<ov::Model> m_model;
    std::shared_ptr<ExecutionLog> m_log;
};

class ExecutorHarness final : public ov::npuw::ISubrequestAccessor {
public:
    static constexpr size_t experts = 4, selected = 2, hidden = 4, chunk = 3;

    ExecutorHarness(size_t tokens, const ov::element::Type& score_type, size_t pool)
        : token_count(tokens),
          scores(score_type, ov::Shape{experts, tokens, 1}),
          weights(ov::element::f32, ov::Shape{experts, 1, hidden}),
          log(std::make_shared<ExecutionLog>()) {
        const bool decode = tokens == 1;
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{decode ? 1 : chunk, hidden});
        ov::ParameterVector params{x};
        ov::OutputVector contributions;
        auto state = std::make_shared<ov::npuw::compiled::MoEExperts>();
        state->num_experts = experts;
        state->num_active_experts = selected;
        state->expert_hidden_dim = hidden;
        state->input_token_count = tokens;
        state->_expert_input = {0, 0};
        state->_router_scores = {1, 1};
        for (size_t slot = 0; slot < (decode ? selected : 1); ++slot) {
            auto score = std::make_shared<ov::op::v0::Parameter>(score_type, ov::Shape{1, decode ? 1 : chunk, 1});
            auto weight = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, hidden});
            state->_param_mapping[1].push_back(params.size());
            params.push_back(score);
            state->_param_mapping[2].push_back(params.size());
            params.push_back(weight);
            contributions.push_back(
                std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v1::Multiply>(x, weight),
                                                       std::make_shared<ov::op::v0::Convert>(score, ov::element::f32)));
        }
        ov::Output<ov::Node> output = contributions[0];
        if (decode) {
            output = std::make_shared<ov::op::v1::Add>(output, contributions[1]);
            output = std::make_shared<ov::op::v0::Squeeze>(
                output,
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
        }
        auto model = std::make_shared<ov::Model>(ov::OutputVector{output}, params);
        desc.compiled_model = {std::make_shared<EvaluatedModel>(model, log), {}};
        state->_compiled_models[decode ? 0 : chunk] = desc.compiled_model;
        ov::npuw::moe::put_compiled_experts(desc.pipeline.context, state);
        desc.replaced_by = 0;
        desc.param_base = 2;
        desc.closure.get().closure = {weights};
        for (size_t expert = 0; expert < experts; ++expert)
            std::fill_n(weights.data<float>() + expert * hidden, hidden, 1024.0f * static_cast<float>(expert + 1));
        request = {desc.compiled_model->create_infer_request(), {}};
        executor = std::make_unique<ov::npuw::moe::MoEExecutor>(
            *this,
            [this](const ov::element::Type& type, const ov::Shape& shape, const std::string& device) {
                auto result = allocate_mem(type, shape, device);
                accumulator = ov::make_tensor(result);
                return result;
            });
        executor->prepare(0, 0, 1, decode ? pool : 0);
        log->calls.clear();  // Exclude pool warmups.
    }

    void set_score(size_t expert, size_t token, float value) {
        const auto offset = expert * token_count + token;
        if (scores.get_element_type() == ov::element::f32)
            scores.data<float>()[offset] = value;
        else
            scores.data<ov::float16>()[offset] = ov::float16(value);
    }
    float score(size_t expert, size_t token) const {
        const auto offset = expert * token_count + token;
        return scores.get_element_type() == ov::element::f32 ? scores.data<float>()[offset]
                                                             : static_cast<float>(scores.data<ov::float16>()[offset]);
    }
    void bind(float input_value) {
        // New input AND output tensors each time detect stale bindings on cache hits.
        input = ov::Tensor(ov::element::f32, ov::Shape{token_count, hidden});
        output = ov::Tensor(ov::element::f32, ov::Shape{1, hidden});
        std::fill_n(input.data<float>(), input.get_size(), input_value);
        std::fill_n(output.data<float>(), output.get_size(), -100.0f);
        executor->function_prologue_moe_input(0, 0, 0, ov::get_tensor_impl(input));
        executor->function_prologue_moe_input(0, 0, 1, ov::get_tensor_impl(scores));
        executor->function_prologue_moe_output(0, 0, ov::get_tensor_impl(output));
    }
    void check() {
        executor->run(0, 0);
        EXPECT_EQ(log->pending.load(), 0u);
        for (size_t token = 0; token < token_count; ++token) {
            for (size_t h = 0; h < hidden; ++h) {
                float expected = 0;
                for (size_t expert = 0; expert < experts; ++expert)
                    expected += input.data<float>()[token * hidden + h] * weights.data<float>()[expert * hidden + h] *
                                score(expert, token);
                float actual = output.data<float>()[h];
                if (token_count != 1) {
                    actual = 0;
                    for (size_t slot = 0; slot < selected; ++slot)
                        actual += accumulator.data<float>()[(slot * token_count + token) * hidden + h];
                }
                EXPECT_NEAR(actual, expected, 1e-5f);
            }
        }
    }

    ov::SoPtr<ov::IAsyncInferRequest> get_subrequest(size_t) override {
        return request;
    }
    const void* get_submodel_desc(size_t) override {
        return &desc;
    }
    ov::npuw::util::TensorPtr allocate_mem(const ov::element::Type& type,
                                           const ov::Shape& shape,
                                           const std::string&) override {
        return ov::get_tensor_impl(ov::Tensor(type, shape));
    }
    bool is_gather_closure(size_t, size_t) override {
        return false;
    }
    bool unpack_required(size_t, size_t) override {
        return false;
    }
    bool needs_copy_closure(size_t, size_t) override {
        return false;
    }
    std::string subgraph_device(size_t) override {
        return "CPU";
    }

    size_t token_count;
    ov::Tensor scores, weights, input, output, accumulator;
    std::shared_ptr<ExecutionLog> log;
    std::unique_ptr<ov::npuw::moe::MoEExecutor> executor;

private:
    ov::npuw::CompiledModelDescTestAccessor::SubmodelDesc desc;
    ov::SoPtr<ov::IAsyncInferRequest> request;
};

class MoEExecutorRuntimeTest : public ::testing::TestWithParam<std::tuple<bool, size_t, size_t>> {};

TEST_P(MoEExecutorRuntimeTest, PreservesScoresAndBindingsAcrossSelections) {
    const auto [half_scores, tokens, pool] = GetParam();
    ExecutorHarness harness(tokens, half_scores ? ov::element::f16 : ov::element::f32, pool);
    const std::vector<std::array<float, 2>> coefficients{{0.5f, 0.25f},
                                                         {0.0f, 0.5f},
                                                         {-0.5f, 0.25f},
                                                         {0.0f, 0.0f},
                                                         {0.5f, 0.0f},
                                                         {0.5f, 0.25f},
                                                         {0.0f, 0.5f},
                                                         {0x1p-24f, -0x1p-23f}};
    for (size_t iteration = 0; iteration < coefficients.size(); ++iteration) {
        std::memset(harness.scores.data(), 0, harness.scores.get_byte_size());
        for (size_t token = 0; token < tokens; ++token) {
            const size_t a = ((iteration % 2 == 0 ? 0 : 1) + token) % 4;
            const size_t b = ((iteration % 2 == 0 ? 1 : 3) + token) % 4;
            harness.set_score(a, token, coefficients[iteration][0]);
            harness.set_score(b, token, coefficients[iteration][1]);
        }
        harness.bind(static_cast<float>(iteration + 1) / 8.0f);
        const auto before = harness.log->calls.size();
        harness.check();
        if (iteration == 3)
            EXPECT_EQ(harness.log->calls.size(), before);
        else if (tokens == 1)
            EXPECT_EQ(harness.log->calls.size(), before + 1);
        else
            EXPECT_GT(harness.log->calls.size(), before);
    }
    if (tokens == 1 && pool > 0) {
        ASSERT_EQ(harness.log->calls.size(), 7u);
        EXPECT_EQ(harness.log->calls[0], harness.log->calls[2]);
        EXPECT_NE(harness.log->calls[0], harness.log->calls[1]);
    }
}

TEST_P(MoEExecutorRuntimeTest, RejectsNonfiniteScoresInsteadOfReturningZeroAndRemainsReusable) {
    const auto [half_scores, tokens, pool] = GetParam();
    ExecutorHarness harness(tokens, half_scores ? ov::element::f16 : ov::element::f32, pool);
    for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity()}) {
        SCOPED_TRACE(invalid);
        for (const bool all_nonfinite : {true, false}) {
            SCOPED_TRACE(all_nonfinite);
            std::memset(harness.scores.data(), 0, harness.scores.get_byte_size());
            if (all_nonfinite) {
                for (size_t expert = 0; expert < ExecutorHarness::experts; ++expert) {
                    for (size_t token = 0; token < tokens; ++token)
                        harness.set_score(expert, token, invalid);
                }
            } else {
                // Prefill can already have work in flight when a later expert's
                // row fails validation; decode must validate before dispatch.
                for (size_t token = 0; token < tokens; ++token) {
                    harness.set_score(0, token, 0.5f);
                    harness.set_score(1, token, invalid);
                }
            }
            harness.bind(0.5f);
            const auto before = harness.log->calls.size();
            try {
                harness.executor->run(0, 0);
                FAIL() << "Non-finite routing scores must not return a successful zero result";
            } catch (const ov::Exception& error) {
                EXPECT_NE(std::string(error.what()).find("MoE router produced a non-finite mixing score"),
                          std::string::npos);
            }
            EXPECT_EQ(harness.log->pending.load(), 0u);
            if (tokens == 1 || all_nonfinite)
                EXPECT_EQ(harness.log->calls.size(), before);
            else
                EXPECT_GT(harness.log->calls.size(), before);

            // A finite all-zero tensor remains a valid, no-inference case.
            const auto after_failure = harness.log->calls.size();
            std::memset(harness.scores.data(), 0, harness.scores.get_byte_size());
            harness.bind(0.25f);
            harness.check();
            EXPECT_EQ(harness.log->calls.size(), after_failure);

            // Reuse the same executor/requests with new bindings and selections.
            for (size_t token = 0; token < tokens; ++token) {
                harness.set_score(2, token, 0.25f);
                harness.set_score(3, token, -0.5f);
            }
            harness.bind(0.75f);
            harness.check();
            EXPECT_GT(harness.log->calls.size(), after_failure);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(ScoreTypesAndModes,
                         MoEExecutorRuntimeTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Values(size_t{1}, size_t{7}),
                                            ::testing::Values(size_t{0}, size_t{2})));

TEST(MoEExecutorCleanupTest, DrainsDeferredRequestsAndRemainsReusableAfterFailures) {
    for (const auto type : {ov::element::f32, ov::element::f16}) {
        for (const bool inference_failure : {false, true}) {
            ExecutorHarness harness(7, type, 0);
            std::memset(harness.scores.data(), 0, harness.scores.get_byte_size());
            for (size_t token = 0; token < 7; ++token) {
                harness.set_score(0, token, 0.5f);
                harness.set_score(1, token, inference_failure ? 0.25f : std::numeric_limits<float>::quiet_NaN());
            }
            harness.bind(0.5f);
            harness.log->fail_wait = inference_failure;
            EXPECT_THROW(harness.executor->run(0, 0), ov::Exception);
            EXPECT_EQ(harness.log->pending.load(), 0u);
            for (size_t token = 0; token < 7; ++token)
                harness.set_score(1, token, 0.25f);
            harness.bind(0.25f);
            harness.check();
        }
    }
}

TEST(MoEExecutorUtilitiesTest, OnlyExactZeroIsInactiveAndNonfiniteScoresFail) {
    for (const float value : {0.0f, -0.0f})
        EXPECT_FALSE(ov::npuw::moe::is_nonzero(value));
    for (const float value : {1e-9f, -1e-10f, 0.5f})
        EXPECT_TRUE(ov::npuw::moe::is_nonzero(value));
    for (const float value : {std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()}) {
        EXPECT_THROW(ov::npuw::moe::is_nonzero(value), ov::Exception);
        EXPECT_THROW(ov::npuw::moe::is_nonzero(ov::float16(value)), ov::Exception);
    }
}

TEST(MoEExecutorUtilitiesTest, ValidatesExpertSlicesAndPackedBoundaries) {
    ov::Tensor aligned(ov::element::i4, ov::Shape{4, 2, 4});
    EXPECT_EQ(ov::npuw::moe::slice_expert_weight(aligned, 3, 4).get_shape(), (ov::Shape{1, 2, 4}));
    EXPECT_THROW(ov::npuw::moe::slice_expert_weight(aligned, 4, 4), ov::Exception);
    EXPECT_THROW(ov::npuw::moe::slice_expert_weight(aligned, 0, 0), ov::Exception);
    ov::Tensor unaligned(ov::element::i4, ov::Shape{4, 3, 1});
    EXPECT_THROW(ov::npuw::moe::slice_expert_weight(unaligned, 1, 4), ov::Exception);
}

}  // namespace
