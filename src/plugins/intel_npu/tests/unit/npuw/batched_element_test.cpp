// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Unit tests for the batched v1 element (ov::npuw::batched), a model-agnostic fan-out
// decorator that unrolls a batched [N, ...] request into N batch-1 inner inferences,
// resets inner variable state between rows, and stacks the per-row outputs back to [N, ...].
// The element itself is the unit under test; only the inner compiled model is mocked, which
// keeps the tests deviceless and lets them observe the per-row infer/reset. It is driven
// through a synthetic Qwen-style reranker (realistic multi-input, stateful signature), and the
// mock echoes each row's first token so a test can assert output row r reflects input row r.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "llm_test_helpers.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "v1/elements/batched.hpp"

namespace {

using ov::test::npuw::build_reranker_test_model;
using ov::test::npuw::NullPlugin;

// Per-output value offset the mock applies so multi-output tests can tell stacked outputs apart.
constexpr float kOutputChannelOffset = 1000.0f;

// Ordered log of the lifecycle the element drives on the inner: "reset" before each row,
// "infer" for each row. Mirrors the events-vector pattern in failsafe.cpp / accuracy_checked.cpp.
struct Recorder {
    std::vector<std::string> events;
};

class MockState : public ov::IVariableState {
public:
    explicit MockState(std::shared_ptr<Recorder> rec) : ov::IVariableState("mock_state"), m_rec(std::move(rec)) {}
    void reset() override {
        m_rec->events.emplace_back("reset");
    }
    void set_state(const ov::SoPtr<ov::ITensor>&) override {}
    ov::SoPtr<ov::ITensor> get_state() const override {
        return {};
    }

private:
    std::shared_ptr<Recorder> m_rec;
};

ov::Output<const ov::Node> port_named(const std::vector<ov::Output<const ov::Node>>& ports,
                                      const std::string& needle) {
    for (const auto& port : ports) {
        if (port.get_any_name().find(needle) != std::string::npos) {
            return port;
        }
    }
    OPENVINO_THROW("port not found: ", needle);
}

std::string error_message(const ov::Exception& ex) {
    return ex.what() == nullptr ? std::string{} : std::string(ex.what());
}

// Minimal two-output model (one input_ids input, two f32 outputs) for the multi-output test. The
// mock fills the outputs itself, so only the I/O signature matters.
std::shared_ptr<ov::Model> build_two_output_model() {
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    ids->set_friendly_name("input_ids");
    ids->output(0).set_names({"input_ids"});

    auto as_f32 = std::make_shared<ov::op::v0::Convert>(ids, ov::element::f32);
    auto two = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {2.0f});
    auto scaled = std::make_shared<ov::op::v1::Multiply>(as_f32, two);

    return std::make_shared<ov::Model>(ov::OutputVector{as_f32, scaled}, ov::ParameterVector{ids}, "two_output");
}

// Batch-1 inner stand-in: echoes the row's first input_ids token across its (generically
// shaped) output. Pure in its input, so the same row always yields the same value -- which
// is what lets callers check that batched scoring equals per-row scoring.
class MockInnerSync : public ov::ISyncInferRequest {
public:
    MockInnerSync(std::shared_ptr<const ov::ICompiledModel> compiled_model, std::shared_ptr<Recorder> rec)
        : ov::ISyncInferRequest(std::move(compiled_model)),
          m_rec(std::move(rec)),
          m_state(std::make_shared<MockState>(m_rec)) {}

    void infer() override {
        m_rec->events.emplace_back("infer");

        const auto ids = get_tensor(port_named(get_inputs(), "input_ids"));
        const float token = static_cast<float>(static_cast<const int64_t*>(ids->data())[0]);

        // Echo the row's token across every output; output k is offset by k so multi-output
        // tests can tell the stacked outputs apart.
        const auto& ports = get_outputs();
        for (std::size_t k = 0; k < ports.size(); ++k) {
            const auto& port = ports[k];
            ov::Shape shape;
            for (const auto& dim : port.get_partial_shape()) {
                shape.push_back(dim.is_static() ? static_cast<std::size_t>(dim.get_length()) : std::size_t{1});
            }
            auto out = ov::get_tensor_impl(ov::Tensor(port.get_element_type(), shape));
            std::fill_n(static_cast<float*>(out->data()),
                        out->get_size(),
                        token + static_cast<float>(k) * kOutputChannelOffset);
            set_tensor(port, out);
        }
    }

    void check_tensors() const override {}
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {ov::SoPtr<ov::IVariableState>(m_state)};
    }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }

private:
    std::shared_ptr<Recorder> m_rec;
    std::shared_ptr<MockState> m_state;
};

class MockInnerCompiled : public ov::ICompiledModel {
public:
    MockInnerCompiled(const std::shared_ptr<ov::Model>& model,
                      const std::shared_ptr<const ov::IPlugin>& plugin,
                      std::shared_ptr<Recorder> rec)
        : ov::ICompiledModel(model, plugin),
          m_rec(std::move(rec)) {}

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        return std::make_shared<MockInnerSync>(shared_from_this(), m_rec);
    }
    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override {
        return std::make_shared<ov::IAsyncInferRequest>(create_sync_infer_request(),
                                                        get_task_executor(),
                                                        get_callback_executor());
    }
    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return nullptr;
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&) const override {
        return {};
    }

private:
    std::shared_ptr<Recorder> m_rec;
};

class NPUWBatchedElementTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
        m_model = build_reranker_test_model();
        m_recorder = std::make_shared<Recorder>();
    }

    ov::SoPtr<ov::ICompiledModel> make_inner() {
        return {std::make_shared<MockInnerCompiled>(m_model, m_plugin, m_recorder), {}};
    }

    ov::SoPtr<ov::ICompiledModel> wrap(bool enabled) {
        return ov::npuw::batched::CompiledModel::create(m_model, m_plugin, make_inner(), enabled);
    }

    // Resize a named input in place and return it. Templated so it serves both the async wrapper
    // and a bare sync request -- both expose get_inputs()/get_tensor().
    template <class Req>
    static ov::SoPtr<ov::ITensor> resize_input(const std::shared_ptr<Req>& req,
                                               const std::string& name,
                                               const ov::Shape& shape) {
        const auto tensor = req->get_tensor(port_named(req->get_inputs(), name));
        tensor->set_shape(shape);
        return tensor;
    }

    // Set input_ids to [batch, len] and fill it row-major -- the only input the mock reads.
    template <class Req>
    static void set_input_ids(const std::shared_ptr<Req>& req, const std::vector<std::vector<int64_t>>& ids) {
        const std::size_t batch = ids.size();
        const std::size_t len = ids.empty() ? 0 : ids.front().size();
        auto* tokens = static_cast<int64_t*>(resize_input(req, "input_ids", {batch, len})->data());
        for (std::size_t r = 0; r < batch; ++r) {
            std::copy(ids[r].begin(), ids[r].end(), tokens + r * len);
        }
    }

    // Bind the full reranker input set; the non-input_ids inputs just need a matching batch dim.
    template <class Req>
    static void bind_inputs(const std::shared_ptr<Req>& req, const std::vector<std::vector<int64_t>>& ids) {
        const std::size_t batch = ids.size();
        const std::size_t len = ids.empty() ? 0 : ids.front().size();
        set_input_ids(req, ids);
        resize_input(req, "attention_mask", {batch, len});
        resize_input(req, "position_ids", {batch, len});
        resize_input(req, "beam_idx", {batch});
    }

    // First element of output row r (the echoed token), independent of the output's rank.
    static float row_value(const ov::SoPtr<ov::ITensor>& out, std::size_t row) {
        const std::size_t stride = out->get_size() / out->get_shape()[0];
        return static_cast<const float*>(out->data())[row * stride];
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
    std::shared_ptr<ov::Model> m_model;
    std::shared_ptr<Recorder> m_recorder;
};

TEST_F(NPUWBatchedElementTest, DisabledReturnsInnerUnwrapped) {
    auto inner = make_inner();
    auto wrapped = ov::npuw::batched::CompiledModel::create(m_model, m_plugin, inner, /*enabled=*/false);
    EXPECT_EQ(wrapped._ptr, inner._ptr);
}

TEST_F(NPUWBatchedElementTest, EnabledWrapsInner) {
    auto inner = make_inner();
    auto wrapped = ov::npuw::batched::CompiledModel::create(m_model, m_plugin, inner, /*enabled=*/true);
    EXPECT_NE(wrapped._ptr, inner._ptr);
    EXPECT_NE(std::dynamic_pointer_cast<ov::npuw::batched::CompiledModel>(wrapped._ptr), nullptr);
}

TEST_F(NPUWBatchedElementTest, EachRowScoredIndependentlyAndStacked) {
    auto wrapped = wrap(/*enabled=*/true);
    auto req = wrapped->create_infer_request();

    // Distinct first tokens so a misrouted row would surface as a wrong output value.
    const std::vector<std::vector<int64_t>> ids = {{11, 2, 3, 4}, {22, 5, 6, 7}, {33, 8, 9, 10}};
    bind_inputs(req, ids);

    req->infer();

    const auto out = req->get_tensor(wrapped->outputs()[0]);
    ASSERT_EQ(out->get_shape()[0], ids.size());
    for (std::size_t r = 0; r < ids.size(); ++r) {
        EXPECT_FLOAT_EQ(row_value(out, r), static_cast<float>(ids[r].front())) << "output row " << r;
    }

    // One inner inference per row, each preceded by a state reset, in order.
    EXPECT_EQ(m_recorder->events,
              (std::vector<std::string>{"reset", "infer", "reset", "infer", "reset", "infer"}));
}

TEST_F(NPUWBatchedElementTest, SingleRowScored) {
    auto wrapped = wrap(/*enabled=*/true);
    auto req = wrapped->create_infer_request();

    const std::vector<std::vector<int64_t>> ids = {{42, 8, 9, 10}};
    bind_inputs(req, ids);

    req->infer();

    const auto out = req->get_tensor(wrapped->outputs()[0]);
    ASSERT_EQ(out->get_shape()[0], 1u);
    EXPECT_FLOAT_EQ(row_value(out, 0), static_cast<float>(ids[0].front()));
    EXPECT_EQ(m_recorder->events, (std::vector<std::string>{"reset", "infer"}));
}

// A zero-row batch has nothing to score and would publish an unpopulated output.
TEST_F(NPUWBatchedElementTest, ZeroBatchIsRejected) {
    auto wrapped = wrap(/*enabled=*/true);
    auto req = wrapped->create_infer_request();

    resize_input(req, "input_ids", {0, 4});

    try {
        req->infer();
        FAIL() << "expected a zero-batch rejection";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(error_message(ex).find("batch size must be > 0"), std::string::npos);
    }
    EXPECT_TRUE(m_recorder->events.empty());
}

// input_ids fixes N=3; an attention_mask with batch 2 (neither N nor 1) cannot be sliced
// per row and must be rejected rather than fed whole into the batch-1 inner.
TEST_F(NPUWBatchedElementTest, MismatchedBatchRejected) {
    auto wrapped = wrap(/*enabled=*/true);
    auto req = wrapped->create_infer_request();

    bind_inputs(req, {{1, 2}, {3, 4}, {5, 6}});
    resize_input(req, "attention_mask", {2, 2});

    try {
        req->infer();
        FAIL() << "expected a batch-mismatch rejection";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(error_message(ex).find("neither the inferred batch size"), std::string::npos);
    }
    EXPECT_TRUE(m_recorder->events.empty());
}

// The production LLM/rerank wiring builds the element from a *sync* inner request (driven on the
// calling thread to dodge a nested-executor deadlock), bypassing batched::CompiledModel. Exercise
// that constructor directly.
TEST_F(NPUWBatchedElementTest, SyncInnerConstructorScoresEachRow) {
    auto inner_compiled = std::make_shared<MockInnerCompiled>(m_model, m_plugin, m_recorder);
    std::shared_ptr<ov::ISyncInferRequest> inner_sync = inner_compiled->create_sync_infer_request();
    std::shared_ptr<const ov::ICompiledModel> compiled = inner_compiled;
    auto req = std::make_shared<ov::npuw::batched::InferRequest>(compiled, std::move(inner_sync));

    const std::vector<std::vector<int64_t>> ids = {{11, 2, 3, 4}, {22, 5, 6, 7}, {33, 8, 9, 10}};
    bind_inputs(req, ids);

    req->infer();

    const auto out = req->get_tensor(req->get_outputs()[0]);
    ASSERT_EQ(out->get_shape()[0], ids.size());
    for (std::size_t r = 0; r < ids.size(); ++r) {
        EXPECT_FLOAT_EQ(row_value(out, r), static_cast<float>(ids[r].front())) << "output row " << r;
    }
    EXPECT_EQ(m_recorder->events,
              (std::vector<std::string>{"reset", "infer", "reset", "infer", "reset", "infer"}));
}

// A shared/broadcast input (batch 1) is fed to every row unsliced -- the slice guard only fires
// when the leading dim equals the batch, so a [1, L] input is never sliced out of range.
TEST_F(NPUWBatchedElementTest, BroadcastInputSharedAcrossRows) {
    auto wrapped = wrap(/*enabled=*/true);
    auto req = wrapped->create_infer_request();

    const std::vector<std::vector<int64_t>> ids = {{11, 1, 1, 1}, {22, 1, 1, 1}, {33, 1, 1, 1}};
    bind_inputs(req, ids);
    resize_input(req, "attention_mask", {1, 4});  // shared across the 3 rows

    req->infer();

    const auto out = req->get_tensor(wrapped->outputs()[0]);
    ASSERT_EQ(out->get_shape()[0], ids.size());
    for (std::size_t r = 0; r < ids.size(); ++r) {
        EXPECT_FLOAT_EQ(row_value(out, r), static_cast<float>(ids[r].front())) << "output row " << r;
    }
    EXPECT_EQ(m_recorder->events,
              (std::vector<std::string>{"reset", "infer", "reset", "infer", "reset", "infer"}));
}

// Regression: a leading batch-1 input (here input_ids, shared) must not pin the batch to 1 when a
// later input carries the real batch. The old "first input with a batch dim wins" logic set batch=1
// and rejected attention_mask as a mismatch; detection now takes the largest leading dim.
TEST_F(NPUWBatchedElementTest, BatchSizeFromNonLeadingInput) {
    auto wrapped = wrap(/*enabled=*/true);
    auto req = wrapped->create_infer_request();

    bind_inputs(req, {{7, 1, 1, 1}});             // input_ids = [1, 4], shared prompt (token 7)
    resize_input(req, "attention_mask", {3, 4});  // real batch N = 3 on a non-leading input

    req->infer();

    const auto out = req->get_tensor(wrapped->outputs()[0]);
    ASSERT_EQ(out->get_shape()[0], 3u);
    for (std::size_t r = 0; r < 3; ++r) {
        EXPECT_FLOAT_EQ(row_value(out, r), 7.0f) << "output row " << r;  // input_ids broadcast to every row
    }
    EXPECT_EQ(m_recorder->events,
              (std::vector<std::string>{"reset", "infer", "reset", "infer", "reset", "infer"}));
}

// Each model output is stacked into its own [N, ...] tensor. The mock offsets output k by k, so the
// two outputs must differ per row.
TEST_F(NPUWBatchedElementTest, MultipleOutputsStackedIndependently) {
    auto model = build_two_output_model();
    ov::SoPtr<ov::ICompiledModel> inner{std::make_shared<MockInnerCompiled>(model, m_plugin, m_recorder), {}};
    auto wrapped = ov::npuw::batched::CompiledModel::create(model, m_plugin, inner, /*enabled=*/true);
    auto req = wrapped->create_infer_request();

    const std::vector<std::vector<int64_t>> ids = {{11, 1}, {22, 1}, {33, 1}};
    set_input_ids(req, ids);  // the two-output model has only input_ids

    req->infer();

    ASSERT_EQ(wrapped->outputs().size(), 2u);
    const auto score = req->get_tensor(wrapped->outputs()[0]);
    const auto hidden = req->get_tensor(wrapped->outputs()[1]);
    ASSERT_EQ(score->get_shape()[0], ids.size());
    ASSERT_EQ(hidden->get_shape()[0], ids.size());
    for (std::size_t r = 0; r < ids.size(); ++r) {
        EXPECT_FLOAT_EQ(row_value(score, r), static_cast<float>(ids[r].front()));
        EXPECT_FLOAT_EQ(row_value(hidden, r), static_cast<float>(ids[r].front()) + kOutputChannelOffset);
    }
    EXPECT_EQ(m_recorder->events,
              (std::vector<std::string>{"reset", "infer", "reset", "infer", "reset", "infer"}));
}

}  // namespace
