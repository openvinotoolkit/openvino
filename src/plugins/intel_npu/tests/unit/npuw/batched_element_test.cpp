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
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "v1/elements/batched.hpp"

namespace {

using ov::test::npuw::build_reranker_test_model;
using ov::test::npuw::NullPlugin;

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

        const auto port = get_outputs()[0];
        ov::Shape shape;
        for (const auto& dim : port.get_partial_shape()) {
            shape.push_back(dim.is_static() ? static_cast<std::size_t>(dim.get_length()) : std::size_t{1});
        }
        auto out = ov::get_tensor_impl(ov::Tensor(port.get_element_type(), shape));
        std::fill_n(static_cast<float*>(out->data()), out->get_size(), token);
        set_tensor(port, out);
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

    // Resize a named input on the request, in place, and return it.
    static ov::SoPtr<ov::ITensor> resize_input(const std::shared_ptr<ov::IAsyncInferRequest>& req,
                                               const ov::SoPtr<ov::ICompiledModel>& wrapped,
                                               const std::string& name,
                                               const ov::Shape& shape) {
        const auto tensor = req->get_tensor(port_named(wrapped->inputs(), name));
        tensor->set_shape(shape);
        return tensor;
    }

    // Bind the reranker inputs at the batch/length implied by `ids`. Only input_ids carries
    // values the mock reads; attention_mask, position_ids and beam_idx just need a matching
    // batch dimension, so they are sized but left unset.
    static void bind_inputs(const std::shared_ptr<ov::IAsyncInferRequest>& req,
                            const ov::SoPtr<ov::ICompiledModel>& wrapped,
                            const std::vector<std::vector<int64_t>>& ids) {
        const std::size_t batch = ids.size();
        const std::size_t len = ids.empty() ? 0 : ids.front().size();

        auto* tokens = static_cast<int64_t*>(resize_input(req, wrapped, "input_ids", {batch, len})->data());
        for (std::size_t r = 0; r < batch; ++r) {
            std::copy(ids[r].begin(), ids[r].end(), tokens + r * len);
        }
        resize_input(req, wrapped, "attention_mask", {batch, len});
        resize_input(req, wrapped, "position_ids", {batch, len});
        resize_input(req, wrapped, "beam_idx", {batch});
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
    bind_inputs(req, wrapped, ids);

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
    bind_inputs(req, wrapped, ids);

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

    resize_input(req, wrapped, "input_ids", {0, 4});

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

    bind_inputs(req, wrapped, {{1, 2}, {3, 4}, {5, 6}});
    resize_input(req, wrapped, "attention_mask", {2, 2});

    try {
        req->infer();
        FAIL() << "expected a batch-mismatch rejection";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(error_message(ex).find("neither the inferred batch size"), std::string::npos);
    }
    EXPECT_TRUE(m_recorder->events.empty());
}

}  // namespace
