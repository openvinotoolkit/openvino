// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pipelines/vision_merger/vision_merger_compiled_model.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"

namespace {

using ov::test::npuw::MockSubCompiledModel;
using ov::test::npuw::NullPlugin;

constexpr std::size_t kMaxLen = 8;
constexpr std::size_t kHidden = 4;
// "merged" keeps one row out of every kMergeRatio input rows, like the Qwen-VL patch merger.
constexpr std::size_t kMergeRatio = 4;

// hidden_states [N, 4] and attention_mask [1, N, N] in; "copy" [N, 4] and "merged" [N / 4, 4] out.
// N is dynamic for the public model and kMaxLen for the model wrapped by the merger.
std::shared_ptr<ov::Model> make_model(bool dynamic) {
    const ov::Dimension len = dynamic ? ov::Dimension::dynamic() : ov::Dimension(kMaxLen);
    auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{len, kHidden});
    hidden->output(0).set_names({"hidden_states"});
    auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, len, len});
    mask->output(0).set_names({"attention_mask"});

    auto copy = std::make_shared<ov::op::v0::Relu>(hidden);
    copy->output(0).set_names({"copy"});
    auto start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto stop = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {std::numeric_limits<int64_t>::max()});
    auto step = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(kMergeRatio)});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto merged = std::make_shared<ov::op::v8::Slice>(hidden, start, stop, step, axis);
    merged->output(0).set_names({"merged"});

    return std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(copy), std::make_shared<ov::op::v0::Result>(merged)},
        ov::ParameterVector{hidden, mask},
        "vision_merger_test_model");
}

// What the inner (static) model received on its last inference.
struct Seen {
    std::vector<float> hidden;
    std::vector<float> mask;
};

// Copies hidden_states to "copy" and every kMergeRatio-th row of it to "merged".
class ScriptedRequest final : public ov::ISyncInferRequest {
public:
    ScriptedRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model, std::shared_ptr<Seen> seen)
        : ov::ISyncInferRequest(compiled_model),
          m_seen(std::move(seen)) {
        for (const auto& ports : {get_inputs(), get_outputs()}) {
            for (const auto& port : ports) {
                ov::ISyncInferRequest::set_tensor(
                    port,
                    ov::get_tensor_impl(ov::Tensor(port.get_element_type(), port.get_shape())));
            }
        }
    }

    void infer() override {
        const auto hidden = get_tensor(get_inputs()[0]);
        const auto mask = get_tensor(get_inputs()[1]);
        const auto* hidden_data = static_cast<const float*>(hidden->data());
        const auto* mask_data = static_cast<const float*>(mask->data());
        m_seen->hidden.assign(hidden_data, hidden_data + ov::shape_size(hidden->get_shape()));
        m_seen->mask.assign(mask_data, mask_data + ov::shape_size(mask->get_shape()));

        std::memcpy(get_tensor(get_outputs()[0])->data(), hidden_data, hidden->get_byte_size());
        auto* merged = static_cast<float*>(get_tensor(get_outputs()[1])->data());
        for (std::size_t row = 0; row < kMaxLen / kMergeRatio; ++row) {
            std::memcpy(merged + row * kHidden, hidden_data + row * kMergeRatio * kHidden, kHidden * sizeof(float));
        }
    }

    void check_tensors() const override {}
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }

private:
    std::shared_ptr<Seen> m_seen;
};

class ScriptedInner final : public MockSubCompiledModel {
public:
    ScriptedInner(const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  std::shared_ptr<Seen> seen)
        : MockSubCompiledModel(model, plugin, {}),
          m_seen(std::move(seen)) {}

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        return std::make_shared<ScriptedRequest>(shared_from_this(), m_seen);
    }
    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override {
        return wrap_async_infer_request({});
    }

private:
    std::shared_ptr<Seen> m_seen;
};

class VisionMergerTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
        m_seen = std::make_shared<Seen>();
        auto inner = std::make_shared<ScriptedInner>(make_model(false), m_plugin, m_seen);
        m_compiled = std::make_shared<ov::npuw::VisionMergerCompiledModel>(make_model(true), inner, m_plugin, kMaxLen);
        m_request = m_compiled->create_sync_infer_request();
    }

    // hidden_states holds 1, 2, 3, ...; every attention_mask element is `mask_value`.
    void set_inputs(std::size_t hidden_len, std::size_t mask_len, float mask_value) {
        ov::Tensor hidden(ov::element::f32, ov::Shape{hidden_len, kHidden});
        std::iota(hidden.data<float>(), hidden.data<float>() + hidden.get_size(), 1.0f);
        ov::Tensor mask(ov::element::f32, ov::Shape{1, mask_len, mask_len});
        std::fill_n(mask.data<float>(), mask.get_size(), mask_value);
        m_request->set_tensor(m_compiled->inputs()[0], ov::get_tensor_impl(hidden));
        m_request->set_tensor(m_compiled->inputs()[1], ov::get_tensor_impl(mask));
    }

    std::shared_ptr<NullPlugin> m_plugin;
    std::shared_ptr<Seen> m_seen;
    std::shared_ptr<ov::npuw::VisionMergerCompiledModel> m_compiled;
    std::shared_ptr<ov::ISyncInferRequest> m_request;
};

TEST_F(VisionMergerTest, PublicPortsStayDynamic) {
    EXPECT_EQ(m_compiled->inputs()[0].get_partial_shape(), (ov::PartialShape{-1, kHidden}));
    EXPECT_EQ(m_compiled->inputs()[1].get_partial_shape(), (ov::PartialShape{1, -1, -1}));
    EXPECT_EQ(m_compiled->dynamic_input_dims()[0], (std::vector<std::size_t>{0}));
    EXPECT_EQ(m_compiled->dynamic_input_dims()[1], (std::vector<std::size_t>{1, 2}));
    // Callers read the static dimensions off the tensor of a dynamic port before filling it.
    EXPECT_EQ(m_request->get_tensor(m_compiled->inputs()[0])->get_shape(), (ov::Shape{0, kHidden}));
}

TEST_F(VisionMergerTest, ReportsItsOptions) {
    EXPECT_TRUE(m_compiled->get_property(ov::intel_npu::npuw::vision_merger::enabled.name()).as<bool>());
    EXPECT_EQ(m_compiled->get_property(ov::intel_npu::npuw::vision_merger::max_len.name()).as<uint64_t>(), kMaxLen);
}

TEST_F(VisionMergerTest, PadsInputsAndHidesPaddedKeys) {
    constexpr std::size_t len = 4;
    set_inputs(len, len, 0.5f);
    m_request->infer();

    ASSERT_EQ(m_seen->hidden.size(), kMaxLen * kHidden);
    for (std::size_t i = 0; i < m_seen->hidden.size(); ++i) {
        EXPECT_EQ(m_seen->hidden[i], i < len * kHidden ? static_cast<float>(i + 1) : 0.0f)
            << "hidden_states[" << i << "]";
    }

    ASSERT_EQ(m_seen->mask.size(), kMaxLen * kMaxLen);
    for (std::size_t row = 0; row < kMaxLen; ++row) {
        for (std::size_t col = 0; col < kMaxLen; ++col) {
            const float value = m_seen->mask[row * kMaxLen + col];
            if (row >= len) {
                EXPECT_EQ(value, 0.0f) << "padded query row " << row << ", col " << col;
            } else if (col >= len) {
                EXPECT_TRUE(std::isinf(value) && value < 0.0f) << "valid query row " << row << ", padded key " << col;
            } else {
                EXPECT_EQ(value, 0.5f) << "valid query row " << row << ", valid key " << col;
            }
        }
    }
}

TEST_F(VisionMergerTest, ReturnsOnlyTheValidRowsOfTheOutputs) {
    constexpr std::size_t len = 4;
    set_inputs(len, len, 0.0f);
    m_request->infer();

    const auto copy = m_request->get_tensor(m_compiled->outputs()[0]);
    ASSERT_EQ(copy->get_shape(), (ov::Shape{len, kHidden}));
    const auto* copy_data = static_cast<const float*>(copy->data());
    for (std::size_t i = 0; i < len * kHidden; ++i) {
        EXPECT_EQ(copy_data[i], static_cast<float>(i + 1)) << "copy[" << i << "]";
    }

    const auto merged = m_request->get_tensor(m_compiled->outputs()[1]);
    ASSERT_EQ(merged->get_shape(), (ov::Shape{len / kMergeRatio, kHidden}));
    const auto* merged_data = static_cast<const float*>(merged->data());
    for (std::size_t i = 0; i < kHidden; ++i) {
        EXPECT_EQ(merged_data[i], static_cast<float>(i + 1)) << "merged[" << i << "]";
    }
}

TEST_F(VisionMergerTest, FullLengthNeedsNoPadding) {
    set_inputs(kMaxLen, kMaxLen, 0.5f);
    m_request->infer();

    for (const float value : m_seen->mask) {
        EXPECT_EQ(value, 0.5f);
    }
    EXPECT_EQ(m_request->get_tensor(m_compiled->outputs()[1])->get_shape(),
              (ov::Shape{kMaxLen / kMergeRatio, kHidden}));
}

TEST_F(VisionMergerTest, ThrowsWhenDynamicDimensionsDisagree) {
    set_inputs(4, 3, 0.0f);
    OV_EXPECT_THROW_HAS_SUBSTRING(m_request->infer(), ov::Exception, "dynamic dimensions of the inputs disagree");
}

TEST_F(VisionMergerTest, ThrowsWhenInputExceedsMaxLen) {
    set_inputs(kMaxLen + 1, kMaxLen + 1, 0.0f);
    OV_EXPECT_THROW_HAS_SUBSTRING(m_request->infer(), ov::Exception, "do not fit NPUW_VISION_MERGER_MAX_LEN=8");
}

TEST_F(VisionMergerTest, ThrowsWhenOutputRowsAreFractional) {
    // 2 of 8 tokens would leave half a row of "merged".
    set_inputs(2, 2, 0.0f);
    OV_EXPECT_THROW_HAS_SUBSTRING(m_request->infer(), ov::Exception, "do not map to a whole number of rows");
}

TEST(VisionMergerCtorTest, ThrowsOnZeroMaxLen) {
    const auto plugin = std::make_shared<NullPlugin>();
    const ov::AnyMap properties{{ov::intel_npu::npuw::vision_merger::max_len.name(), std::size_t{0}}};
    OV_EXPECT_THROW_HAS_SUBSTRING(ov::npuw::VisionMergerCompiledModel(make_model(true), plugin, properties),
                                  ov::Exception,
                                  "NPUW_VISION_MERGER_MAX_LEN must be greater than 0");
}

TEST(VisionMergerCtorTest, ThrowsOnModelWithoutDynamicInputs) {
    const auto plugin = std::make_shared<NullPlugin>();
    OV_EXPECT_THROW_HAS_SUBSTRING(ov::npuw::VisionMergerCompiledModel(make_model(false), plugin, {}),
                                  ov::Exception,
                                  "no dynamic input dimension");
}

}  // namespace
