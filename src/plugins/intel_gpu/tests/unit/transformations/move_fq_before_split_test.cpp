// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/move_fq_before_split.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <tuple>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/variadic_split.hpp"

using namespace ov::intel_gpu;

namespace ov::test::intel_gpu {

namespace {

const ov::PartialShape static_shape{1, 8, 4, 4};

std::shared_ptr<ov::Node> make_fq(const ov::Output<ov::Node>& data,
                                  float out_high = 10.f,
                                  const ov::Shape& range_shape = {},
                                  size_t levels = 256) {
    auto c = [&](float v) {
        return ov::op::v0::Constant::create(ov::element::f32, range_shape, {v});
    };
    return std::make_shared<ov::op::v0::FakeQuantize>(data, c(-0.25f), c(out_high), c(-0.25f), c(out_high), levels);
}

std::shared_ptr<ov::Node> make_split(const ov::Output<ov::Node>& data, bool variadic) {
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    if (variadic) {
        return std::make_shared<ov::op::v1::VariadicSplit>(data, axis, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {4, 4}));
    }
    return std::make_shared<ov::op::v1::Split>(data, axis, 2);
}

std::shared_ptr<ov::Node> make_conv(const ov::Output<ov::Node>& data) {
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4, 4, 1, 1}, {0.5f});
    return std::make_shared<ov::op::v1::Convolution>(data,
                                                     weights,
                                                     ov::Strides{1, 1},
                                                     ov::CoordinateDiff{0, 0},
                                                     ov::CoordinateDiff{0, 0},
                                                     ov::Strides{1, 1});
}

// yolo26 C3k2 block: Swish -> chunk(2) -> {FQ -> Concat, FQ -> {Concat, Conv, Conv}}.
std::shared_ptr<ov::Model> get_model(bool variadic, const ov::PartialShape& shape) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto swish = std::make_shared<ov::op::v4::Swish>(input);
    auto split = make_split(swish, variadic);
    auto fq0 = make_fq(split->output(0));
    auto fq1 = make_fq(split->output(1));
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{fq0, fq1, make_conv(fq1), make_conv(fq1)}, 1);
    return std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> get_model_ref(bool variadic, const ov::PartialShape& shape) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto swish = std::make_shared<ov::op::v4::Swish>(input);
    auto fq = make_fq(swish);
    auto split = make_split(fq, variadic);
    auto concat = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{split->output(0), split->output(1), make_conv(split->output(1)), make_conv(split->output(1))},
        1);
    return std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{input});
}

// Split -> {FQ0, FQ1} -> Concat, with FQ0 built from the given range shape/levels/high value.
std::shared_ptr<ov::Model> get_model_with_first_fq(const ov::PartialShape& shape,
                                                   const ov::Shape& range_shape,
                                                   float out_high,
                                                   size_t levels) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto relu = std::make_shared<ov::op::v0::Relu>(input);
    auto split = make_split(relu, false);
    auto fq0 = make_fq(split->output(0), out_high, range_shape, levels);
    auto fq1 = make_fq(split->output(1), 10.f, range_shape);
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{fq0, fq1}, 1);
    return std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{input});
}

class MoveFakeQuantizeBeforeSplitTest : public TransformationTestsF {
public:
    MoveFakeQuantizeBeforeSplitTest() {
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<MoveFakeQuantizeBeforeSplit>();
    }
};

using MoveFakeQuantizeBeforeSplitParams = std::tuple<bool /* variadic */, ov::PartialShape>;

class MoveFakeQuantizeBeforeSplitParamTest : public MoveFakeQuantizeBeforeSplitTest,
                                             public testing::WithParamInterface<MoveFakeQuantizeBeforeSplitParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MoveFakeQuantizeBeforeSplitParams>& info) {
        const auto& [variadic, shape] = info.param;
        std::ostringstream ss;
        ss << (variadic ? "VariadicSplit" : "Split") << "_shape=" << shape;
        return ss.str();
    }

protected:
    void SetUp() override {
        MoveFakeQuantizeBeforeSplitTest::SetUp();
        const auto& [variadic, shape] = GetParam();
        model = get_model(variadic, shape);
        model_ref = get_model_ref(variadic, shape);
    }
};

}  // namespace

TEST_P(MoveFakeQuantizeBeforeSplitParamTest, CompareWithRef) {}

INSTANTIATE_TEST_SUITE_P(smoke_MoveFakeQuantizeBeforeSplit,
                         MoveFakeQuantizeBeforeSplitParamTest,
                         ::testing::Combine(::testing::Values(false, true),
                                            ::testing::Values(static_shape, ov::PartialShape{-1, 8, -1, -1})),
                         MoveFakeQuantizeBeforeSplitParamTest::getTestCaseName);

TEST_F(MoveFakeQuantizeBeforeSplitTest, ResultConsumerNotMoved) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, static_shape);
    auto split = make_split(input, false);
    auto fq0 = make_fq(split->output(0));
    auto fq1 = make_fq(split->output(1));
    model = std::make_shared<ov::Model>(ov::OutputVector{fq0, fq1}, ov::ParameterVector{input});
}

TEST_F(MoveFakeQuantizeBeforeSplitTest, DifferentRangesNotMoved) {
    model = get_model_with_first_fq(static_shape, {}, 5.f, 256);
}

TEST_F(MoveFakeQuantizeBeforeSplitTest, DifferentLevelsNotMoved) {
    model = get_model_with_first_fq(static_shape, {}, 10.f, 16);
}

TEST_F(MoveFakeQuantizeBeforeSplitTest, PerChannelRangesNotMoved) {
    model = get_model_with_first_fq(static_shape, {1, 4, 1, 1}, 10.f, 256);
}

TEST_F(MoveFakeQuantizeBeforeSplitTest, RangesOfHigherRankMoved) {
    // FakeQuantize output keeps the data shape, so single-element ranges of any rank commute with the split.
    const ov::PartialShape shape{1, 8, 16};
    const ov::Shape range_shape{1, 1, 1, 1};
    model = get_model_with_first_fq(shape, range_shape, 10.f, 256);
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto relu = std::make_shared<ov::op::v0::Relu>(input);
    auto split = make_split(make_fq(relu, 10.f, range_shape), false);
    auto concat = std::make_shared<ov::op::v0::Concat>(split->outputs(), 1);
    model_ref = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{input});
}

TEST_F(MoveFakeQuantizeBeforeSplitTest, NonQuantizedConsumerNotMoved) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, static_shape);
    auto split = make_split(input, false);
    auto fq0 = make_fq(split->output(0));
    auto fq1 = make_fq(split->output(1));
    auto other = std::make_shared<ov::op::v0::Relu>(split->output(1));
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{fq0, fq1, other}, 1);
    model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{input});
}

}  // namespace ov::test::intel_gpu
