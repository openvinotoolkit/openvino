// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/rt_info/decompression.hpp"

namespace {
using namespace ov;

std::shared_ptr<Node> decompress(const Output<Node>& data) {
    auto convert = std::make_shared<op::v0::Convert>(data, element::f32);
    mark_as_decompression(convert);
    pass::disable_constant_folding(convert);
    pass::disable_constant_folding(data.get_node_shared_ptr());
    return convert;
}

class MoveDecompressionAfterSplitTest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<pass::MoveDecompressionAfterSplit>();
        manager.register_pass<pass::MoveDecompressionAfterSplit>();
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
};

using SplitParams = std::tuple<element::Type, bool, int64_t, bool>;

class MoveDecompressionAfterSplitParameterizedTest : public MoveDecompressionAfterSplitTest,
                                                     public testing::WithParamInterface<SplitParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SplitParams>& info) {
        const auto& [type, variadic, axis, shared] = info.param;
        return type.get_type_name() + std::string(variadic ? "_VariadicSplit" : "_Split") +
               (axis < 0 ? "_NegativeAxis" : "_PositiveAxis") + (shared ? "_SharedConvert" : "_SingleConsumer");
    }
};

TEST_P(MoveDecompressionAfterSplitParameterizedTest, FoldLowPrecisionWeights) {
    comparator.enable(FunctionsComparator::TENSOR_NAMES);
    const auto& [type, variadic, axis_value, shared] = GetParam();
    auto weights = op::v0::Constant::create(type, Shape{2, 6}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto convert = decompress(weights);
    auto axis = op::v0::Constant::create(element::i64, Shape{}, {axis_value});
    std::shared_ptr<Node> split;
    if (variadic) {
        auto lengths = op::v0::Constant::create(element::i64, Shape{2}, {2, -1});
        split = std::make_shared<op::v1::VariadicSplit>(convert, axis, lengths);
    } else {
        split = std::make_shared<op::v1::Split>(convert, axis, 2);
    }
    split->output(0).set_names({"gate"});
    split->output(1).set_names({"up"});
    auto outputs = split->outputs();
    if (shared)
        outputs.push_back(convert);
    model = std::make_shared<Model>(outputs, ParameterVector{});

    const size_t first_width = variadic ? 2 : 3;
    OutputVector expected;
    for (size_t i = 0; i < 2; ++i) {
        const auto start = i == 0 ? 0 : first_width;
        const auto width = i == 0 ? first_width : 6 - first_width;
        std::vector<float> values;
        for (size_t row = 0; row < 2; ++row)
            for (size_t column = start; column < start + width; ++column)
                values.push_back(static_cast<float>(row * 6 + column + 1));
        auto chunk = op::v0::Constant::create(type, Shape{2, width}, values);
        auto result = decompress(chunk);
        result->output(0).set_names({i == 0 ? "gate" : "up"});
        expected.push_back(result);
    }
    if (shared)
        expected.push_back(decompress(weights));
    model_ref = std::make_shared<Model>(expected, ParameterVector{});

    manager.register_pass<pass::ConstantFolding>();
}

INSTANTIATE_TEST_SUITE_P(
    CompressedWeights,
    MoveDecompressionAfterSplitParameterizedTest,
    testing::Combine(testing::Values(element::f16, element::bf16, element::f8e4m3, element::f8e5m2),
                     testing::Bool(),
                     testing::Values(1, -1),
                     testing::Bool()),
    MoveDecompressionAfterSplitParameterizedTest::getTestCaseName);

TEST_F(MoveDecompressionAfterSplitTest, UnmarkedConvertIsUnchanged) {
    auto weights = op::v0::Constant::create(element::bf16, Shape{2, 6}, {1});
    auto convert = std::make_shared<op::v0::Convert>(weights, element::f32);
    auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto split = std::make_shared<op::v1::Split>(convert, axis, 2);
    model = std::make_shared<Model>(split->outputs(), ParameterVector{});
}

TEST_F(MoveDecompressionAfterSplitTest, SubByteWeightsAreUnchanged) {
    auto weights = op::v0::Constant::create(element::f4e2m1, Shape{2, 6}, {1});
    auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto split = std::make_shared<op::v1::Split>(decompress(weights), axis, 2);
    model = std::make_shared<Model>(split->outputs(), ParameterVector{});
}

TEST_F(MoveDecompressionAfterSplitTest, DynamicAxisIsUnchanged) {
    auto weights = op::v0::Constant::create(element::bf16, Shape{2, 6}, {1});
    auto axis = std::make_shared<op::v0::Parameter>(element::i64, Shape{});
    auto split = std::make_shared<op::v1::Split>(decompress(weights), axis, 2);
    model = std::make_shared<Model>(split->outputs(), ParameterVector{axis});
}

TEST_F(MoveDecompressionAfterSplitTest, DynamicLengthsAreUnchanged) {
    auto weights = op::v0::Constant::create(element::bf16, Shape{2, 6}, {1});
    auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto lengths = std::make_shared<op::v0::Parameter>(element::i64, Shape{2});
    auto split = std::make_shared<op::v1::VariadicSplit>(decompress(weights), axis, lengths);
    model = std::make_shared<Model>(split->outputs(), ParameterVector{lengths});
}

TEST_F(MoveDecompressionAfterSplitTest, NonConstantWeightsAreUnchanged) {
    auto weights = std::make_shared<op::v0::Parameter>(element::bf16, Shape{2, 6});
    auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto split = std::make_shared<op::v1::Split>(decompress(weights), axis, 2);
    model = std::make_shared<Model>(split->outputs(), ParameterVector{weights});
}

TEST_F(MoveDecompressionAfterSplitTest, DisabledSplitIsUnchanged) {
    auto weights = op::v0::Constant::create(element::bf16, Shape{2, 6}, {1});
    auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto split = std::make_shared<op::v1::Split>(decompress(weights), axis, 2);
    pass::disable_constant_folding(split);
    model = std::make_shared<Model>(split->outputs(), ParameterVector{});
}
}  // namespace
