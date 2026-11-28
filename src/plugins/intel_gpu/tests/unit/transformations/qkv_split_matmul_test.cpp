// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/qkv_split_matmul.hpp"

#include <sstream>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"

namespace ov::test::intel_gpu {
namespace {

constexpr int64_t L = 5, K = 4, P = 3, H = 2, D = 2, N = P * H * D, NP = H * D;

std::vector<int8_t> weight_values(int64_t first_col, int64_t cols) {
    std::vector<int8_t> values;
    for (int64_t k = 0; k < K; ++k)
        for (int64_t n = first_col; n < first_col + cols; ++n)
            values.push_back(static_cast<int8_t>((k * N + n) % 7 - 3));
    return values;
}

std::vector<int8_t> zp_values(int64_t first_col, int64_t cols) {
    std::vector<int8_t> values;
    for (int64_t n = first_col; n < first_col + cols; ++n)
        values.push_back(static_cast<int8_t>(n % 3 - 1));
    return values;
}

std::vector<float> scale_values(int64_t first_col, int64_t cols) {
    std::vector<float> values;
    for (int64_t n = first_col; n < first_col + cols; ++n)
        values.push_back(0.01f * static_cast<float>(n + 1));
    return values;
}

// Multiply(Subtract(Convert(w_i8), Reshape(Convert(zp_i8), [1, cols])), scale[1, cols]) for columns [first_col, first_col + cols)
ov::Output<ov::Node> dequantized_weight(int64_t first_col, int64_t cols) {
    const auto n = static_cast<size_t>(cols);
    auto weight = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{K, n}, weight_values(first_col, cols));
    auto zp = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{n}, zp_values(first_col, cols));
    auto zp_2d = std::make_shared<ov::op::v1::Reshape>(std::make_shared<ov::op::v0::Convert>(zp, ov::element::f32),
                                                       ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, cols}),
                                                       false);
    auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, n}, scale_values(first_col, cols));
    auto shifted = std::make_shared<ov::op::v1::Subtract>(std::make_shared<ov::op::v0::Convert>(weight, ov::element::f32), zp_2d);
    return std::make_shared<ov::op::v1::Multiply>(shifted, scale);
}

std::shared_ptr<ov::Node> gather_part(const ov::Output<ov::Node>& data, int64_t index) {
    return std::make_shared<ov::op::v8::Gather>(data,
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {index}),
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
}

// x[1, L, K] -> MatMul(W[K, P*H*D]) -> Reshape [1, L, P, H, D] -> Transpose(order) -> Gather(k) for k in `parts`
std::shared_ptr<ov::Model> getModel(const std::vector<int64_t>& parts,
                                    const ov::Output<ov::Node>& weight = dequantized_weight(0, N),
                                    const std::vector<int64_t>& order = {2, 0, 3, 1, 4}) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, L, K});
    ov::ParameterVector params{x};
    if (auto weight_param = ov::as_type_ptr<ov::op::v0::Parameter>(weight.get_node_shared_ptr()))
        params.push_back(weight_param);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(x, weight);
    auto split =
        std::make_shared<ov::op::v1::Reshape>(matmul, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{1, L, P, H, D}), false);
    auto transpose = std::make_shared<ov::op::v1::Transpose>(split, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, order));
    ov::OutputVector outputs;
    for (auto k : parts)
        outputs.push_back(gather_part(transpose, k));
    return std::make_shared<ov::Model>(outputs, params);
}

std::shared_ptr<ov::Model> getModelRef(const std::vector<int64_t>& parts) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, L, K});
    ov::OutputVector outputs;
    for (auto k : parts) {
        const auto part = k < 0 ? k + P : k;
        auto matmul = std::make_shared<ov::op::v0::MatMul>(x, dequantized_weight(part * NP, NP));
        auto heads =
            std::make_shared<ov::op::v1::Reshape>(matmul, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, H, D}), true);
        outputs.push_back(std::make_shared<ov::op::v1::Transpose>(heads, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3})));
    }
    return std::make_shared<ov::Model>(outputs, ov::ParameterVector{x});
}

class QKVSplitMatMulTests : public TransformationTestsF {
public:
    QKVSplitMatMulTests() {
        comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::intel_gpu::QKVSplitMatMul>();
    }
};

class QKVSplitMatMulPartsTests : public QKVSplitMatMulTests, public testing::WithParamInterface<std::vector<int64_t>> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::vector<int64_t>>& info) {
        std::ostringstream name;
        name << "parts";
        for (auto k : info.param)
            name << "_" << (k < 0 ? "m" : "") << std::abs(k);
        return name.str();
    }
};

}  // namespace

TEST_P(QKVSplitMatMulPartsTests, Split) {
    model = getModel(GetParam());
    model_ref = getModelRef(GetParam());
}

INSTANTIATE_TEST_SUITE_P(smoke_QKVSplitMatMul,
                         QKVSplitMatMulPartsTests,
                         testing::Values(std::vector<int64_t>{0, 1, 2}, std::vector<int64_t>{0, 2}, std::vector<int64_t>{-3, -1}),
                         QKVSplitMatMulPartsTests::getTestCaseName);

TEST_F(QKVSplitMatMulTests, OtherTransposeOrderNotSplit) {
    model = getModel({0, 1, 2}, dequantized_weight(0, N), {0, 2, 3, 1, 4});
}

// 0 and -3 are the same part, so it would be computed twice
TEST_F(QKVSplitMatMulTests, DuplicatePartsNotSplit) {
    model = getModel({0, -3});
}

TEST_F(QKVSplitMatMulTests, NonConstantWeightNotSplit) {
    model = getModel({0, 1, 2}, std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{K, N}));
}

// The fused MatMul would still be computed for the non-Gather consumer
TEST_F(QKVSplitMatMulTests, NonGatherConsumerNotSplit) {
    model = getModel({0, 1, 2});
    auto transpose = model->get_results()[0]->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(0);
    model->add_results({std::make_shared<ov::op::v0::Result>(std::make_shared<ov::op::v0::Relu>(transpose))});
}

}  // namespace ov::test::intel_gpu