// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/transpose.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

using TransposeOrderParams = std::tuple<std::vector<int64_t>,  // allowed transpose orders
                                        ov::element::Type>;    // input precision
class TransposeMatmulFuseTest : public ::testing::Test, public testing::WithParamInterface<TransposeOrderParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TransposeOrderParams> obj) {
        std::vector<int64_t> target_order;
        ov::element::Type input_precision;

        std::tie(target_order, input_precision) = obj.param;

        std::ostringstream result;
        result << "transpose_order=[";
        for (const auto& order : target_order) {
            result << order << "_";
        }
        result << "]_input_precision=" << input_precision;
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> init_subgraph(ov::element::Type& input_precision, const std::vector<int64_t>& target_transpose_order) {
        ov::PartialShape input_a_shape = ov::PartialShape{-1, -1, -1, -1};
        ov::PartialShape input_b_shape = ov::PartialShape{-1, -1};

        auto input_a = std::make_shared<ov::op::v0::Parameter>(input_precision, input_a_shape);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(input_precision, input_b_shape);

        auto transpose_order_a = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{target_transpose_order.size()}, target_transpose_order);
        auto transpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, transpose_order_a);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(transpose_a, input_b, false, false);

        auto model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input_a, input_b});
        return model;
    }

private:
    ov::element::Type input_precision = ov::element::f16;
    std::vector<int64_t> order = {0, 1, 2, 3};
};

TEST_P(TransposeMatmulFuseTest, smoke_allowed_transpose_order) {
    std::vector<int64_t> target_order;
    ov::element::Type input_precision;
    std::tie(target_order, input_precision) = GetParam();
    auto function = init_subgraph(input_precision, target_order);

    std::string targetDevice = ov::test::utils::DEVICE_GPU;
    ov::Shape input_a_shape = {5, 6, 7, 8};
    ov::Shape input_b_shape = {8, 7};
    ov::Shape permuted_shape_b(input_b_shape.size());
    for (size_t i = 0; i < permuted_shape_b.size(); ++i) {
        permuted_shape_b[i] = input_a_shape[target_order[target_order.size() - 1 - i]];
    }

    auto input_tensor_a = ov::test::utils::create_and_fill_tensor(input_precision, input_a_shape, 0.0f, 1.0f);
    auto input_tensor_b = ov::test::utils::create_and_fill_tensor(input_precision, permuted_shape_b, 0.0f, 1.0f);

    auto core = ov::test::utils::PluginCache::get().core();
    ov::CompiledModel cM = core->compile_model(function, targetDevice, {ov::hint::inference_precision(input_precision)});
    auto request = cM.create_infer_request();
    request.set_input_tensor(0, ov::Tensor(input_precision, input_a_shape, input_tensor_a.data()));
    request.set_input_tensor(1, ov::Tensor(input_precision, permuted_shape_b, input_tensor_b.data()));
    request.infer();
}

const std::vector<ov::element::Type> input_precisions = {ov::element::f32, ov::element::f16};

const std::vector<std::vector<int64_t>> allowed_order = {
    {0, 3, 1, 2},
    {0, 1, 2, 3},
    {0, 1, 3, 2},
    {1, 2, 3, 0},
    {0, 2, 1, 3},
    {1, 2, 0, 3},
};

INSTANTIATE_TEST_SUITE_P(smoke_TransposeMatMulFusion_basic,
                         TransposeMatmulFuseTest,
                         ::testing::Combine(::testing::ValuesIn(allowed_order), ::testing::ValuesIn(input_precisions)),
                         TransposeMatmulFuseTest::getTestCaseName);
}  // namespace
