// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "subgraphs_builders.hpp"

namespace {

using TransposeOrderParams = std::tuple<std::vector<int64_t>,  // allowed transpose orders
                                        ov::element::Type>;    // input precision
class TransposeMatmulFuseTest : public testing::WithParamInterface<TransposeOrderParams>, virtual public ov::test::SubgraphBaseTest {
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
    std::shared_ptr<ov::Model> init_subgraph(std::vector<ov::PartialShape>& input_shapes,
                                             ov::element::Type& input_precision,
                                             const std::vector<int64_t>& target_transpose_order) {
        ov::PartialShape input_a_shape = input_shapes[0];
        ov::PartialShape input_b_shape = input_shapes[1];

        auto input_a = std::make_shared<ov::op::v0::Parameter>(input_precision, input_a_shape);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(input_precision, input_b_shape);

        auto transpose_order_a = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{target_transpose_order.size()}, target_transpose_order);
        auto transpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, transpose_order_a);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(transpose_a, input_b, false, false);

        auto model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input_a, input_b});
        return model;
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        std::tie(order, input_precision) = GetParam();
        ov::Shape input_a_shape = {5, 6, 7, 8};
        ov::Shape input_b_shape = {8, 7};
        ov::Shape permuted_shape_a(input_a_shape.size());
        for (size_t i = 0; i < permuted_shape_a.size(); ++i) {
            permuted_shape_a[i] = input_a_shape[order[i]];
        }
        ov::Shape permuted_shape_b(input_b_shape.size());
        for (size_t i = 0; i < permuted_shape_b.size(); ++i) {
            permuted_shape_b[i] = input_a_shape[order[order.size() - 1 - i]];
        }
        std::vector<ov::test::InputShape> permuted_input_shapes_dyn = {
            std::make_pair(ov::PartialShape{-1, -1, -1, -1}, std::vector<ov::Shape>{input_a_shape}),
            std::make_pair(ov::PartialShape{-1, -1}, std::vector<ov::Shape>{permuted_shape_b}),
        };

        init_input_shapes(permuted_input_shapes_dyn);
        function = init_subgraph(inputDynamicShapes, input_precision, order);
    }

private:
    ov::element::Type input_precision = ov::element::f16;
    std::vector<int64_t> order = {0, 1, 2, 3};
};

TEST_P(TransposeMatmulFuseTest, smoke_allowed_transpose_order) {
    std::vector<int64_t> target_order;
    ov::element::Type input_precision;
    std::tie(target_order, input_precision) = GetParam();
    ov::CompiledModel cM = core->compile_model(function, targetDevice, {ov::hint::inference_precision(input_precision)});
    auto request = cM.create_infer_request();
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
