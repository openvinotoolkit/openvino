// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace ngraph;
using namespace ov::test;

namespace SubgraphTestsDefinitions {
/*
 *           Input(F32) Const(F32)
 *              |  \     /
 *              |  Power(F32) Const(I64)
 *              |      \       /
 *              |   ReduceMean(F32)
 *              |       |  Const(F32)
 *              |       |  /
 *              |      Add(F32)
 *              |       |
 *              |     Sqrt(F32) Const(F32)
 *              |       |      /
 *              |    Divide(F32)
 *              |      /
 *  Const(F32) Multiply(F32)
 *         \    |
 *         Multiply(F32)
 *              |
 *          Convert(F16)
 */
using RMSNormDecompositionParams = std::tuple<std::vector<InputShape>,             // input shapes
                                              ov::test::ElementType,               // input precision
                                              std::map<std::string, std::string>>; // additional config

class RMSNormDecomposition : public testing::WithParamInterface<RMSNormDecompositionParams>, public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RMSNormDecompositionParams> obj) {
        std::vector<InputShape> input_shapes;
        ElementType input_precision;
        std::map<std::string, std::string> additional_config;

        std::tie(input_shapes, input_precision, additional_config) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : input_shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=";
        for (const auto& shape : input_shapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "input_precision=" << input_precision << "_";

        result << "config=(";
        for (const auto& configEntry : additional_config) {
            result << configEntry.first << ", " << configEntry.second << ":";
        }
        result << ")";

        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> init_subgraph(std::vector<ov::PartialShape>& input_shapes,
                                             const ov::Shape& target_shape,
                                             const ov::element::Type input_precision) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(input_precision, input_shapes[0])};

        // x^2
        auto power_const = ov::opset10::Constant::create(input_precision, {}, {2.f});
        auto power = std::make_shared<ov::opset10::Power>(params[0], power_const);

        // ReduceMean(x^2,axes)
        auto mean_axes = ov::opset10::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto mean = std::make_shared<ov::opset10::ReduceMean>(power, mean_axes, true);

        // ReduceMean(x^2,axes)+eps
        auto eps = ov::opset10::Constant::create(input_precision, {}, {1e-5f});
        auto add_eps = std::make_shared<ov::opset10::Add>(mean, eps);

        // Sqrt(ReduceMean(x^2,axes)+eps)
        auto sqrt = std::make_shared<ov::opset10::Sqrt>(add_eps);

        // 1/Sqrt(ReduceMean(x^2,axes)+eps)
        auto div_const = ov::opset10::Constant::create(input_precision, {}, {1});
        auto div = std::make_shared<ov::opset10::Divide>(div_const, sqrt);

        // x * 1/Sqrt(ReduceMean(x^2,axes)+eps)
        auto mul1 = std::make_shared<ov::opset10::Multiply>(params[0], div);

        // x * 1/Sqrt(ReduceMean(x^2,axes)+eps) * gamma
        auto dim = *target_shape.rbegin();
        auto gamma = ngraph::builder::makeConstant<float>(input_precision, ov::Shape{dim}, std::vector<float>{}, true);
        auto mul2 = std::make_shared<ov::opset10::Multiply>(gamma, mul1);

        auto comp = std::make_shared<ov::opset10::Convert>(mul2, ov::element::f16);

        return std::make_shared<ov::Model>(NodeVector{comp}, params, "RMSNormDecomposition");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        std::vector<InputShape> input_shapes;
        ElementType input_precision;
        std::map<std::string, std::string> additional_config;

        std::tie(input_shapes, input_precision, additional_config) = GetParam();

        configuration.insert(additional_config.begin(), additional_config.end());
        init_input_shapes(input_shapes);

        inType = outType = input_precision;

        function = init_subgraph(inputDynamicShapes, targetStaticShapes.front().front(), input_precision);
    }
};

TEST_P(RMSNormDecomposition, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

namespace {

const std::vector<ov::test::ElementType> input_precisions = {ov::element::f32, ov::element::f16};

const std::vector<std::vector<InputShape>> input_shapes_basic = {
    {{{-1, -1, 96}, {{1, 4, 96}}}},
    {{{-1, -1, -1}, {{1, 2, 16}}}},
    {{{}, {{1, 2, 6}}}},
    {{{}, {{1, 2, 18}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_RMSNormDecomposition_basic,
                         RMSNormDecomposition,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::Values(std::map<std::string, std::string>())),
                         RMSNormDecomposition::getTestCaseName);
} // namespace

} // namespace SubgraphTestsDefinitions
