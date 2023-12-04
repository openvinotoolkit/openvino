// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/op/parameter.hpp>
#include "shared_test_classes/single_layer/random_uniform.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace LayerTestsDefinitions {


std::string RandomUniformLayerTest::getTestCaseName(
        const testing::TestParamInfo<RandomUniformParamsTuple> &obj) {
    RandomUniformTypeSpecificParams randomUniformTypeSpecificParams;
    ov::Shape output_shape;
    int64_t global_seed;
    int64_t op_seed;
    std::string targetName;
    std::tie(output_shape, randomUniformTypeSpecificParams, global_seed, op_seed, targetName) = obj.param;

    std::ostringstream result;
    result << "outputShape=" << ov::test::utils::vec2str(output_shape) << "_";
    result << "global_seed=" << global_seed << "_";
    result << "op_seed=" << op_seed << "_";
    result << "outputType=" << randomUniformTypeSpecificParams.precision.name() << "_";
    result << "min_val=" << randomUniformTypeSpecificParams.min_value << "_";
    result << "max_val=" << randomUniformTypeSpecificParams.max_value;
    return result.str();
}

namespace {

template<InferenceEngine::Precision::ePrecision p>
std::shared_ptr<ov::op::v0::Constant>
createRangeConst(const typename InferenceEngine::PrecisionTrait<p>::value_type &value) {
    return std::make_shared<ov::op::v0::Constant>(FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(p), ov::Shape{1},
                                                  std::vector<typename InferenceEngine::PrecisionTrait<p>::value_type>{
                                                          value});
}

std::shared_ptr<ov::op::v0::Constant> createConstant(InferenceEngine::Precision p, double value) {
    using namespace InferenceEngine;
    switch (p) {
        case Precision::FP32:
            return createRangeConst<InferenceEngine::Precision::FP32>(
                    static_cast<PrecisionTrait<Precision::FP32>::value_type>(value));
        case Precision::FP16:
            return createRangeConst<InferenceEngine::Precision::FP16>(
                    static_cast<PrecisionTrait<Precision::FP16>::value_type>(value));
        default:
            return createRangeConst<InferenceEngine::Precision::I32>(
                    static_cast<PrecisionTrait<Precision::I32>::value_type>(value));
    }
}

} // unnamed namespace

void RandomUniformLayerTest::SetUp() {
    RandomUniformTypeSpecificParams randomUniformParams;
    int64_t global_seed;
    int64_t op_seed;
    ov::Shape output_shape;
    std::string targetName;
    std::tie(output_shape, randomUniformParams, global_seed, op_seed, targetDevice) = this->GetParam();
    const auto precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(randomUniformParams.precision);

    // Use Parameter as input with desired precision to properly configure execution configuration
    // in CoreConfiguration() function
    auto input = std::make_shared<ov::op::v0::Parameter>(precision, output_shape);
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(input);

    auto min_value = createConstant(randomUniformParams.precision, randomUniformParams.min_value);
    auto max_value = createConstant(randomUniformParams.precision, randomUniformParams.max_value);
    auto random_uniform = std::make_shared<ngraph::op::v8::RandomUniform>(shape_of,
                                                                          min_value,
                                                                          max_value,
                                                                          precision,
                                                                          global_seed,
                                                                          op_seed);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(random_uniform)};

    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{input}, "random_uniform");
}

void RandomUniformLayerTest::ConvertRefsParams() {
    // we shouldn't use default conversion from f16 to f32
    ngraph::pass::ConvertPrecision<ngraph::element::Type_t::bf16, ngraph::element::Type_t::f32>().run_on_model(
            functionRefs);
}

}  // namespace LayerTestsDefinitions
