
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/convolution_test.hpp"

#include <memory>
#include <tuple>

//#include "ngraph_ops/type_relaxed.hpp
#include "ngraph/pass/visualize_tree.hpp"
#include "convolution_function.hpp"

namespace LayerTestsDefinitions {

std::string ConvolutionTest::getTestCaseName(testing::TestParamInfo<testsParams> obj) {
    std::ostringstream result;
    auto values = std::get<0>(obj.param);
    const size_t input_branch = std::get<1>(obj.param);
    values.input_shape[0] = input_branch;
    const auto input_type = std::get<2>(obj.param);
    const auto operations_number = std::get<3>(obj.param);
    const auto targetDevice = std::get<4>(obj.param);

    result << "IS=" << CommonTestUtils::vec2str(values.input_shape) << "_";
    result << "netPRC=" << input_type << "_";
    result << "D=" << targetDevice << "_";
    result << "IN=" << input_type << "_";

    for (const auto& convolution_param : values.convolution_params) {
        result << "P_S=" << CommonTestUtils::vec2str(convolution_param.strides) << "_";
        result << "P_PB=" << CommonTestUtils::vec2str(convolution_param.pads_begin) << "_";
        result << "P_PE=" << CommonTestUtils::vec2str(convolution_param.pads_end) << "_";
        result << "P_K=" << CommonTestUtils::vec2str(convolution_param.dilations) << "_";
    }

    result << "NN1=" << operations_number.first;
    result << "NN2=" << operations_number.second;
    return result.str();
}

namespace {
ov::test::snippets::ConvolutionFunction::ConvolutionParams to_param(
    const LayerTestsDefinitions::ConvolutionTestValues::ConvolutionParams& param) {
    std::vector<ov::test::snippets::ConvolutionFunction::ConvolutionParams> result;
    return {
        param.strides,
        param.pads_begin,
        param.pads_end,
        param.dilations,
        param.auto_pad,
        param.weights_shape };
}

std::vector<ov::test::snippets::ConvolutionFunction::ConvolutionParams> to_params(
const std::vector<LayerTestsDefinitions::ConvolutionTestValues::ConvolutionParams>& params) {
    std::vector<ov::test::snippets::ConvolutionFunction::ConvolutionParams> result;
    for (const auto& param : params) {
        result.push_back(to_param(param));
    }
    return result;
}
} // namespace

void ConvolutionTest::SetUp() {
    // not initialized by default
    abs_threshold = 0.05f;
    rel_threshold = 0.01f;

    auto& testsParams = this->GetParam();

    auto values = std::get<0>(testsParams);

    auto input_batch = std::get<1>(testsParams);
    values.input_shape[0] = input_batch;
    const auto input_type = std::get<2>(testsParams);
    targetDevice = std::get<4>(testsParams);

    init_input_shapes({{values.input_shape, {values.input_shape}}});

    function = ov::test::snippets::ConvolutionFunction::get(
            values.input_shape,
            input_type,
            {
                values.prerequisites_params.strides,
                values.prerequisites_params.pads_begin,
                values.prerequisites_params.pads_end,
                values.prerequisites_params.kernel
            },
            to_params(values.convolution_params));

    ngraph::pass::Validate().run_on_model(function);

#define CPU_DEBUG_CAPS_SNIPPETS
#ifdef CPU_DEBUG_CAPS_SNIPPETS
    ngraph::pass::Serialize("svg/test.actual.xml", "svg/test.actual.bin").run_on_model(function);
    ngraph::pass::VisualizeTree("svg/test.actual.svg").run_on_model(function);
    ngraph::pass::Serialize("svg/test.actual.xml", "svg/test.actual.bin").run_on_model(function);
#endif
}

void ConvolutionTest::run() {
    SubgraphBaseTest::run();

    const auto operations_number = std::get<3>(GetParam());
    ref_num_nodes = operations_number.first;
    ref_num_subgraphs = operations_number.second;

    validateNumSubgraphs();
}

TEST_P(ConvolutionTest, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
