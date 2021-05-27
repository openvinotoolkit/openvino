// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include <shared_test_classes/single_layer/convolution.hpp>

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;
using LayerTestsDefinitions::convSpecificParams;

namespace SubgraphTestsDefinitions {

using SplitConvConcatPatternTestParams = std::tuple<convSpecificParams,
                                                    SizeVector,          // input shape
                                                    size_t>;             // convolution num

class SplitConvConcatPatternTest : public testing::WithParamInterface<SplitConvConcatPatternTestParams>,
                                   virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SplitConvConcatPatternTestParams> obj) {
        convSpecificParams convParams;
        SizeVector IS;
        size_t convNum;
        std::tie(convParams, IS, convNum) = obj.param;

        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        ngraph::op::PadType padType;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(IS) << "_";
        result << "K" << CommonTestUtils::vec2str(kernel) << "_";
        result << "S" << CommonTestUtils::vec2str(stride) << "_";
        result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
        result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
        result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
        result << "O=" << convOutChannels << "_";
        result << "AP=" << padType << "_";
        result << "CONV_NUM=" << convNum;

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        convSpecificParams convParams;
        SizeVector IS;
        size_t convNum;
        std::tie(convParams, IS, convNum) = this->GetParam();

        const auto dataType = element::f32;
        auto inputParams = builder::makeParams(dataType, {IS});
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(inputParams));

        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        ngraph::op::PadType padType;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

        const auto axis = 1;
        const auto split = builder::makeSplit(paramOuts[0], dataType, convNum, axis);
        NodeVector convs;
        for (size_t i = 0; i < split->get_output_size(); i++) {
            convs.push_back(builder::makeConvolution(split->output(i), dataType, kernel, stride, padBegin, padEnd, dilation, padType, convOutChannels));
        }
        const auto concat = std::make_shared<opset1::Concat>(convs, axis);

        function = std::make_shared<ngraph::Function>(concat, inputParams, "SplitConvConcatPattern");
    }
};

TEST_P(SplitConvConcatPatternTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckNodeOfTypeCount(executableNetwork, "Convolution", 1);
}

const SizeVector numConv = { 2, 4, 16 };

namespace conv1D {

const std::vector<SizeVector> kernels = { {3}, {1} };
const std::vector<SizeVector> strides = { {1}, {2} };
const std::vector<std::vector<ptrdiff_t>> padBegins = { {0}, {1} };
const std::vector<std::vector<ptrdiff_t>> padEnds = { {0} };
const std::vector<SizeVector> dilations = { {1}, {2} };
const SizeVector numOutChannels = { 4, 8, 16 };
const auto convParams = ::testing::Combine(
    ::testing::ValuesIn(kernels),
    ::testing::ValuesIn(strides),
    ::testing::ValuesIn(padBegins),
    ::testing::ValuesIn(padEnds),
    ::testing::ValuesIn(dilations),
    ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const SizeVector IS{2, 32, 5};

INSTANTIATE_TEST_CASE_P(smoke_SplitConvConcat_1D, SplitConvConcatPatternTest,
                        ::testing::Combine(convParams,
                            ::testing::Values(IS),
                            ::testing::ValuesIn(numConv)),
                        SplitConvConcatPatternTest::getTestCaseName);

} // namespace conv1D

namespace conv2D {

const std::vector<SizeVector> kernels = { {3, 3}, {1, 1} };
const std::vector<SizeVector> strides = { {1, 1}, {2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins = { {0, 0}, {1, 1} };
const std::vector<std::vector<ptrdiff_t>> padEnds = { {0, 0} };
const std::vector<SizeVector> dilations = { {1, 1}, {2, 2} };
const SizeVector numOutChannels = { 4, 8, 16 };
const auto convParams = ::testing::Combine(
    ::testing::ValuesIn(kernels),
    ::testing::ValuesIn(strides),
    ::testing::ValuesIn(padBegins),
    ::testing::ValuesIn(padEnds),
    ::testing::ValuesIn(dilations),
    ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const SizeVector IS{2, 32, 5, 5};

INSTANTIATE_TEST_CASE_P(smoke_SplitConvConcat_2D, SplitConvConcatPatternTest,
                        ::testing::Combine(convParams,
                            ::testing::Values(IS),
                            ::testing::ValuesIn(numConv)),
                        SplitConvConcatPatternTest::getTestCaseName);

} // namespace conv2D

namespace conv3D {

const std::vector<SizeVector> kernels = { {3, 3, 3}, {1, 1, 1} };
const std::vector<SizeVector> strides = { {1, 1, 1}, {2, 2, 1} };
const std::vector<std::vector<ptrdiff_t>> padBegins = { {0, 0, 0}, {1, 1, 1} };
const std::vector<std::vector<ptrdiff_t>> padEnds = { {0, 0, 0} };
const std::vector<SizeVector> dilations = { {1, 1, 1}, {2, 2, 2} };
const SizeVector numOutChannels = { 4, 8, 16 };
const auto convParams = ::testing::Combine(
    ::testing::ValuesIn(kernels),
    ::testing::ValuesIn(strides),
    ::testing::ValuesIn(padBegins),
    ::testing::ValuesIn(padEnds),
    ::testing::ValuesIn(dilations),
    ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const SizeVector IS{2, 32, 5, 5, 5};

INSTANTIATE_TEST_CASE_P(smoke_SplitConvConcat_3D, SplitConvConcatPatternTest,
                        ::testing::Combine(convParams,
                            ::testing::Values(IS),
                            ::testing::ValuesIn(numConv)),
                        SplitConvConcatPatternTest::getTestCaseName);

} // namespace conv3D

} // namespace SubgraphTestsDefinitions
