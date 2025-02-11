// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/node_builders/group_convolution.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using Conv3dReshapeTestParams = std::tuple<nodeType, size_t>;

class Conv3dReshapeTest : public testing::WithParamInterface<Conv3dReshapeTestParams>,
                          virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<Conv3dReshapeTestParams> obj) {
        nodeType conv;
        size_t numOut;
        std::tie(conv, numOut) = obj.param;

        std::ostringstream result;
        result << nodeType2str(conv) << "_";
        result << "NUM_OUTPUTS=" << numOut;

        return result.str();
    }

protected:
    std::string cpuNodeType;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        nodeType convType;
        size_t numOut;
        std::tie(convType, numOut) = this->GetParam();

        cpuNodeType = nodeType2PluginType(convType);

        ov::ParameterVector inputParams{
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1024, 64})};

        std::shared_ptr<Node> conv;
        const std::vector<size_t> kernelSize = {1};
        const std::vector<size_t> strides = {1};
        const std::vector<ptrdiff_t> padBegin = {0};
        const std::vector<ptrdiff_t> padEnd = {0};
        const std::vector<size_t> dilation = {1};
        const size_t numOutChannels = 30;
        const size_t numOfGroups = 2;
        const op::PadType paddingType = op::PadType::EXPLICIT;
        switch (convType) {
        case nodeType::convolution: {
            conv = ov::test::utils::make_convolution(inputParams[0],
                                                     element::f32,
                                                     kernelSize,
                                                     strides,
                                                     padBegin,
                                                     padEnd,
                                                     dilation,
                                                     paddingType,
                                                     numOutChannels);
            break;
        }
        case nodeType::groupConvolution: {
            conv = ov::test::utils::make_group_convolution(inputParams[0],
                                                           element::f32,
                                                           kernelSize,
                                                           strides,
                                                           padBegin,
                                                           padEnd,
                                                           dilation,
                                                           paddingType,
                                                           numOutChannels,
                                                           numOfGroups);
            break;
        }
        default: {
            throw std::runtime_error("Conv3dReshapeTest doesn't support this type of operation");
        }
        }

        ResultVector results;
        for (size_t i = 0; i < numOut; i++) {
            auto mockNode =
                std::make_shared<ov::op::v1::Multiply>(conv->output(0),
                                                       ov::op::v0::Constant::create(element::f32, Shape{1}, {1}));
            results.push_back(std::make_shared<ov::op::v0::Result>(mockNode));
        }

        function = std::make_shared<ov::Model>(results, inputParams, "Conv3dReshape");
    }
};

TEST_P(Conv3dReshapeTest, CompareWithRefs) {
    run();
}

namespace {

const std::vector<nodeType> convType = {nodeType::convolution, nodeType::groupConvolution};
const std::vector<size_t> numOut = {1, 2, 5};
const auto conv3dReshapeParams = ::testing::Combine(::testing::ValuesIn(convType), ::testing::ValuesIn(numOut));

INSTANTIATE_TEST_SUITE_P(smoke_Conv3dReshapeTest,
                         Conv3dReshapeTest,
                         conv3dReshapeParams,
                         Conv3dReshapeTest::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
