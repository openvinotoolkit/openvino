// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ov_ops/type_relaxed.hpp>
#include "test_utils/fusing_test_utils.hpp"
#include "test_utils/convolution_params.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

using namespace CPUTestUtils;
using namespace InferenceEngine;
using namespace ov::test;

namespace SubgraphTestsDefinitions {
typedef std::tuple<
        InputShape, //convShape
        InputShape,  //second term shape
        bool,       // bias flag
        fusingSpecificParams,
        std::map<std::string, std::string> // config
> convSumBroadcastParamSet;

class ConvSumInPlaceTest : public testing::WithParamInterface<convSumBroadcastParamSet>,
                           virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convSumBroadcastParamSet>& obj);
    virtual ngraph::ParameterVector makeParams();
    virtual std::shared_ptr<ngraph::Node> makeConv(const ngraph::ParameterVector& inputParams);
    virtual std::shared_ptr<ngraph::Node> addSum(std::shared_ptr<ngraph::Node> lastNode, const ngraph::ParameterVector& inputParams);
    virtual ov::element::Type getNetType() const;
    void SetUp() override;

protected:
    ov::element::Type runtimeType;
    InferenceEngine::SizeVector _kernel = {3, 3};
    InferenceEngine::SizeVector _stride = {1, 1};
    InferenceEngine::SizeVector _dilation = {1, 1};
    std::vector<ptrdiff_t> _padBegin = {0, 0};
    std::vector<ptrdiff_t> _padEnd = {0, 0};
    size_t _convOutChannels = 64;
};

class ConvSumInPlaceStrided : public ConvSumInPlaceTest {
public:
    ConvSumInPlaceStrided();
};

class ConvSumInPlaceTestInt8 : public ConvSumInPlaceTest {
public:
    ngraph::ParameterVector makeParams() override;
    std::shared_ptr<ngraph::Node> makeConv(const ngraph::ParameterVector& inputParams) override;
    std::shared_ptr<ngraph::Node> addSum(std::shared_ptr<ngraph::Node> lastNode, const ngraph::ParameterVector& inputParams) override;
    void SetUp() override;
};

class ConvSumInPlaceTestSeveralConsumers : public ConvSumInPlaceTest {
public:
    std::shared_ptr<ngraph::Node> addSum(std::shared_ptr<ngraph::Node> lastNode, const ngraph::ParameterVector& inputParams) override;
};

namespace ConvSumBroadcast {
const InputShape convInpShape();
const std::vector<InputShape> secondInp();
} // namespace ConvSumBroadcast
} // namespace SubgraphTestsDefinitions