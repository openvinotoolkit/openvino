// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/function.hpp>

#include <gtest/gtest.h>
#include <common_test_utils/test_common.hpp>
#include <common_test_utils/test_constants.hpp>
#include <ie_core.hpp>
#include <ngraph/ops.hpp>

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"

#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace {

using DataType  = ngraph::element::Type_t;
using DimsType  = ngraph::element::Type_t;
using DataShape = ngraph::Shape;

class DynamicShapeResolverTests : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<std::tuple<DataType, DataShape>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& dataType   = std::get<0>(parameters);
        const auto& dataShape  = std::get<1>(parameters);

        data = std::make_shared<ngraph::opset3::Parameter>(dataType, dataShape);
        dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataShape.size()});
    }

protected:
    std::shared_ptr<ngraph::opset3::Parameter> data;
    std::shared_ptr<ngraph::opset3::Parameter> dims;
};

TEST_P(DynamicShapeResolverTests, CanValidateAndInferTypes) {
    std::shared_ptr<ngraph::vpu::op::DynamicShapeResolver> dynamicShapeResolver;
    ASSERT_NO_THROW(dynamicShapeResolver = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims));
    ASSERT_NO_THROW(auto fun = std::make_shared<ngraph::Function>(ngraph::NodeVector{dynamicShapeResolver}, ngraph::ParameterVector{data, dims}));
}

std::set<ngraph::element::Type_t> allNGraphTypes() {
    return {
        ngraph::element::dynamic,
        ngraph::element::boolean,
        ngraph::element::bf16,
        ngraph::element::f16,
        ngraph::element::f32,
        ngraph::element::f64,
        ngraph::element::i8,
        ngraph::element::i16,
        ngraph::element::i32,
        ngraph::element::i64,
        ngraph::element::u1,
        ngraph::element::u8,
        ngraph::element::u16,
        ngraph::element::u32,
        ngraph::element::u64
    };
}

std::set<ngraph::element::Type_t> allNGraphIntegralNumberTypes() {
    return {
        ngraph::element::i8,
        ngraph::element::i16,
        ngraph::element::i32,
        ngraph::element::i64,
        ngraph::element::u1,
        ngraph::element::u8,
        ngraph::element::u16,
        ngraph::element::u32,
        ngraph::element::u64
    };
}

std::set<ngraph::element::Type_t> allNGraphStaticTypes() {
    auto staticTypes = std::set<ngraph::element::Type_t>{};
    const auto& allTypes = allNGraphTypes();
    const auto& allDynamicTypes = std::set<ngraph::element::Type_t>{ngraph::element::dynamic};
    std::set_difference(allTypes.cbegin(), allTypes.cend(), allDynamicTypes.cbegin(), allDynamicTypes.cend(),
        std::inserter(staticTypes, staticTypes.begin()));
    return staticTypes;
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicShapeResolverTests, testing::Combine(
    testing::ValuesIn(allNGraphStaticTypes()),
    testing::Values(DataShape{1, 800}, DataShape{1, 1})));


using DataPartialShape = ngraph::PartialShape;
using DimsPartialShape = ngraph::PartialShape;
class DynamicShapeResolverNegativeTests
    : public CommonTestUtils::TestsCommon
    , public testing::WithParamInterface<std::tuple<DataType, DimsType, DataPartialShape, DimsPartialShape>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& dataType   = std::get<0>(parameters);
        const auto& dimsType   = std::get<1>(parameters);
        const auto& dataPartialShape  = std::get<2>(parameters);
        const auto& dimsPartialShape  = std::get<3>(parameters);

        data = std::make_shared<ngraph::opset3::Parameter>(dataType, dataPartialShape);
        dims = std::make_shared<ngraph::opset3::Parameter>(dimsType, dimsPartialShape);
    }

protected:
    std::shared_ptr<ngraph::opset3::Parameter> data;
    std::shared_ptr<ngraph::opset3::Parameter> dims;
};

using DynamicShapeResolverNegativeTestsDataType = DynamicShapeResolverNegativeTests;
TEST_P(DynamicShapeResolverNegativeTestsDataType, ThrowsOnInvalidDimsType) {
    ASSERT_THROW(auto dynamicShapeResolver = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims), ngraph::ngraph_error);
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicShapeResolverNegativeTestsDataType, testing::Combine(
    testing::Values(ngraph::element::dynamic),
    testing::Values(ngraph::element::i64),
    testing::Values(DataPartialShape{1, 800}),
    testing::Values(DataPartialShape{2})));

using DynamicShapeResolverNegativeTestsDimsType = DynamicShapeResolverNegativeTests;
TEST_P(DynamicShapeResolverNegativeTestsDimsType, ThrowsOnInvalidDimsType) {
    ASSERT_THROW(auto dynamicShapeResolver = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims), ngraph::ngraph_error);
}

std::set<ngraph::element::Type_t> allNGraphNotIntegralTypes() {
    auto notIntegralTypes = std::set<ngraph::element::Type_t>{};
    const auto& allTypes = allNGraphTypes();
    const auto& allIntegralTypes = allNGraphIntegralNumberTypes();
    std::set_difference(allTypes.cbegin(), allTypes.cend(), allIntegralTypes.cbegin(), allIntegralTypes.cend(),
        std::inserter(notIntegralTypes, notIntegralTypes.begin()));
    return notIntegralTypes;
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicShapeResolverNegativeTestsDimsType, testing::Combine(
    testing::ValuesIn(allNGraphStaticTypes()),
    testing::ValuesIn(allNGraphNotIntegralTypes()),
    testing::Values(DataPartialShape{1, 800}),
    testing::Values(DataPartialShape{2})));

using DynamicShapeResolverNegativeTestsDataShape = DynamicShapeResolverNegativeTests;
TEST_P(DynamicShapeResolverNegativeTestsDataShape, ThrowsOnInvalidDimsType) {
    ASSERT_THROW(auto dynamicShapeResolver = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims), ngraph::ngraph_error);
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicShapeResolverNegativeTestsDataShape, testing::Combine(
    testing::ValuesIn(allNGraphStaticTypes()),
    testing::Values(ngraph::element::i64),
    testing::Values(
        DataPartialShape::dynamic(),
        DataPartialShape{{1, ngraph::Dimension::dynamic()}},
        DataPartialShape{{ngraph::Dimension::dynamic(), 1}},
        DataPartialShape{{ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()}}),
    testing::Values(DataShape{2})));

using DynamicShapeResolverNegativeTestsDimsShape = DynamicShapeResolverNegativeTests;
TEST_P(DynamicShapeResolverNegativeTestsDimsShape, ThrowsOnInvalidDimsType) {
    ASSERT_THROW(auto dynamicShapeResolver = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims), ngraph::ngraph_error);
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicShapeResolverNegativeTestsDimsShape, testing::Combine(
    testing::ValuesIn(allNGraphTypes()),
    testing::Values(ngraph::element::i64),
    testing::Values(DataShape{1, 800}),
    testing::Values(
        DataPartialShape::dynamic(),
        DataPartialShape{{1, ngraph::Dimension::dynamic()}},
        DataPartialShape{{ngraph::Dimension::dynamic(), 1}},
        DataPartialShape{{ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()}},
        DataPartialShape{0},
        DataPartialShape{1},
        DataPartialShape{3})));

typedef std::vector<int32_t> InputData;

typedef std::tuple<
        InputData,
        std::string> dsrParams;

class DynamicShapeResolverPluginTests : public testing::WithParamInterface<dsrParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<dsrParams> &obj) {
        InputData inputData;
        std::string targetDevice;
        std::tie(inputData,
                 targetDevice) = obj.param;

        std::ostringstream result;
        const char separator = '_';
        result << "inputData=" << CommonTestUtils::vec2str(inputData) << separator;
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
    void SetUp() override {
        std::tie(inputData, targetDevice) = GetParam();

        const auto& inPrecision = ::ngraph::element::Type(::ngraph::element::Type_t::i32);

        const auto& tensor = std::make_shared<ngraph::op::Parameter>(inPrecision, ngraph::Shape{inputData.size()});
        const auto& nonZero = std::make_shared<ngraph::op::NonZero>(tensor);
        const auto& gatherIndices = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                               ngraph::Shape{1},
                                                                               std::vector<int64_t>{0});
        const auto& gatherAxis = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                            ngraph::Shape{1},
                                                                            std::vector<int64_t>{1});
        const auto& gather = std::make_shared<ngraph::opset1::Gather>(nonZero->output(0), gatherIndices, gatherAxis);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{tensor});
    }

protected:
    InputData inputData;
};

TEST_P(DynamicShapeResolverPluginTests, DynamicNetworkWithStaticOutput) {
    // TODO: reimplement with normal reference function
    // Currently the network gets the index of the first non-zero element
    int32_t refOutput{};
    for (size_t i = 0; i < inputData.size(); i++) {
        if (inputData[i] != 0) {
            refOutput = static_cast<int32_t>(i);
            break;
        }
    }

    InferenceEngine::CNNNetwork cnnNet(function);

    for (const auto& outputInfo : cnnNet.getOutputsInfo()) {
        outputInfo.second->setPrecision(InferenceEngine::Precision::I32);
    }

    auto ie = PluginCache::get().ie();
    InferenceEngine::ExecutableNetwork execNet;
    ASSERT_NO_THROW(execNet = ie->LoadNetwork(cnnNet, targetDevice));
    auto req = execNet.CreateInferRequest();

    for (const auto &inputItem : cnnNet.getInputsInfo()) {
        auto blob = make_blob_with_precision(inputItem.second->getTensorDesc());
        blob->allocate();
        std::copy_n(inputData.begin(), inputData.size(), blob->buffer().as<int32_t*>());
        req.SetBlob(inputItem.first, blob);
    }

    ASSERT_NO_THROW(req.Infer());

    for (const auto &output : cnnNet.getOutputsInfo()) {
        auto outBlob = req.GetBlob(output.first);
        auto outBuffer = outBlob->cbuffer().as<int32_t*>();

        ASSERT_EQ(outBlob->size() , 1);
        ASSERT_EQ(refOutput, outBuffer[0]);
    }
}

const std::vector<InputData> inputDatas = {
    {1, 0, 0, 0, 0},
    {0, 1, 0, 1, 0},
    {0, 0, 42, 0, 1},
    {0, 0, 0, 0, -42}
};

const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(inputDatas),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)
);


INSTANTIATE_TEST_SUITE_P(smoke_DynamicShapeResolverPluginTests, DynamicShapeResolverPluginTests,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputDatas),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        DynamicShapeResolverPluginTests::getTestCaseName);

}  // namespace
