// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/function.hpp>

#include <gtest/gtest.h>
#include <common_test_utils/test_common.hpp>
#include <common_test_utils/test_constants.hpp>
#include <details/ie_exception.hpp>
#include <ie_core.hpp>
#include <ngraph/ops.hpp>

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"

#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

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
    ASSERT_NO_THROW(std::make_shared<ngraph::Function>(ngraph::NodeVector{dynamicShapeResolver}, ngraph::ParameterVector{data, dims}));
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

INSTANTIATE_TEST_CASE_P(NGraph, DynamicShapeResolverTests, testing::Combine(
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
    ASSERT_THROW(std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims), ngraph::ngraph_error);
}

INSTANTIATE_TEST_CASE_P(NGraph, DynamicShapeResolverNegativeTestsDataType, testing::Combine(
    testing::Values(ngraph::element::dynamic),
    testing::Values(ngraph::element::i64),
    testing::Values(DataPartialShape{1, 800}),
    testing::Values(DataPartialShape{2})));

using DynamicShapeResolverNegativeTestsDimsType = DynamicShapeResolverNegativeTests;
TEST_P(DynamicShapeResolverNegativeTestsDimsType, ThrowsOnInvalidDimsType) {
    ASSERT_THROW(std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims), ngraph::ngraph_error);
}

std::set<ngraph::element::Type_t> allNGraphNotIntegralTypes() {
    auto notIntegralTypes = std::set<ngraph::element::Type_t>{};
    const auto& allTypes = allNGraphTypes();
    const auto& allIntegralTypes = allNGraphIntegralNumberTypes();
    std::set_difference(allTypes.cbegin(), allTypes.cend(), allIntegralTypes.cbegin(), allIntegralTypes.cend(),
        std::inserter(notIntegralTypes, notIntegralTypes.begin()));
    return notIntegralTypes;
}

INSTANTIATE_TEST_CASE_P(NGraph, DynamicShapeResolverNegativeTestsDimsType, testing::Combine(
    testing::ValuesIn(allNGraphStaticTypes()),
    testing::ValuesIn(allNGraphNotIntegralTypes()),
    testing::Values(DataPartialShape{1, 800}),
    testing::Values(DataPartialShape{2})));

using DynamicShapeResolverNegativeTestsDataShape = DynamicShapeResolverNegativeTests;
TEST_P(DynamicShapeResolverNegativeTestsDataShape, ThrowsOnInvalidDimsType) {
    ASSERT_THROW(std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims), ngraph::ngraph_error);
}

INSTANTIATE_TEST_CASE_P(NGraph, DynamicShapeResolverNegativeTestsDataShape, testing::Combine(
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
    ASSERT_THROW(std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, dims), ngraph::ngraph_error);
}

INSTANTIATE_TEST_CASE_P(NGraph, DynamicShapeResolverNegativeTestsDimsShape, testing::Combine(
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

typedef std::tuple<
        InferenceEngine::SizeVector,
        std::string> dsrParams;

class DynamicShapeResolverPluginTests : public testing::WithParamInterface<dsrParams>, public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<dsrParams> &obj) {
        InferenceEngine::SizeVector inputData;
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

        const auto& tensor = std::make_shared<ngraph::op::Parameter>(inPrecision, ngraph::Shape(inputData));
        const auto& nonZero = std::make_shared<ngraph::op::NonZero>(tensor);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{nonZero}, ngraph::ParameterVector{tensor});
    }

protected:
    InferenceEngine::SizeVector inputData;
};

static void ref_nonZero(const InferenceEngine::Blob::Ptr& src,
                 InferenceEngine::Blob::Ptr& outIndices) {
    auto outIndicesPtr = outIndices->buffer().as<int32_t*>();

    const auto srcTotalDimSize = src->size();

    const auto getCoord = [&src](int offset){
        std::vector<size_t> coord;
        for (const size_t& stride : src->getTensorDesc().getBlockingDesc().getStrides()) {
            coord.insert(coord.begin(), offset / stride);
            offset %= stride;
        }
        return coord;
    };

    const auto addCoordToIndices = [&outIndicesPtr, &srcTotalDimSize](const std::vector<size_t> &coord,
                                                                      const size_t numNonZeros) {
        for (int j = 0; j < coord.size(); ++j) {
            outIndicesPtr[j * srcTotalDimSize + numNonZeros] = coord[j];
        }
    };

    const auto isNonZero = [&src](const size_t i) {
        if (src->getTensorDesc().getPrecision() == InferenceEngine::Precision::I32) {
            const auto srcPtr = src->cbuffer().as<const int32_t*>();
            return srcPtr[i] != 0;
        }
    };

    size_t numNonZeros = 0;
    for (size_t i = 0; i < srcTotalDimSize; ++i) {
        if (isNonZero(i)) {
            addCoordToIndices(getCoord(i), numNonZeros++);
        }
    }

    auto rank = src->getTensorDesc().getDims().size();
    outIndices->getTensorDesc().reshape({rank, numNonZeros}, InferenceEngine::Layout::NC);
}

static void GenRandomNonZeroData(InferenceEngine::Blob::Ptr& blob) {
    std::mt19937 generator(43);

    const auto getRandomValue = [&generator]() {
        // Each third value will be a zero for test NonZero functionality
        return generator() % 3 ? static_cast<float>(generator()) / generator.max() * 255.f : 0.f;
    };

    size_t count = blob->size();
    if (blob->getTensorDesc().getPrecision() == InferenceEngine::Precision::I32) {
        auto blobPtr = blob->buffer().as<int32_t*>();
        for (size_t idx = 0; idx < count; ++idx) {
            blobPtr[idx] = static_cast<int32_t>(getRandomValue());
        }
    }
}

TEST_P(DynamicShapeResolverPluginTests, DynamicNetworkWithStaticOutput) {
    InferenceEngine::CNNNetwork cnnNet(function);

    for (const auto& outputInfo : cnnNet.getOutputsInfo()) {
        outputInfo.second->setPrecision(InferenceEngine::Precision::I32);
    }

    auto ie = PluginCache::get().ie();
    InferenceEngine::ExecutableNetwork execNet;
    ASSERT_NO_THROW(execNet = ie->LoadNetwork(cnnNet, targetDevice));
    auto req = execNet.CreateInferRequest();

    auto inputBlob = req.GetBlob(cnnNet.getInputsInfo().cbegin()->first);
    GenRandomNonZeroData(inputBlob);

    ASSERT_NO_THROW(req.Infer());

    auto outputIndicesBlob = req.GetBlob(cnnNet.getOutputsInfo().cbegin()->first);
    auto refIndicesBlob = make_blob_with_precision(outputIndicesBlob->getTensorDesc());
    refIndicesBlob->allocate();
    ref_nonZero(inputBlob, refIndicesBlob);

    ASSERT_EQ(refIndicesBlob->size(), outputIndicesBlob->size());
    ASSERT_EQ(refIndicesBlob->byteSize(), outputIndicesBlob->byteSize());
    const auto outPtr = outputIndicesBlob->cbuffer().as<const int32_t*>();
    const auto refPtr = refIndicesBlob->cbuffer().as<const int32_t*>();
    for (size_t idx = 0; idx < outputIndicesBlob->size(); idx++) {
        ASSERT_EQ(refPtr[idx], outPtr[idx]);
    }
}

std::vector<InferenceEngine::SizeVector> inputDims = {
//    { 7 },
    { 1000 },
//    { 3, 5 },
//    { 65, 33 },
//    { 33, 65 },
//    { 1, 1000 },
//    { 223, 217, 21 },
//    { 3, 4, 5, 1 },
//    { 3, 4, 1, 5, 1 }
};

const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(inputDims),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)
);


INSTANTIATE_TEST_CASE_P(DynamicShapeResolverPluginTests, DynamicShapeResolverPluginTests,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputDims),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        DynamicShapeResolverPluginTests::getTestCaseName);

}  // namespace
