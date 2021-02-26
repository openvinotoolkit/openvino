// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/topk.hpp"

namespace LayerTestsDefinitions {
    std::string TopKLayerTest::getTestCaseName(testing::TestParamInfo<TopKParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout;
    InferenceEngine::SizeVector inputShape;
    std::string targetDevice;
    int64_t keepK, axis;
    ngraph::opset4::TopK::Mode mode;
    ngraph::opset4::TopK::SortType sort;
    std::tie(keepK, axis, mode, sort, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "k=" << keepK << "_";
    result << "axis=" << axis << "_";
    result << "mode=" << mode << "_";
    result << "sort=" << sort << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

template <typename T, typename U>
void sortOutputs(T* expectedValues, U* expectedIndices, T* actualValues, U* actualIndices, size_t size,
                 InferenceEngine::SizeVector& inputShape, int64_t k, int64_t axis) {
    size_t numSorts =
        std::accumulate(inputShape.begin(), inputShape.end(), size_t{1}, std::multiplies<size_t>()) / inputShape[axis];
    size_t kStride =
        std::accumulate(inputShape.begin() + axis + 1, inputShape.end(), size_t{1}, std::multiplies<size_t>());

    std::vector<std::pair<T, U>> expected(k);
    std::vector<std::pair<T, U>> actual(k);
    for (size_t i = 0; i < numSorts; ++i) {
        size_t chunkId = i / kStride;
        size_t chunkOffset = chunkId * k * kStride;
        size_t startOffset = chunkOffset + i % kStride;
        for (int j = 0; j < k; ++j) {
            expected[j] =
                std::make_pair(expectedValues[startOffset + j * kStride], expectedIndices[startOffset + j * kStride]);
            actual[j] =
                std::make_pair(actualValues[startOffset + j * kStride], actualIndices[startOffset + j * kStride]);
        }

        auto compPairs = [](std::pair<T, U> const& lhs, std::pair<T, U> const& rhs) {
            if (lhs.first == rhs.first) {
                return lhs.second < rhs.second;
            }
            return lhs.first < rhs.first;
        };
        std::sort(expected.begin(), expected.end(), compPairs);
        std::sort(actual.begin(), actual.end(), compPairs);

        for (int j = 0; j < k; ++j) {
            expectedValues[startOffset + j * kStride] = expected[j].first;
            expectedIndices[startOffset + j * kStride] = expected[j].second;
            actualValues[startOffset + j * kStride] = actual[j].first;
            actualIndices[startOffset + j * kStride] = actual[j].second;
        }
    }
}

template <typename T>
void sortOutputs(T* expectedValues, uint8_t* expectedIndices, T* actualValues, uint8_t* actualIndices, size_t size,
                 InferenceEngine::SizeVector& inputShape, int64_t k, int64_t axis,
                 const InferenceEngine::Precision& indicesPrecision) {
    switch (indicesPrecision) {
    case InferenceEngine::Precision::I32:
        return sortOutputs(expectedValues, reinterpret_cast<int32_t*>(expectedIndices), actualValues,
                           reinterpret_cast<int32_t*>(actualIndices), size, inputShape, k, axis);
    case InferenceEngine::Precision::I64:
        return sortOutputs(expectedValues, reinterpret_cast<int64_t*>(expectedIndices), actualValues,
                           reinterpret_cast<int64_t*>(actualIndices), size, inputShape, k, axis);
    default:
        FAIL() << indicesPrecision << " precision isn't supported for topK index tensor.";
    }
}

void TopKLayerTest::Validate() {
    auto expectedOutputs = LayerTestsCommon::CalculateRefs();
    const auto& actualOutputs = LayerTestsCommon::GetOutputs();

    if (expectedOutputs.empty()) {
        return;
    }

    IE_ASSERT(actualOutputs.size() == expectedOutputs.size())
        << "nGraph interpreter has " << expectedOutputs.size() << " outputs, while IE " << actualOutputs.size();

    auto params = this->GetParam();
    auto k = std::get<0>(params);
    auto sort = std::get<3>(params);

    // If SortType is NONE, the order of top k elements is undefined and may be implementation-dependent.
    // This section sorts expected and actual values in a way that retains correctness of the results and ensures
    // that the test won't fail because of different ordering of elements in the actual and reference implementation.
    if (sort == ngraph::opset4::TopK::SortType::NONE && k > 1) {
        ASSERT_EQ(expectedOutputs.size(), 2);
        auto& expectedValues = expectedOutputs[0];
        auto& expectedIndices = expectedOutputs[1];
        auto& actualValues = actualOutputs[0];
        auto& actualIndices = actualOutputs[1];

        ASSERT_EQ(expectedValues.size(), actualValues->byteSize());
        ASSERT_EQ(expectedIndices.size(), actualIndices->byteSize());
        auto expectedValuesBuffer = expectedValues.data();
        auto expectedIndicesBuffer = expectedIndices.data();

        auto actualValuesMemory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actualValues);
        auto actualIndicesMemory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actualIndices);
        IE_ASSERT(actualValuesMemory);
        IE_ASSERT(actualIndicesMemory);
        auto lockedValuesMemory = actualValuesMemory->wmap();
        auto lockedIndicesMemory = actualIndicesMemory->wmap();
        auto actualValuesBuffer = lockedValuesMemory.as<std::uint8_t*>();
        auto actualIndicesBuffer = lockedIndicesMemory.as<std::uint8_t*>();

        const auto& valuesPrecision = actualValues->getTensorDesc().getPrecision();
        const auto& indicesPrecision = actualIndices->getTensorDesc().getPrecision();
        const auto& size = actualValues->size();
        auto axis = std::get<1>(params);
        auto inputShape = std::get<8>(params);

        switch (valuesPrecision) {
        case InferenceEngine::Precision::FP32:
            sortOutputs(reinterpret_cast<float*>(expectedValuesBuffer), expectedIndicesBuffer,
                        reinterpret_cast<float*>(actualValuesBuffer), actualIndicesBuffer, size, inputShape, k, axis,
                        indicesPrecision);
            break;
        case InferenceEngine::Precision::I32:
            sortOutputs(reinterpret_cast<int32_t*>(expectedValuesBuffer), expectedIndicesBuffer,
                        reinterpret_cast<int32_t*>(actualValuesBuffer), actualIndicesBuffer, size, inputShape, k, axis,
                        indicesPrecision);
            break;
        case InferenceEngine::Precision::I64:
            sortOutputs(reinterpret_cast<int64_t*>(expectedValuesBuffer), expectedIndicesBuffer,
                        reinterpret_cast<int64_t*>(actualValuesBuffer), actualIndicesBuffer, size, inputShape, k, axis,
                        indicesPrecision);
            break;
        case InferenceEngine::Precision::I8:
            sortOutputs(reinterpret_cast<int8_t*>(expectedValuesBuffer), expectedIndicesBuffer,
                        reinterpret_cast<int8_t*>(actualValuesBuffer), actualIndicesBuffer, size, inputShape, k, axis,
                        indicesPrecision);
            break;
        case InferenceEngine::Precision::U16:
            sortOutputs(reinterpret_cast<uint16_t*>(expectedValuesBuffer), expectedIndicesBuffer,
                        reinterpret_cast<uint16_t*>(actualValuesBuffer), actualIndicesBuffer, size, inputShape, k, axis,
                        indicesPrecision);
            break;
        case InferenceEngine::Precision::I16:
            sortOutputs(reinterpret_cast<int16_t*>(expectedValuesBuffer), expectedIndicesBuffer,
                        reinterpret_cast<int16_t*>(actualValuesBuffer), actualIndicesBuffer, size, inputShape, k, axis,
                        indicesPrecision);
            break;
        case InferenceEngine::Precision::BOOL:
        case InferenceEngine::Precision::U8:
            sortOutputs(reinterpret_cast<uint8_t*>(expectedValuesBuffer), expectedIndicesBuffer,
                        reinterpret_cast<uint8_t*>(actualValuesBuffer), actualIndicesBuffer, size, inputShape, k, axis,
                        indicesPrecision);
            break;
        case InferenceEngine::Precision::U64:
            sortOutputs(reinterpret_cast<uint64_t*>(expectedValuesBuffer), expectedIndicesBuffer,
                        reinterpret_cast<uint64_t*>(actualValuesBuffer), actualIndicesBuffer, size, inputShape, k, axis,
                        indicesPrecision);
            break;
        case InferenceEngine::Precision::BF16:
            sortOutputs(reinterpret_cast<ngraph::bfloat16*>(expectedValuesBuffer), expectedIndicesBuffer,
                        reinterpret_cast<ngraph::bfloat16*>(actualValuesBuffer), actualIndicesBuffer, size, inputShape,
                        k, axis, indicesPrecision);
            break;
        case InferenceEngine::Precision::FP16:
            sortOutputs(reinterpret_cast<ngraph::float16*>(expectedValuesBuffer), expectedIndicesBuffer,
                        reinterpret_cast<ngraph::float16*>(actualValuesBuffer), actualIndicesBuffer, size, inputShape,
                        k, axis, indicesPrecision);
            break;
        default:
            FAIL() << indicesPrecision << " precision isn't supported for topK values tensor.";
        }
    }

    LayerTestsCommon::Compare(expectedOutputs, actualOutputs);
}

void TopKLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    int64_t keepK, axis;
    ngraph::opset4::TopK::Mode mode;
    ngraph::opset4::TopK::SortType sort;
    std::tie(keepK, axis, mode, sort, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramIn = ngraph::helpers::convert2OutputVector(
                        ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    auto k = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{}, &keepK);
    auto topk = std::dynamic_pointer_cast<ngraph::opset4::TopK>(
            std::make_shared<ngraph::opset4::TopK>(paramIn[0], k, axis, mode, sort));

    ngraph::ResultVector results;
    for (int i = 0; i < topk->get_output_size(); i++) {
        results.push_back(std::make_shared<ngraph::opset4::Result>(topk->output(i)));
    }
    function = std::make_shared<ngraph::Function>(results, params, "TopK");
}
}  // namespace LayerTestsDefinitions
