// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/const_strided_slice_concat.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

std::string ConstStridedSliceConcatTest::getTestCaseName(const testing::TestParamInfo<ConstStridedSliceConcatParams>& obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    uint32_t inputSliceSize;
    uint32_t constSliceSize;
    uint32_t inputSlices;
    uint32_t constSlices;
    std::tie(netPrecision, targetDevice, configuration, inputSliceSize, inputSlices, constSliceSize, constSlices) = obj.param;

    std::ostringstream result;
    result << "ISS=" << inputSliceSize << "_";
    result << "ISN=" << inputSlices << "_";
    result << "CSS=" << constSliceSize << "_";
    result << "CSN=" << constSlices << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

InferenceEngine::Blob::Ptr ConstStridedSliceConcatTest::GenerateInput(const InferenceEngine::InputInfo& info) const {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
    blob->allocate();

    auto* rawBlobDataPtr = blob->buffer().as<float*>();
    std::vector<float> values = ov::test::utils::generate_float_numbers(blob->size(), -0.5f, 0.5f);
    for (size_t i = 0; i < blob->size(); i++) {
        rawBlobDataPtr[i] = values[i];
    }
    return blob;
}

namespace {
template <class A, class B, class C>
void appendSlices(A&& destVector, B&& src, const int64_t chunkSize, const int64_t totalSize, C precission) {
    for (int64_t start = 0; start < totalSize; start += chunkSize) {
        ov::Shape constShape = {2};
        auto beginNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, std::vector<int64_t>{ 0, start });
        auto endNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, std::vector<int64_t>{ 0, start + chunkSize });
        auto strideNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, std::vector<int64_t>{ 1, 1 });
        auto ssNode = std::make_shared<ov::op::v1::StridedSlice>(src,
                                                                beginNode,
                                                                endNode,
                                                                strideNode,
                                                                std::vector<int64_t>{ 1, 0 },
                                                                std::vector<int64_t>{ 1, 0 },
                                                                std::vector<int64_t>{},
                                                                std::vector<int64_t>{},
                                                                std::vector<int64_t>{});
        destVector.push_back(ssNode);
    }
}
} // namespace

//  Topology:
//
//       Constant                Parameter
//        |   |                    |   |
//    +---+   +---+            +---+   +---+
//    |           |            |           |
//  SS_1c  ...  SS_Nc        SS_1p  ...  SS_Np
//    |           |            |           |
//    |           +----+  +----+           |
//    |                |  |                |
//    +-------------+  |  |  +-------------+
//                   \ |  | /
//                    Concat
//
//  Legend:
//      SS == Strided Slice
void ConstStridedSliceConcatTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    uint32_t inputSliceSize;
    uint32_t constSliceSize;
    uint32_t inputSlices;
    uint32_t constSlices;
    std::tie(netPrecision, targetDevice, tempConfig, inputSliceSize, inputSlices, constSliceSize, constSlices) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    std::vector<size_t> inputShape;
    const size_t totalInputSize = static_cast<size_t>(inputSlices) * inputSliceSize;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, totalInputSize})};

    const auto totalConstantSize = constSlices * constSliceSize;
    auto constantValues = ov::test::utils::generate_float_numbers(totalConstantSize, -0.2f, 0.2f);
    auto constant = ngraph::builder::makeConstant(ngPrc, { 1, totalConstantSize }, constantValues);

    std::vector<ngraph::Output<ngraph::Node>> allToConcat;
    appendSlices(allToConcat, params[0], inputSliceSize, totalInputSize, ngPrc);
    appendSlices(allToConcat, constant, constSliceSize, totalConstantSize, ngPrc);
    auto concat = std::make_shared<ov::op::v0::Concat>(allToConcat, 1);

    function = std::make_shared<ngraph::Function>(concat, params, "ConstStridedSliceConcatTest");
}
}  // namespace SubgraphTestsDefinitions
