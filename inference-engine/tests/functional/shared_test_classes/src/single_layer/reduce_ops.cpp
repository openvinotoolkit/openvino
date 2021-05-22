// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/reduce_ops.hpp"

namespace LayerTestsDefinitions {

std::string ReduceOpsLayerTest::getTestCaseName(testing::TestParamInfo<reduceMeanParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout;
    bool keepDims;
    ngraph::helpers::ReductionType reductionType;
    std::vector<size_t> inputShape;
    std::vector<int> axes;
    CommonTestUtils::OpType opType;
    std::string targetDevice;
    std::tie(axes, opType, keepDims, reductionType, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "axes=" << CommonTestUtils::vec2str(axes) << "_";
    result << "opType=" << opType << "_";
    result << "type=" << reductionType << "_";
    if (keepDims) result << "KeepDims_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void ReduceOpsLayerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    bool keepDims;
    ngraph::helpers::ReductionType reductionType;
    std::vector<size_t> inputShape;
    std::vector<int> axes;
    CommonTestUtils::OpType opType;
    std::tie(axes, opType, keepDims, reductionType, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) = GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    std::vector<size_t> shapeAxes;
    switch (opType) {
        case CommonTestUtils::OpType::SCALAR: {
            if (axes.size() > 1)
                FAIL() << "In reduce op if op type is scalar, 'axis' input's must contain 1 element";
            break;
        }
        case CommonTestUtils::OpType::VECTOR: {
            shapeAxes.push_back(axes.size());
            break;
        }
        default:
            FAIL() << "Reduce op doesn't support operation type: " << opType;
    }
    auto reductionAxesNode = std::dynamic_pointer_cast<ngraph::Node>(
                             std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape(shapeAxes), axes));

    const auto reduce = ngraph::builder::makeReduce(paramOuts[0], reductionAxesNode, keepDims, reductionType);
    const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(reduce)};
    function = std::make_shared<ngraph::Function>(results, params, "Reduce");
}
InferenceEngine::Blob::Ptr ReduceOpsLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::helpers::ReductionType reductionType = std::get<3>(GetParam());
    InferenceEngine::Precision netPrecision = std::get<4>(GetParam());
    if (reductionType == ngraph::helpers::ReductionType::LogicalOr ||
        reductionType == ngraph::helpers::ReductionType::LogicalAnd) {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 2, 0);
    } else if (!netPrecision.is_float()) {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 5, 0);
    }
    auto td = info.getTensorDesc();
    auto blob = make_blob_with_precision(td);
    blob->allocate();
    if (reductionType == ngraph::helpers::ReductionType::Max) {
        CommonTestUtils::fill_data_random_float<InferenceEngine::Precision::FP32>(blob, 5, -5, 1000);
    } else {
        CommonTestUtils::fill_data_random_float<InferenceEngine::Precision::FP32>(blob, 5, 0, 1000);
    }
    return blob;
}

InferenceEngine::Blob::Ptr ReduceOpsLayerWithSpecificInputTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    auto axis_vec = std::get<0>(GetParam());
    IE_ASSERT(axis_vec.size() == 1);

    auto axis = axis_vec[0];
    auto td = info.getTensorDesc();
    auto dims = td.getDims();

    // Slice of tensor through axis is {1, 0, 0, ....}, the mean value is 1/slice_size
    auto raw_values = std::vector<float>(dims[axis], 0);
    raw_values[0] = 1;

    auto blob = make_blob_with_precision(td);
    blob->allocate();
    CommonTestUtils::fill_data_with_broadcast(blob, axis, raw_values);
    return blob;
}

}  // namespace LayerTestsDefinitions