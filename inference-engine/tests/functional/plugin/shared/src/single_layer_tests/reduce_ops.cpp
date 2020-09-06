// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/reduce_ops.hpp"

namespace LayerTestsDefinitions {

std::string ReduceOpsLayerTest::getTestCaseName(testing::TestParamInfo<reduceMeanParams> obj) {
    InferenceEngine::Precision netPrecision;
    bool keepDims;
    ngraph::helpers::ReductionType reductionType;
    std::vector<size_t> inputShape;
    std::vector<int> axes;
    CommonTestUtils::OpType opType;
    std::string targetDevice;
    std::tie(axes, opType, keepDims, reductionType, netPrecision, inputShape, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "axes=" << CommonTestUtils::vec2str(axes) << "_";
    result << "opType=" << opType << "_";
    result << "type=" << reductionType << "_";
    if (keepDims) result << "KeepDims_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void ReduceOpsLayerTest::SetUp() {
    // TODO: Issue 33151
    // Failed to create function on SetUp stage with some parameters
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Precision netPrecision;
    bool keepDims;
    ngraph::helpers::ReductionType reductionType;
    std::vector<size_t> inputShape;
    std::vector<int> axes;
    CommonTestUtils::OpType opType;
    std::tie(axes, opType, keepDims, reductionType, netPrecision, inputShape, targetDevice) = GetParam();

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

TEST_P(ReduceOpsLayerTest, CompareWithRefs) {
    Run();
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

TEST_P(ReduceOpsLayerWithSpecificInputTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions