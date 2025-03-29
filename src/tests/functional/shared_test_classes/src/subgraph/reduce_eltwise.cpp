// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/reduce_eltwise.hpp"

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/test_enums.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reduce_sum.hpp"

namespace ov {
namespace test {
std::string ReduceEltwiseTest::getTestCaseName(const testing::TestParamInfo<ReduceEltwiseParamsTuple> &obj) {
    ov::Shape inputShapes;
    std::vector<int> axes;
    ov::test::utils::OpType opType;
    bool keepDims;
    ov::element::Type type;
    std::string targetName;
    std::tie(inputShapes, axes, opType, keepDims, type, targetName) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "axes=" << ov::test::utils::vec2str(axes) << "_";
    result << "opType=" << opType << "_";
    if (keepDims) result << "KeepDims_";
    result << "netPRC=" << type.get_type_name() << "_";
    result << "targetDevice=" << targetName;
    return result.str();
}

void ReduceEltwiseTest::SetUp() {
    ov::Shape inputShape;
    std::vector<int> axes;
    ov::test::utils::OpType opType;
    bool keepDims;
    ov::element::Type type;
    std::tie(inputShape, axes, opType, keepDims, type, targetDevice) = this->GetParam();

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(type, inputShape)};

    std::vector<size_t> shapeAxes;
    switch (opType) {
        case ov::test::utils::OpType::SCALAR: {
            if (axes.size() > 1)
                FAIL() << "In reduce op if op type is scalar, 'axis' input's must contain 1 element";
            break;
        }
        case ov::test::utils::OpType::VECTOR: {
            shapeAxes.push_back(axes.size());
            break;
        }
        default:
            FAIL() << "Reduce op doesn't support operation type: " << opType;
    }
    auto reductionAxesNode = std::dynamic_pointer_cast<ov::Node>(
                             std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape(shapeAxes), axes));

    auto reduce = std::make_shared<ov::op::v1::ReduceSum>(params[0], reductionAxesNode, keepDims);

    std::vector<size_t> constShape(reduce.get()->get_output_partial_shape(0).rank().get_length(), 1);
    ASSERT_GT(constShape.size(), 2);
    constShape[2] = inputShape.back();
    auto constant_tensor = ov::test::utils::create_and_fill_tensor(type, constShape);
    auto constant = std::make_shared<ov::op::v0::Constant>(constant_tensor);
    auto eltw = ov::test::utils::make_eltwise(reduce, constant, ov::test::utils::EltwiseTypes::MULTIPLY);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltw)};
    function = std::make_shared<ov::Model>(results, params, "ReduceEltwise");
}
} // namespace test
} // namespace ov
