// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/reduce_ops.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/node_builders/reduce.hpp"

namespace ov {
namespace test {
std::string ReduceOpsLayerTest::getTestCaseName(const testing::TestParamInfo<reduceOpsParams>& obj) {
    std::vector<size_t> input_shape;
    ov::element::Type model_type;
    bool keep_dims;
    ov::test::utils::ReductionType reduction_type;
    std::vector<int> axes;
    ov::test::utils::OpType op_type;
    std::string target_device;
    std::tie(axes, op_type, keep_dims, reduction_type, model_type, input_shape, target_device) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
    result << "axes=" << ov::test::utils::vec2str(axes) << "_";
    result << "opType=" << op_type << "_";
    result << "type=" << reduction_type << "_";
    if (keep_dims) result << "KeepDims_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void ReduceOpsLayerTest::SetUp() {
    std::vector<size_t> input_shape;
    ov::element::Type model_type;
    bool keep_dims;
    ov::test::utils::ReductionType reduction_type;
    std::vector<int> axes;
    ov::test::utils::OpType op_type;
    std::tie(axes, op_type, keep_dims, reduction_type, model_type, input_shape, targetDevice) = GetParam();

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_shape));

    std::vector<size_t> shape_axes;
    switch (op_type) {
        case ov::test::utils::OpType::SCALAR: {
            if (axes.size() > 1)
                FAIL() << "In reduce op if op type is scalar, 'axis' input's must contain 1 element";
            break;
        }
        case ov::test::utils::OpType::VECTOR: {
            shape_axes.push_back(axes.size());
            break;
        }
        default:
            FAIL() << "Reduce op doesn't support operation type: " << op_type;
    }
    auto reduction_axes_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape(shape_axes), axes);

    const auto reduce = ov::test::utils::make_reduce(param, reduction_axes_node, keep_dims, reduction_type);
    function = std::make_shared<ov::Model>(reduce->outputs(), ov::ParameterVector{param}, "Reduce");
}

void ReduceOpsLayerWithSpecificInputTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    auto param = function->get_parameters()[0];
    auto axes = std::get<0>(GetParam());
    auto axis = axes[0];
    auto dims = targetInputStaticShapes[0];

    // Slice of tensor through axis is {1, 0, 0, ....}, the mean value is 1/slice_size
    auto raw_values = std::vector<float>(dims[axis], 0);
    raw_values[0] = 1;

    auto tensor = ov::Tensor(param->get_element_type(), dims);
    ov::test::utils::fill_data_with_broadcast(tensor, axis, raw_values);

    inputs.insert({param, tensor});
}
}  // namespace test
}  // namespace ov
