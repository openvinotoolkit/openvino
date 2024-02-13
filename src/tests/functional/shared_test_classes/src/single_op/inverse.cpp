// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/inverse.hpp"

#include "ov_models/builders.hpp"

using namespace ov::test;

namespace ov {
namespace test {
std::string InverseLayerTest::getTestCaseName(const testing::TestParamInfo<InverseTestParams>& obj) {
    ov::Shape input_shape;
    ov::element::Type element_type;
    bool adjoint;
    std::string device_name;

    std::tie(input_shape, element_type, adjoint, device_name) = obj.param;

    const char separator = '_';
    std::ostringstream result;
    result << "IS=" << input_shape.to_string() << separator;
    result << "type=" << element_type.to_string() << separator;
    result << "adjoint=" << ov::test::utils::bool2str(adjoint) << separator;
    result << "device=" << device_name;

    return result.str();
}

void InverseLayerTest::SetUp() {
    InverseTestParams test_params;

    ov::Shape input_shape;
    ov::element::Type element_type;
    bool adjoint;

    std::tie(input_shape, element_type, adjoint, targetDevice) = GetParam();

    auto input = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
    auto inverse = std::make_shared<ov::op::v14::Inverse>(input, adjoint);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(inverse)};
    function = std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "Inverse");
}
}  // namespace test
}  // namespace ov
