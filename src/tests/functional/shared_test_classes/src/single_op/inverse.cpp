// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/inverse.hpp"

#include <numeric>
#include <random>

#include "common_test_utils/ov_tensor_utils.hpp"

using namespace ov::test;

namespace ov {
namespace test {
std::string InverseLayerTest::getTestCaseName(const testing::TestParamInfo<InverseTestParams>& obj) {
    std::vector<InputShape> input_shape;
    ov::element::Type element_type;
    bool adjoint;
    int32_t seed;
    std::string device_name;

    std::tie(input_shape, element_type, adjoint, seed, device_name) = obj.param;

    const char separator = '_';
    std::ostringstream result;
    result << "IS=" << input_shape[0].first.to_string() << separator;
    result << "dtype=" << element_type.to_string() << separator;
    result << "adjoint=" << ov::test::utils::bool2str(adjoint) << separator;
    result << "seed=" << seed << separator;
    result << "device=" << device_name;

    return result.str();
}

void InverseLayerTest::SetUp() {
    std::vector<InputShape> input_shape;
    ov::element::Type element_type;
    bool adjoint;

    std::tie(input_shape, element_type, adjoint, m_seed, targetDevice) = GetParam();

    init_input_shapes(input_shape);

    const auto data = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape[0].first);
    data->set_friendly_name("data");

    const auto inverse = std::make_shared<ov::op::v14::Inverse>(data, adjoint);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(inverse)};
    function = std::make_shared<ov::Model>(results, ParameterVector{data}, "InverseTestCPU");
}

void InverseLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& func_inputs = function->inputs();
    const auto& func_input = func_inputs[0];
    const auto& name = func_input.get_node()->get_friendly_name();
    const auto& in_prc = func_input.get_element_type();

    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = 5;
    in_data.range = 5;
    in_data.resolution = 1;
    in_data.seed = m_seed;
    auto tensor = ov::test::utils::create_and_fill_tensor(in_prc, targetInputStaticShapes[0], in_data);
    inputs.insert({func_input.get_node_shared_ptr(), tensor});
}
}  // namespace test
}  // namespace ov
