// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/inverse.hpp"

#include <numeric>
#include <random>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "ov_models/builders.hpp"

using namespace ov::test;

namespace ov {
namespace test {
std::string InverseLayerTest::getTestCaseName(const testing::TestParamInfo<InverseTestParams>& obj) {
    std::vector<InputShape> input_shape;
    ov::element::Type element_type;
    bool adjoint;
    bool test_static;
    int32_t seed;
    std::string device_name;

    std::tie(input_shape, element_type, adjoint, test_static, seed, device_name) = obj.param;

    const char separator = '_';
    std::ostringstream result;
    result << "IS=" << input_shape[0].first.to_string() << separator;
    result << "dtype=" << element_type.to_string() << separator;
    result << "adjoint=" << ov::test::utils::bool2str(adjoint) << separator;
    result << "static=" << ov::test::utils::bool2str(test_static) << separator;
    result << "seed=" << seed << separator;
    result << "device=" << device_name;

    return result.str();
}

void InverseLayerTest::SetUp() {
    InverseTestParams test_params;

    std::vector<InputShape> input_shape;
    ov::element::Type element_type;
    bool adjoint;
    bool test_static;

    std::tie(input_shape, element_type, adjoint, test_static, m_seed, targetDevice) = GetParam();

    ov::PartialShape parameter_input_shape;
    if (!test_static) {
        init_input_shapes(input_shape);
        parameter_input_shape = input_shape[0].first;
    } else {
        std::vector<InputShape> static_input_shape;
        static_input_shape.push_back({input_shape[0].second[0], {input_shape[0].second[0]}});
        init_input_shapes({static_input_shape});
        parameter_input_shape = static_input_shape[0].first;
    }

    const auto data = std::make_shared<ov::op::v0::Parameter>(element_type, parameter_input_shape);
    data->set_friendly_name("data");

    auto inverse = std::make_shared<ov::op::v14::Inverse>(data, adjoint);

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
    in_data.start_from = -10.0;
    in_data.range = 20;
    in_data.resolution = 256;
    in_data.seed = m_seed;
    auto tensor = ov::test::utils::create_and_fill_tensor(in_prc, targetInputStaticShapes[0], in_data);
    inputs.insert({func_input.get_node_shared_ptr(), tensor});
}
}  // namespace test
}  // namespace ov
