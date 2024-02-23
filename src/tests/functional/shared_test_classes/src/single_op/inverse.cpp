// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/inverse.hpp"

#include <numeric>
#include <random>

using namespace ov::test;

namespace ov {
namespace test {
std::string InverseLayerTest::getTestCaseName(const testing::TestParamInfo<InverseTestParams>& obj) {
    ov::Shape input_shape;
    ov::element::Type element_type;
    bool adjoint;
    bool test_static;
    unsigned int seed;
    std::string device_name;

    std::tie(input_shape, element_type, adjoint, test_static, seed, device_name) = obj.param;

    const char separator = '_';
    std::ostringstream result;
    result << "IS=" << input_shape.to_string() << separator;
    result << "dtype=" << element_type.to_string() << separator;
    result << "adjoint=" << ov::test::utils::bool2str(adjoint) << separator;
    result << "static=" << ov::test::utils::bool2str(test_static) << separator;
    result << "seed=" << seed << separator;
    result << "device=" << device_name;

    return result.str();
}

void InverseLayerTest::SetUp() {
    InverseTestParams test_params;

    ov::Shape input_shape;
    ov::element::Type element_type;
    bool adjoint;
    bool test_static;

    std::tie(input_shape, element_type, adjoint, test_static, m_seed, targetDevice) = GetParam();

    std::vector<InputShape> in_shapes;
    if (!test_static) {
        in_shapes.push_back({{}, {input_shape}});
    } else {
        in_shapes.push_back({input_shape, {input_shape}});
    }
    init_input_shapes(in_shapes);

    const auto data = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
    data->set_friendly_name("data");

    auto inverse = std::make_shared<ov::op::v14::Inverse>(data, adjoint);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(inverse)};
    function = std::make_shared<ov::Model>(results, ParameterVector{data}, "InverseTestCPU");
}

template <typename T>
void fill_data(T* dst, const size_t count, const unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0f, 10.0f);

    for (size_t i = 0; i < count; i++) {
        dst[i] = static_cast<T>(dis(gen));
    }
}

void InverseLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& func_inputs = function->inputs();
    const auto& func_input = func_inputs[0];
    const auto& name = func_input.get_node()->get_friendly_name();
    const auto& in_prc = func_input.get_element_type();
    auto tensor = ov::Tensor(in_prc, targetInputStaticShapes[0]);
    size_t count = std::accumulate(targetInputStaticShapes[0].begin(),
                                   targetInputStaticShapes[0].end(),
                                   1,
                                   std::multiplies<size_t>());

    switch (in_prc) {
    case ov::element::f32:
        fill_data(tensor.data<ov::element_type_traits<ov::element::f32>::value_type>(), count, m_seed);
        break;
    case ov::element::f16:
        fill_data(tensor.data<ov::element_type_traits<ov::element::f16>::value_type>(), count, m_seed);
        break;
    case ov::element::bf16:
        fill_data(tensor.data<ov::element_type_traits<ov::element::bf16>::value_type>(), count, m_seed);
        break;
    default:
        OPENVINO_THROW("Inverse does not support precision ", in_prc, " for the 'data' input.");
    }

    inputs.insert({func_input.get_node_shared_ptr(), tensor});
}
}  // namespace test
}  // namespace ov
