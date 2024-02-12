// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/inverse.hpp"

#include "ov_models/builders.hpp"

using namespace ov::test;

namespace ov {
namespace test {
std::string InverseLayerTest::getTestCaseName(const testing::TestParamInfo<InverseTestParams>& obj) {
    std::string test_type;
    ov::Tensor input;
    bool adjoint;
    std::string device_name;

    std::tie(test_type, input, adjoint, device_name) = obj.param;

    const char separator = '_';
    std::ostringstream result;
    result << test_type << separator;
    result << "input_shape=" << input.get_shape().to_string() << separator;
    result << "type=" << input.get_element_type() << separator;
    result << "adjoint=" << adjoint << separator;
    result << "device=" << device_name;

    return result.str();
}

void InverseLayerTest::SetUp() {
    InverseTestParams test_params;

    std::string test_type;
    ov::Tensor input;
    bool adjoint;

    std::tie(test_type, input, adjoint, targetDevice) = GetParam();

    m_input = input;

    InputShape input_shape;
    const ov::Shape input_tensor_shape = input.get_shape();
    if (test_type == "static") {
        input_shape = {ov::PartialShape(input_tensor_shape), {input_tensor_shape}};
    } else {  // dynamic
        input_shape = {ov::PartialShape::dynamic(ov::Rank(input_tensor_shape.size())), {input_tensor_shape}};
    }
    init_input_shapes({input_shape});

    ov::ParameterVector params;
    std::vector<std::shared_ptr<ov::Node>> inputs;

    auto input_param = std::make_shared<ov::op::v0::Parameter>(input.get_element_type(), input_shape.first);
    input_param->set_friendly_name("input");
    inputs.push_back(input_param);
    params.push_back(input_param);

    auto inverse = std::make_shared<ov::op::v14::Inverse>(inputs[0], adjoint);

    ov::ResultVector results{std::make_shared<ov::opset10::Result>(inverse)};
    function = std::make_shared<ov::Model>(results, params, "Inverse");
}

void InverseLayerTest::generate_inputs(const std::vector<ov::Shape>& target_shapes) {
    inputs.clear();
    const auto& func_inputs = function->inputs();

    auto& probs = func_inputs[0];
    inputs.insert({probs.get_node_shared_ptr(), m_input});
}

void InverseLayerTest::compare(const std::vector<ov::Tensor>& expected,
                                         const std::vector<ov::Tensor>& actual) {
    auto shape = expected[0].get_shape();
    auto element_type = expected[0].get_element_type();

    if (shape.size() == 3) {
        for (size_t b = 0 ; b < shape.front(); ++b) {
            for (size_t x = 0; x < shape.back(); ++x) {
                for(size_t y = 0; y < shape.back(); y++) {
                    std::cout << ((float*)(expected[0].data()))[b * shape.back() * shape.back() + x * shape.back() + y] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            for (size_t x = 0; x < shape.back(); ++x) {
                for(size_t y = 0; y < shape.back(); y++) {
                    std::cout << ((float*)(actual[0].data()))[b * shape.back() * shape.back() + x * shape.back() + y] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    } else if (shape.size() == 2) {
        for (size_t x = 0; x < shape.back(); ++x) {
            for(size_t y = 0; y < shape.back(); y++) {
                std::cout << ((float*)(expected[0].data()))[x * shape.back() + y] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for (size_t x = 0; x < shape.back(); ++x) {
            for(size_t y = 0; y < shape.back(); y++) {
                std::cout << ((float*)(actual[0].data()))[x * shape.back() + y] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }


    SubgraphBaseTest::compare(expected, actual);
}
}  // namespace test
}  // namespace ov
