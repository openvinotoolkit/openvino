// Copyright (C) 2018-2024 Intel Corporation
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

    if (element_type == ov::element::bf16) {
        rel_threshold = 1.2f;
        abs_threshold = 1.0f;
    }

    const auto data = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape[0].first);
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
    in_data.start_from = 10.0;
    in_data.range = 5;
    in_data.resolution = 16;
    in_data.seed = m_seed;
    auto tensor = ov::test::utils::create_and_fill_tensor(in_prc, targetInputStaticShapes[0], in_data);
    inputs.insert({func_input.get_node_shared_ptr(), tensor});
}

void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) {

    auto& t1 = expected[0];
    auto& t2 = actual[0];

    if(t1.get_shape().size() == 3) {
        for (size_t i = 0; i < t1.get_shape()[0]; ++i) {
            for (size_t x = 0; x < t1.get_shape()[1]; ++x) {
                for (size_t y = 0; y < t1.get_shape()[2]; ++y) {
                    float val = ((float*)t1.data())[i * t1.get_shape()[0] + x * t1.get_shape()[1] + y];
                    std::cout << val << ' ';
                }
                std::cout << '\n';
            }
            std::cout << '\n';
        }
    } else if (t1.get_shape().size() == 2) {
        for (size_t x = 0; x < t1.get_shape()[0]; ++x) {
            for (size_t y = 0; y < t1.get_shape()[1]; ++y) {
                float val = ((float*)t1.data())[x * t1.get_shape()[0] + y];
                std::cout << val << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }

    if (expected[0].get_element_type() == ov::element::bf16) {
        ov::test::utils::compare(t1, t2, 1.0f, 1.2f);
    } else {
        ov::test::utils::compare(t1, t2);
    }
}
}  // namespace test
}  // namespace ov
