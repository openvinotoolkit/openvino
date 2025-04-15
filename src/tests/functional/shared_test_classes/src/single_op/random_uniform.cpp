// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/random_uniform.hpp"
#include "openvino/core/type/element_type_traits.hpp"


namespace ov {
namespace test {
std::string RandomUniformLayerTest::getTestCaseName(const testing::TestParamInfo<RandomUniformParamsTuple> &obj) {
    RandomUniformTypeSpecificParams random_uniform_params;
    ov::Shape input_shape;
    int64_t global_seed;
    int64_t op_seed;
    ov::op::PhiloxAlignment alignment;
    std::string target_device;
    std::tie(input_shape, random_uniform_params, global_seed, op_seed, alignment, target_device) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
    result << "global_seed=" << global_seed << "_";
    result << "op_seed=" << op_seed << "_";
    result << "min_val=" << random_uniform_params.min_value << "_";
    result << "max_val=" << random_uniform_params.max_value << "_";
    result << "modelType=" << random_uniform_params.model_type.to_string() << "_";
    result << "alignment=" << alignment << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void RandomUniformLayerTest::SetUp() {
    RandomUniformTypeSpecificParams random_uniform_params;
    ov::Shape input_shape;
    int64_t global_seed;
    int64_t op_seed;
    ov::op::PhiloxAlignment alignment;
    std::tie(input_shape, random_uniform_params, global_seed, op_seed, alignment, targetDevice) = this->GetParam();
    auto model_type = random_uniform_params.model_type;

    // Use Parameter as input with desired model_type to properly configure execution configuration
    // in CoreConfiguration() function
    auto input = std::make_shared<ov::op::v0::Parameter>(model_type, input_shape);
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(input);

    std::shared_ptr<ov::op::v0::Constant> min_value, max_value;
    if (model_type == ov::element::f32) {
            auto min_val = static_cast<ov::fundamental_type_for<ov::element::f32>>(random_uniform_params.min_value);
            auto max_val = static_cast<ov::fundamental_type_for<ov::element::f32>>(random_uniform_params.max_value);
            min_value = std::make_shared<ov::op::v0::Constant>(model_type, ov::Shape{1},
                                                  std::vector<ov::fundamental_type_for<ov::element::f32>>{min_val});
            max_value = std::make_shared<ov::op::v0::Constant>(model_type, ov::Shape{1},
                                                  std::vector<ov::fundamental_type_for<ov::element::f32>>{max_val});
    } else if (model_type == ov::element::f16) {
            auto min_val = static_cast<ov::fundamental_type_for<ov::element::f16>>(random_uniform_params.min_value);
            auto max_val = static_cast<ov::fundamental_type_for<ov::element::f16>>(random_uniform_params.max_value);
            min_value = std::make_shared<ov::op::v0::Constant>(model_type, ov::Shape{1},
                                                  std::vector<ov::fundamental_type_for<ov::element::f16>>{min_val});
            max_value = std::make_shared<ov::op::v0::Constant>(model_type, ov::Shape{1},
                                                  std::vector<ov::fundamental_type_for<ov::element::f16>>{max_val});
    } else if (model_type == ov::element::i32) {
            auto min_val = static_cast<ov::fundamental_type_for<ov::element::i32>>(random_uniform_params.min_value);
            auto max_val = static_cast<ov::fundamental_type_for<ov::element::i32>>(random_uniform_params.max_value);
            min_value = std::make_shared<ov::op::v0::Constant>(model_type, ov::Shape{1},
                                                  std::vector<ov::fundamental_type_for<ov::element::i32>>{min_val});
            max_value = std::make_shared<ov::op::v0::Constant>(model_type, ov::Shape{1},
                                                  std::vector<ov::fundamental_type_for<ov::element::i32>>{max_val});
    } else {
        GTEST_FAIL() << model_type << " type isn't supported by the test";
    }
    auto random_uniform = std::make_shared<ov::op::v8::RandomUniform>(shape_of,
                                                                      min_value,
                                                                      max_value,
                                                                      model_type,
                                                                      global_seed,
                                                                      op_seed,
                                                                      alignment);

    function = std::make_shared<ov::Model>(random_uniform->outputs(), ov::ParameterVector{input}, "random_uniform");
}
}  // namespace test
}  // namespace ov
