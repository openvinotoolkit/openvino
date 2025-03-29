// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "utils/cpu_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/add.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

struct InputTensorROIParamType {
    ov::PartialShape shape;
    ov::element::Type type;
    ov::Layout layout;
};

class InputTensorROI : public ::testing::TestWithParam<InputTensorROIParamType> {
public:
    static std::string getTestCaseName(::testing::TestParamInfo<InputTensorROIParamType> obj) {
        std::ostringstream result;
        result << "type=" << obj.param.type << "_";
        result << "shape=" << obj.param.shape << "_";
        result << "layout=" << obj.param.layout.to_string();
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> create_test_function(element::Type type,
                                                    const ov::PartialShape& shape,
                                                    const ov::Layout& layout) {
        ResultVector res;
        ParameterVector params;

        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        param->set_friendly_name("input_0");
        param->get_output_tensor(0).set_names({"tensor_input_0"});
        param->set_layout(layout);

        auto constant = ov::op::v0::Constant::create(type, {1}, {1});

        auto add = std::make_shared<ov::op::v1::Add>(param, constant);
        add->set_friendly_name("Add");

        auto result = std::make_shared<ov::op::v0::Result>(add);
        result->set_friendly_name("result_0");
        result->get_output_tensor(0).set_names({"tensor_output_0"});

        params.push_back(param);
        res.push_back(result);

        return std::make_shared<ov::Model>(res, params);
    }

    template <typename T>
    void Run() {
        std::shared_ptr<ov::Core> ie = ov::test::utils::PluginCache::get().core();

        // Compile model
        auto fn_shape = GetParam().shape;
        auto model = create_test_function(GetParam().type, fn_shape, GetParam().layout);
        auto compiled_model = ie->compile_model(model, "CPU");

        // Create InferRequest
        ov::InferRequest req = compiled_model.create_infer_request();

        // Create input tensor
        auto input_shape = Shape{1, 4, 4, 4};
        auto input_shape_size = ov::shape_size(input_shape);
        std::vector<T> data(input_shape_size);
        std::iota(data.begin(), data.end(), 0);
        auto input_tensor = ov::Tensor(GetParam().type, input_shape, &data[0]);

        // Set ROI
        auto roi = ov::Tensor(input_tensor, {0, 1, 1, 1}, {1, 3, 3, 3});
        req.set_tensor("tensor_input_0", roi);

        // Infer
        req.infer();

        // Check result
        auto actual_tensor = req.get_tensor("tensor_output_0");
        auto* actual = actual_tensor.data<T>();
        EXPECT_EQ(actual[0], 21 + 1);
        EXPECT_EQ(actual[1], 22 + 1);
        EXPECT_EQ(actual[2], 25 + 1);
        EXPECT_EQ(actual[3], 26 + 1);
        EXPECT_EQ(actual[4], 37 + 1);
        EXPECT_EQ(actual[5], 38 + 1);
        EXPECT_EQ(actual[6], 41 + 1);
        EXPECT_EQ(actual[7], 42 + 1);
    }
};

TEST_P(InputTensorROI, SetInputTensorROI) {
    switch (GetParam().type) {
    case ov::element::Type_t::f32: {
        Run<float>();
        break;
    }
    case ov::element::Type_t::u8: {
        Run<uint8_t>();
        break;
    }
    default:
        break;
    }
}

static InputTensorROI::ParamType InputTensorROIParams[] = {
    {ov::PartialShape{1, 2, 2, 2}, element::f32, "NCHW"},
    {ov::PartialShape{1, 2, ov::Dimension::dynamic(), ov::Dimension::dynamic()}, element::f32, "NCHW"},
    {ov::PartialShape{1, 2, 2, 2}, element::u8, "NCHW"},
    {ov::PartialShape{1, 2, ov::Dimension::dynamic(), ov::Dimension::dynamic()}, element::u8, "NCHW"},
};

INSTANTIATE_TEST_SUITE_P(smoke_InputTensorROI,
                         InputTensorROI,
                         ::testing::ValuesIn(InputTensorROIParams),
                         InputTensorROI::getTestCaseName);

}  // namespace test
}  // namespace ov
