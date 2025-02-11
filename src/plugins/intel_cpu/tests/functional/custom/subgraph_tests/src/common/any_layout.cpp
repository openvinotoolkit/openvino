// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils/cpu_test_utils.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/add.hpp"

namespace ov {
namespace test {

class AnyLayoutOnInputsAndOutputs : public ::testing::TestWithParam<ov::Shape> {
public:
    static std::string getTestCaseName(::testing::TestParamInfo<ov::Shape> obj) {
        std::ostringstream result;
        result << "shape=" << obj.param;
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model>
    create_test_function(const ov::Shape & shape) {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);

        float shift = 1.0f;
        auto shift_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, &shift);

        auto add = std::make_shared<ov::op::v1::Add>(param, shift_node);

        auto result = std::make_shared<ov::op::v0::Result>(add);

        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }

    void Run() {
        const ov::Shape & shape = GetParam();
        auto shape_size = ov::shape_size(shape);

        std::vector<float> input_data(shape_size, 2);
        std::vector<float> output_data(shape_size);
        std::vector<float> expected_output(shape_size, 3);

        // Create model
        auto function = create_test_function(shape);

        auto ppp_model = ov::preprocess::PrePostProcessor(function);
        ppp_model.input().tensor().set_layout("...");
        ppp_model.output().tensor().set_layout("...");
        function = ppp_model.build();

        auto input = ov::Tensor(ov::element::f32, shape, input_data.data());
        auto output = ov::Tensor(ov::element::f32, shape, output_data.data());

        // Load model
        ov::Core core;
        auto compiled_model = core.compile_model(function, "CPU");

        // Infer
        auto infer_req = compiled_model.create_infer_request();
        infer_req.set_input_tensor(input);
        infer_req.set_output_tensor(output);
        infer_req.infer();

        ASSERT_EQ(output_data, expected_output);
    }
};

TEST_P(AnyLayoutOnInputsAndOutputs, CheckExpectedResult) {
    Run();
}

static AnyLayoutOnInputsAndOutputs::ParamType AnyLayoutOnInputsAndOutputsParams[] = {
    ov::Shape{ 1, 2, 3, 4 },
    ov::Shape{ 1, 2, 3, 4, 5 },
    ov::Shape{ 1, 2, 3, 4, 5, 6 },
};

INSTANTIATE_TEST_SUITE_P(AnyLayoutOnInputsAndOutputs,
                         AnyLayoutOnInputsAndOutputs,
                         ::testing::ValuesIn(AnyLayoutOnInputsAndOutputsParams),
                         AnyLayoutOnInputsAndOutputs::getTestCaseName);

}  // namespace test
}  // namespace ov
