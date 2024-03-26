// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/runtime/core.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

class ParameterResultShareTensor : public ::testing::TestWithParam<ov::Shape> {
public:
    static std::string getTestCaseName(::testing::TestParamInfo<ov::Shape> obj) {
        std::ostringstream result;
        result << "shape=" << obj.param;
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> create_test_function(const ov::Shape& shape) {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, shape);
        auto result = std::make_shared<ov::op::v0::Result>(param);
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }

    void Run() {
        const ov::Shape& shape = GetParam();

        // Create model
        auto function = create_test_function(shape);

        // Create input
        auto shape_size = ov::shape_size(shape);
        std::vector<std::int8_t> input_data(shape_size);
        auto input = ov::Tensor(ov::element::i8, shape, input_data.data());

        // Load model
        ov::Core core;
        auto compiled_model = core.compile_model(function, "CPU");

        // Infer
        auto infer_req = compiled_model.create_infer_request();
        infer_req.set_input_tensor(input);
        infer_req.infer();

        ASSERT_EQ(infer_req.get_input_tensor().data(), infer_req.get_output_tensor().data());
    }
};

TEST_P(ParameterResultShareTensor, CheckResult) {
    Run();
}

static ParameterResultShareTensor::ParamType ParameterResultShareTensorParams[] = {
    ov::Shape{1, 2, 3, 4},
    ov::Shape{1, 2, 3, 4, 5},
    ov::Shape{1, 2, 3, 4, 5, 6},
};

INSTANTIATE_TEST_SUITE_P(ParameterResultShareTensor,
                         ParameterResultShareTensor,
                         ::testing::ValuesIn(ParameterResultShareTensorParams),
                         ParameterResultShareTensor::getTestCaseName);

}  // namespace test
}  // namespace ov
