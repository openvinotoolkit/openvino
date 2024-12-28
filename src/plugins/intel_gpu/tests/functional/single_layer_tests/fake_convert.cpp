// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/fake_convert.hpp"

namespace {

namespace fp8 {
constexpr float MAX_F8E4M3  = 448.f;
constexpr float MAX_F8E5M2  = 57344.f;
}  // namespace fp8

using namespace std;
using namespace ov;
using namespace testing;
using ov::test::InputShape;

using FakeConvertTestParams = std::tuple<
                                    ov::Shape,                  // Input shapes
                                    ov::Shape,                  // Scale shape
                                    ov::Shape,                  // Shift shape
                                    ov::element::Type,          // input precision
                                    ov::element::Type,          // destination type
                                    std::string >;              // device name

class FakeConvertTest : public testing::WithParamInterface<FakeConvertTestParams>,
                     virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FakeConvertTestParams> obj) {
        ov::Shape input_shape;
        ov::Shape scale_shape;
        ov::Shape shift_shape;
        ov::element::Type prec;
        ov::element::Type destination_type;
        std::string target_device;

        std::tie(input_shape, scale_shape, shift_shape, prec, destination_type, target_device) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        result << ov::test::utils::vec2str(input_shape) << "_";
        result << "scale_shape=" << ov::test::utils::vec2str(scale_shape) << "_";
        result << "shift_shape=" << ov::test::utils::vec2str(shift_shape) << "_";
        result << "input_precision=" << prec << "_";
        result << "destination_type=" << destination_type << "_";
        result << "device_type=" << target_device;
        return result.str();
    }

protected:
    ov::Shape input_shape, scale_shape, shift_shape;
    ov::element::Type destination_type;

    void SetUp() override {
        ov::element::Type prec;
        std::tie(input_shape, scale_shape, shift_shape, prec, destination_type, targetDevice) = GetParam();
        const float MAX_FP8 = (destination_type == ov::element::f8e4m3) ? fp8::MAX_F8E4M3 : fp8::MAX_F8E5M2;
        if (shift_shape.empty()) {
            auto data = make_shared<op::v0::Parameter>(prec, input_shape);
            auto scale = op::v0::Constant::create(prec,
                                                scale_shape,
                                                {MAX_FP8 / (MAX_FP8 / 2.f),
                                                1.0f,
                                                MAX_FP8 / (MAX_FP8 * 3.5f),
                                                MAX_FP8 / (MAX_FP8 * 4.f)});

            auto op = make_shared<op::v13::FakeConvert>(data, scale, destination_type);

            function = make_shared<Model>(OutputVector{op}, ParameterVector{data});
        } else {
            auto data = make_shared<op::v0::Parameter>(prec, input_shape);
            auto scale = op::v0::Constant::create(prec,
                                                scale_shape,
                                                {MAX_FP8 / (MAX_FP8 / 2.f),
                                                1.0f,
                                                MAX_FP8 / (MAX_FP8 * 3.5f),
                                                MAX_FP8 / (MAX_FP8 * 4.f)});
            auto shift = op::v0::Constant::create(prec, shift_shape, {0.f, 0.f, 0.f, 0.f});

            auto op = make_shared<op::v13::FakeConvert>(data, scale, shift, destination_type);

            function = make_shared<Model>(OutputVector{op}, ParameterVector{data});
        }
    }

    void generate_inputs(const std::vector<ov::Shape>& target_shapes) override {
        inputs.clear();
        const float MAX_FP8 = (destination_type == ov::element::f8e4m3) ? fp8::MAX_F8E4M3 : fp8::MAX_F8E5M2;
        const auto& func_inputs = function->inputs();
        auto& data_input = func_inputs[0];
        ov::Tensor tensor = ov::Tensor(data_input.get_element_type(), target_shapes[0]);
        std::vector<float> input_data{MAX_FP8 / 4.f,
                                    MAX_FP8 / 3.f,
                                    MAX_FP8 / 2.f,
                                    MAX_FP8,
                                    MAX_FP8,
                                    MAX_FP8,
                                    MAX_FP8 * 1.2f,
                                    MAX_FP8 * 2.3f,
                                    MAX_FP8 * 3.4f,
                                    MAX_FP8 * 2.f,
                                    MAX_FP8 * 3.f,
                                    MAX_FP8 * 4.f};
        auto* data_ptr = tensor.data<float>();
        for (size_t i = 0; i < input_data.size(); i++) {
            data_ptr[i] = input_data[i];
        }
        inputs.insert({data_input.get_node_shared_ptr(), tensor});
    }
};

TEST_P(FakeConvertTest, Inference) {
    run();
}

const std::vector<ov::element::Type> input_precisions = {ov::element::f32};

const std::vector<ov::Shape> input_shapes = {{4, 3}};

const ov::Shape scale_shape = {4, 1};
const std::vector<ov::Shape> shift_shapes = {{4, 1}, {}};
const std::vector<ov::element::Type> destination_types = {ov::element::f8e4m3, ov::element::f8e5m2};

INSTANTIATE_TEST_SUITE_P(Smoke_FakeConvertTest,
                         FakeConvertTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::Values(scale_shape),
                                            ::testing::ValuesIn(shift_shapes),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::ValuesIn(destination_types),
                                            ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                                            FakeConvertTest::getTestCaseName);
} // namespace
