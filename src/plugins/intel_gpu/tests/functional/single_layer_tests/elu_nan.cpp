// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

using EluNaNPropagationParams = std::tuple<ov::Shape,           // input data shape
                                           ov::element::Type>;  // input precision

// ELU must propagate NaN from the input to the output. The GPU kernel
// returned zero for a NaN input instead (ticket 37729).
class EluNaNPropagationTest : public testing::WithParamInterface<EluNaNPropagationParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<EluNaNPropagationParams>& obj) {
        const auto& [data_shape, precision] = obj.param;
        std::ostringstream result;
        result << "data_shape=" << ov::test::utils::vec2str(data_shape) << "_";
        result << "precision=" << precision.get_type_name();
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [data_shape, precision] = GetParam();

        inType = outType = precision;
        auto data = std::make_shared<ov::op::v0::Parameter>(precision, data_shape);
        data->set_friendly_name("data");
        auto elu = std::make_shared<ov::op::v0::Elu>(data, 1.0);
        elu->set_friendly_name("elu");
        auto result = std::make_shared<ov::op::v0::Result>(elu);
        function = std::make_shared<ov::Model>(result, ov::ParameterVector{data}, "EluNaNPropagation");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            const auto& shape = targetInputStaticShapes[i];
            auto tensor = ov::Tensor(funcInput.get_element_type(), shape);
            const size_t size = tensor.get_size();
            // Negative, positive and NaN values exercise every ELU branch.
            std::vector<float> values;
            values.reserve(size);
            for (size_t j = 0; j < size; ++j) {
                if (j % 3 == 0) {
                    values.push_back(-std::sqrt(static_cast<float>(j + 1)));
                } else if (j % 3 == 1) {
                    values.push_back(std::sqrt(static_cast<float>(j + 1)));
                } else {
                    values.push_back(std::numeric_limits<float>::quiet_NaN());
                }
            }
            if (funcInput.get_element_type() == ov::element::f32) {
                std::copy(values.begin(), values.end(), tensor.data<float>());
            } else if (funcInput.get_element_type() == ov::element::f16) {
                for (size_t j = 0; j < size; ++j) {
                    tensor.data<ov::float16>()[j] = ov::float16(values[j]);
                }
            } else {
                OPENVINO_THROW("Unsupported precision: ", funcInput.get_element_type());
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    // The default comparator may not match NaN against NaN, so compare here:
    // NaN must stay NaN, everything else must match the reference value.
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        ASSERT_EQ(expected.size(), actual.size());
        for (size_t j = 0; j < expected.size(); ++j) {
            ASSERT_EQ(expected[j].get_element_type(), actual[j].get_element_type());
            const size_t size = expected[j].get_size();
            const float tolerance = expected[j].get_element_type() == ov::element::f16 ? 2e-3f : 1e-6f;
            for (size_t i = 0; i < size; ++i) {
                const float exp_value = get_value(expected[j], i);
                const float act_value = get_value(actual[j], i);
                if (std::isnan(exp_value)) {
                    ASSERT_TRUE(std::isnan(act_value)) << "at index " << i << ": expected NaN, got " << act_value;
                } else {
                    ASSERT_NEAR(exp_value, act_value, tolerance) << "at index " << i;
                }
            }
        }
    }

private:
    static float get_value(const ov::Tensor& tensor, size_t index) {
        if (tensor.get_element_type() == ov::element::f32) {
            return tensor.data<const float>()[index];
        }
        return static_cast<float>(tensor.data<const ov::float16>()[index]);
    }
};

TEST_P(EluNaNPropagationTest, NansArePropagated) {
    run();
}

const std::vector<EluNaNPropagationParams> eluNaNParams = {
    {{1, 8}, ov::element::f32},
    {{2, 3, 4}, ov::element::f32},
    {{1, 4, 5, 6}, ov::element::f32},
    {{1, 8}, ov::element::f16},
    {{2, 3, 4}, ov::element::f16},
};

INSTANTIATE_TEST_SUITE_P(smoke_Elu_NaN_Propagation, EluNaNPropagationTest, ::testing::ValuesIn(eluNaNParams), EluNaNPropagationTest::getTestCaseName);
}  // namespace
