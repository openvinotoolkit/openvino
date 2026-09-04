// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sqrt.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

using ExtremumNaNPropagationParams = std::tuple<bool,                // true: Maximum, false: Minimum
                                                bool,                // true: NaN operand is the second input, false: the first
                                                ov::Shape,           // input shape
                                                ov::element::Type>;  // input precision

// Maximum/Minimum must propagate a NaN produced inside the graph (e.g. by
// Sqrt(-1)) regardless of the operand order. The GPU fmax/fmin based kernels
// returned the non-NaN operand instead (ticket 37730).
class ExtremumNaNPropagationTest : public testing::WithParamInterface<ExtremumNaNPropagationParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ExtremumNaNPropagationParams>& obj) {
        const auto& [is_maximum, nan_is_second, shape, precision] = obj.param;
        std::ostringstream result;
        result << "op=" << (is_maximum ? "Maximum" : "Minimum") << "_";
        result << "nan_position=" << (nan_is_second ? "second" : "first") << "_";
        result << "shape=" << ov::test::utils::vec2str(shape) << "_";
        result << "precision=" << precision.get_type_name();
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> init_subgraph(bool is_maximum, bool nan_is_second, const ov::Shape& shape, const ov::element::Type precision) {
        auto data = std::make_shared<ov::op::v0::Parameter>(precision, shape);
        data->set_friendly_name("data");
        // Sqrt(-1) produces NaN inside the graph, as in the issue reproducer.
        auto invalid = std::make_shared<ov::op::v0::Sqrt>(data);
        std::shared_ptr<ov::Node> extremum;
        if (nan_is_second) {
            auto bound = std::make_shared<ov::op::v0::Constant>(precision, ov::Shape{}, std::vector<float>{1.5f});
            extremum = is_maximum ? std::static_pointer_cast<ov::Node>(std::make_shared<ov::op::v1::Maximum>(bound, invalid))
                                  : std::static_pointer_cast<ov::Node>(std::make_shared<ov::op::v1::Minimum>(bound, invalid));
        } else {
            auto bound = std::make_shared<ov::op::v0::Constant>(precision, ov::Shape{}, std::vector<float>{-1.5f});
            extremum = is_maximum ? std::static_pointer_cast<ov::Node>(std::make_shared<ov::op::v1::Maximum>(invalid, bound))
                                  : std::static_pointer_cast<ov::Node>(std::make_shared<ov::op::v1::Minimum>(invalid, bound));
        }
        auto result = std::make_shared<ov::op::v0::Result>(extremum);
        return std::make_shared<ov::Model>(result, ov::ParameterVector{data}, "ExtremumNaNPropagation");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [is_maximum, nan_is_second, shape, precision] = GetParam();

        inType = outType = precision;
        function = init_subgraph(is_maximum, nan_is_second, shape, precision);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInput = function->inputs().front();
        const auto& shape = targetInputStaticShapes.front();
        auto tensor = ov::Tensor(funcInput.get_element_type(), shape);
        const size_t size = tensor.get_size();
        // Values <= -1 make Sqrt produce NaN, values > 0 stay finite so both
        // paths of the operation are exercised.
        std::vector<float> values;
        values.reserve(size);
        for (size_t j = 0; j < size; ++j) {
            values.push_back(static_cast<float>(j % 2 == 0 ? -1.0 : 1.0 + (j / 2)));
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

    // NaN can't be compared with thresholds - check that NaN stays NaN and
    // every other element matches the reference value.
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

TEST_P(ExtremumNaNPropagationTest, NansArePropagated) {
    run();
}

const std::vector<ExtremumNaNPropagationParams> extremumNaNParams = {
    // The issue case: Maximum(0, Sqrt(-1)) with a constant first operand.
    {true, true, {1}, ov::element::f32},
    {true, true, {1, 8}, ov::element::f32},
    {true, true, {2, 3, 4}, ov::element::f32},
    // NaN operand first.
    {true, false, {1, 8}, ov::element::f32},
    {true, false, {2, 3, 4}, ov::element::f32},
    // Minimum propagates NaN the same way.
    {false, true, {1, 8}, ov::element::f32},
    {false, false, {1, 8}, ov::element::f32},
    // f16.
    {true, true, {1, 8}, ov::element::f16},
    {true, false, {2, 3, 4}, ov::element::f16},
    {false, true, {2, 3, 4}, ov::element::f16},
};

INSTANTIATE_TEST_SUITE_P(smoke_Extremum_NaN_Propagation,
                         ExtremumNaNPropagationTest,
                         ::testing::ValuesIn(extremumNaNParams),
                         ExtremumNaNPropagationTest::getTestCaseName);
}  // namespace
