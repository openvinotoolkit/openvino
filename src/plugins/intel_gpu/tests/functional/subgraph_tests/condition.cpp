// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "common_test_utils/node_builders/reduce.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/runtime/core.hpp"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <utility>

namespace {
class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value) : m_name(name) {
        if (const char* previous = std::getenv(name)) {
            m_had_previous = true;
            m_previous = previous;
        }
#ifdef _WIN32
        _putenv_s(m_name.c_str(), value);
#else
        ::setenv(m_name.c_str(), value, 1);
#endif
    }

    ~ScopedEnvVar() {
#ifdef _WIN32
        _putenv_s(m_name.c_str(), m_had_previous ? m_previous.c_str() : "");
#else
        if (m_had_previous)
            ::setenv(m_name.c_str(), m_previous.c_str(), 1);
        else
            ::unsetenv(m_name.c_str());
#endif
    }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    std::string m_name;
    std::string m_previous;
    bool m_had_previous = false;
};

class ConditionMultiPortInputGPUTest : public testing::TestWithParam<bool> {};

TEST_P(ConditionMultiPortInputGPUTest, ExecutesBothBodiesWithProducerPorts) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto predicate = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{});
    auto count = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
    auto producer = std::make_shared<ov::op::v3::TopK>(data,
                                                      count,
                                                      1,
                                                      ov::op::v3::TopK::Mode::MAX,
                                                      ov::op::v3::TopK::SortType::SORT_VALUES,
                                                      ov::element::i32);
    auto then_values = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
    auto then_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 2});
    auto else_values = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
    auto else_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 2});
    auto sum = std::make_shared<ov::op::v1::Add>(
        then_values,
        std::make_shared<ov::op::v0::Convert>(then_indices, ov::element::f32));
    auto product = std::make_shared<ov::op::v1::Multiply>(
        else_values,
        std::make_shared<ov::op::v0::Convert>(else_indices, ov::element::f32));
    auto then_result = std::make_shared<ov::op::v0::Result>(sum);
    auto else_result = std::make_shared<ov::op::v0::Result>(product);
    auto conditional = std::make_shared<ov::op::v8::If>(predicate);
    conditional->set_then_body(std::make_shared<ov::Model>(ov::ResultVector{then_result},
                                                          ov::ParameterVector{then_values, then_indices}));
    conditional->set_else_body(std::make_shared<ov::Model>(ov::ResultVector{else_result},
                                                          ov::ParameterVector{else_values, else_indices}));
    conditional->set_input(producer->output(0), then_values, else_values);
    conditional->set_input(producer->output(1), then_indices, else_indices);
    conditional->set_output(then_result, else_result);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{conditional}, ov::ParameterVector{data, predicate});

    ScopedEnvVar new_shape_infer("OV_GPU_ALLOW_NEW_SHAPE_INFER", "1");
    ov::Core core;
    auto compiled = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    SCOPED_TRACE("then and else bodies return inputs directly");
    if (GetParam()) {
        std::stringstream blob;
        compiled.export_model(blob);
        compiled = core.import_model(blob, ov::test::utils::DEVICE_GPU);
    }
    auto request = compiled.create_infer_request();
    auto input = ov::Tensor(ov::element::f32, ov::Shape{1, 4});
    const std::vector<float> input_values{3.f, 1.f, 4.f, 2.f};
    std::copy(input_values.begin(), input_values.end(), input.data<float>());
    request.set_input_tensor(0, input);
    auto condition = ov::Tensor(ov::element::boolean, ov::Shape{});
    request.set_input_tensor(1, condition);
    for (const bool take_then : {true, false, true}) {
        SCOPED_TRACE(take_then);
        condition.data<bool>()[0] = take_then;
        request.infer();
        auto output = request.get_output_tensor();
        ASSERT_EQ(output.get_shape(), (ov::Shape{1, 2}));
        EXPECT_FLOAT_EQ(output.data<float>()[0], take_then ? 6.f : 8.f);
        EXPECT_FLOAT_EQ(output.data<float>()[1], take_then ? 3.f : 0.f);
    }
}

TEST_P(ConditionMultiPortInputGPUTest, ExecutesBothBodiesWithRepeatedProducerOutput) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto predicate = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{});
    auto count = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
    auto producer = std::make_shared<ov::op::v3::TopK>(data,
                                                     count,
                                                     1,
                                                     ov::op::v3::TopK::Mode::MAX,
                                                     ov::op::v3::TopK::SortType::SORT_VALUES,
                                                     ov::element::i32);
    auto then_first = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
    auto then_second = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
    auto else_first = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
    auto else_second = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
    auto then_result = std::make_shared<ov::op::v0::Result>(
        std::make_shared<ov::op::v1::Add>(then_first, then_second));
    auto else_result = std::make_shared<ov::op::v0::Result>(
        std::make_shared<ov::op::v1::Multiply>(else_first, else_second));
    auto conditional = std::make_shared<ov::op::v8::If>(predicate);
    conditional->set_then_body(std::make_shared<ov::Model>(ov::ResultVector{then_result},
                                                         ov::ParameterVector{then_first, then_second}));
    conditional->set_else_body(std::make_shared<ov::Model>(ov::ResultVector{else_result},
                                                         ov::ParameterVector{else_first, else_second}));
    conditional->set_input(producer->output(0), then_first, else_first);
    conditional->set_input(producer->output(0), then_second, else_second);
    conditional->set_output(then_result, else_result);
    ASSERT_EQ(conditional->get_input_size(), 3);
    ASSERT_EQ(conditional->input_value(1), conditional->input_value(2));
    auto model = std::make_shared<ov::Model>(ov::OutputVector{conditional}, ov::ParameterVector{data, predicate});

    ScopedEnvVar new_shape_infer("OV_GPU_ALLOW_NEW_SHAPE_INFER", "1");
    ov::Core core;
    auto compiled = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    if (GetParam()) {
        std::stringstream blob;
        compiled.export_model(blob);
        compiled = core.import_model(blob, ov::test::utils::DEVICE_GPU);
    }
    auto request = compiled.create_infer_request();
    auto input = ov::Tensor(ov::element::f32, ov::Shape{1, 4});
    request.set_input_tensor(0, input);
    auto condition = ov::Tensor(ov::element::boolean, ov::Shape{});
    request.set_input_tensor(1, condition);
    for (const float offset : {0.f, 2.f}) {
        SCOPED_TRACE(offset);
        const std::vector<float> input_values{3.f + offset, 1.f + offset, 4.f + offset, 2.f + offset};
        std::copy(input_values.begin(), input_values.end(), input.data<float>());
        for (const bool take_then : {true, false, true}) {
            SCOPED_TRACE(take_then);
            condition.data<bool>()[0] = take_then;
            request.infer();
            auto output = request.get_output_tensor();
            ASSERT_EQ(output.get_shape(), (ov::Shape{1, 2}));
            const float first = 4.f + offset;
            const float second = 3.f + offset;
            EXPECT_FLOAT_EQ(output.data<float>()[0], take_then ? first + first : first * first);
            EXPECT_FLOAT_EQ(output.data<float>()[1], take_then ? second + second : second * second);
        }
    }
}

TEST_P(ConditionMultiPortInputGPUTest, SkippedBranchPreservesDeclaredOutputOrder) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ScopedEnvVar new_shape_infer("OV_GPU_ALLOW_NEW_SHAPE_INFER", "1");
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto predicate = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{});
    auto count = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
    auto producer = std::make_shared<ov::op::v3::TopK>(data,
                                                      count,
                                                      1,
                                                      ov::op::v3::TopK::Mode::MAX,
                                                      ov::op::v3::TopK::SortType::SORT_VALUES,
                                                      ov::element::i32);

    auto make_body = [](const std::string& prefix) {
        auto first = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
        auto second = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 2});
        auto first_result = std::make_shared<ov::op::v0::Result>(first);
        auto second_result = std::make_shared<ov::op::v0::Result>(second);
        first_result->set_friendly_name(prefix + "_z_result_0");
        second_result->set_friendly_name(prefix + "_a_result_1");
        return std::make_pair(std::make_shared<ov::Model>(ov::ResultVector{first_result, second_result},
                                                          ov::ParameterVector{first, second}),
                              std::make_pair(first_result, second_result));
    };

    auto then_body = make_body("then");
    auto else_body = make_body("else");
    auto conditional = std::make_shared<ov::op::v8::If>(predicate);
    conditional->set_then_body(then_body.first);
    conditional->set_else_body(else_body.first);
    conditional->set_input(producer->output(0),
                            then_body.first->get_parameters()[0],
                            else_body.first->get_parameters()[0]);
    conditional->set_input(producer->output(1),
                            then_body.first->get_parameters()[1],
                            else_body.first->get_parameters()[1]);
    conditional->set_output(then_body.second.first, else_body.second.first);
    conditional->set_output(then_body.second.second, else_body.second.second);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{conditional->output(0), conditional->output(1)},
                                             ov::ParameterVector{data, predicate});

    ov::Core core;
    auto compiled = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    if (GetParam()) {
        std::stringstream blob;
        compiled.export_model(blob);
        compiled = core.import_model(blob, ov::test::utils::DEVICE_GPU);
    }

    auto request = compiled.create_infer_request();
    auto input = ov::Tensor(ov::element::f32, ov::Shape{1, 4});
    const std::vector<float> input_values{3.f, 1.f, 4.f, 2.f};
    std::copy(input_values.begin(), input_values.end(), input.data<float>());
    request.set_input_tensor(0, input);
    auto condition = ov::Tensor(ov::element::boolean, ov::Shape{});
    request.set_input_tensor(1, condition);
    for (const bool take_then : {true, false}) {
        SCOPED_TRACE(take_then);
        condition.data<bool>()[0] = take_then;
        request.infer();
        for (size_t output_idx = 0; output_idx < 2; ++output_idx) {
            const auto output = request.get_output_tensor(output_idx);
            ASSERT_EQ(output.get_shape(), (ov::Shape{1, 2}));
            if (output_idx == 0) {
                EXPECT_FLOAT_EQ(output.data<float>()[0], 4.f);
                EXPECT_FLOAT_EQ(output.data<float>()[1], 3.f);
            } else {
                EXPECT_EQ(output.data<int32_t>()[0], 2);
                EXPECT_EQ(output.data<int32_t>()[1], 0);
            }
        }
    }
}

TEST_P(ConditionMultiPortInputGPUTest, SkippedBranchReinterpretsGrowingSecondaryOutput) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ScopedEnvVar new_shape_infer("OV_GPU_ALLOW_NEW_SHAPE_INFER", "1");
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 10});
    auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{});
    auto predicate = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{});
    auto producer = std::make_shared<ov::op::v3::TopK>(data,
                                                      k,
                                                      1,
                                                      ov::op::v3::TopK::Mode::MAX,
                                                      ov::op::v3::TopK::SortType::SORT_VALUES,
                                                      ov::element::i32);

    auto make_body = [](const std::string& prefix) {
        auto values = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, ov::Dimension(1, 10)});
        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, ov::Dimension(1, 10)});
        auto values_result = std::make_shared<ov::op::v0::Result>(values);
        auto indices_result = std::make_shared<ov::op::v0::Result>(indices);
        values_result->set_friendly_name(prefix + "_values_result");
        indices_result->set_friendly_name(prefix + "_indices_result");
        return std::make_pair(std::make_shared<ov::Model>(ov::ResultVector{values_result, indices_result},
                                                          ov::ParameterVector{values, indices}),
                              std::make_pair(values_result, indices_result));
    };

    auto then_body = make_body("then");
    auto else_body = make_body("else");
    auto conditional = std::make_shared<ov::op::v8::If>(predicate);
    conditional->set_then_body(then_body.first);
    conditional->set_else_body(else_body.first);
    conditional->set_input(producer->output(0),
                            then_body.first->get_parameters()[0],
                            else_body.first->get_parameters()[0]);
    conditional->set_input(producer->output(1),
                            then_body.first->get_parameters()[1],
                            else_body.first->get_parameters()[1]);
    conditional->set_output(then_body.second.first, else_body.second.first);
    conditional->set_output(then_body.second.second, else_body.second.second);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{conditional->output(0), conditional->output(1)},
                                             ov::ParameterVector{data, k, predicate});

    ov::Core core;
    auto compiled = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    if (GetParam()) {
        std::stringstream blob;
        compiled.export_model(blob);
        compiled = core.import_model(blob, ov::test::utils::DEVICE_GPU);
    }
    auto request = compiled.create_infer_request();
    auto data_tensor = ov::Tensor(ov::element::f32, ov::Shape{1, 10});
    for (size_t i = 0; i < 10; ++i)
        data_tensor.data<float>()[i] = static_cast<float>(i + 1);
    request.set_input_tensor(0, data_tensor);
    auto k_tensor = ov::Tensor(ov::element::i64, ov::Shape{});
    request.set_input_tensor(1, k_tensor);
    auto condition = ov::Tensor(ov::element::boolean, ov::Shape{});
    request.set_input_tensor(2, condition);

    for (const int64_t top_k : {2, 8}) {
        SCOPED_TRACE(top_k);
        k_tensor.data<int64_t>()[0] = top_k;
        condition.data<bool>()[0] = true;
        request.infer();
        auto values = request.get_output_tensor(0);
        auto indices = request.get_output_tensor(1);
        ASSERT_EQ(values.get_shape(), (ov::Shape{1, static_cast<size_t>(top_k)}));
        ASSERT_EQ(indices.get_shape(), (ov::Shape{1, static_cast<size_t>(top_k)}));
        for (size_t i = 0; i < static_cast<size_t>(top_k); ++i) {
            EXPECT_FLOAT_EQ(values.data<float>()[i], static_cast<float>(10 - i));
            EXPECT_EQ(indices.data<int32_t>()[i], static_cast<int32_t>(9 - i));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_ConditionGPU,
                         ConditionMultiPortInputGPUTest,
                         testing::Values(false, true));

class InnerBodyGenerator {
public:
    using ptr = std::shared_ptr<InnerBodyGenerator>;

enum InnerBodyType {
    /**
     * Simple inner body with single constant value
    */
    Type01 = 1,
    /**
     * Inner body with eltwise sum
    */
    Type02 = 2,
    /**
     * Inner body with eltwise multiply
    */
    Type03 = 3,
    /**
     * Inner body with eltwise sum and pooling
     * output shape is different with type02 and type03 for same input shape
    */
    Type04 = 4,
    /**
     * Inner body with nested condition case
     */
    Type05 = 5,
    /**
     * Inner body with single constant with zero dimensions
     */
    Type06 = 6,
    /**
     * Inner body with single constant
    */
    Type07 = 7,
    /**
     * Inner body with single parameter
    */
    Type08 = 8
};

public:
    InnerBodyGenerator()  = default;
    virtual ~InnerBodyGenerator() = default;

    virtual std::shared_ptr<ov::Model> get_function() { return _func; }
    virtual std::shared_ptr<ov::op::v0::Parameter> get_input() { return _param; }
    virtual std::shared_ptr<ov::op::v0::Result> get_result() { return _result; }

    // virtual void create_body(ov::Shape input_shape, ov::element::Type prc) {
    virtual void create_body(ov::PartialShape& input_shape, ov::element::Type prc) {
        _func = generate(input_shape, prc);
        _param = (_func->get_parameters().size() > 0)? _func->get_parameters().front() : nullptr;
        _result = _func->get_results().front();
    }

protected:
    virtual std::shared_ptr<ov::Model> generate(ov::PartialShape& input_shape, ov::element::Type prc) = 0;

    std::shared_ptr<ov::Model> _func;
    std::shared_ptr<ov::op::v0::Parameter> _param;
    std::shared_ptr<ov::op::v0::Result> _result;
};

class InnerBodyType01 : public InnerBodyGenerator {
protected:
    std::shared_ptr<ov::Model> generate(ov::PartialShape& input_shape, ov::element::Type prc) override {
        auto constantA = ov::op::v0::Constant::create(prc, ov::Shape(input_shape.rank().get_length(), 2), {2.0f});
        constantA->set_friendly_name("body1_constantA");

        auto constantB = ov::op::v0::Constant::create(prc, ov::Shape(input_shape.rank().get_length(), 2), {12.0f});
        constantB->set_friendly_name("body1_constantB");

        auto add = std::make_shared<ov::op::v1::Add>(constantA, constantB);
        add->set_friendly_name("body1_add");

        auto result = std::make_shared<ov::op::v0::Result>(add);
        result->set_friendly_name("body1_result");

        auto body = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{}, "constant");
        return body;
    }
};

class InnerBodyType02 : public InnerBodyGenerator {
protected:
    std::shared_ptr<ov::Model> generate(ov::PartialShape& input_shape, ov::element::Type prc) override {
        auto constant = std::make_shared<ov::op::v0::Constant>(prc, ov::Shape{}, 10.0f);
        constant->set_friendly_name("body2_const");

        auto data = std::make_shared<ov::op::v0::Parameter>(prc, input_shape);
        data->set_friendly_name("body2_data");

        auto sum = std::make_shared<ov::op::v1::Multiply>(data, constant);
        sum->set_friendly_name("body2_mul");

        auto result = std::make_shared<ov::op::v0::Result>(sum);
        result->set_friendly_name("body2_result");

        auto body = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{data}, "eltwise_mul");
        return body;
    }
};

class InnerBodyType03 : public InnerBodyGenerator {
protected:
    std::shared_ptr<ov::Model> generate(ov::PartialShape& input_shape, ov::element::Type prc) override {
        auto constant = std::make_shared<ov::op::v0::Constant>(prc, ov::Shape{}, 2.0f);
        constant->set_friendly_name("body3_constant");

        auto data = std::make_shared<ov::op::v0::Parameter>(prc, input_shape);
        data->set_friendly_name("body3_data");

        auto add = std::make_shared<ov::op::v1::Add>(data, constant);
        add->set_friendly_name("body3_add");

        auto result = std::make_shared<ov::op::v0::Result>(add);
        result->set_friendly_name("body3_result");

        auto body = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{data}, "eltwise_sum");
        return body;
    }
};

class InnerBodyType04 : public InnerBodyGenerator {
protected:
    std::shared_ptr<ov::Model> generate(ov::PartialShape& input_shape, ov::element::Type prc) override {
        auto scale = std::make_shared<ov::op::v0::Constant>(prc, ov::Shape{}, 2.0f);
        scale->set_friendly_name("body4_scale");

        auto data = std::make_shared<ov::op::v0::Parameter>(prc, input_shape);
        data->set_friendly_name("body4_data");

        auto mul = std::make_shared<ov::op::v1::Multiply>(data, scale);
        mul->set_friendly_name("body4_mul");

        auto pooling = generate_pooling(mul, input_shape);
        pooling->set_friendly_name("body4_pool");

        auto result = std::make_shared<ov::op::v0::Result>(pooling);
        result->set_friendly_name("body4_result");

        auto body = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{data}, "eltwise_mul_pooling");
        return body;
    }

    struct poolSpecificParams {
            ov::test::utils::PoolingTypes pooling_type;   // Pooling type, max or avg
            std::vector<size_t>           kernel_size;    // Kernel size
            std::vector<size_t>           stride;         // Stride
            std::vector<size_t>           pad_begin;      // Pad begin
            std::vector<size_t>           pad_end;        // Pad end
            ov::op::RoundingType          rounding_type;  // Rounding type
            ov::op::PadType               pad_type;       // Pad type
            bool                          exclued_pad;    // Exclude pad
    };

    std::shared_ptr<ov::Node> generate_pooling(const ov::Output<ov::Node> &in, ov::PartialShape& input_shape) {
        poolSpecificParams params;
        switch (input_shape.rank().get_length()) {
            case 5:
            {
                params = poolSpecificParams{ ov::test::utils::PoolingTypes::MAX,
                                                    {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                                                    ov::op::RoundingType::CEIL,
                                                    ov::op::PadType::SAME_LOWER, true };
                break;
            }
            case 4:
            {
                params = poolSpecificParams{ ov::test::utils::PoolingTypes::MAX,
                                                    {2, 2}, {2, 2}, {0, 0}, {0, 0},
                                                    ov::op::RoundingType::CEIL,
                                                    ov::op::PadType::SAME_LOWER, true };
                break;
            }
            case 3:
            {
                params = poolSpecificParams{ ov::test::utils::PoolingTypes::MAX,
                                                    {2}, {2}, {0}, {0},
                                                    ov::op::RoundingType::CEIL,
                                                    ov::op::PadType::SAME_LOWER, true };
                break;
            }
            default:
            {
                OPENVINO_ASSERT(false, "Not allowed other rank");
            }
        }
        if (ov::test::utils::PoolingTypes::MAX == params.pooling_type) {
            return std::make_shared<ov::op::v1::MaxPool>(in,
                                                         params.stride,
                                                         params.pad_begin,
                                                         params.pad_end,
                                                         params.kernel_size,
                                                         params.rounding_type,
                                                         params.pad_type);
        } else {
            return std::make_shared<ov::op::v1::AvgPool>(in,
                                                         params.stride,
                                                         params.pad_begin,
                                                         params.pad_end,
                                                         params.kernel_size,
                                                         params.exclued_pad,
                                                         params.rounding_type,
                                                         params.pad_type);
        }
    }
};

class InnerBodyType05 : public InnerBodyGenerator {
protected:
    std::shared_ptr<ov::Model> generate(ov::PartialShape& input_shape, ov::element::Type prc) override {
        auto constant = std::make_shared<ov::op::v0::Constant>(prc, ov::Shape{}, 2.0f);
        constant->set_friendly_name("body5_constant");

        auto data = std::make_shared<ov::op::v0::Parameter>(prc, input_shape);
        data->set_friendly_name("body5_data");

        auto add = std::make_shared<ov::op::v1::Add>(data, constant);
        add->set_friendly_name("body5_add");

        std::vector<int> axes;
        for (int i = 0, r = 0; i < input_shape.rank().get_length(); i++) {
            axes.push_back(r--);
        }

        std::vector<size_t> shapeAxes;
        shapeAxes.push_back(axes.size());

        std::shared_ptr<ov::Node> reductionAxesNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape(shapeAxes), axes);

        const auto reduce = ov::test::utils::make_reduce(add, reductionAxesNode, false, ov::test::utils::ReductionType::Min);
        reduce->set_friendly_name("body5_reduce");

        auto constant_ref = std::make_shared<ov::op::v0::Constant>(prc, ov::Shape{}, 10.0f);
        constant_ref->set_friendly_name("body5_ref_constant");

        auto pred = std::make_shared<ov::op::v1::GreaterEqual>(reduce, constant_ref);
        pred->set_friendly_name("nested_pred");

        auto nested_body_then_generator = std::make_shared<InnerBodyType03>();
        auto nested_body_else_generator = std::make_shared<InnerBodyType04>();

        auto nested_input_shape = add->get_output_partial_shape(0);
        nested_body_then_generator->create_body(nested_input_shape, prc);
        nested_body_else_generator->create_body(nested_input_shape, prc);
        nested_body_then_generator->get_function()->set_friendly_name("nested_then_inner_body");
        nested_body_else_generator->get_function()->set_friendly_name("nested_else_inner_body");

        auto cond_nested = std::make_shared<ov::op::v8::If>(pred);
        cond_nested->set_friendly_name("if_operator_nested");
        cond_nested->set_else_body(nested_body_else_generator->get_function());
        cond_nested->set_then_body(nested_body_then_generator->get_function());
        cond_nested->set_input(add, nested_body_then_generator->get_input(), nested_body_else_generator->get_input());
        cond_nested->set_output(nested_body_then_generator->get_result(), nested_body_else_generator->get_result());

        auto result = std::make_shared<ov::op::v0::Result>(cond_nested);
        result->set_friendly_name("body5_result");

        auto body = std::make_shared<ov::Model>(ov::OutputVector {result}, ov::ParameterVector{data}, "eltwise_sum");
        return body;
    }
};

class InnerBodyType06 : public InnerBodyGenerator {
protected:
    std::shared_ptr<ov::Model> generate(ov::PartialShape& input_shape, ov::element::Type prc) override {
        auto constant = ov::op::v0::Constant::create(prc, ov::Shape(input_shape.rank().get_length(), 1), {2.0f});
        constant->set_friendly_name("body6_constant");

        auto result = std::make_shared<ov::op::v0::Result>(constant);
        result->set_friendly_name("body6_result");

        auto body = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{}, "constant_only");
        return body;
    }
};

class InnerBodyType07 : public InnerBodyGenerator {
protected:
    std::shared_ptr<ov::Model> generate(ov::PartialShape& input_shape, ov::element::Type prc) override {
        auto constant = ov::op::v0::Constant::create(prc, input_shape.to_shape(), {2.0f});
        constant->set_friendly_name("body7_constant");

        auto result = std::make_shared<ov::op::v0::Result>(constant);
        result->set_friendly_name("body7_result");

        auto body = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{}, "constant_to_result");
        return body;
    }
};

class InnerBodyType08 : public InnerBodyGenerator {
protected:
    std::shared_ptr<ov::Model> generate(ov::PartialShape& input_shape, ov::element::Type prc) override {
        auto constant = std::make_shared<ov::op::v0::Constant>(prc, ov::Shape{}, 10.0f);
        constant->set_friendly_name("body8_const");

        auto data = std::make_shared<ov::op::v0::Parameter>(prc, input_shape);
        data->set_friendly_name("body8_data");

        auto result = std::make_shared<ov::op::v0::Result>(data);
        result->set_friendly_name("body8_result");

        auto body = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{data}, "parameter_to_result");
        return body;
    }
};

static std::shared_ptr<InnerBodyGenerator> get_inner_body_generator(InnerBodyGenerator::InnerBodyType type) {
    std::shared_ptr<InnerBodyGenerator> generator_ptr;
    switch (type) {
        case InnerBodyGenerator::InnerBodyType::Type01:
        {
            return std::make_shared<InnerBodyType01>();
        }
        case InnerBodyGenerator::InnerBodyType::Type02:
        {
            return std::make_shared<InnerBodyType02>();
        }
        case InnerBodyGenerator::InnerBodyType::Type03:
        {
            return std::make_shared<InnerBodyType03>();
        }
        case InnerBodyGenerator::InnerBodyType::Type04:
        {
            return std::make_shared<InnerBodyType04>();
        }
        case InnerBodyGenerator::InnerBodyType::Type05:
        {
            return std::make_shared<InnerBodyType05>();
        }
        case InnerBodyGenerator::InnerBodyType::Type06:
        {
            return std::make_shared<InnerBodyType06>();
        }
        case InnerBodyGenerator::InnerBodyType::Type07:
        {
            return std::make_shared<InnerBodyType07>();
        }
        case InnerBodyGenerator::InnerBodyType::Type08:
        {
            return std::make_shared<InnerBodyType08>();
        }
        default:
        {
            OPENVINO_ASSERT(false, "Not supported type");
        }
    }
}

class TestModelGenerator {
public:
    enum PredicateTypes {
        PARAM,
        NODE
    };

public:
    TestModelGenerator(InnerBodyGenerator::InnerBodyType then_body_type,
                       InnerBodyGenerator::InnerBodyType else_body_type,
                       PredicateTypes pred_type,
                       ov::element::Type prc,
                       ov::PartialShape input_shape,
                       bool cond_execution_value = false) {
        body_then_generator = get_inner_body_generator(then_body_type);
        body_else_generator = get_inner_body_generator(else_body_type);

        body_then_generator->create_body(input_shape, prc);
        body_else_generator->create_body(input_shape, prc);
        body_else_generator->get_function()->set_friendly_name("else_inner_body");
        body_then_generator->get_function()->set_friendly_name("then_inner_body");

        ov::ParameterVector params{};
        auto predicate = create_cond_execution(pred_type, params, ov::element::boolean, ov::Shape{});
        predicate->set_friendly_name("if_predicate");

        auto data = create_condition_input(params, prc, input_shape);
        data->set_friendly_name("input_data");

        auto cond = std::make_shared<ov::op::v8::If>(predicate);
        cond->set_friendly_name("if_operator");
        cond->set_else_body(body_else_generator->get_function());
        cond->set_then_body(body_then_generator->get_function());
        cond->set_input(data, body_then_generator->get_input(), body_else_generator->get_input());
        cond->set_output(body_then_generator->get_result(), body_else_generator->get_result());

        if (then_body_type == InnerBodyGenerator::InnerBodyType::Type06 || else_body_type == InnerBodyGenerator::InnerBodyType::Type06) {
            auto constant = create_condition_input(params, prc, ov::Shape{1}, 0, true);
            auto addition = std::make_shared<ov::op::v1::Add>(cond, constant);
            auto shapeof1 = std::make_shared<ov::op::v3::ShapeOf>(addition);
            auto convert = std::make_shared<ov::op::v0::Convert>(shapeof1, prc);
            auto mul = std::make_shared<ov::op::v1::Multiply>(convert, constant);
            auto shapePatternsNode = create_condition_input(params, ov::element::i64, ov::Shape{1}, 0, true);
            auto reshapeOp = std::make_shared<ov::op::v1::Reshape>(mul, shapePatternsNode, true);
            auto result = std::make_shared<ov::op::v0::Result>(reshapeOp);
            result->set_friendly_name("outer_result");
            function = std::make_shared<ov::Model>(ov::OutputVector {result}, params);
        } else {
            auto result = std::make_shared<ov::op::v0::Result>(cond);
            result->set_friendly_name("outer_result");
            function = std::make_shared<ov::Model>(ov::OutputVector {result}, params);
        }
    }
    std::shared_ptr<ov::Model> get_function() { return function; }

private:
    std::shared_ptr<ov::Node> create_condition_input(ov::ParameterVector& params,
        const ov::element::Type prc, const ov::PartialShape& shape,
        int value = 0, bool is_static = false) {
        if (is_static)
            return std::make_shared<ov::op::v0::Constant>(prc, shape.to_shape(), value);

        auto input = std::make_shared<ov::op::v0::Parameter>(prc, shape);
        params.push_back(input);
        return input;
    }

    std::shared_ptr<ov::Node> create_cond_execution(PredicateTypes pred_type,
                                                    ov::ParameterVector& params,
                                                    const ov::element::Type prc = ov::element::u8,
                                                    const ov::Shape shape = ov::Shape{}) {
        std::shared_ptr<ov::Node> pred;
        switch (pred_type) {
            case PredicateTypes::PARAM:
            {
                pred = create_condition_input(params, prc, shape);
                break;
            }
            case PredicateTypes::NODE:
            {
                auto param_cond = create_condition_input(params, prc, shape);
                param_cond->set_friendly_name("param_cond");
                auto const_cond = create_condition_input(params, prc, ov::Shape{}, 1, true);
                const_cond->set_friendly_name("const_cond");
                pred = std::make_shared<ov::op::v1::GreaterEqual>(param_cond, const_cond);
                pred->set_friendly_name("pred");
                break;
            }
            default:
            {
                OPENVINO_ASSERT(false, "Not supported type");
            }
        }
        return pred;
    }

private:
    std::shared_ptr<ov::Model> function;
    InnerBodyGenerator::ptr body_then_generator;
    InnerBodyGenerator::ptr body_else_generator;
};

static std::ostream& operator<<(std::ostream& os, const InnerBodyGenerator::InnerBodyType type) {
    switch (type) {
        case InnerBodyGenerator::InnerBodyType::Type01:
        {
            os << "Type01";
            break;
        }
        case InnerBodyGenerator::InnerBodyType::Type02:
        {
            os << "Type02";
            break;
        }
        case InnerBodyGenerator::InnerBodyType::Type03:
        {
            os << "Type03";
            break;
        }
        case InnerBodyGenerator::InnerBodyType::Type04:
        {
            os << "Type04";
            break;
        }
        case InnerBodyGenerator::InnerBodyType::Type05:
        {
            os << "Type05";
            break;
        }
        case InnerBodyGenerator::InnerBodyType::Type06:
        {
            os << "Type06";
            break;
        }
        case InnerBodyGenerator::InnerBodyType::Type07:
        {
            os << "Type07";
            break;
        }
        case InnerBodyGenerator::InnerBodyType::Type08:
        {
            os << "Type08";
            break;
        }
        default:
        {
            os << "NONE";
            break;
        }
    }
    return os;
}

static std::ostream& operator<<(std::ostream& os, const TestModelGenerator::PredicateTypes type) {
    switch (type) {
        case TestModelGenerator::PredicateTypes::PARAM:
        {
            os << "PARAM";
            break;
        }
        case TestModelGenerator::PredicateTypes::NODE:
        {
            os << "NODE";
            break;
        }
        default:
        {
            os << "NONE";
            break;
        }
    }
    return os;
}

using ConditionParams = typename std::tuple<
        ov::Shape,                           // Shape
        ov::element::Type,                   // Precision
        TestModelGenerator::PredicateTypes,  // if predicate type
        std::string>;                        // Device name


class StaticConditionLayerGPUTest : public testing::WithParamInterface<ConditionParams>,
                                    virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConditionParams>& obj) {
        const auto& [data_shape, model_type, pred, targetDevice] = obj.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::vec2str(data_shape) << "_";
        result << "netPRC=" << model_type << "_";
        result << "ifCond=" << pred << "_";
        result << "targetDevice=" << targetDevice << "_";
        auto res_str = result.str();
        std::replace(res_str.begin(), res_str.end(), '-', '_');
        return res_str;
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [_data_shape, _model_type, pred, _targetDevice] = GetParam();
        data_shape = _data_shape;
        model_type = _model_type;
        targetDevice = _targetDevice;
        const auto ngShape = ov::PartialShape{data_shape};

        TestModelGenerator model_generator(InnerBodyGenerator::InnerBodyType::Type02,
                                           InnerBodyGenerator::InnerBodyType::Type03,
                                           pred,
                                           model_type,
                                           ngShape);
        function = model_generator.get_function();
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& model_inputs = function->inputs();
        for (size_t i = 0; i < model_inputs.size(); ++i) {
            const auto& model_input = model_inputs[i];
            auto type = model_input.get_element_type();
            ov::Tensor tensor;
            if (ov::element::boolean == type) {
                tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(), target_input_static_shapes[i], 0, 0);
            } else {
                tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(), target_input_static_shapes[i], 0, 20);
            }
            inputs.insert({model_input.get_node_shared_ptr(), tensor});
        }
    }

    ov::Shape data_shape;
    ov::element::Type model_type;
};

TEST_P(StaticConditionLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

std::vector<ov::element::Type> netPrecisions_static = {
    ov::element::f32,
    ov::element::f16,
    ov::element::i8
};

std::vector<ov::Shape> inputs_shape = {
    {3, 6}
};

std::vector<TestModelGenerator::PredicateTypes> if_cond_types = {
    TestModelGenerator::PredicateTypes::PARAM
};

INSTANTIATE_TEST_SUITE_P(smoke_ConditionGPUTest_static, StaticConditionLayerGPUTest,
                testing::Combine(
                    testing::ValuesIn(inputs_shape),
                    testing::ValuesIn(netPrecisions_static),
                    testing::ValuesIn(if_cond_types),
                    testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                StaticConditionLayerGPUTest::getTestCaseName);


/// Static shape single layer test
class StaticConditionSingleLayerGPUTest : public testing::WithParamInterface<ConditionParams>,
                                          virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConditionParams>& obj) {
        const auto& [data_shape, model_type, pred, targetDevice] = obj.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::vec2str(data_shape) << "_";
        result << "netPRC=" << model_type << "_";
        result << "ifCond=" << pred << "_";
        result << "targetDevice=" << targetDevice << "_";
        auto res_str = result.str();
        std::replace(res_str.begin(), res_str.end(), '-', '_');
        return res_str;
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [_data_shape, _model_type, pred, _targetDevice] = GetParam();
        data_shape = _data_shape;
        model_type = _model_type;
        targetDevice = _targetDevice;
        const auto ngShape = ov::PartialShape{data_shape};

        TestModelGenerator model_generator(InnerBodyGenerator::InnerBodyType::Type07,
                                            InnerBodyGenerator::InnerBodyType::Type08,
                                            pred,
                                            model_type,
                                            ngShape);
        function = model_generator.get_function();
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& model_inputs = function->inputs();
        for (size_t i = 0; i < model_inputs.size(); ++i) {
            const auto& model_input = model_inputs[i];
            auto type = model_input.get_element_type();
            ov::Tensor tensor;
            if (ov::element::boolean == type) {
                tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(), target_input_static_shapes[i], 0, 0);
            } else {
                tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(), target_input_static_shapes[i], 0, 20);
            }
            inputs.insert({model_input.get_node_shared_ptr(), tensor});
        }
    }

    ov::Shape data_shape;
    ov::element::Type model_type;
};

TEST_P(StaticConditionSingleLayerGPUTest, Inference) {
    run();
}

std::vector<ov::element::Type> model_types_static_single = {
    ov::element::f32,
    ov::element::f16,
    ov::element::i8
};

std::vector<ov::Shape> inputs_shape_single = {
    {64}
};

INSTANTIATE_TEST_SUITE_P(smoke_ConditionGPUTest_static, StaticConditionSingleLayerGPUTest,
                testing::Combine(
                    testing::ValuesIn(inputs_shape_single),
                    testing::ValuesIn(model_types_static_single),
                    testing::ValuesIn(if_cond_types),
                    testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                StaticConditionLayerGPUTest::getTestCaseName);


/// Dynamic shape test
struct InnerBodyTypeParams {
    InnerBodyGenerator::InnerBodyType then_body_type;
    InnerBodyGenerator::InnerBodyType else_body_type;
};

using ov::test::InputShape;

using ConditionGPUParams = typename std::tuple<
        InputShape,                         // Input Shapes
        InnerBodyTypeParams,                // Inner body type
        ov::element::Type,                  // Type
        TestModelGenerator::PredicateTypes, // if predicate type
        std::string>;                       // Device name

class DynamicConditionLayerGPUTest : public testing::WithParamInterface<ConditionGPUParams>,
                                virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConditionGPUParams>& obj) {
        const auto& [inputShapes, bodyParams, model_type, condType, targetDevice] = obj.param;
        std::ostringstream result;
        result << "IS=(";
        result << ov::test::utils::partialShape2str({inputShapes.first}) << "_";
        for (size_t i = 0lu; i < inputShapes.second.size(); i++) {
            result << "{";
            result << ov::test::utils::vec2str(inputShapes.second[i]) << "_";
            result << "}_";
        }
        result << ")_";
        result << "innerBody={" << bodyParams.then_body_type << ", " << bodyParams.else_body_type << "}_";
        result << "netPRC=" << model_type << "_";
        result << "ifCond=" << condType << "_";
        result << "targetDevice=" << targetDevice << "_";
        auto res_str = result.str();
        std::replace(res_str.begin(), res_str.end(), '-', '_');
        return res_str;
    }

protected:
    void SetUp() override {
        const auto& [inputShapes, bodyParams, model_type, condType, _targetDevice] = GetParam();
        targetDevice = _targetDevice;
        auto num_second = inputShapes.second.size();
        std::vector<ov::Shape> condSecondVec;
        for (size_t i = 0; i < num_second; i++) {
            condSecondVec.push_back({});
        }
        auto condShapes = ov::test::InputShape(ov::PartialShape({}), condSecondVec);
        init_input_shapes({condShapes, inputShapes});

        TestModelGenerator model_generator(bodyParams.then_body_type,
                                            bodyParams.else_body_type,
                                            condType,
                                            model_type,
                                            inputShapes.first);
        function = model_generator.get_function();
        function->set_friendly_name("if_operator_outer");
    }

    /**
     * @brief Override generate_inputs to support boolean param for if(condition) operator.
     *
     * @param targetInputStaticShapes
     */
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        ov::Shape input_shape;
        for (auto& shape : targetInputStaticShapes) {
            // Change condition to cover 1 dim input shape
            if (shape.size() > 0) {
                input_shape = shape;
                break;
            }
        }

        inputs.clear();
        for (const auto &param : function->get_parameters()) {
            if (param->get_output_element_type(0) == ov::element::boolean) {
                auto tensor = ov::Tensor{ov::element::boolean, {}};
                auto p_data = tensor.data<ov::element_type_traits<ov::element::boolean>::value_type>();
                p_data[0] = (niter++ % 2);

                inputs.insert({param, tensor});
            } else {
                ov::test::utils::InputGenerateData inGenData;
                inGenData.range         = 10;
                inGenData.start_from    = 0;
                inGenData.resolution    = 128;
                inGenData.seed          = 1;
                auto tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(), input_shape, inGenData);
                inputs.insert({param, tensor});
            }
        }
    }

    size_t niter = 0;
};

TEST_P(DynamicConditionLayerGPUTest, CompareWithRefs) {
    run();
}

const std::vector<ov::test::InputShape> dynamicInputShapes_f32 = {
    ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1}), {{4, 1, 1, 64, 32}, {6, 1, 1, 8, 4}, {8, 1, 1, 24, 16}}),
    ov::test::InputShape(ov::PartialShape({1, 1, -1, -1}), {{1, 1, 64, 32}, {1, 1, 8, 4}, {1, 1, 24, 16}})
};

const std::vector<ov::test::InputShape> dynamicInputShapes_f16 = {
    ov::test::InputShape(ov::PartialShape({1, 1, -1, -1}), {{1, 1, 64, 32}, {1, 1, 8, 4}, {1, 1, 24, 16}}),
    ov::test::InputShape(ov::PartialShape({-1, -1, -1}), {{2, 24, 16}, {2, 64, 32}, {2, 8, 4}})
};

const std::vector<ov::test::InputShape> dynamicInputShapes_zero_dims = {
    ov::test::InputShape(ov::PartialShape({-1}), {{24}, {64}, {8}})
};

const std::vector<InnerBodyTypeParams> innerBodyTypes_f32 = {
    {
        InnerBodyGenerator::InnerBodyType::Type01,
        InnerBodyGenerator::InnerBodyType::Type02
    },
    {
        InnerBodyGenerator::InnerBodyType::Type02,
        InnerBodyGenerator::InnerBodyType::Type03
    }
};

const std::vector<InnerBodyTypeParams> innerBodyTypes_f16 = {
    {
        InnerBodyGenerator::InnerBodyType::Type04,
        InnerBodyGenerator::InnerBodyType::Type03
    },
    {
        InnerBodyGenerator::InnerBodyType::Type02,
        InnerBodyGenerator::InnerBodyType::Type05
    }
};

const std::vector<InnerBodyTypeParams> innerBodyTypes_zero_dims = {
    {
        InnerBodyGenerator::InnerBodyType::Type02,
        InnerBodyGenerator::InnerBodyType::Type06
    },
};

const std::vector<TestModelGenerator::PredicateTypes> condTypes = {
    TestModelGenerator::PredicateTypes::PARAM,
    TestModelGenerator::PredicateTypes::NODE
};

const std::vector<TestModelGenerator::PredicateTypes> condTypes_zero_dims = {
    TestModelGenerator::PredicateTypes::PARAM
};

INSTANTIATE_TEST_SUITE_P(smoke_ConditionGPUTest_dynamic_f32, DynamicConditionLayerGPUTest,
                testing::Combine(
                    testing::ValuesIn(dynamicInputShapes_f32),                          // input shapes
                    testing::ValuesIn(innerBodyTypes_f32),                              // inner body type
                    testing::Values(ov::element::f32),                               // network precision
                    testing::ValuesIn(condTypes),                                       // cond type
                    testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),         // device type
                DynamicConditionLayerGPUTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_ConditionGPUTest_dynamic_f16, DynamicConditionLayerGPUTest,
                testing::Combine(
                    testing::ValuesIn(dynamicInputShapes_f16),                          // input shapes
                    testing::ValuesIn(innerBodyTypes_f16),                              // inner body type
                    testing::Values(ov::element::f16),                               // network precision
                    testing::ValuesIn(condTypes),                                       // cond type
                    testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),         // device type
                DynamicConditionLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConditionGPUTest_zero_dims, DynamicConditionLayerGPUTest,
                testing::Combine(
                    testing::ValuesIn(dynamicInputShapes_zero_dims),                    // input shapes
                    testing::ValuesIn(innerBodyTypes_zero_dims),                        // inner body type
                    testing::Values(ov::element::f32),                               // network precision
                    testing::ValuesIn(condTypes_zero_dims),                             // cond type
                    testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),         // device type
                DynamicConditionLayerGPUTest::getTestCaseName);
} // namespace
