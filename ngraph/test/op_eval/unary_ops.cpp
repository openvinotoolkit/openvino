// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/hsigmoid.hpp"
#include "ngraph/op/hswish.hpp"
#include "ngraph/op/mish.hpp"
#include "ngraph/op/softplus.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;
using output_generator = std::function<float(float)>;

class unary_test_param
{
public:
    unary_test_param(const element::Type_t& _net_prec,
                     const Shape& _input_shape,
                     const std::function<void(unary_test_param*)> _func,
                     const std::vector<float>& _input,
                     const std::vector<float>& _output,
                     const std::string _op_name,
                     const double& _tolerance)
        : net_prec(_net_prec)
        , input_shape(_input_shape)
        , run_eval(_func)
        , input(_input)
        , output(_output)
        , gen_output(nullptr)
        , op_name(_op_name)
        , tolerance(_tolerance)
    {
    }

    unary_test_param(const element::Type_t& _net_prec,
                     const Shape& _input_shape,
                     const std::function<void(unary_test_param*)> _func,
                     const std::vector<float>& _input,
                     const output_generator _gen_output,
                     const std::string _op_name,
                     const double& _tolerance)
        : net_prec(_net_prec)
        , input_shape(_input_shape)
        , run_eval(_func)
        , input(_input)
        , output(_input)
        , gen_output(_gen_output)
        , op_name(_op_name)
        , tolerance(_tolerance)
    {
    }
    const element::Type_t net_prec;
    const Shape input_shape;
    const std::function<void(unary_test_param*)> run_eval;
    const std::vector<float> input;
    std::vector<float> output;
    const output_generator gen_output;
    const std::string op_name;
    const double tolerance;
};

template <typename T,
          element::Type_t net_precision,
          typename value_type = fundamental_type_for<net_precision>>
void run_eval(unary_test_param* test_param)
{
    auto p = make_shared<op::Parameter>(net_precision, test_param->input_shape);
    auto op = make_shared<T>(p);
    auto func = make_shared<Function>(OutputVector{op}, ParameterVector{p});
    auto result = make_shared<HostTensor>();
    std::vector<value_type> input_vec;
    std::vector<value_type> expected_vec;

    if (test_param->gen_output != nullptr)
        std::transform(test_param->output.begin(),
                       test_param->output.end(),
                       test_param->output.begin(),
                       test_param->gen_output);

    for (auto input_data : test_param->input)
        input_vec.push_back((value_type)input_data);
    for (auto output_data : test_param->output)
        expected_vec.push_back((value_type)output_data);

    ASSERT_TRUE(func->evaluate(
        {result}, {make_host_tensor<net_precision>(test_param->input_shape, input_vec)}));

    EXPECT_EQ(result->get_element_type(), net_precision);
    EXPECT_EQ(result->get_shape(), test_param->input_shape);
    std::vector<value_type> result_data = read_vector<value_type>(result);
    for (size_t i = 0; i < expected_vec.size(); i++)
        EXPECT_NEAR(result_data[i], expected_vec[i], test_param->tolerance);
}

template <typename T, element::Type_t net_precision>
unary_test_param make_test_param(const Shape input_shape,
                                 const std::vector<float> input,
                                 const std::vector<float> output,
                                 const double tolerance = 1e-5)
{
    std::function<void(unary_test_param*)> eval_func = run_eval<T, net_precision>;
    return unary_test_param(net_precision,
                            input_shape,
                            eval_func,
                            input,
                            output,
                            T::get_type_info_static().name,
                            tolerance);
}

template <typename T, element::Type_t net_precision>
unary_test_param make_test_param(const Shape input_shape,
                                 const std::vector<float> input,
                                 const output_generator gen_func,
                                 const double tolerance = 1e-5)
{
    std::function<void(unary_test_param*)> eval_func = run_eval<T, net_precision>;
    return unary_test_param(net_precision,
                            input_shape,
                            eval_func,
                            input,
                            gen_func,
                            T::get_type_info_static().name,
                            tolerance);
}

class UnaryEval : public ::testing::TestWithParam<unary_test_param>
{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<unary_test_param>& obj)
    {
        unary_test_param param = obj.param;
        element::Type pres_type{param.net_prec};
        std::ostringstream result;
        result << "OP_" << param.op_name << "_";
        result << "shape" << vec2str(param.input_shape) << "_";
        result << "netprcision" << pres_type.get_type_name() << "_";
        result << (param.gen_output == nullptr ? "manual" : "auto");
        return result.str();
    }

private:
    template <typename T>
    static std::string vec2str(const std::vector<T>& vec)
    {
        if (!vec.empty())
        {
            std::ostringstream result;
            result << "_";
            std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<T>(result, "_"));
            result << vec.back();
            return result.str();
        }
        return std::string("()");
    }
};

TEST_P(UnaryEval, compareWithExpected)
{
    unary_test_param param = GetParam();
    param.run_eval(&param);
}

const static std::vector<unary_test_param> unary_test_array = {
    make_test_param<ngraph::op::v5::HSigmoid, element::f32>(
        {3}, {-0.5f, 0.0f, 0.5f}, {0.416667f, 0.5f, 0.583333f}),
    make_test_param<ngraph::op::v4::HSwish, element::f32>(
        {3}, {-0.5f, 0.0f, 0.5f}, {-0.208333f, 0.0f, 0.29166667f}),
    make_test_param<ngraph::op::v4::Mish, element::f32>(
        {3}, {-1.0, 1.0, 20.0}, {-0.303401, 0.86509835720062256, 20.0}),
    make_test_param<ngraph::op::v4::SoftPlus, element::f32>(
        {4}, {-1.0, 0.0, 1.0, 20.0}, {0.31326166, 0.69314718, 1.3132616, 20.0})

};

INSTANTIATE_TEST_CASE_P(Unary,
                        UnaryEval,
                        ::testing::ValuesIn(unary_test_array),
                        UnaryEval::getTestCaseName);
