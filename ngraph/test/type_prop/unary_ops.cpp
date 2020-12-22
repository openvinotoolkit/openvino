//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//  Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace ngraph;
using InputType = element::Type_t;
using UnaryParams = std::tuple<InputType, Shape>;

template <class Op>
void basic_param_inference(InputType type, Shape shape)
{
    const auto param = std::make_shared<op::Parameter>(type, shape);
    const auto op = std::make_shared<Op>(param);
    ASSERT_EQ(op->get_shape(), (shape));
    ASSERT_EQ(op->get_element_type(), type);
};

template <class Op>
void incompatible_input_type(InputType type, Shape shape)
{
    const auto param = std::make_shared<op::Parameter>(type, shape);
    ASSERT_THROW(std::make_shared<Op>(param), ngraph::NodeValidationFailure);
};

template <class Op>
void dynamic_rank_input_shape(InputType type)
{
    const auto param = std::make_shared<op::Parameter>(type, PartialShape::dynamic());
    const auto op = std::make_shared<Op>(param);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
};

class BasicParamInference : public testing::WithParamInterface<UnaryParams>, public testing::Test
{
public:
    InputType input_type;
    Shape input_shape;

    static std::string getTestCaseName(testing::TestParamInfo<UnaryParams> obj)
    {
        InputType type;
        Shape shape;
        std::ostringstream result;
        std::tie(type, shape) = obj.param;
        result << type << "_";
        result << shape;
        auto test_name = result.str();
        std::replace_if(
            test_name.begin(), test_name.end(), [](char c) { return !std::isalnum(c); }, '_');
        test_name.erase(test_name.end() - 1);
        return test_name;
    }

    void SetUp()
    {
        const auto& type = std::get<0>(GetParam());
        const auto& shape = std::get<1>(GetParam());

        input_type = type;
        input_shape = shape;
    }
};

class IncompatibleInputType : public testing::WithParamInterface<UnaryParams>, public testing::Test
{
public:
    InputType input_type;
    Shape input_shape;

    static std::string getTestCaseName(testing::TestParamInfo<UnaryParams> obj)
    {
        InputType type;
        Shape shape;
        std::ostringstream result;
        std::tie(type, shape) = obj.param;
        result << type << "_";
        result << shape;
        auto test_name = result.str();
        std::replace_if(
            test_name.begin(), test_name.end(), [](char c) { return !std::isalnum(c); }, '_');
        test_name.erase(test_name.end() - 1);
        return test_name;
    }

    void SetUp()
    {
        const auto& type = std::get<0>(GetParam());
        const auto& shape = std::get<1>(GetParam());

        input_type = type;
        input_shape = shape;
    }
};

class DynamicRankInputShape : public testing::WithParamInterface<std::tuple<InputType>>,
                              public testing::Test
{
public:
    InputType input_type;

    static std::string getTestCaseName(testing::TestParamInfo<std::tuple<InputType>> obj)
    {
        InputType type;
        std::ostringstream result;
        type = std::get<0>(obj.param);
        result << type;
        return result.str();
    }

    void SetUp()
    {
        const auto& type = std::get<0>(GetParam());
        input_type = type;
    }
};

TEST_P(BasicParamInference, acos)
{
    basic_param_inference<op::Acos>(input_type, input_shape);
}

TEST_P(IncompatibleInputType, acos)
{
    incompatible_input_type<op::Acos>(input_type, input_shape);
}

TEST_P(DynamicRankInputShape, acos)
{
    dynamic_rank_input_shape<op::Acos>(input_type);
}

INSTANTIATE_TEST_CASE_P(type_prop,
                        BasicParamInference,
                        testing::Values(std::make_tuple(InputType{element::f32}, Shape{2, 2}),
                                        std::make_tuple(InputType{element::i32}, Shape{2, 2}),
                                        std::make_tuple(InputType{element::i16}, Shape{100, 2, 50}),
                                        std::make_tuple(InputType{element::u16}, Shape{5, 243, 51}),
                                        std::make_tuple(InputType{element::f64}, Shape{17, 2, 50}),
                                        std::make_tuple(InputType{element::i64}, Shape{76, 15, 50}),
                                        std::make_tuple(InputType{element::f16},
                                                        Shape{2, 20, 5, 5})),
                        BasicParamInference::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    type_prop,
    IncompatibleInputType,
    testing::Values(std::make_tuple(InputType{element::boolean}, Shape{4, 2}),
                    std::make_tuple(InputType{element::boolean}, Shape{3, 3}),
                    std::make_tuple(InputType{element::boolean}, Shape{100, 3, 6}),
                    std::make_tuple(InputType{element::boolean}, Shape{3, 3, 10})),
    IncompatibleInputType::getTestCaseName);

INSTANTIATE_TEST_CASE_P(type_prop,
                        DynamicRankInputShape,
                        testing::Values(std::make_tuple(InputType{element::f64}),
                                        std::make_tuple(InputType{element::f32}),
                                        std::make_tuple(InputType{element::f16}),
                                        std::make_tuple(InputType{element::u64}),
                                        std::make_tuple(InputType{element::i64}),
                                        std::make_tuple(InputType{element::i32}),
                                        std::make_tuple(InputType{element::i16}),
                                        std::make_tuple(InputType{element::i8}),
                                        std::make_tuple(InputType{element::u32})),
                        DynamicRankInputShape::getTestCaseName);
