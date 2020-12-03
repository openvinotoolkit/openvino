//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
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
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

static const std::vector<ngraph::element::Type> base_types = {
    ngraph::element::from<float>(),
    ngraph::element::from<int32_t>(),
    ngraph::element::from<int64_t>(),
    ngraph::element::from<uint32_t>(),
    ngraph::element::from<uint64_t>(),
};

template <typename OP>
void make_unary_empty_test(const string& backend_name)
{
    Shape shape{0};

    ParameterVector params;
    NodeVector result_list;
    for (size_t i = 0; i < base_types.size(); i++)
    {
        shared_ptr<op::Parameter> p = make_shared<op::Parameter>(base_types[i], shape);
        params.push_back(p);
        result_list.push_back(make_shared<OP>(p));
    }

    auto f = make_shared<Function>(result_list, params);
    auto backend = runtime::Backend::create(backend_name);

    vector<shared_ptr<runtime::Tensor>> inputs;
    vector<shared_ptr<runtime::Tensor>> outputs;
    for (size_t i = 0; i < base_types.size(); i++)
    {
        inputs.push_back(backend->create_tensor(base_types[i], shape));
        outputs.push_back(backend->create_tensor(base_types[i], shape));
    }

    auto handle = backend->compile(f);
    handle->call_with_validate(outputs, inputs);

    EXPECT_EQ(read_vector<float>(inputs[0]).size(), 0);
    EXPECT_EQ(read_vector<int32_t>(inputs[1]).size(), 0);
    EXPECT_EQ(read_vector<int64_t>(inputs[2]).size(), 0);
    EXPECT_EQ(read_vector<uint32_t>(inputs[3]).size(), 0);
    EXPECT_EQ(read_vector<uint64_t>(inputs[4]).size(), 0);

    EXPECT_EQ(read_vector<float>(outputs[0]).size(), 0);
    EXPECT_EQ(read_vector<int32_t>(outputs[1]).size(), 0);
    EXPECT_EQ(read_vector<int64_t>(outputs[2]).size(), 0);
    EXPECT_EQ(read_vector<uint32_t>(outputs[3]).size(), 0);
    EXPECT_EQ(read_vector<uint64_t>(outputs[4]).size(), 0);
}

template <typename OP>
void make_binary_empty_test(const string& backend_name, bool is_comparison = false)
{
    Shape shape{0};
    ParameterVector A;
    for (size_t i = 0; i < base_types.size(); i++)
    {
        A.push_back(make_shared<op::Parameter>(base_types[i], shape));
    }

    NodeVector result_list;
    for (shared_ptr<op::Parameter> p : A)
    {
        result_list.push_back(make_shared<OP>(p, p));
    }

    auto f = make_shared<Function>(result_list, A);
    auto backend = runtime::Backend::create(backend_name);

    vector<shared_ptr<runtime::Tensor>> inputs;
    vector<shared_ptr<runtime::Tensor>> outputs;
    for (size_t i = 0; i < base_types.size(); i++)
    {
        inputs.push_back(backend->create_tensor(base_types[i], shape));
        if (is_comparison)
        {
            outputs.push_back(backend->create_tensor(element::from<char>(), shape));
        }
        else
        {
            outputs.push_back(backend->create_tensor(base_types[i], shape));
        }
    }

    auto handle = backend->compile(f);
    handle->call_with_validate(outputs, inputs);

    EXPECT_EQ(read_vector<float>(inputs[0]).size(), 0);
    EXPECT_EQ(read_vector<int32_t>(inputs[1]).size(), 0);
    EXPECT_EQ(read_vector<int64_t>(inputs[2]).size(), 0);
    EXPECT_EQ(read_vector<uint32_t>(inputs[3]).size(), 0);
    EXPECT_EQ(read_vector<uint64_t>(inputs[4]).size(), 0);

    if (is_comparison)
    {
        EXPECT_EQ(read_vector<char>(outputs[0]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[1]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[2]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[3]).size(), 0);
        EXPECT_EQ(read_vector<char>(outputs[4]).size(), 0);
    }
    else
    {
        EXPECT_EQ(read_vector<float>(outputs[0]).size(), 0);
        EXPECT_EQ(read_vector<int32_t>(outputs[1]).size(), 0);
        EXPECT_EQ(read_vector<int64_t>(outputs[2]).size(), 0);
        EXPECT_EQ(read_vector<uint32_t>(outputs[3]).size(), 0);
        EXPECT_EQ(read_vector<uint64_t>(outputs[4]).size(), 0);
    }
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_abs)
{
    make_unary_empty_test<op::Abs>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_ceiling)
{
    make_unary_empty_test<op::Ceiling>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_erf)
{
    make_unary_empty_test<op::Erf>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_exp)
{
    make_unary_empty_test<op::Exp>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_floor)
{
    make_unary_empty_test<op::Floor>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_log)
{
    make_unary_empty_test<op::Log>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_negative)
{
    make_unary_empty_test<op::Negative>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_sign)
{
    make_unary_empty_test<op::Sign>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_sqrt)
{
    make_unary_empty_test<op::Sqrt>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_sin)
{
    make_unary_empty_test<op::Sin>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_sinh)
{
    make_unary_empty_test<op::Sinh>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_cos)
{
    make_unary_empty_test<op::Cos>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_cosh)
{
    make_unary_empty_test<op::Cosh>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_tan)
{
    make_unary_empty_test<op::Tan>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_tanh)
{
    make_unary_empty_test<op::Tanh>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_asin)
{
    make_unary_empty_test<op::Asin>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_acos)
{
    make_unary_empty_test<op::Acos>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_atan)
{
    make_unary_empty_test<op::Atan>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_add)
{
    make_binary_empty_test<op::v1::Add>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_divide)
{
    make_binary_empty_test<op::v1::Divide>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_eq)
{
    make_binary_empty_test<op::v1::Equal>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_greater)
{
    make_binary_empty_test<op::v1::Greater>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_greatereq)
{
    make_binary_empty_test<op::v1::GreaterEqual>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_less)
{
    make_binary_empty_test<op::v1::Less>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_lesseq)
{
    make_binary_empty_test<op::v1::LessEqual>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_maximum)
{
    make_binary_empty_test<op::v1::Maximum>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_minimum)
{
    make_binary_empty_test<op::v1::Minimum>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_multiply)
{
    make_binary_empty_test<op::v1::Multiply>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_not_equal)
{
    make_binary_empty_test<op::v1::NotEqual>("${BACKEND_NAME}", true);
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_power)
{
    make_binary_empty_test<op::v1::Power>("${BACKEND_NAME}");
}

NGRAPH_TEST(${BACKEND_NAME}, zero_sized_subtract)
{
    make_binary_empty_test<op::v1::Subtract>("${BACKEND_NAME}");
}
