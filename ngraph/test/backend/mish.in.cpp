// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

template <element::Type_t Type, typename T = fundamental_type_for<Type>>
static void mish_test(const PartialShape& dynamic_shape,
                      const Shape& static_shape,
                      const double fp_tolerance = 1e-5)
{
    bool must_support_dynamic = dynamic_shape.is_dynamic();
    auto data = make_shared<op::Parameter>(Type, dynamic_shape);
    auto f = make_shared<Function>(make_shared<op::v4::Mish>(data), ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", must_support_dynamic);

    auto create_output_tensor = [&]() {
        if (must_support_dynamic)
            return backend->create_dynamic_tensor(Type, dynamic_shape);
        return backend->create_tensor(Type, dynamic_shape.get_shape());
    };

    auto a = backend->create_tensor(Type, static_shape);
    auto result = create_output_tensor();

    // generate input tensor (with possible type conversion)
    auto static_size = shape_size(static_shape);
    std::vector<T> expected;
    std::vector<T> input;
    {
        std::mt19937 gen{0}; // use fixed seed for reproducibility of the test
        std::normal_distribution<> d{0.0, 20.0};

        for (auto i = static_size; i > 0; i--)
        {
            auto x = static_cast<T>(d(gen));
            auto y =
                static_cast<T>(static_cast<double>(x) * std::tanh(std::log(1.0 + std::exp(x))));
            input.push_back(x);
            expected.push_back(y);
        }

        copy_data(a, input);
    }

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    auto actual = read_vector<T>(result);

    // size equility test
    EXPECT_EQ(actual.size(), static_size);
    EXPECT_EQ(result->get_shape(), static_shape);

    // backend is allowed to trade off accuracy for performance
    for (size_t i = 0; i < static_size; i++)
        EXPECT_NEAR(actual[i], expected[i], fp_tolerance) << "input[i] is " << input[i];
}

NGRAPH_TEST(${BACKEND_NAME}, mish_f32)
{
    mish_test<element::f32>({2, 5}, {2, 5});
    mish_test<element::f32>({2, 3, 4, 5}, {2, 3, 4, 5});
}

NGRAPH_TEST(${BACKEND_NAME}, mish_f16)
{
    mish_test<element::f16>({2, 5}, {2, 5});
    mish_test<element::f16>({2, 3, 4, 5}, {2, 3, 4, 5});
}

NGRAPH_TEST(${BACKEND_NAME}, mish_dynamic)
{
    mish_test<element::f32>(PartialShape::dynamic(), {2, 3, 4, 5});
    mish_test<element::f32>({2, Dimension::dynamic(), 4, 5}, {2, 3, 4, 5});
}