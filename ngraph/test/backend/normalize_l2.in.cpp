// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"
#include "util/engine/test_engines.hpp"


using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

// ----------------------- eps_mode = ngraph::op::EpsMode::ADD ----------------------- //

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_all_2d_add)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 1});
    float eps = 1e-7;
    auto f = make_shared<Function>(
        make_shared<op::v0::NormalizeL2>(A, axes, eps, ngraph::op::EpsMode::ADD),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0.18257418, 0.36514837, 0.5477226, 0.73029673}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_empty_2d_add)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{0}, vector<int64_t>{});
    float eps = 1e-7;
    auto f = make_shared<Function>(
        make_shared<op::v0::NormalizeL2>(A, axes, eps, ngraph::op::EpsMode::ADD),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{0, 3, 0, 8});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0, 1, 0, 1}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_0_2d_add)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{0});
    float eps = 1e-7;
    auto f = make_shared<Function>(
        make_shared<op::v0::NormalizeL2>(A, axes, eps, ngraph::op::EpsMode::ADD),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0.31622776, 0.4472136, 0.94868326, 0.8944272}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_1_3d_add)
{
    Shape shape{1, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{1});
    float eps = 1e-7;
    auto f = make_shared<Function>(
        make_shared<op::v0::NormalizeL2>(A, axes, eps, ngraph::op::EpsMode::ADD),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0.31622776, 0.4472136, 0.94868326, 0.8944272}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_1_2d_add)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{1});
    float eps = 1e-7;
    auto f = make_shared<Function>(
        make_shared<op::v0::NormalizeL2>(A, axes, eps, ngraph::op::EpsMode::ADD),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0.4472136, 0.8944272, 0.6, 0.8}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_2_3d_add)
{
    Shape shape{1, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{2});
    float eps = 1e-7;
    auto f = make_shared<Function>(
        make_shared<op::v0::NormalizeL2>(A, axes, eps, ngraph::op::EpsMode::ADD),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0.4472136, 0.8944272, 0.6, 0.8}),
                                  read_vector<float>(result)));
}

// ----------------------- eps_mode = ngraph::op::EpsMode::MAX ----------------------- //

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_all_2d_max)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 1});
    float eps = 1e-7;
    auto f = make_shared<Function>(
        make_shared<op::v0::NormalizeL2>(A, axes, eps, ngraph::op::EpsMode::ADD),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0.18257419, 0.36514837, 0.54772256, 0.73029674}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_empty_2d_max)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{0}, vector<int64_t>{});
    float eps = 1e-7;
    auto f = make_shared<Function>(
        make_shared<op::v0::NormalizeL2>(A, axes, eps, ngraph::op::EpsMode::MAX),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{0, 3, 0, 8});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0, 1, 0, 1}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_0_2d_max)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{0});
    float eps = 1e-7;
    auto f = make_shared<Function>(
        make_shared<op::v0::NormalizeL2>(A, axes, eps, ngraph::op::EpsMode::MAX),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0.31622777, 0.4472136, 0.9486833, 0.89442719}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_1_2d_max)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{1});
    float eps = 1e-7;
    auto f = make_shared<Function>(
        make_shared<op::v0::NormalizeL2>(A, axes, eps, ngraph::op::EpsMode::MAX),
        ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0.4472136, 0.89442719, 0.6, 0.8}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_123_4d_add)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{3}, vector<int64_t>{1, 2, 3});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.01428571f, 0.02857143f, 0.04285714f, 0.05714286f, 0.07142857f, 0.08571429f,
                     0.1f,        0.11428571f, 0.12857144f, 0.14285715f, 0.15714286f, 0.17142858f,
                     0.18571429f, 0.2f,        0.21428572f, 0.22857143f, 0.24285714f, 0.25714287f,
                     0.27142859f, 0.2857143f,  0.30000001f, 0.31428573f, 0.32857144f, 0.34285715f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_empty_4D_add)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{0}, vector<int64_t>{});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape,
        vector<float>(shape_size(data_shape), 1));
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_empty_4D_max)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{0}, vector<int64_t>{});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape,
        vector<float>(shape_size(data_shape), 1));
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_1_4d_add)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{1});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.0766965f,  0.14142136f, 0.19611613f, 0.24253564f, 0.28216633f, 0.31622776f,
                     0.34570536f, 0.37139067f, 0.39391932f, 0.41380295f, 0.43145549f, 0.44721359f,
                     0.99705452f, 0.98994946f, 0.98058069f, 0.97014254f, 0.95936549f, 0.94868332f,
                     0.93834311f, 0.92847669f, 0.91914505f, 0.91036648f, 0.90213418f, 0.89442718f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_1_5d_add)
{
    Shape data_shape{1, 2, 2, 2, 3};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{1});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.0766965f,  0.14142136f, 0.19611613f, 0.24253564f, 0.28216633f, 0.31622776f,
                     0.34570536f, 0.37139067f, 0.39391932f, 0.41380295f, 0.43145549f, 0.44721359f,
                     0.99705452f, 0.98994946f, 0.98058069f, 0.97014254f, 0.95936549f, 0.94868332f,
                     0.93834311f, 0.92847669f, 0.91914505f, 0.91036648f, 0.90213418f, 0.89442718f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_123_5d_add)
{
    Shape data_shape{1, 2, 2, 2, 3};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{3}, vector<int64_t>{1, 2, 3});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.02638899f, 0.04956816f, 0.070014f,   0.10555596f, 0.1239204f,  0.140028f,
                     0.18472293f, 0.19827265f, 0.210042f,   0.26388991f, 0.27262488f, 0.280056f,
                     0.34305686f, 0.34697714f, 0.35007f,    0.42222384f, 0.42132938f, 0.420084f,
                     0.50139081f, 0.49568161f, 0.49009803f, 0.58055776f, 0.57003385f, 0.560112f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_1_2x2_add)
{
    Shape data_shape{2, 2};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{1});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(data_shape,
                                         {0.44721353f, 0.89442706f, 0.60000002f, 0.80000001f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_1_2x4_add)
{
    Shape data_shape{2, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{1});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(data_shape,
                                         {0.18257418f,
                                          0.36514837f,
                                          0.54772252f,
                                          0.73029673f,
                                          0.37904903f,
                                          0.45485884f,
                                          0.53066862f,
                                          0.60647845f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_123_4d_max)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{3}, vector<int64_t>{1, 2, 3});
    float eps{5000};
    auto eps_mode = op::EpsMode::MAX;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::TestCase<TestEngine>(function);

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.01414214f, 0.02828427f, 0.04242641f, 0.05656854f, 0.07071068f, 0.08485281f,
                     0.09899495f, 0.11313709f, 0.12727922f, 0.14142136f, 0.15556349f, 0.16970563f,
                     0.18384777f, 0.1979899f,  0.21213204f, 0.22627418f, 0.2404163f,  0.25455844f,
                     0.26870057f, 0.28284273f, 0.29698485f, 0.31112698f, 0.32526913f, 0.33941126f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}
