// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

// ----------------------- keep dims = false ----------------------- //

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_to_scalar) {
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, false), ParameterVector{A});

    // Create some tensors for input/output
    std::vector<float> a{1, 2, 3, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>({4});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_to_scalar_int8) {
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i8, shape);
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, false), ParameterVector{A});

    // Create some tensors for input/output
    std::vector<int8_t> a{1, 2, 3, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int8_t>({a});
    test_case.add_expected_output<int8_t>({4});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_matrix_columns) {
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, false), ParameterVector{A});

    // Create some tensors for input/output
    std::vector<float> a{1, 2, 3, 4, 5, 6};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape_rt, {5, 6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_matrix_rows) {
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, false), ParameterVector{A});

    // Create some tensors for input/output
    std::vector<float> a{1, 2, 3, 4, 5, 6};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape_rt, {2, 4, 6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_matrix_rows_int32) {
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, false), ParameterVector{A});

    // Create some tensors for input/output
    std::vector<int32_t> a{1, 2, 3, 4, 5, 6};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>({a});
    test_case.add_expected_output<int32_t>(shape_rt, {2, 4, 6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_3d_to_matrix_most_sig) {
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, false), ParameterVector{A});

    // Create some tensors for input/output
    std::vector<float> a{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape_rt, {19, 20, 21, 22, 23, 24, 25, 26, 27});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_3d_to_matrix_least_sig) {
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 2);
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, false), ParameterVector{A});

    // Create some tensors for input/output
    std::vector<float> a{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape_rt, {3, 6, 9, 12, 15, 18, 21, 24, 27});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_3d_to_vector) {
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, false), ParameterVector{A});

    // Create some tensors for input/output
    std::vector<float> a{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape_rt, {25.0f, 26.0f, 27.0f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_3d_to_scalar) {
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto axes = make_shared<op::Constant>(element::i32, Shape{3}, vector<int32_t>{0, 1, 2});
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, false), ParameterVector{A});

    // Create some tensors for input/output
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape_rt, {14.0f});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_3d_to_scalar_int32) {
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{};
    auto axes = make_shared<op::Constant>(element::i32, Shape{3}, vector<int32_t>{0, 1, 2});
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                 13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->create_tensor(element::i32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{14}), read_vector<int32_t>(result));
}

// ----------------------- keep dims = true ----------------------- //

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_keep_to_scalar) {
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, Shape{1, 1});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{4}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_keep_to_scalar_int8) {
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i8, shape);
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i8, shape);
    copy_data(a, vector<int8_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::i8, Shape{1, 1});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int8_t>{4}), read_vector<int8_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_keep_matrix_columns) {
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1, 2};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{5, 6}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_keep_matrix_rows) {
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 4, 6}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_keep_matrix_rows_int32) {
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{3, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::i32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{2, 4, 6}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_keep_3d_to_matrix_most_sig) {
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1, 3, 3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{19, 20, 21, 22, 23, 24, 25, 26, 27}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_keep_3d_to_matrix_least_sig) {
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 2);
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{3, 6, 9, 12, 15, 18, 21, 24, 27}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_keep_3d_to_vector) {
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1, 1, 3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(
        test::all_close_f((vector<float>{25.0f, 26.0f, 27.0f}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_keep_3d_to_scalar) {
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1, 1, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{3}, vector<int32_t>{0, 1, 2});
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{14.0f}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_keep_3d_to_scalar_int32) {
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{1, 1, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{3}, vector<int32_t>{0, 1, 2});
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                 13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->create_tensor(element::i32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{14}), read_vector<int32_t>(result));
}

// Dynamic

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_matrix_columns_dynamic) {
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    // Create some tensors for input/output
    Shape shape_a{3, 2};
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{5, 6}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_matrix_rows_dynamic) {
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    // Create some tensors for input/output
    Shape shape_a{3, 2};
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 4, 6}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_keep_matrix_columns_dynamic) {
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    // Create some tensors for input/output
    Shape shape_a{3, 2};
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{5, 6}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_max_keep_matrix_rows_dynamic) {
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f = make_shared<Function>(make_shared<op::v1::ReduceMax>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    // Create some tensors for input/output
    Shape shape_a{3, 2};
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 4, 6}), read_vector<float>(result)));
}
