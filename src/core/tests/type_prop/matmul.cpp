// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/matmul.hpp"

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"

using namespace std;
using namespace ov;
using namespace testing;

TEST(type_prop, matmul_2D_same) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{2, 2}));
}

TEST(type_prop, matmul_4D_same) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2, 3, 3});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2, 3, 3});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{2, 2, 3, 3}));
}

TEST(type_prop, matmul_2D) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 6});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{6, 4});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{3, 4}));
}

TEST(type_prop, matmul_4D) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2, 3, 6});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2, 6, 4});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{2, 2, 3, 4}));
}

TEST(type_prop, matmul_5D_x_3D_transpose_a_transpose_b) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 1, 6, 3});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{7, 1, 5, 4, 6});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, true, true);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{7, 2, 5, 3, 4}));
}

TEST(type_prop, matmul_2D_transpose_a) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{6, 3});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{6, 4});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, 1);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{3, 4}));
}

TEST(type_prop, matmul_4D_transpose_a) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2, 6, 3});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2, 6, 4});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, 1);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{2, 2, 3, 4}));
}

TEST(type_prop, matmul_2D_transpose_b) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 6});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 6});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, 0, 1);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{3, 4}));
}

TEST(type_prop, matmul_4D_transpose_b) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2, 3, 6});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2, 4, 6});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, 0, 1);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{2, 2, 3, 4}));
}

TEST(type_prop, matmul_dynamic_5D_transpose_b) {
    Dimension dynamic = Dimension::dynamic();
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{dynamic, 4, dynamic, dynamic, 6});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, dynamic, dynamic, 4, 6});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, 0, 1);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_output_partial_shape(0), (PartialShape{Dimension(-1), 4, dynamic, dynamic, 4}));
}

TEST(type_prop, matmul_dynamic_2D_transpose_a) {
    Dimension dynamic = Dimension::dynamic();
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{dynamic, 3});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{4, dynamic});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, 1, 0);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_output_partial_shape(0), (PartialShape{3, dynamic}));
}

TEST(type_prop, matmul_dynamic_1D_3D) {
    Dimension dynamic = Dimension::dynamic();
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{dynamic});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, 4, dynamic});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_output_partial_shape(0), (PartialShape{2, dynamic}));
}

// Transpose attributes are ignored for 1D
// 1D x 1D
TEST(type_prop, matmul_1D_x_1D_false_false) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{}));
}

TEST(type_prop, matmul_1D_x_1D_false_true) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, false, true);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{}));
}

TEST(type_prop, matmul_1D_x_1D_true_false) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, true, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{}));
}

TEST(type_prop, matmul_1D_x_1D_true_true) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, true, true);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{}));
}

TEST(type_prop, matmul_1D_x_1D_incompatible) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});

    try {
        auto matmul = make_shared<ov::op::v0::MatMul>(A, B);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible matrix dimensions not detected. ";
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Incompatible MatMul matrix dimension"));
    } catch (...) {
        FAIL() << "MatMul shape validation failed for unexpected reason";
    }
}

// 2D x 1D
TEST(type_prop, matmul_2D_x_1D_false_false) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{1}));
}

TEST(type_prop, matmul_2D_x_1D_false_true) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, false, true);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{1}));
}

TEST(type_prop, matmul_2D_x_1D_true_false) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});

    try {
        auto matmul = make_shared<ov::op::v0::MatMul>(A, B, true, false);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible matrix dimensions not detected. ";
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Incompatible MatMul matrix dimension"));
    } catch (...) {
        FAIL() << "MatMul shape validation failed for unexpected reason";
    }
}

TEST(type_prop, matmul_2D_x_1D_true_true) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});

    try {
        auto matmul = make_shared<ov::op::v0::MatMul>(A, B, true, true);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible matrix dimensions not detected. ";
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Incompatible MatMul matrix dimension"));
    } catch (...) {
        FAIL() << "MatMul shape validation failed for unexpected reason";
    }
}

// 1D x 2D
TEST(type_prop, matmul_1D_x_2D_false_false) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 1});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{1}));
}

TEST(type_prop, matmul_1D_x_2D_false_true) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 1});

    try {
        auto matmul = make_shared<ov::op::v0::MatMul>(A, B, false, true);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible matrix dimensions not detected. ";
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Incompatible MatMul matrix dimension"));
    } catch (...) {
        FAIL() << "MatMul shape validation failed for unexpected reason";
    }
}

TEST(type_prop, matmul_1D_x_2D_true_false) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 1});
    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, true, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{1}));
}

TEST(type_prop, matmul_1D_x_2D_true_true) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 1});

    try {
        auto matmul = make_shared<ov::op::v0::MatMul>(A, B, true, true);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible matrix dimensions not detected. ";
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Incompatible MatMul matrix dimension"));
    } catch (...) {
        FAIL() << "MatMul shape validation failed for unexpected reason";
    }
}

// 1D x 4D
TEST(type_prop, matmul_1D_x_4D_false_false) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{1, 2, 4}));
}

// 4D x 1D
TEST(type_prop, matmul_4D_x_1D_false_false) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{1, 2, 3}));
}

// Batch broadcast
TEST(type_prop, matmul_batch_broadcast) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 1, 1, 4, 3});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 6, 3, 2});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{5, 1, 6, 4, 2}));
}

TEST(type_prop, matmul_batch_broadcast_expand_to_A) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 3});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{7, 8, 5, 3, 2});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{7, 8, 5, 4, 2}));
}

TEST(type_prop, matmul_batch_broadcast_expand_to_B) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{8, 7, 6, 1, 4, 3});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 5, 3, 2});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{8, 7, 6, 5, 4, 2}));
}

TEST(type_prop, matmul_incompatible_batch_dims) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{7, 4, 3});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, Shape{6, 3, 2});

    try {
        auto matmul = make_shared<ov::op::v0::MatMul>(A, B);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible batch dimensions not detected. ";
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Incompatible MatMul batch dimension"));
    } catch (...) {
        FAIL() << "MatMul shape validation failed for unexpected reason";
    }
}

TEST(type_prop, matmul_matrix_dynamic_bounds) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(2, 5), Dimension(6, 10)});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(7, 8), Dimension(15, 20)});

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_output_partial_shape(0), (PartialShape{Dimension(2, 5), Dimension(15, 20)}));
}

TEST(type_prop, matmul_batch_dynamic_bounds) {
    // Input A and input B dim bounds => output dim bound
    // Dimension 1 can be expanded to any bigger

    Dimension dynamic = Dimension::dynamic();

    auto A_shape = PartialShape{dynamic,           // 0
                                Dimension(1, 5),   // 1
                                Dimension(2, 10),  // 2
                                Dimension(5, 7),   // 3
                                Dimension(4, 7),   // 4
                                Dimension(5, 10),  // 5
                                Dimension(1, 4),   // 6
                                Dimension(0, 1),   // 7
                                Dimension(0, 3),   // 8
                                1,                 // 9
                                Dimension(1, -1),  // 10
                                Dimension(1, 10),  // 11
                                Dimension(2, -1),  // 12
                                Dimension(1, -1),  // 13
                                Dimension(2, -1),  // 14
                                Dimension(1, -1),  // 15
                                1,                 // 16
                                1,                 // 17
                                5,                 // 18
                                6};                // 19

    auto B_shape = PartialShape{dynamic,            // 0
                                Dimension(10, 20),  // 1
                                Dimension(10, 20),  // 2
                                Dimension(4, 10),   // 3
                                Dimension(5, 10),   // 4
                                Dimension(4, 7),    // 5
                                dynamic,            // 6
                                Dimension(0, 1),    // 7
                                Dimension(2, 5),    // 8
                                Dimension(5, 10),   // 9
                                Dimension(1, 5),    // 10
                                Dimension(1, 5),    // 11
                                Dimension(1, 5),    // 12
                                Dimension(2, -1),   // 13
                                Dimension(2, -1),   // 14
                                Dimension(1, -1),   // 15
                                dynamic,            // 16
                                3,                  // 17
                                6,                  // 18
                                4};                 // 19

    auto expected_output_shape = PartialShape{dynamic,            // 0
                                              Dimension(10, 20),  // 1
                                              10,                 // 2
                                              Dimension(5, 7),    // 3
                                              Dimension(5, 7),    // 4
                                              Dimension(5, 7),    // 5
                                              Dimension(-1),      // 6
                                              Dimension(0, 1),    // 7
                                              Dimension(2, 5),    // 8
                                              Dimension(5, 10),   // 9
                                              Dimension(1, -1),   // 10
                                              Dimension(1, 10),   // 11
                                              Dimension(2, -1),   // 12
                                              Dimension(2, -1),   // 13
                                              Dimension(2, -1),   // 14
                                              Dimension(1, -1),   // 15
                                              Dimension(-1),      // 16
                                              3,                  // 17
                                              5,                  // 18
                                              4};                 // 19

    auto A = make_shared<ov::op::v0::Parameter>(element::f32, A_shape);
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, B_shape);

    auto matmul = make_shared<ov::op::v0::MatMul>(A, B);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_output_partial_shape(0), expected_output_shape);
}

TEST(type_prop, matmul_incompatible_matrix_dim_bounds) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(2, 5), Dimension(3, 4)});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(1, 2), Dimension(15, 20)});

    auto expected_output_shape = PartialShape{Dimension(2, 5), Dimension(15, 20)};

    // No error for backward compatibility
    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_output_partial_shape(0), expected_output_shape);
}

TEST(type_prop, matmul_incompatible_batch_dim_bounds) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(2, 5), 4, 3});
    auto B = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(6, 10), 3, 2});

    Dimension dynamic = Dimension::dynamic();
    auto expected_output_shape = PartialShape{dynamic, 4, 2};

    // No error for backward compatibility
    auto matmul = make_shared<ov::op::v0::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_output_partial_shape(0), expected_output_shape);
}

TEST(type_prop, matmul_propagate_symbols) {
    auto A = std::make_shared<ov::Symbol>();

    const auto a_symbols = ov::TensorSymbol{A,
                                            nullptr,
                                            std::make_shared<ov::Symbol>(),
                                            std::make_shared<ov::Symbol>(),
                                            std::make_shared<ov::Symbol>()};
    const auto b_symbols = ov::TensorSymbol{nullptr,
                                            A,
                                            std::make_shared<ov::Symbol>(),
                                            std::make_shared<ov::Symbol>(),
                                            std::make_shared<ov::Symbol>()};

    auto a_shape = PartialShape{4, 2, 3, 6, 4};
    auto b_shape = PartialShape{4, 2, 3, 4, 2};

    set_shape_symbols(a_shape, a_symbols);
    set_shape_symbols(b_shape, b_symbols);

    const auto a = make_shared<ov::op::v0::Parameter>(element::f32, a_shape);
    const auto b = make_shared<ov::op::v0::Parameter>(element::f32, b_shape);
    const auto matmul = make_shared<ov::op::v0::MatMul>(a, b, false, false);

    const auto& output_shape = matmul->get_output_partial_shape(0);
    const auto symbols = get_shape_symbols(output_shape);

    ASSERT_THAT(symbols,
                ElementsAre(a_symbols[0],  // use a symbol, b is not set
                            b_symbols[1],  // use b symbol, a is not set
                            a_symbols[2],  // use a symbol, equal dimension
                            a_symbols[3],  // use symbol from a, b is lost
                            b_symbols[4]   // use symbol from b, a is lost
                            ));
}

TEST(type_prop, matmul_propagate_symbols_on_interval_dims) {
    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>(), C = std::make_shared<ov::Symbol>();

    const auto a_symbols = ov::TensorSymbol{A, nullptr, B, C, std::make_shared<ov::Symbol>()};
    const auto b_symbols = ov::TensorSymbol{nullptr, A, B, C, std::make_shared<ov::Symbol>()};

    auto a_shape = PartialShape{Dimension(1, 3), 1, Dimension(2, 3), Dimension(3, 4), 4};
    auto b_shape = PartialShape{1, Dimension(1, 5), Dimension(1, 3), 4, Dimension::dynamic()};

    set_shape_symbols(a_shape, a_symbols);
    set_shape_symbols(b_shape, b_symbols);

    const auto a = make_shared<ov::op::v0::Parameter>(element::f32, a_shape);
    const auto b = make_shared<ov::op::v0::Parameter>(element::f32, b_shape);
    const auto matmul = make_shared<ov::op::v0::MatMul>(a, b, false, false);

    const auto& output_shape = matmul->get_output_partial_shape(0);
    const auto symbols = get_shape_symbols(output_shape);

    ASSERT_THAT(symbols,
                ElementsAre(a_symbols[0],  // use a symbol, b is not set
                            b_symbols[1],  // use b symbol, a is not set
                            a_symbols[2],  // use a symbol, b is same
                            a_symbols[3],  // use symbol from a, b is lost
                            b_symbols[4]   // use symbol from a, b is lost
                            ));
}

TEST(type_prop, matmul_propagate_symbol_on_b_input_after_reshape) {
    auto my_symbol = std::make_shared<ov::Symbol>();
    auto marked_dim = Dimension(2, 3);
    marked_dim.set_symbol(my_symbol);

    const auto a_shape = PartialShape{Dimension::dynamic(), 5, 3};
    const auto b_shape = PartialShape{3, marked_dim, 2};

    const auto b = make_shared<ov::op::v0::Parameter>(element::f32, b_shape);
    const auto shape_of_b = std::make_shared<op::v0::ShapeOf>(b);
    const auto gather = std::make_shared<op::v7::Gather>(
        shape_of_b,
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{1, 0}),
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0));
    const auto concat = std::make_shared<op::v0::Concat>(
        OutputVector{gather, std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 8)},
        0);
    const auto reshape_b = make_shared<op::v1::Reshape>(b, concat, false);

    const auto a = make_shared<ov::op::v0::Parameter>(element::f32, a_shape);
    const auto matmul = make_shared<ov::op::v0::MatMul>(a, reshape_b, false, false);

    const auto& output_shape = matmul->get_output_partial_shape(0);
    const auto symbols = get_shape_symbols(output_shape);

    ASSERT_THAT(symbols, ElementsAre(my_symbol, nullptr, nullptr));
    ASSERT_EQ(output_shape, (PartialShape{marked_dim, 5, 8}));
}
