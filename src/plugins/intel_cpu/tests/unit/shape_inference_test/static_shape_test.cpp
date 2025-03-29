// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "shape_inference/static_shape.hpp"

using namespace testing;
using namespace ov::intel_cpu;

class StaticShapeAdapterTest : public Test {};

TEST_F(StaticShapeAdapterTest, make_from_partial_shape) {
    const auto partial_shape = ov::PartialShape{2, 6, 1};

    OV_EXPECT_THROW(std::ignore = StaticShapeRef(partial_shape),
                    ov::Exception,
                    HasSubstr("[shape infer] Shouldn't convert from PartialShape to StaticShape at runtime"));

    OV_EXPECT_THROW(std::ignore = StaticShape(partial_shape),
                    ov::Exception,
                    HasSubstr("[shape infer] Shouldn't convert from PartialShape to StaticShape at runtime"));
}

TEST_F(StaticShapeAdapterTest, create_empty_ref) {
    auto shape = StaticShapeRef();

    EXPECT_TRUE(shape.is_static());
    EXPECT_FALSE(shape.is_dynamic());
    EXPECT_TRUE(shape.empty());
    EXPECT_EQ(shape.rank(), ov::Rank(0));
}

TEST_F(StaticShapeAdapterTest, create_as_container) {
    auto shape = StaticShape(VectorDims{2, 4, 5, 7});

    shape[2] = 10;

    EXPECT_TRUE(shape.is_static());
    EXPECT_FALSE(shape.is_dynamic());
    EXPECT_EQ(shape.rank(), ov::Rank(4));

    EXPECT_THAT(shape, ElementsAre(2, 4, 10, 7));
}

TEST_F(StaticShapeAdapterTest, create_from_list) {
    auto shape = StaticShape({2, 4, 5, 7});

    shape[1] = 10;

    EXPECT_TRUE(shape.is_static());
    EXPECT_FALSE(shape.is_dynamic());
    EXPECT_EQ(shape.rank(), ov::Rank(4));

    EXPECT_THAT(shape, ElementsAre(2, 10, 5, 7));
}

TEST_F(StaticShapeAdapterTest, make_from_const_dims) {
    const auto dims = VectorDims{2, 5, 3, 8, 9, 1};

    auto shape = StaticShapeRef(dims);

    EXPECT_TRUE(shape.is_static());
    EXPECT_FALSE(shape.is_dynamic());
    EXPECT_EQ(shape.rank(), ov::Rank(6));

    EXPECT_THAT(shape, ElementsAre(2, 5, 3, 8, 9, 1));
}

TEST_F(StaticShapeAdapterTest, create_as_conatiner_from_dims) {
    auto dims = VectorDims{2, 5, 3, 8, 9, 1};

    auto shape = StaticShape(dims);
    std::iota(shape.begin(), shape.end(), 10);

    EXPECT_TRUE(shape.is_static());
    EXPECT_FALSE(shape.is_dynamic());
    EXPECT_EQ(shape.rank(), ov::Rank(6));

    EXPECT_THAT(shape, ElementsAre(10, 11, 12, 13, 14, 15));
    EXPECT_THAT(dims, ElementsAre(2, 5, 3, 8, 9, 1));
}

TEST_F(StaticShapeAdapterTest, copy_const_reference_to_dims) {
    const auto dims = VectorDims{10, 12, 1, 3};
    auto shape = StaticShapeRef(dims);
    auto shape_copy = shape;

    EXPECT_EQ(std::addressof(*shape), std::addressof(dims));
    EXPECT_EQ(std::addressof(*shape_copy), std::addressof(dims));
    EXPECT_THAT(dims, ElementsAre(10, 12, 1, 3));
    EXPECT_THAT(shape, ElementsAre(10, 12, 1, 3));
    EXPECT_THAT(shape_copy, ElementsAre(10, 12, 1, 3));
}

TEST_F(StaticShapeAdapterTest, make_copy_from_empty_ref) {
    auto shape = StaticShapeRef();
    StaticShape shape_copy = shape;
    shape_copy.resize(5);
    shape_copy[shape_copy.size() - 1] = 1000;

    EXPECT_NE(std::addressof(*shape_copy), std::addressof(*shape));
    EXPECT_THAT(shape_copy, ElementsAre(0, 0, 0, 0, 1000));
}

TEST_F(StaticShapeAdapterTest, make_copy_from_other) {
    const auto dims = VectorDims{10, 12, 1, 3};
    auto shape = StaticShapeRef(dims);
    StaticShape shape_copy = shape;
    shape_copy.resize(shape_copy.size() + 1);
    shape_copy[shape_copy.size() - 1] = 1000;

    EXPECT_EQ(std::addressof(*shape), std::addressof(dims));
    EXPECT_NE(std::addressof(*shape_copy), std::addressof(dims));
    EXPECT_THAT(dims, ElementsAre(10, 12, 1, 3));
    EXPECT_THAT(shape, ElementsAre(10, 12, 1, 3));
    EXPECT_THAT(shape_copy, ElementsAre(10, 12, 1, 3, 1000));
}

TEST_F(StaticShapeAdapterTest, emplace_and_push_back_dims) {
    auto dims = VectorDims{2, 3, 4, 7, 8, 9};

    auto shape = StaticShape(dims);
    shape.resize(4);
    shape.emplace_back(11);
    shape.push_back(21);
    shape[2] = 13;

    EXPECT_THAT(dims, ElementsAre(2, 3, 4, 7, 8, 9));
    EXPECT_THAT(shape, ElementsAre(2, 3, 13, 7, 11, 21));
}

TEST_F(StaticShapeAdapterTest, are_compatible_same_object) {
    const auto dims = VectorDims{2, 7, 1, 2, 3, 4};

    auto shape1 = StaticShapeRef(dims);
    auto shape2 = StaticShapeRef(dims);

    EXPECT_TRUE(shape1.compatible(shape2));
    EXPECT_TRUE(shape2.compatible(shape1));
}

TEST_F(StaticShapeAdapterTest, are_compatible) {
    auto dims1 = VectorDims({2, 5, 6, 7});
    auto dims2 = VectorDims({2, 5, 6, 7});
    auto shape1 = StaticShapeRef(dims1);
    auto shape2 = StaticShapeRef(dims2);
    auto shape3 = StaticShape(dims1);

    EXPECT_TRUE(shape1.compatible(shape2));
    EXPECT_TRUE(shape1.compatible(shape3));
    EXPECT_TRUE(shape2.compatible(shape1));
    EXPECT_TRUE(shape2.compatible(shape3));
    EXPECT_TRUE(shape3.compatible(shape2));
    EXPECT_TRUE(shape3.compatible(shape1));
}

TEST_F(StaticShapeAdapterTest, not_compatible_different_rank) {
    auto dims1 = VectorDims{2, 5, 6, 7, 8};
    auto dims2 = VectorDims{2, 5, 6, 7};
    auto shape1 = StaticShapeRef(dims1);
    auto shape2 = StaticShapeRef(dims2);

    EXPECT_FALSE(shape1.compatible(shape2));
    EXPECT_FALSE(shape2.compatible(shape1));
}

TEST_F(StaticShapeAdapterTest, not_compatible_different_dimensions) {
    auto dims1 = VectorDims{2, 5, 68, 7};
    auto dims2 = VectorDims{2, 5, 6, 7};
    auto shape1 = StaticShapeRef(dims1);
    auto shape2 = StaticShapeRef(dims2);

    EXPECT_FALSE(shape2.compatible(shape1));
}

TEST_F(StaticShapeAdapterTest, merge_rank) {
    auto dims = VectorDims{2, 5, 6, 7};
    auto shape1 = StaticShapeRef(dims);

    EXPECT_TRUE(shape1.merge_rank(ov::Rank(4)));
    EXPECT_FALSE(shape1.merge_rank(ov::Rank(10)));
    EXPECT_FALSE(shape1.merge_rank(ov::Rank(1)));
}

TEST_F(StaticShapeAdapterTest, dereference_as_rvalue_and_move) {
    auto dims = VectorDims{2, 6, 3, 2, 5};
    auto shape = StaticShape(dims);

    const auto shape_data_address = reinterpret_cast<uintptr_t>((*shape).data());
    auto out_dims = std::move(*shape);
    const auto shape_data_mv_address = reinterpret_cast<uintptr_t>((*shape).data());
    const auto out_dims_address = reinterpret_cast<uintptr_t>(out_dims.data());

    EXPECT_EQ(out_dims_address, shape_data_address);
    EXPECT_NE(shape_data_mv_address, shape_data_address);
}

TEST_F(StaticShapeAdapterTest, dereference_as_lvalue_and_copy) {
    auto dims = VectorDims{2, 6, 3, 2, 5};
    auto shape = StaticShapeRef(dims);

    const auto shape_data_address = reinterpret_cast<uintptr_t>((*shape).data());
    const auto out_dims = *shape;
    const auto shape_data_cp_address = reinterpret_cast<uintptr_t>((*shape).data());
    const auto out_dims_address = reinterpret_cast<uintptr_t>(out_dims.data());

    EXPECT_NE(out_dims_address, shape_data_address);
    EXPECT_EQ(shape_data_cp_address, shape_data_address);
    EXPECT_THAT(out_dims, ElementsAreArray(*shape));
}

TEST_F(StaticShapeAdapterTest, move_shapes_into_vector_of_vector_dims) {
    auto output_shapes = std::vector<StaticShape>{{1, 2, 3}, {10, 12}, {100, 1, 2, 3, 4}};

    std::vector<VectorDims> output_vec_dims;
    output_vec_dims.reserve(output_shapes.size());
    for (auto&& s : output_shapes) {
        output_vec_dims.emplace_back(std::move(*s));
    }

    EXPECT_EQ(output_shapes.size(), 3);
    EXPECT_THAT(output_vec_dims, ElementsAre(VectorDims{1, 2, 3}, VectorDims{10, 12}, VectorDims{100, 1, 2, 3, 4}));
}

TEST_F(StaticShapeAdapterTest, compare_empty_ref_and_container) {
    auto shape1 = StaticShapeRef();
    auto shape2 = StaticShape();

    EXPECT_TRUE(shape1 == shape2);
}

TEST_F(StaticShapeAdapterTest, compare_ref_and_container) {
    auto dims = VectorDims{2, 5, 3, 4};
    auto shape1 = StaticShape{2, 5, 3};
    const auto shape2 = StaticShape{2, 5, 3, 4};
    auto shape3 = StaticShapeRef(dims);

    EXPECT_FALSE(shape1 == shape2);
    EXPECT_FALSE(shape1 == shape3);
    EXPECT_FALSE(shape2 == shape1);
    EXPECT_FALSE(shape3 == shape1);

    EXPECT_TRUE(shape2 == shape3);
    EXPECT_TRUE(shape3 == shape2);

    std::cout << "ref  " << shape2 << std::endl;
    for (auto& d : shape2) {
        std::cout << d << " ";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < shape2.size(); ++i) {
        std::cout << shape2[i] << " ";
    }
    std::cout << std::endl << "----------------" << std::endl;

    // std::cout << "dims " << dims << std::endl;
    std::cout << "ref  " << shape3 << std::endl;
    for (auto& d : shape3) {
        std::cout << d << " ";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < shape3.size(); ++i) {
        std::cout << shape3[i] << " ";
    }
    std::cout << std::endl;
}

TEST_F(StaticShapeAdapterTest, subscript_op_on_reference) {
    auto dims = VectorDims{2, 5, 3, 4};
    auto shape = StaticShapeRef(dims);

    EXPECT_EQ(shape[3], StaticDimension(4));
    EXPECT_EQ(shape[2], StaticDimension(3));
    EXPECT_EQ(shape[0], StaticDimension(2));
    EXPECT_EQ(shape[1], StaticDimension(5));
}

TEST_F(StaticShapeAdapterTest, subscript_op_on_container) {
    auto shape = StaticShape{10, 2, 123, 4, 3};

    EXPECT_EQ(shape[3], StaticDimension(4));
    EXPECT_EQ(shape[1], StaticDimension(2));
    EXPECT_EQ(shape[2], StaticDimension(123));
    EXPECT_EQ(shape[0], StaticDimension(10));
    EXPECT_EQ(shape[4], StaticDimension(3));
}
