/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <gtest/gtest.h>
#include <api/CPP/tensor.hpp>

TEST(tensor_api, order_new_notation)
{
    cldnn::tensor test{ cldnn::feature(3), cldnn::batch(4), cldnn::spatial(2, 1) };

    //sizes
    EXPECT_EQ(test.batch.size(), size_t(1));
    EXPECT_EQ(test.feature.size(), size_t(1));
    EXPECT_EQ(test.spatial.size(), size_t(CLDNN_TENSOR_SPATIAL_DIM_MAX));

    //passed values
    EXPECT_EQ(test.spatial[0], cldnn::tensor::value_type(2));
    EXPECT_EQ(test.spatial[1], cldnn::tensor::value_type(1));
    EXPECT_EQ(test.feature[0], cldnn::tensor::value_type(3));
    EXPECT_EQ(test.batch[0], cldnn::tensor::value_type(4));

    //reverse
    auto sizes = test.sizes();
    EXPECT_EQ(sizes[0], cldnn::tensor::value_type(4));
    EXPECT_EQ(sizes[1], cldnn::tensor::value_type(3));
    EXPECT_EQ(sizes[2], cldnn::tensor::value_type(2));
    EXPECT_EQ(sizes[3], cldnn::tensor::value_type(1));
}

TEST(tensor_api, order_new_notation_feature_default)
{
    cldnn::tensor test{ cldnn::feature(3), cldnn::spatial(2) };

    //sizes
    EXPECT_EQ(test.batch.size(), size_t(1));
    EXPECT_EQ(test.feature.size(), size_t(1));
    EXPECT_EQ(test.spatial.size(), size_t(CLDNN_TENSOR_SPATIAL_DIM_MAX));

    //passed values
    EXPECT_EQ(test.spatial[0], cldnn::tensor::value_type(2));
    EXPECT_EQ(test.spatial[1], cldnn::tensor::value_type(1));
    EXPECT_EQ(test.feature[0], cldnn::tensor::value_type(3));
    EXPECT_EQ(test.batch[0], cldnn::tensor::value_type(1));

    //reverse
    auto sizes = test.sizes();
    EXPECT_EQ(sizes[0], cldnn::tensor::value_type(1));
    EXPECT_EQ(sizes[1], cldnn::tensor::value_type(3));
    EXPECT_EQ(sizes[2], cldnn::tensor::value_type(2));
    EXPECT_EQ(sizes[3], cldnn::tensor::value_type(1));
}

TEST(tensor_api, order)
{
    cldnn::tensor test{ 1, 2, 3, 4 };

    //sizes
    EXPECT_EQ(test.batch.size(), size_t(1));
    EXPECT_EQ(test.feature.size(), size_t(1));
    EXPECT_EQ(test.spatial.size(), size_t(CLDNN_TENSOR_SPATIAL_DIM_MAX));

    //passed values
    EXPECT_EQ(test.spatial[1], cldnn::tensor::value_type(4));
    EXPECT_EQ(test.spatial[0], cldnn::tensor::value_type(3));
    EXPECT_EQ(test.feature[0], cldnn::tensor::value_type(2));
    EXPECT_EQ(test.batch[0], cldnn::tensor::value_type(1));

    //reverse
    auto sizes = test.sizes();
    EXPECT_EQ(sizes[0], cldnn::tensor::value_type(1));
    EXPECT_EQ(sizes[1], cldnn::tensor::value_type(2));
    EXPECT_EQ(sizes[2], cldnn::tensor::value_type(3));
    EXPECT_EQ(sizes[3], cldnn::tensor::value_type(4));
}

static void test_tensor_offset(cldnn::tensor shape, cldnn::tensor coord, cldnn::format fmt, size_t ref_offset) {
    auto offset = shape.get_linear_offset(coord, fmt);
    EXPECT_EQ(ref_offset, offset)
        << "format: " << fmt << ", shape: " << shape << ", coord: " << coord;
}

TEST(tensor_api, linear_offsets) {
    // Simple formats:
    test_tensor_offset({ 2, 5, 4, 3 }, { 1, 3, 1, 2 }, cldnn::format::bfyx, 105);
    test_tensor_offset({ 2, 5, 4, 3 }, { 1, 3, 1, 2 }, cldnn::format::yxfb, 97);
    test_tensor_offset({ 2, 5, 4, 3 }, { 1, 3, 1, 2 }, cldnn::format::fyxb, 91);
    test_tensor_offset({ 2, 5, 4, 3 }, { 1, 3, 1, 2 }, cldnn::format::byxf, 108);
    test_tensor_offset({ 2, 5, 4, 3, 5 }, { 1, 3, 1, 2, 4 }, cldnn::format::bfzyx, 537);

    // Blocked formats:
    test_tensor_offset({ 2, 5, 4, 3 }, { 1, 3, 1, 2 }, cldnn::format::bfyx_f16, 339);
    test_tensor_offset({ 2, 19, 4, 3 }, { 1, 18, 3, 2 }, cldnn::format::bfyx_f16, 754);
    test_tensor_offset({ 2, 5, 4, 3 }, { 1, 3, 1, 2 }, cldnn::format::fs_b_yx_fsv32, 675);
    test_tensor_offset({ 2, 37, 4, 3 }, { 1, 35, 3, 2 }, cldnn::format::fs_b_yx_fsv32, 1507);

    // Formats with alignment:
    test_tensor_offset({ 2, 5, 4, 3 }, { 1, 3, 1, 2 }, cldnn::format::byxf_af32, 675);
    test_tensor_offset({ 2, 37, 4, 3 }, { 1, 35, 3, 2 }, cldnn::format::byxf_af32, 1507);
    test_tensor_offset({ 2, 5, 4, 3 }, { 1, 3, 1, 2 }, cldnn::format::byx8_f4, 331);
    test_tensor_offset({ 2, 37, 4, 3 }, { 1, 35, 3, 2 }, cldnn::format::byx8_f4, 1755);

    // Non-standard blocked formats:
    // bf8_xy16 - b_fs_es_fsv8_esv16, where e is flattened yx := x + y * size_x
    test_tensor_offset({ 2, 5, 4, 3 }, { 1, 3, 1, 2 }, cldnn::format::bf8_xy16, 185);
    test_tensor_offset({ 2, 19, 7, 3 }, { 1, 18, 3, 2 }, cldnn::format::bf8_xy16, 1441);

}
