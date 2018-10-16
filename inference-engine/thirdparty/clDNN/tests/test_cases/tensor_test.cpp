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
    EXPECT_EQ(test.spatial.size(), size_t(2));

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
    EXPECT_EQ(test.spatial.size(), size_t(2));

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
    EXPECT_EQ(test.spatial.size(), size_t(2));

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