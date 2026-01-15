// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "transformations/hash.hpp"

namespace v0 = ov::op::v0;
namespace ov::test {
using v0::Parameter, v0::Constant, ov::op::v1::Add;

TEST(HashTest, same_model_hashed_without_weights) {
    uint64_t hash1 = 0, hash2 = 0;
    constexpr auto skip_weights = true;

    // create same model twice and hash without weights
    // detects issue when first 8-byte value used instead weights which may have some bits in uninitialized state
    {
        int data_value = 121;
        const auto data = Tensor(element::i32, {1}, &data_value);
        auto out = std::make_shared<Add>(std::make_shared<Parameter>(element::i32, Shape{1}),
                                         std::make_shared<Constant>(data));
        auto model = std::make_shared<Model>(OutputVector{out}, "TestModel");

        ov::pass::Hash hasher(hash1, skip_weights);
        hasher.run_on_model(model);
    }
    {
        int data_value = 121;
        const auto data = ov::Tensor(element::i32, {1}, &data_value);
        auto out = std::make_shared<Add>(std::make_shared<Parameter>(element::i32, Shape{1}),
                                         std::make_shared<Constant>(data));
        auto model = std::make_shared<Model>(OutputVector{out}, "TestModel");

        ov::pass::Hash hasher(hash2, skip_weights);
        hasher.run_on_model(model);
    }

    EXPECT_EQ(hash1, hash2);
}
}  // namespace ov::test
