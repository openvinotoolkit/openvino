// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/nf4.hpp"

#include "common_test_utils/test_tools.hpp"
#include "gtest/gtest.h"
#include "openvino/op/constant.hpp"

using namespace std;

TEST(nf4, convert_nf4_to_string) {
    vector<uint8_t> values{186, 17};
    auto constant = make_shared<ov::op::v0::Constant>(ov::element::nf4, ov::Shape{3}, &values[0]);

    vector<string> ref{"10", "11", "1"};
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(constant->convert_value_to_string(i), ref[i]);
    }
}

TEST(nf4, tensor_or_constant_size) {
    vector<uint8_t> values{171, 16};
    auto constant = make_shared<ov::op::v0::Constant>(ov::element::nf4, ov::Shape{3}, &values[0]);
    EXPECT_EQ(2, constant->get_byte_size());

    ov::Tensor runtime_tensor(ov::element::nf4, ov::Shape{3});
    EXPECT_EQ(constant->get_byte_size(), runtime_tensor.get_byte_size());
}

template <typename T>
void test_nf4_convert() {
    vector<float> const_data_f{-1.5f,   -1.425f, -1.35f,  -1.275f, -1.2f,   -1.125f, -1.05f,  -0.975f, -0.9f,
                               -0.825f, -0.75f,  -0.675f, -0.6f,   -0.525f, -0.45f,  -0.375f, -0.3f,   -0.225f,
                               -0.15f,  -0.075f, 0.0f,    0.075f,  0.15f,   0.225f,  0.3f,    0.375f,  0.45f,
                               0.525f,  0.6f,    0.675f,  0.75f,   0.825f,  0.9f,    0.975f,  1.05f,   1.125f,
                               1.2f,    1.275f,  1.35f,   1.425f,  1.5};

    vector<float> target_f{-1.0f,
                           -0.6961928009986877f,
                           -0.5250730514526367f,
                           -0.39491748809814453f,
                           -0.28444138169288635f,
                           -0.18477343022823334f,
                           -0.09105003625154495f,
                           0.0f,
                           0.07958029955625534f,
                           0.16093020141124725f,
                           0.24611230194568634f,
                           0.33791524171829224f,
                           0.44070982933044434f,
                           0.5626170039176941f,
                           0.7229568362236023f,
                           1.0f};

    vector<T> const_data;
    const_data.reserve(const_data_f.size());
    for (auto& val : const_data_f) {
        const_data.push_back(static_cast<T>(val));
    }
    vector<T> target;
    target.reserve(target_f.size());
    for (auto& val : target_f) {
        target.push_back(static_cast<T>(val));
    }

    auto constant = ov::op::v0::Constant::create(ov::element::nf4, ov::Shape{const_data.size()}, const_data);

    const uint8_t* p = static_cast<const uint8_t*>(constant->get_data_ptr());
    EXPECT_NE(p, nullptr);
    std::vector<uint8_t> packed_data(p, p + const_data.size() / 2 + const_data.size() % 2);

    std::vector<T> decompressed_data(const_data.size(), 0);
    for (size_t i = 0; i < const_data.size(); i++) {
        ov::ConvertNF4::unpack(&decompressed_data[0], &packed_data[0], i);
    }

    auto it = std::unique(decompressed_data.begin(), decompressed_data.end());
    decompressed_data.resize(std::distance(decompressed_data.begin(), it));

    EXPECT_EQ(16, decompressed_data.size());

    float max_diff = 0.0;
    for (size_t i = 0; i < 16; i++) {
        float diff = fabs(static_cast<float>(decompressed_data[i] - target[i]));
        max_diff = std::max(max_diff, diff);
    }
    EXPECT_LE(max_diff, 0.001);
}

TEST(nf4, convert_from_float) {
    test_nf4_convert<float>();
}

TEST(nf4, convert_from_float16) {
    test_nf4_convert<ov::float16>();
}

TEST(nf4, convert_from_bfloat16) {
    test_nf4_convert<ov::bfloat16>();
}
