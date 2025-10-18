// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define OPENVINO_ARCH_X86_64

#include "nodes/kernels/scaled_attn/xattention.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include "nodes/kernels/scaled_attn/transpose_kernel.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

using namespace ov::intel_cpu;
using namespace ov::Extensions::Cpu::XARCH;

namespace XAttentionUnitTest {
struct XAttr {
    size_t block_size;
    size_t stride;
    float threshold;
};

using XAttentionParams =
    std::tuple<ov::element::Type, std::vector<size_t>, std::vector<size_t>, XAttr, std::vector<bool>>;

class XAttentionTest : public ov::test::TestsCommon, public testing::WithParamInterface<XAttentionParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<XAttentionParams>& obj) {
        auto shape_to_str = [](const std::vector<size_t>& shape) {
            std::string msg;
            for (size_t i = 0; i < shape.size(); i++) {
                if (i != 0)
                    msg += "x";
                msg += std::to_string(shape[i]);
            }
            return msg;
        };
        const auto& [data_type, query_shape, key_shape, xattr, expected_values] = obj.param;
        std::ostringstream result;
        result << "DT=" << data_type.to_string();
        result << ",Q=" << shape_to_str(query_shape);
        result << ",K=" << shape_to_str(key_shape);
        result << ",block_size=" << xattr.block_size;
        result << ",stride=" << xattr.stride;
        result << ",threshold=" << xattr.threshold;
        return result.str();
    }
};

TEST_P(XAttentionTest, simpleTest) {
    const auto& [data_type, query_shape, key_shape, xattr, expected_values] = this->GetParam();
    size_t element_size = ov::element::Type(data_type).size();
    PlainTensor query;
    PlainTensor key;
    query.resize(query_shape, element_size, data_type);
    key.resize(key_shape, element_size, data_type);

    for (size_t h = 0; h < query_shape[1]; h++) {
        for (size_t b = 0; b < query_shape[0]; b++) {
            for (size_t s = 0; s < query_shape[3]; s++) {
                float v = static_cast<float>(b + s) / 100.0f;
                if (data_type == ov::element::f32) {
                    *key.ptr<float>(b, h, 0, s) = v;
                    *query.ptr<float>(b, h, 0, s) = 1.0f;
                } else if (data_type == ov::element::bf16) {
                    *key.ptr<ov::bfloat16, ov::element::bf16>(b, h, 0, s) = v;
                    *query.ptr<ov::bfloat16, ov::element::bf16>(b, h, 0, s) = 1.0f;
                } else {
                    OPENVINO_THROW("xattention: unsupported precision: ", data_type);
                }
            }
        }
    }

    Xattn xattn;
    xattn.init(query.size(0),
               query.size(1),
               query.size(2),
               query.size(3),
               key.size(2),
               xattr.stride,
               xattr.block_size,
               ov::element::f32);
    PlainTensor mask;
    xattn.estimate(query, key, xattr.block_size, xattr.stride, xattr.threshold, mask);
    size_t k_num_blocks = div_up(key_shape[0], xattr.block_size);
    for (size_t i = 0; i < expected_values.size(); i++) {
        EXPECT_EQ(*mask.ptr<bool>(0, i / k_num_blocks, i % k_num_blocks), expected_values[i]);
    }
}

const std::vector<XAttentionParams> params = {
    {ov::element::f32, {8, 1, 1, 8}, {8, 1, 1, 8}, {4, 2, 0.8f}, {true, false, true, true}},
    {ov::element::f32,
     {16, 1, 1, 32},
     {16, 1, 1, 32},
     {4, 2, 0.3f},
     {true, false, false, false, true, true, false, false, true, false, true, false, true, false, false, true}},
    {ov::element::f32,
     {16, 1, 1, 32},
     {16, 1, 1, 32},
     {4, 2, 0.7f},
     {true, false, false, false, true, true, false, false, true, true, true, false, true, false, true, true}}};

INSTANTIATE_TEST_SUITE_P(XAttentionUnitTest,
                         XAttentionTest,
                         ::testing::ValuesIn(params),
                         XAttentionTest::getTestCaseName);

}  // namespace XAttentionUnitTest
