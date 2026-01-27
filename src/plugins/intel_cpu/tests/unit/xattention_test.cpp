// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define OPENVINO_ARCH_X86_64

#include "nodes/kernels/scaled_attn/xattention.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "nodes/kernels/scaled_attn/transpose_kernel.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/reference/xattention.hpp"
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
    std::tuple<ov::element::Type, std::vector<size_t>, std::vector<size_t>, XAttr>;

template <class T>
void inline fill_data_random(T* pointer, std::size_t size, const float min, const float max, const int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(min, max);
    for (std::size_t i = 0; i < size; i++) {
        pointer[i] = static_cast<T>(dist(gen));
    }
}

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
        const auto& [data_type, query_shape, key_shape, xattr] = obj.param;
        std::ostringstream result;
        result << "DT=" << data_type.to_string();
        result << ",Q=" << shape_to_str(query_shape);
        result << ",K=" << shape_to_str(key_shape);
        result << ",block_size=" << xattr.block_size;
        result << ",stride=" << xattr.stride;
        result << ",threshold=" << xattr.threshold;
        return result.str();
    }

public:
    template <typename T>
    void compareWithRef() {
        const auto& [data_type, query_shape, key_shape, xattr] = this->GetParam();
        auto element_size = ov::element::Type(data_type).size();

        PlainTensor query;
        PlainTensor key;
        query.resize(query_shape, element_size, data_type);
        key.resize(key_shape, element_size, data_type);

        fill_data_random<T>(query.ptr<T>(), ov::shape_size(query_shape.begin(), query_shape.end()), 0, 2, 0);
        fill_data_random<T>(key.ptr<T>(), ov::shape_size(key_shape.begin(), key_shape.end()), 0, 1, 0);

        Xattn xattn;
        xattn.init(query.size(0),
                   query.size(1),
                   query.size(2),
                   query.size(3),
                   key.size(1),
                   xattr.stride,
                   xattr.block_size,
                   ov::element::f32);
        PlainTensor mask;
        xattn.estimate(query, key, xattr.block_size, xattr.stride, xattr.threshold, mask);

        ov::reference::XAttentionBlockSelector<T> selector(xattr.threshold, xattr.block_size, xattr.stride);
        ov::Shape query_shape_ref = {query_shape[1], query_shape[0], query_shape[3]};
        ov::Shape key_shape_ref = {key_shape[1], key_shape[0], key_shape[3]};
        auto ref_result = selector.select_blocks(query.ptr<T>(), query_shape_ref, key.ptr<T>(), key_shape_ref);

        for (size_t h = 0; h < query.size(1); h++) {
            for (const auto& idx : ref_result[h]) {
                auto row = idx.first;
                auto col = idx.second;
                EXPECT_EQ(*mask.ptr<bool>(h, row, col), true);
            }
        }
    }
};

TEST_P(XAttentionTest, testWithRef) {
        compareWithRef<float>();
}

const std::vector<XAttentionParams> params = {
    {ov::element::f32,
     {8, 1, 1, 8},  // {B, H, L, S}
     {8, 1, 1, 8},
     {4, 2, 0.8f}},
    {ov::element::f32,
     {16, 1, 1, 32},
     {16, 1, 1, 32},
     {4, 2, 0.3f}},
    {ov::element::f32,
     {16, 2, 1, 32},
     {16, 2, 1, 32},
     {4, 2, 0.7f}}};

INSTANTIATE_TEST_SUITE_P(XAttentionUnitTest,
                         XAttentionTest,
                         ::testing::ValuesIn(params),
                         XAttentionTest::getTestCaseName);

}  // namespace XAttentionUnitTest
