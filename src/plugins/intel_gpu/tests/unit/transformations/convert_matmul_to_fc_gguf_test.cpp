// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/convert_matmul_to_fc_gguf.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/fully_connected_gguf.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gguf_dequantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/pass/manager.hpp"

using namespace testing;
using namespace ov::intel_gpu;
using ov::op::internal::GGUFDequantize;

namespace {
std::shared_ptr<ov::op::v0::Constant> blocks(size_t n, size_t row_bytes, uint8_t seed) {
    std::vector<uint8_t> bytes(n * row_bytes);
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<uint8_t>(i * 7 + seed);
    }
    return ov::op::v0::Constant::create(ov::element::u8, ov::Shape{n, row_bytes}, bytes);
}

// The Q6_K blocks of blocks(n, row_bytes, seed) regrouped as the pass does: QL [nblk][128] | QH [nblk][64] |
// SC [nblk][16] | D [nblk].
std::shared_ptr<ov::op::v0::Constant> q6_k_planes(size_t n, size_t row_bytes, uint8_t seed) {
    const auto src = blocks(n, row_bytes, seed)->get_vector<uint8_t>();
    const size_t nblk = src.size() / 210;
    std::vector<uint8_t> dst(src.size());
    for (size_t i = 0; i < nblk; ++i) {
        std::memcpy(dst.data() + i * 128, src.data() + i * 210, 128);
        std::memcpy(dst.data() + nblk * 128 + i * 64, src.data() + i * 210 + 128, 64);
        std::memcpy(dst.data() + nblk * 192 + i * 16, src.data() + i * 210 + 192, 16);
        std::memcpy(dst.data() + nblk * 208 + i * 2, src.data() + i * 210 + 208, 2);
    }
    return ov::op::v0::Constant::create(ov::element::u8, ov::Shape{n, row_bytes}, dst);
}
}  // namespace

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedGGUF_Q6_K) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 512});
        auto deq = std::make_shared<GGUFDequantize>(blocks(16, 2 * 210, 1), GGUFDequantize::Format::Q6_K);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input, deq, false, true);
        model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input});
        manager.register_pass<ConvertMatMulToFullyConnectedGGUF>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 512});
        auto fc = std::make_shared<op::FullyConnectedGGUF>(input, q6_k_planes(16, 2 * 210, 1), GGUFDequantize::Format::Q6_K);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedGGUF_Q6_K_with_convert) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 256});
        auto deq = std::make_shared<GGUFDequantize>(blocks(8, 210, 2), GGUFDequantize::Format::Q6_K, ov::element::f32);
        auto convert = std::make_shared<ov::op::v0::Convert>(deq, ov::element::f16);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input, convert, false, true);
        model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input});
        manager.register_pass<ConvertMatMulToFullyConnectedGGUF>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 256});
        auto fc = std::make_shared<op::FullyConnectedGGUF>(input, q6_k_planes(8, 210, 2), GGUFDequantize::Format::Q6_K);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// Q5_K blocks are 16-byte aligned and passed as stored.
TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedGGUF_Q5_K) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 512});
        auto deq = std::make_shared<GGUFDequantize>(blocks(16, 2 * 176, 3), GGUFDequantize::Format::Q5_K);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input, deq, false, true);
        model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input});
        manager.register_pass<ConvertMatMulToFullyConnectedGGUF>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 512});
        auto fc = std::make_shared<op::FullyConnectedGGUF>(input, blocks(16, 2 * 176, 3), GGUFDequantize::Format::Q5_K);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// Negative cases: model_ref stays unset, the model must not change.

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedGGUF_not_transposed) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{16, 4});
    auto deq = std::make_shared<GGUFDequantize>(blocks(4, 210, 4), GGUFDequantize::Format::Q6_K);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, deq, false, false);
    model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input});
    manager.register_pass<ConvertMatMulToFullyConnectedGGUF>();
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedGGUF_other_format) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 256});
    auto deq = std::make_shared<GGUFDequantize>(blocks(8, 144, 5), GGUFDequantize::Format::Q4_K);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, deq, false, true);
    model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input});
    manager.register_pass<ConvertMatMulToFullyConnectedGGUF>();
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedGGUF_weights_with_another_consumer) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 256});
    auto deq = std::make_shared<GGUFDequantize>(blocks(8, 210, 6), GGUFDequantize::Format::Q6_K);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, deq, false, true);
    auto relu = std::make_shared<ov::op::v0::Relu>(deq);
    model = std::make_shared<ov::Model>(ov::OutputVector{matmul, relu}, ov::ParameterVector{input});
    manager.register_pass<ConvertMatMulToFullyConnectedGGUF>();
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedGGUF_vector_input) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{256});
    auto deq = std::make_shared<GGUFDequantize>(blocks(8, 210, 7), GGUFDequantize::Format::Q6_K);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, deq, false, true);
    model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input});
    manager.register_pass<ConvertMatMulToFullyConnectedGGUF>();
}
