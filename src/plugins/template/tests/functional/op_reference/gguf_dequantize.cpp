// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gguf_dequantize.hpp"

#include <gtest/gtest.h>

#include <cstring>

#include "base_reference_test.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;
using namespace reference_tests;
using ov::op::internal::GGUFDequantize;

namespace {

// Blocks built by hand from the ggml block definitions; the expected values follow the formulas, not the reference.

void put_f16(uint8_t* p, float v) {
    const uint16_t bits = ov::float16(v).to_bits();
    std::memcpy(p, &bits, sizeof(bits));
}

// Packs 6-bit scales sc[8] and mins m[8] as ggml's Q4_K / Q5_K (inverse of get_scale_min_k4).
void pack_scales_k4(const int* sc, const int* m, uint8_t* q) {
    for (int j = 0; j < 4; ++j) {
        q[j] = static_cast<uint8_t>((sc[j] & 63) | ((sc[j + 4] >> 4) << 6));
        q[j + 4] = static_cast<uint8_t>((m[j] & 63) | ((m[j + 4] >> 4) << 6));
        q[j + 8] = static_cast<uint8_t>((sc[j + 4] & 0xF) | ((m[j + 4] & 0xF) << 4));
    }
}

// One Q4_K or Q5_K block per row: w = d * sc * q - dmin * m, per sub-block of 32 values.
void make_q4_k_q5_k(bool q5, size_t rows, std::vector<uint8_t>& bytes, std::vector<float>& expected) {
    const size_t bb = q5 ? 176 : 144;
    bytes.assign(rows * bb, 0);
    for (size_t row = 0; row < rows; ++row) {
        uint8_t* blk = bytes.data() + row * bb;
        const float d = row ? -0.375f : 0.5f, dmin = row ? 0.0625f : 0.25f;
        int sc[8], m[8], q[256];
        for (int j = 0; j < 8; ++j) {
            sc[j] = (7 * j + 3 + static_cast<int>(row)) % 64;  // up to 52 + row, uses the high bits for j >= 4
            m[j] = 63 - 5 * j;                                 // up to 63
        }
        for (int i = 0; i < 256; ++i) {
            q[i] = q5 ? (i * 7 + static_cast<int>(row)) % 32 : (i * 5 + static_cast<int>(row)) % 16;
        }
        put_f16(blk, d);
        put_f16(blk + 2, dmin);
        pack_scales_k4(sc, m, blk + 4);
        // value 64c + l: low nibble of qs[32c + l]; value 64c + 32 + l: high nibble; 5th bits in qh[l], bits 2c / 2c+1
        uint8_t* qh = q5 ? blk + 16 : nullptr;
        uint8_t* qs = blk + (q5 ? 48 : 16);
        for (int c = 0; c < 4; ++c) {
            for (int l = 0; l < 32; ++l) {
                const int lo = q[64 * c + l], hi = q[64 * c + 32 + l];
                qs[32 * c + l] = static_cast<uint8_t>((lo & 0xF) | ((hi & 0xF) << 4));
                if (qh) {
                    qh[l] |= static_cast<uint8_t>(((lo >> 4) & 1) << (2 * c));
                    qh[l] |= static_cast<uint8_t>(((hi >> 4) & 1) << (2 * c + 1));
                }
            }
        }
        for (int i = 0; i < 256; ++i) {
            expected.push_back(d * sc[i / 32] * q[i] - dmin * m[i / 32]);
        }
    }
}

// One Q6_K block per row: w = d * sc * (q - 32), per sub-block of 16 values.
void make_q6_k(size_t rows, std::vector<uint8_t>& bytes, std::vector<float>& expected) {
    bytes.assign(rows * 210, 0);
    for (size_t row = 0; row < rows; ++row) {
        uint8_t* blk = bytes.data() + row * 210;
        const float d = row ? -0.125f : 0.75f;
        int sc[16], q[256];
        for (int j = 0; j < 16; ++j) {
            sc[j] = (j % 2 ? -1 : 1) * (9 * j + static_cast<int>(row)) % 128;
        }
        for (int i = 0; i < 256; ++i) {
            q[i] = (i * 11 + static_cast<int>(row)) % 64;  // stored 0..63, value q - 32
        }
        put_f16(blk + 208, d);
        for (int j = 0; j < 16; ++j) {
            blk[192 + j] = static_cast<uint8_t>(static_cast<int8_t>(sc[j]));
        }
        // half h (128 values): ql[64h + l] low nibble = value 128h + l, ql[64h + 32 + l] low = 128h + 32 + l,
        // high nibbles = values + 64; qh[32h + l] bits 0-1, 2-3, 4-5, 6-7 = high bits of values l, l+32, l+64, l+96
        for (int h = 0; h < 2; ++h) {
            const int* v = q + 128 * h;
            for (int l = 0; l < 32; ++l) {
                blk[64 * h + l] = static_cast<uint8_t>((v[l] & 0xF) | ((v[l + 64] & 0xF) << 4));
                blk[64 * h + 32 + l] = static_cast<uint8_t>((v[l + 32] & 0xF) | ((v[l + 96] & 0xF) << 4));
                blk[128 + 32 * h + l] = static_cast<uint8_t>((v[l] >> 4) | ((v[l + 32] >> 4) << 2) |
                                                             ((v[l + 64] >> 4) << 4) | ((v[l + 96] >> 4) << 6));
            }
        }
        for (int i = 0; i < 256; ++i) {
            expected.push_back(d * sc[i / 16] * (q[i] - 32));
        }
    }
}

struct GGUFDequantizeParams {
    GGUFDequantize::Format format;
    element::Type output_type;
    size_t rows;
    std::string name;
    reference_tests::Tensor blocks;
    reference_tests::Tensor expected;
};

template <class T>
std::vector<T> cast_to(const std::vector<float>& values) {
    std::vector<T> result;
    result.reserve(values.size());
    for (const auto v : values) {
        result.push_back(static_cast<T>(v));
    }
    return result;
}

GGUFDequantizeParams make_params(GGUFDequantize::Format format, element::Type output_type, size_t rows) {
    std::vector<uint8_t> bytes;
    std::vector<float> expected;
    if (format == GGUFDequantize::Format::Q6_K) {
        make_q6_k(rows, bytes, expected);
    } else {
        make_q4_k_q5_k(format == GGUFDequantize::Format::Q5_K, rows, bytes, expected);
    }
    const Shape in_shape{rows, bytes.size() / rows};
    const Shape out_shape{rows, GGUFDequantize::block_size};
    GGUFDequantizeParams p{format, output_type, rows, {}, reference_tests::Tensor{in_shape, element::u8, bytes}, {}};
    if (output_type == element::f16) {
        p.expected = reference_tests::Tensor{out_shape, output_type, cast_to<ov::float16>(expected)};
    } else if (output_type == element::bf16) {
        p.expected = reference_tests::Tensor{out_shape, output_type, cast_to<ov::bfloat16>(expected)};
    } else {
        p.expected = reference_tests::Tensor{out_shape, output_type, expected};
    }
    std::ostringstream name;
    name << format << "_" << output_type << "_rows" << rows;
    p.name = name.str();
    return p;
}

class ReferenceGGUFDequantizeLayerTest : public testing::TestWithParam<GGUFDequantizeParams>,
                                         public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        const auto data = std::make_shared<op::v0::Parameter>(element::u8, params.blocks.shape);
        const auto op = std::make_shared<GGUFDequantize>(data, params.format, params.output_type);
        function = std::make_shared<Model>(OutputVector{op}, ParameterVector{data});
        inputData = {params.blocks.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<GGUFDequantizeParams>& obj) {
        return obj.param.name;
    }
};

TEST_P(ReferenceGGUFDequantizeLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

std::vector<GGUFDequantizeParams> generate_params() {
    std::vector<GGUFDequantizeParams> params;
    for (const auto format :
         {GGUFDequantize::Format::Q4_K, GGUFDequantize::Format::Q5_K, GGUFDequantize::Format::Q6_K}) {
        for (const auto output_type : {element::f32, element::f16, element::bf16}) {
            params.push_back(make_params(format, output_type, 2));
        }
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(smoke_GGUFDequantize_With_Hardcoded_Refs,
                         ReferenceGGUFDequantizeLayerTest,
                         ::testing::ValuesIn(generate_params()),
                         ReferenceGGUFDequantizeLayerTest::getTestCaseName);

}  // namespace
