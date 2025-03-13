// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal_properties.hpp"
#include "utils/cpu_test_utils.hpp"
#include "openvino/runtime/system_conf.hpp"

namespace ov {
namespace test {
namespace snippets {

#define SNIPPETS_TESTS_STATIC_SHAPES(...) static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>{__VA_ARGS__})

static inline bool is_bf16_supported_by_brgemm() {
    return ov::with_cpu_x86_bfloat16() || ov::with_cpu_x86_avx512_core_amx_bf16();
}

static inline bool is_fp16_supported_by_brgemm() {
    return ov::with_cpu_x86_avx512_core_amx_fp16();
}

static inline bool is_i8_supported_by_brgemm() {
    return ov::with_cpu_x86_avx512_core_vnni() || ov::with_cpu_x86_avx512_core_amx_int8();
}

static inline std::vector<std::vector<element::Type>> precision_f32(size_t count) {
    std::vector<std::vector<element::Type>> prc;
    prc.emplace_back(std::vector<element::Type>(count, element::f32));
    return prc;
}

static inline std::vector<std::vector<element::Type>> precision_bf16_if_supported(size_t count) {
    std::vector<std::vector<element::Type>> prc;
    if (is_bf16_supported_by_brgemm())
        prc.emplace_back(std::vector<element::Type>(count, element::bf16));
    return prc;
}

static inline std::vector<std::vector<element::Type>> precision_fp16_if_supported(size_t count) {
    std::vector<std::vector<element::Type>> prc;
    if (is_fp16_supported_by_brgemm())
        prc.emplace_back(std::vector<element::Type>(count, element::f16));
    return prc;
}

static inline std::vector<std::vector<element::Type>> quantized_precisions_if_supported() {
    std::vector<std::vector<element::Type>> prc = {};
    // In Snippets MatMul INT8 is supported only on VNNI/AMX platforms
    if (is_i8_supported_by_brgemm()) {
        prc.emplace_back(std::vector<element::Type>{element::i8, element::i8});
        prc.emplace_back(std::vector<element::Type>{element::u8, element::i8});
    }
    return prc;
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
