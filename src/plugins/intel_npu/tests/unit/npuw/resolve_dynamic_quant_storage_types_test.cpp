// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

#include "util.hpp"

namespace {

using DQMode = ov::npuw::util::DynamicQuantDecomposeMode;

struct ResolveStorageTypesParams {
    DQMode mode;
    bool is_symmetric;
    ov::element::Type quant_dt;
    ov::element::Type scale_dt;
    ov::element::Type expected_quantized_data_type;
    ov::element::Type expected_zero_point_type;
    ov::element::Type expected_scale_type;
};

const char* mode_to_token(DQMode mode) {
    switch (mode) {
    case DQMode::HandcraftedSymmetricI8:
        return "V1";
    case DQMode::OnnxDynamicQuantizeLinear:
        return "V2";
    case DQMode::CompilerPatternI8:
        return "V3";
    default:
        return "Unknown";
    }
}

const char* type_to_token(const ov::element::Type& type) {
    if (type == ov::element::u8) {
        return "u8";
    }
    if (type == ov::element::i8) {
        return "i8";
    }
    if (type == ov::element::i4) {
        return "i4";
    }
    if (type == ov::element::dynamic) {
        return "dyn";
    }
    return "other";
}

std::string make_test_name(const ResolveStorageTypesParams& p) {
    std::ostringstream os;
    os << mode_to_token(p.mode) << "_" << (p.is_symmetric ? "Sym" : "Asym") << "_Q" << type_to_token(p.quant_dt)
       << "_S" << type_to_token(p.scale_dt)
       << "_RQ" << type_to_token(p.expected_quantized_data_type) << "_RZ"
       << type_to_token(p.expected_zero_point_type) << "_RS" << type_to_token(p.expected_scale_type);
    return os.str();
}

std::vector<ResolveStorageTypesParams> make_all_cases() {
    return {
        {DQMode::HandcraftedSymmetricI8, true, ov::element::u8, ov::element::f32, ov::element::u8,
         ov::element::dynamic, ov::element::f32},
        {DQMode::HandcraftedSymmetricI8, true, ov::element::i8, ov::element::f32, ov::element::i8,
         ov::element::dynamic, ov::element::f32},
        {DQMode::HandcraftedSymmetricI8, true, ov::element::i4, ov::element::f32, ov::element::i4,
         ov::element::dynamic, ov::element::f32},
        {DQMode::HandcraftedSymmetricI8, false, ov::element::u8, ov::element::f32, ov::element::u8,
         ov::element::u8, ov::element::f32},
        {DQMode::HandcraftedSymmetricI8, false, ov::element::i8, ov::element::f32, ov::element::i8,
         ov::element::i8, ov::element::f32},
        {DQMode::HandcraftedSymmetricI8, false, ov::element::i4, ov::element::f32, ov::element::i4,
         ov::element::i4, ov::element::f32},

        {DQMode::OnnxDynamicQuantizeLinear, true, ov::element::u8, ov::element::f32, ov::element::u8,
         ov::element::dynamic, ov::element::f32},
        {DQMode::OnnxDynamicQuantizeLinear, true, ov::element::i8, ov::element::f32, ov::element::i8,
         ov::element::dynamic, ov::element::f32},
        {DQMode::OnnxDynamicQuantizeLinear, true, ov::element::i4, ov::element::f32, ov::element::i4,
         ov::element::dynamic, ov::element::f32},
        {DQMode::OnnxDynamicQuantizeLinear, false, ov::element::u8, ov::element::f32, ov::element::u8,
         ov::element::u8, ov::element::f32},
        {DQMode::OnnxDynamicQuantizeLinear, false, ov::element::i8, ov::element::f32, ov::element::i8,
         ov::element::i8, ov::element::f32},
        {DQMode::OnnxDynamicQuantizeLinear, false, ov::element::i4, ov::element::f32, ov::element::i4,
         ov::element::i4, ov::element::f32},

        {DQMode::CompilerPatternI8, true, ov::element::u8, ov::element::f32, ov::element::u8,
         ov::element::dynamic, ov::element::f32},
        {DQMode::CompilerPatternI8, true, ov::element::i8, ov::element::f32, ov::element::i8,
         ov::element::dynamic, ov::element::f32},
        {DQMode::CompilerPatternI8, true, ov::element::i4, ov::element::f32, ov::element::i4,
         ov::element::dynamic, ov::element::f32},
        {DQMode::CompilerPatternI8, false, ov::element::u8, ov::element::f32, ov::element::u8,
         ov::element::u8, ov::element::f32},
        {DQMode::CompilerPatternI8, false, ov::element::i8, ov::element::f32, ov::element::u8,
         ov::element::u8, ov::element::f32},
        {DQMode::CompilerPatternI8, false, ov::element::i4, ov::element::f32, ov::element::i4,
         ov::element::i4, ov::element::f32},

        {DQMode::CompilerPatternI8, false, ov::element::i8, ov::element::f16, ov::element::u8,
         ov::element::u8, ov::element::f16},
        {DQMode::OnnxDynamicQuantizeLinear, true, ov::element::u8, ov::element::f16, ov::element::u8,
         ov::element::dynamic, ov::element::f16},
    };
}

class ResolveDynamicQuantStorageTypesTest : public ::testing::TestWithParam<ResolveStorageTypesParams> {};

TEST_P(ResolveDynamicQuantStorageTypesTest, CoversAllQuantDtSymmetryAndDecomposeModeCombinations) {
    const auto& p = GetParam();

    const auto resolved =
        ov::npuw::util::resolve_dynamic_quant_storage_types(p.mode, p.is_symmetric, p.quant_dt, p.scale_dt);

    EXPECT_EQ(resolved.quantized_data_type, p.expected_quantized_data_type);
    EXPECT_EQ(resolved.zero_point_type, p.expected_zero_point_type);
    EXPECT_EQ(resolved.scale_type, p.expected_scale_type);
}

INSTANTIATE_TEST_SUITE_P(AllCombinations,
                         ResolveDynamicQuantStorageTypesTest,
                         ::testing::ValuesIn(make_all_cases()),
                         [](const ::testing::TestParamInfo<ResolveStorageTypesParams>& info) {
                             return make_test_name(info.param);
                         });

}  // namespace
