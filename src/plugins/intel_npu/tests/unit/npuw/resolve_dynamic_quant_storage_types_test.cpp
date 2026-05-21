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
    ov::element::Type expected_quantized_data_type;
    ov::element::Type expected_zero_point_type;
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
       << "_RQ" << type_to_token(p.expected_quantized_data_type) << "_RZ"
       << type_to_token(p.expected_zero_point_type);
    return os.str();
}

std::vector<ResolveStorageTypesParams> make_all_cases() {
    static const std::vector<DQMode> modes = {
        DQMode::HandcraftedSymmetricI8,
        DQMode::OnnxDynamicQuantizeLinear,
        DQMode::CompilerPatternI8,
    };
    static const std::vector<ov::element::Type> quant_dtypes = {
        ov::element::u8,
        ov::element::i8,
        ov::element::i4,
    };

    std::vector<ResolveStorageTypesParams> cases;
    cases.reserve(modes.size() * 2 * quant_dtypes.size());

    for (const auto mode : modes) {
        for (const bool is_symmetric : {true, false}) {
            for (const auto& quant_dt : quant_dtypes) {
                ResolveStorageTypesParams p{};
                p.mode = mode;
                p.is_symmetric = is_symmetric;
                p.quant_dt = quant_dt;

                p.expected_quantized_data_type = quant_dt;
                p.expected_zero_point_type = is_symmetric ? ov::element::dynamic : quant_dt;

                if (!is_symmetric && mode == DQMode::CompilerPatternI8 && quant_dt == ov::element::i8) {
                    p.expected_quantized_data_type = ov::element::u8;
                    p.expected_zero_point_type = ov::element::u8;
                }

                cases.push_back(p);
            }
        }
    }

    return cases;
}

class ResolveDynamicQuantStorageTypesTest : public ::testing::TestWithParam<ResolveStorageTypesParams> {};

TEST_P(ResolveDynamicQuantStorageTypesTest, CoversAllQuantDtSymmetryAndDecomposeModeCombinations) {
    const auto& p = GetParam();

    const auto resolved = ov::npuw::util::resolve_dynamic_quant_storage_types(p.mode, p.is_symmetric, p.quant_dt);

    EXPECT_EQ(resolved.quantized_data_type, p.expected_quantized_data_type);
    EXPECT_EQ(resolved.zero_point_type, p.expected_zero_point_type);
}

INSTANTIATE_TEST_SUITE_P(AllCombinations,
                         ResolveDynamicQuantStorageTypesTest,
                         ::testing::ValuesIn(make_all_cases()),
                         [](const ::testing::TestParamInfo<ResolveStorageTypesParams>& info) {
                             return make_test_name(info.param);
                         });

}  // namespace
