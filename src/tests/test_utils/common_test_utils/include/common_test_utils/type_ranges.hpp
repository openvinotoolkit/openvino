// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <map>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/type/element_type_traits.hpp"

namespace ov {
namespace test {
namespace utils {

static const std::vector<element::Type>& get_known_types() {
    static const auto known_types = [] {
        using namespace ov::element;
        constexpr size_t enum_count = static_cast<std::underlying_type_t<Type_t>>(Type_t::f8e8m0) - 1;

        std::vector<Type> types(enum_count);
        for (size_t idx = 1, i = 0; i < types.size(); ++idx, ++i) {
            types[i] = Type{static_cast<Type_t>(idx)};
        }
        return types;
    }();

    return known_types;
}

static ov::test::utils::InputGenerateData get_range_by_type(
    ov::element::Type elemType,
    uint32_t max_range_limit = testing::internal::Random::kMaxRange) {
    double min_start = 0 - static_cast<int32_t>(round(max_range_limit / 2));

    ov::test::utils::InputGenerateData inData(min_start, max_range_limit);
    auto apply_data = [&inData, &max_range_limit, &min_start](double new_range, double start_from) {
        if (new_range < max_range_limit) {
            inData.start_from = start_from;
            inData.range = static_cast<int32_t>(round(new_range));
        } else {
            inData.start_from = start_from > min_start ? start_from : min_start;
            inData.range = max_range_limit;
        }
    };

#define CASE_OV_TYPE(X)                                                   \
    case X: {                                                             \
        using dataType = typename ov::element_type_traits<X>::value_type; \
        dataType lowest_tmp = std::numeric_limits<dataType>::lowest();    \
        dataType max_tmp = std::numeric_limits<dataType>::max();          \
        double lowest = 0 - static_cast<double>(lowest_tmp.to_bits());    \
        double max = max_tmp.to_bits();                                   \
        double tmp_range = max - lowest;                                  \
        apply_data(tmp_range, lowest);                                    \
        break;                                                            \
    }

#define CASE_C_TYPE(X)                                                                       \
    case X: {                                                                                \
        auto lowest = std::numeric_limits<ov::element_type_traits<X>::value_type>::lowest(); \
        auto max = std::numeric_limits<ov::element_type_traits<X>::value_type>::max();       \
        double tmp_range = static_cast<double>(max) - static_cast<double>(lowest);           \
        apply_data(tmp_range, lowest);                                                       \
        break;                                                                               \
    }

    switch (elemType) {
    case (ov::element::Type_t::dynamic): {
        inData.start_from = min_start;
        inData.range = max_range_limit;
        break;
    }
    case (ov::element::Type_t::boolean): {
        inData.start_from = 0;
        inData.range = 2;
        break;
    }
    case ov::element::Type_t::string: {
        auto lowest = std::numeric_limits<char>::lowest();
        auto max = std::numeric_limits<char>::max();

        double tmp_range = static_cast<double>(max) - static_cast<double>(lowest);
        apply_data(tmp_range, lowest);

        break;
    }
        CASE_OV_TYPE(ov::element::Type_t::f8e4m3)
        CASE_OV_TYPE(ov::element::Type_t::f8e5m2)
        CASE_OV_TYPE(ov::element::Type_t::f4e2m1)
        CASE_OV_TYPE(ov::element::Type_t::f8e8m0)
        CASE_OV_TYPE(ov::element::Type_t::bf16)
        CASE_OV_TYPE(ov::element::Type_t::f16)
        CASE_C_TYPE(ov::element::Type_t::f32)
        CASE_C_TYPE(ov::element::Type_t::f64)
        CASE_C_TYPE(ov::element::Type_t::i4)
        CASE_C_TYPE(ov::element::Type_t::i8)
        CASE_C_TYPE(ov::element::Type_t::i16)
        CASE_C_TYPE(ov::element::Type_t::i32)
        CASE_C_TYPE(ov::element::Type_t::i64)
        CASE_C_TYPE(ov::element::Type_t::u1)
        CASE_C_TYPE(ov::element::Type_t::u2)
        CASE_C_TYPE(ov::element::Type_t::u3)
        CASE_C_TYPE(ov::element::Type_t::u4)
        CASE_C_TYPE(ov::element::Type_t::u6)
        CASE_C_TYPE(ov::element::Type_t::u8)
        CASE_C_TYPE(ov::element::Type_t::nf4)
        CASE_C_TYPE(ov::element::Type_t::u16)
        CASE_C_TYPE(ov::element::Type_t::u32)
        CASE_C_TYPE(ov::element::Type_t::u64)
        break;
    }

    return inData;
}

struct RangeByType {
    std::map<ov::element::Type, ov::test::utils::InputGenerateData> data;

    RangeByType() {
        for (const auto& type : get_known_types()) {
            data[type] = get_range_by_type(type);
        }
    }

    ov::test::utils::InputGenerateData get_range(ov::element::Type type) {
        if (data.count(type) > 0) {
            return data.at(type);
        } else {
            throw std::runtime_error("Couln't find Type in typeMap: " + type.to_string());
        }
    }
};

static RangeByType rangeByType;

}  // namespace utils
}  // namespace test
}  // namespace ov
