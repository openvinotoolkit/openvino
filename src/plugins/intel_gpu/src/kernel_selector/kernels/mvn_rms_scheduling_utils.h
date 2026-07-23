// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cctype>
#include <limits>
#include <string>

namespace kernel_selector {

class StaticDimExpressionParser {
public:
    static bool IsDecimalNumber(const std::string& value) {
        return !value.empty() && std::all_of(value.begin(), value.end(), [](char c) {
            return std::isdigit(static_cast<unsigned char>(c));
        });
    }

    // Accepts forms like "(a*b*c)" where each factor is a non-negative decimal integer.
    // Returns false on malformed input or overflow.
    static bool TryFoldMulExpression(const std::string& expression, size_t& folded_value) {
        if (expression.size() < 3 || expression.front() != '(' || expression.back() != ')')
            return false;

        const size_t max_size_t = std::numeric_limits<size_t>::max();
        const size_t end = expression.size() - 1;
        size_t pos = 1;
        size_t result = 1;

        while (pos < end) {
            if (!std::isdigit(static_cast<unsigned char>(expression[pos])))
                return false;

            size_t factor = 0;
            while (pos < end && std::isdigit(static_cast<unsigned char>(expression[pos]))) {
                const size_t digit = static_cast<size_t>(expression[pos] - '0');
                if (factor > (max_size_t - digit) / 10)
                    return false;
                factor = factor * 10 + digit;
                ++pos;
            }

            if (factor != 0 && result > max_size_t / factor)
                return false;
            result *= factor;

            if (pos == end)
                break;
            if (expression[pos] != '*')
                return false;
            ++pos;
        }

        folded_value = result;
        return true;
    }
};

class MvnSchedulingPolicy {
public:
    struct Policy {
        size_t target_items;
        size_t stack_cap;
    };

    static constexpr size_t kMaxRegisterStack = 16;

    // Generalized rule from ce_test_AI_job/aboutSHW: pick the largest power-of-two
    // LWS that keeps about target_items normalized elements per work-item.
    static size_t GetGeneralizedLws(size_t data_set_size, size_t max_lws, size_t target_items = kTargetItemsPerWi) {
        size_t lws = 1;
        const size_t limit = std::max<size_t>(1, std::min(max_lws, data_set_size / target_items));
        while (2 * lws <= limit) {
            lws *= 2;
        }
        return lws;
    }

    static size_t GetStackSize(size_t data_set_size, size_t lws) {
        return (data_set_size + lws - 1) / lws;
    }

    // Adaptive policy from omni benchmarks:
    // - Prefer t16 in the stable mid-range window.
    // - Prefer t8 outside that window.
    // - Tighten register stack cap for tiny/very-wide rows to reduce pressure.
    static Policy GetAdaptivePolicy(size_t data_set_size) {
        const size_t target_items =
            (data_set_size >= kAdaptiveT16MinDataSetSize && data_set_size <= kAdaptiveT16MaxDataSetSize)
                ? kTargetItemsPerWiWide
                : kTargetItemsPerWi;
        const size_t stack_cap = GetAdaptiveStackCap(data_set_size, target_items);
        return {target_items, stack_cap};
    }

private:
    static constexpr size_t kTargetItemsPerWi = 8;
    static constexpr size_t kTargetItemsPerWiWide = 16;
    static constexpr size_t kAdaptiveStackCap = 12;
    static constexpr size_t kAdaptiveT16MinDataSetSize = 1024;
    static constexpr size_t kAdaptiveT16MaxDataSetSize = 6144;

    static size_t GetAdaptiveStackCap(size_t data_set_size, size_t target_items, size_t base_cap = kMaxRegisterStack) {
        if (target_items >= kTargetItemsPerWiWide &&
            data_set_size >= kAdaptiveT16MinDataSetSize &&
            data_set_size <= kAdaptiveT16MaxDataSetSize) {
            return base_cap;
        }

        if (data_set_size < 512 || data_set_size >= 8192) {
            return std::min(base_cap, kAdaptiveStackCap);
        }

        return base_cap;
    }
};

class RmsSchedulingPolicy {
public:
    struct Policy {
        size_t target_items;
        size_t stack_cap;
    };

    static constexpr size_t kSubgroupSize = 16;
    static constexpr size_t kMaxRegisterStack = 16;

    // Generalized rule from ce_test_AI_job/aboutSHW: pick largest power-of-two LWS
    // while keeping approximately target_items normalized elements per work-item.
    static size_t GetGeneralizedLws(size_t data_size, size_t max_lws, size_t target_items = kTargetItemsPerWi) {
        size_t lws = kSubgroupSize;
        const size_t limit = std::max(kSubgroupSize, std::min(max_lws, data_size / target_items));
        while (2 * lws <= limit) {
            lws *= 2;
        }
        return lws;
    }

    static size_t GetStackSize(size_t data_size, size_t lws) {
        return (data_size + lws - 1) / lws;
    }

    static size_t GetAdaptiveSubgroupBlockSize(size_t items, size_t preferred = 8) {
        return std::min(preferred, GetSubgroupBlockSize(items));
    }

    // Adaptive policy from omni benchmarks:
    // - Prefer t16 for wide-ish rows.
    // - Prefer t8 elsewhere.
    // - Reduce stack cap on tiny/huge rows to avoid pressure-driven regressions.
    static Policy GetAdaptivePolicy(size_t data_size) {
        if (data_size >= kAdaptiveT16MinDataSize && data_size < kAdaptiveT16MaxDataSize) {
            return {kTargetItemsPerWiWide, GetAdaptiveStackCap(data_size, kMaxRegisterStack)};
        }

        const size_t base_cap = data_size <= 2048 ? kAdaptiveStackCap : kMaxRegisterStack;
        return {kTargetItemsPerWi, GetAdaptiveStackCap(data_size, base_cap)};
    }

private:
    static constexpr size_t kTargetItemsPerWi = 8;
    static constexpr size_t kTargetItemsPerWiWide = 16;
    static constexpr size_t kAdaptiveStackCap = 12;
    static constexpr size_t kAdaptiveT16MinDataSize = 6144;
    static constexpr size_t kAdaptiveT16MaxDataSize = 16384;

    static size_t GetSubgroupBlockSize(size_t items) {
        if ((items >> 3) != 0) {
            return 8;
        }
        if ((items >> 2) != 0) {
            return 4;
        }
        if ((items >> 1) != 0) {
            return 2;
        }
        return 1;
    }

    static size_t GetAdaptiveStackCap(size_t data_size, size_t base_cap) {
        if (data_size < 512 || data_size >= 8192) {
            return std::min(base_cap, kAdaptiveStackCap);
        }
        return base_cap;
    }
};

}  // namespace kernel_selector
