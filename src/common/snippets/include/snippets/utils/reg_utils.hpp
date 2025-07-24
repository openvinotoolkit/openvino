// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>

#include "openvino/core/except.hpp"
#include "snippets/emitter.hpp"

namespace ov::snippets::utils {
inline static std::vector<size_t> transform_snippets_regs_to_idxs(const std::vector<Reg>& regs, RegType expected_type) {
    std::vector<size_t> idxs;
    idxs.reserve(regs.size());
    for (const auto& reg : regs) {
        const bool valid_reg_type =
            reg.type == RegType::address || expected_type == RegType::undefined || reg.type == expected_type;
        OPENVINO_ASSERT(valid_reg_type, "Reg type mismatch during to_idxs conversion");
        idxs.emplace_back(reg.idx);
    }
    return idxs;
}
inline static std::vector<size_t> transform_snippets_regs_to_idxs(const std::vector<Reg>& regs) {
    std::vector<size_t> idxs;
    std::transform(regs.begin(), regs.end(), std::back_inserter(idxs), [](const Reg& r) {
        return r.idx;
    });
    return idxs;
}

}  // namespace ov::snippets::utils
