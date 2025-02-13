// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_cpu::tpp::pass {

/**
 * @interface BrgemmToBrgemmTPP
 * @brief Converts Snippets Brgemm to BrgemmTPP operation, if possible. Only fp32 Brgemms are currently converted.
 * @ingroup snippets
 */
class BrgemmToBrgemmTPP : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("BrgemmToBrgemmTPP");
    BrgemmToBrgemmTPP();

    static bool is_supported_brgemm_configuration(const std::vector<std::vector<size_t>>& layouts,
                                                  const ov::element::TypeVector& precisions);
};

}  // namespace ov::intel_cpu::tpp::pass
