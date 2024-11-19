// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
/**
 * @ingroup ov_transformation_common_api
 * @brief MarkDequantizationAndDecompression is a set of transformation which mark
 * Dequantization and Decompression patterns with the keep_const_precision, disable_const_folding and
 * dequantization attributes. Also it calls ConstantFolding.
 */
class TRANSFORMATIONS_API MarkDequantizationAndDecompression : public ModelPass {
public:
    OPENVINO_RTTI("MarkDequantizationAndDecompression", "0");
    explicit MarkDequantizationAndDecompression(element::TypeVector precisions,
                                                const bool fold_subtract_const = false,
                                                const bool fold_multiply_const = true)
        : m_fold_subtract_const(fold_subtract_const),
          m_fold_multiply_const(fold_multiply_const),
          m_precisions(std::move(precisions)) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    bool m_fold_subtract_const = false;
    bool m_fold_multiply_const = true;
    element::TypeVector m_precisions;
};

}  // namespace pass
}  // namespace ov
