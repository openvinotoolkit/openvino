// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface FuseLoadStoreConvert
 * @brief Fuse Load and ConvertSaturation into one op LoadConvertSaturation
 *        Fuse Load and ConvertTruncation into one op LoadConvertTruncation
 *        Fuse Store and ConvertSaturation into one op StoreConvertSaturation
 *        Fuse Store and ConvertTruncation into one op StoreConvertTruncation
 * @ingroup snippets
 */
class FuseLoadStoreConvert: public snippets::lowered::pass::RangedPass {
public:
    FuseLoadStoreConvert() = default;
    OPENVINO_RTTI("FuseLoadStoreConvert", "RangedPass");
    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;

private:
    bool fuse_load_convert(snippets::lowered::LinearIR& linear_ir,
                           snippets::lowered::LinearIR::constExprIt& convert_it);
    bool fuse_store_convert(snippets::lowered::LinearIR& linear_ir,
                            snippets::lowered::LinearIR::constExprIt& convert_it);
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
