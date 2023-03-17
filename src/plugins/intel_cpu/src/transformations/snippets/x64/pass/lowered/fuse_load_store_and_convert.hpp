// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/pass/lowered/linear_IR_transformation.hpp"

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
class FuseLoadStoreConvert: public ngraph::snippets::pass::lowered::LinearIRTransformation {
public:
    FuseLoadStoreConvert() = default;
    OPENVINO_RTTI("FuseLoadStoreConvert", "LinearIRTransformation");
    bool run(ngraph::snippets::LoweredExprIR& linear_ir) override;

private:
    bool fuse_load_convert(ngraph::snippets::LoweredExprIR& linear_ir,
                           ngraph::snippets::LoweredExprIR::constExprIt& convert_it);
    bool fuse_store_convert(ngraph::snippets::LoweredExprIR& linear_ir,
                            ngraph::snippets::LoweredExprIR::constExprIt& convert_it);
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
