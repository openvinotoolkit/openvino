// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pattern/matcher.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface FuseLoadConvert
 * @brief Fuse Load and ConvertSaturation into one op LoadConvertSaturation
 *        Fuse Load and ConvertTruncation into one op LoadConvertTruncation
 * @ingroup snippets
 */
class FuseLoadConvert: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseLoadConvert", "0");
    FuseLoadConvert();
};

/**
 * @interface FuseStoreConvert
 * @brief Fuse Store and ConvertSaturation into one op StoreConvertSaturation
 *        Fuse Store and ConvertTruncation into one op StoreConvertTruncation
 * @ingroup snippets
 */
class FuseStoreConvert: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseStoreConvert",  "0");
    FuseStoreConvert();
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov
