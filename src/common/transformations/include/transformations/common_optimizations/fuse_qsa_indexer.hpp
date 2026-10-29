// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses QSAPrepare and SparseAttentionIndexer into the stateful internal QSAIndexer op.
 *
 * The pass recognizes the supported RMSNorm -> SparseAttentionIndexer path and the
 * QSAPrepare state inputs, replacing them with one operation that produces selections
 * and updated cache state.
 */
class TRANSFORMATIONS_API FuseQSAIndexer : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseQSAIndexer");
    FuseQSAIndexer();
};

}  // namespace ov::pass
