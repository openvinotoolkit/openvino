// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
/**
 * @ingroup ov_transformation_common_api
 * @brief Converts EmbeddingBagPacked v15 to EmbeddingBagPacked v3.
 */
class TRANSFORMATIONS_API ConvertEmbeddingBagPacked15ToEmbeddingBagPackedSum3 : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertEmbeddingBagPacked15ToEmbeddingBagPackedSum3");
    ConvertEmbeddingBagPacked15ToEmbeddingBagPackedSum3();
};

}  // namespace pass
}  // namespace ov
