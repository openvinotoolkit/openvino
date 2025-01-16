// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
/**
 * @ingroup ov_transformation_common_api
 * @brief Converts EmbeddingBagOffsets v15 to EmbeddingBagOffsets v3.
 */
class TRANSFORMATIONS_API ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3 : public MatcherPass {
public:
    OPENVINO_RTTI("ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3", "0");
    ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3();
};

}  // namespace pass
}  // namespace ov
