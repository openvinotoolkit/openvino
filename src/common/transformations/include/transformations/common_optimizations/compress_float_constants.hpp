// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API CompressFloatConstantsImpl;
class TRANSFORMATIONS_API CompressFloatConstants;

void TRANSFORMATIONS_API compress_model_to_f16(const std::shared_ptr<Model>& model, bool postponed = false);
bool TRANSFORMATIONS_API is_model_optimized(const std::shared_ptr<ov::Model>& model);

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief CompressFloatConstantsImpl transformation replaces FP32/FP64 Constants with FP16 ones.
 */
class ov::pass::CompressFloatConstantsImpl : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("CompressFloatConstantsImpl", "0");
    /// @brief Transformation constructor
    /// @param postponed If true then the transformation won't compress the constants
    ///                  keeping them in the original type but still will insert Converts. This is
    ///                  a special mode of operation that requires another transformation to
    ///                  apply a real compression on constants. Constants eligible for
    ///                  postponed compression are marked with a special rt_info tag.
    CompressFloatConstantsImpl(bool postponed = false);
};

/**
 * @ingroup ov_transformation_common_api
 * @brief CompressFloatConstants transformation replaces FP32/FP64 Constants with FP16 ones.
 */
class ov::pass::CompressFloatConstants : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("CompressFloatConstants", "0");
    /// @brief Transformation constructor
    /// @param postponed Postponed compression, see ov::pass::CompressFloatConstantsImpl for details.
    CompressFloatConstants(bool postponed = false) {
        add_matcher<ov::pass::CompressFloatConstantsImpl>(postponed);
    }
};
