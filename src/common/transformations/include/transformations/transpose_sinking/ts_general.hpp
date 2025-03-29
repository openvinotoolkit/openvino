// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
namespace transpose_sinking {

class TRANSFORMATIONS_API TSGeneralForward;
class TRANSFORMATIONS_API TSGeneralBackward;
class TRANSFORMATIONS_API TSGeneral;

}  // namespace transpose_sinking

using TransposeSinkingGeneral = ov::pass::transpose_sinking::TSGeneral;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief TSGeneralForward transformation combines all TransposeSinkingForward* transformations into
 * single GraphRewrite pass.
 */
class ov::pass::transpose_sinking::TSGeneralForward : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("TSGeneralForward");
    TSGeneralForward();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief TSGeneralBackward transformation combines all TransposeSinkingBackward* transformations into
 * single GraphRewrite pass.
 */
class ov::pass::transpose_sinking::TSGeneralBackward : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("TSGeneralBackward");
    TSGeneralBackward();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief TSGeneral transformation combines TSGeneralForward and
 * TSGeneralBackward transformations into single ModelPass pass and inserts
 * ConstantFolding pass after them.
 */
class ov::pass::transpose_sinking::TSGeneral : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("TSGeneral");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
