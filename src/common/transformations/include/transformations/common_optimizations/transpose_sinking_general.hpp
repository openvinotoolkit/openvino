// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinkingGeneralForward;
class TRANSFORMATIONS_API TransposeSinkingGeneralBackward;
class TRANSFORMATIONS_API TransposeSinkingGeneral;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeSinkingGeneralForward transformation combines all TransposeSinkingForward* transformations into
 * single GraphRewrite pass.
 */
class ov::pass::TransposeSinkingGeneralForward : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("TransposeSinkingGeneralForward", "0");
    TransposeSinkingGeneralForward();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeSinkingGeneralBackward transformation combines all TransposeSinkingBackward* transformations into
 * single GraphRewrite pass.
 */
class ov::pass::TransposeSinkingGeneralBackward : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("TransposeSinkingGeneralBackward", "0");
    TransposeSinkingGeneralBackward();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeSinkingGeneral transformation combines TransposeSinkingGeneralForward and
 * TransposeSinkingGeneralBackward transformations into single ModelPass pass and inserts
 * ConstantFolding pass after them.
 */
class ov::pass::TransposeSinkingGeneral : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("TransposeSinkingGeneral", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
