// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RemoveMultiSubGraphOpDanglingParamsResults;

}  // namespace pass
}  // namespace ov

/*
 * @ingroup ov_transformation_common_api
 * @brief RemoveMultiSubGraphOpDanglingParamsResults transformation removes MultiSubGraphOp inputs which are not
 * connected to other nodes in the bodies of a MultiSubGraphOp and outputs that are not used in the Model
 */

class ov::pass::RemoveMultiSubGraphOpDanglingParamsResults : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("RemoveMultiSubGraphOpDanglingParamsResults");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
