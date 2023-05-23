// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @brief
 * @ingroup ov_pass_cpp_api
 */
// tj notes:
//      - when to apply? before or after others:
//          * before might break some other transforms (not aware of multi-comsumers)
//      - apply after ConstFold
class TRANSFORMATIONS_API MergeSimilarBranches : public ModelPass {
public:
    OPENVINO_RTTI("MergeSimilarBranches");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace ov
