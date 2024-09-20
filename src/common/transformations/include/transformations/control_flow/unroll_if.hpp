// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API UnrollIf;

}  // namespace pass
}  // namespace ov

// clang-format off
/**
 * @ingroup ov_transformation_common_api
 * @brief The transformation replaces 'If' operations with one of the internal functions (bodies) if the provided condition is constant.
 * The condition is true: 'If' op is replaced with then_body
 * The condition is false 'If' op is replaced with else_body
 */
// clang-format on

class ov::pass::UnrollIf : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("UnrollIf", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
