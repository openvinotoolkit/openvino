// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface Validate
 * @brief The pass validates OV model on correctness after all common optimizations
 * @ingroup snippets
 */
class Validate: public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("Validate", "0");
    Validate() = default;

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
