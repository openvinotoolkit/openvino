// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace transformation_sample {
namespace passes {

/**
 * @brief Inserts Add after Paramter.
 *
 * Note: This is just examplary transformation.
 *
 * [Parameter]          [Parameter]
 *      |                    |
 *  [Any Layer]     =>    [Add]
 *                          |
 *                      [Any Layer]
 *
 */
class InsertAddAfterParameter : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertAddAfterParameter", "0");
    InsertAddAfterParameter();
};

}  // namespace passes
}  // namespace transformation_sample
