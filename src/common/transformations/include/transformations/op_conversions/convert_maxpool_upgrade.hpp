// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertMaxPool1ToMaxPool8;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertMaxPool1ToMaxPool8 converts v1::MaxPool into v8::MaxPool.
 */

class ov::pass::ConvertMaxPool1ToMaxPool8 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertMaxPool1ToMaxPool8");
    ConvertMaxPool1ToMaxPool8();
};
