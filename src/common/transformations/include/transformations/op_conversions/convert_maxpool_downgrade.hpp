// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertMaxPool8ToMaxPool1;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertMaxPool8ToMaxPool1 converts v8::MaxPool into v1::MaxPool.
 */
class ov::pass::ConvertMaxPool8ToMaxPool1 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertMaxPool8ToMaxPool1");
    ConvertMaxPool8ToMaxPool1();
};
