// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/MatcherPass.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RMSFusion;

}  // namespace pass
}  // namespace ov

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RMSFusion;

}  // namespace pass
}  // namespace ov

namespace ov {
namespace pass {

class RMSFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RMSFusion", "0");
    RMSFusion();
};

}  // namespace pass
}  // namespace ov
