// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Split over channels for Eltwise to avoid GNA-HW kBufferMaxSize limitation per eltwise
 */
class SplitEltwise : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SplitEltwise", "0");
    SplitEltwise();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
