// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <list>
#include <memory>
#include <vector>

#include "ngraph/deprecated.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/pass_config.hpp"
#include "ngraph/util.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {

class Manager;

}
}  // namespace ov
namespace ngraph {
namespace pass {
using FunctionPass = ov::pass::ModelPass;
using ov::pass::Manager;
using ov::pass::PassBase;
using ov::pass::PassProperty;
using ov::pass::PassPropertyMask;
NGRAPH_API_DEPRECATED
const PassPropertyMask all_pass_property_off;

class NGRAPH_API_DEPRECATED NGRAPH_API NodePass : public PassBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ~NodePass() override;
    virtual bool run_on_node(std::shared_ptr<ngraph::Node>) = 0;
};

enum class NGRAPH_API_DEPRECATED FusionType : uint32_t {
    //`DIFFERENTIABLE_FUSIONS` produce ops that support autodiff
    // i.e. implement `generate_adjoints`
    DIFFERENTIABLE_FUSIONS = 0x1,
    REGULAR_FUSIONS = 0x2,
    //`FOP_FUSIONS` produce ops in the FusedOps category that might
    // not be supported by all backends
    FOP_FUSIONS = 0x4,
    ALL_FUSIONS = 0xFFFFFFFF
};

NGRAPH_SUPPRESS_DEPRECATED_START
using FusionTypeMask = ov::EnumMask<FusionType>;
NGRAPH_SUPPRESS_DEPRECATED_END
}  // namespace pass
}  // namespace ngraph
