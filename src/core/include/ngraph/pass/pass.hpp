// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
NGRAPH_DEPRECATED("This variable is deprecated and will be removed soon.")
const PassPropertyMask all_pass_property_off;

class NGRAPH_DEPRECATED("Use MatcherPass or FunctionPass instead.") NGRAPH_API NodePass : public PassBase {
public:
    NGRAPH_RTTI_DECLARATION;
    ~NodePass() override;
    virtual bool run_on_node(std::shared_ptr<ngraph::Node>) = 0;
};

enum class NGRAPH_DEPRECATED("FusionType is no longer used anywhere. Please do no use it.") FusionType : uint32_t {
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
