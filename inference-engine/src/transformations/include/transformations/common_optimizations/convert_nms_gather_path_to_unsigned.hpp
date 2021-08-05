// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/pass.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertNmsGatherPathToUnsigned;

}  // namespace pass
}  // namespace ngraph

/**
 * Converts Gather indices to unsigned if indices are from NMS selected indices output.
 * NMS returns -1 for not selected boxes, old version of Gather fill corresponding
 * output for such indices with zero.
 * But new * Gather-8 has support negative indices indicating count from the end.
 * In order to keep such behaviour (until dynamism is not supported) instead of -1 new
 * Gather-8 will accept UINT32_MAX which is always outside of the bounds
 * and corresponding output for such indices in gather always will be filled with zeros.
 */
class ngraph::pass::ConvertNmsGatherPathToUnsigned: public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};
