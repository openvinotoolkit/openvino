// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief ConvertPadded2ValidConv transformation breaks down padded convolutions into a set of unpadded ones
 */
class ConvertPadded2ValidConv : public ngraph::pass::FunctionPass {
public:
  NGRAPH_RTTI_DECLARATION;
  bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

} // namespace GNAPluginNS
