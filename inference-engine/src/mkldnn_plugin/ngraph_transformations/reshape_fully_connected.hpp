// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

/*
 * Description:
 *     ReshapeFullyConnected transformation detects FullyConnected operations
 *     and for each operation where input shape is greater than 2 inserts Reshape
 *     operations before and after FullyConnected operation. This transformation is
 *     required because of IE restrictions.
 */

namespace MKLDNNPlugin {

class ReshapeFullyConnected: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ReshapeFullyConnected();
};

}  // namespace MKLDNNPlugin
