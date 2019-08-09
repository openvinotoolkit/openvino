// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"

namespace ngraph {
namespace pass {

class ConvertBroadcastToTiles;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertBroadcastToTiles: public ngraph::pass::GraphRewrite {
public:
    ConvertBroadcastToTiles() : GraphRewrite() {
        convert_broadcast_to_tiles();
    }

private:
    void convert_broadcast_to_tiles();
};
