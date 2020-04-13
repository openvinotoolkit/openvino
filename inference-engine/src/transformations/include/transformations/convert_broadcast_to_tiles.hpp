// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertBroadcastToTiles);

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
