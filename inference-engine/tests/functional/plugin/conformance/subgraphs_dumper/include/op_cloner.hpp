// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <memory>

#include <ngraph/ngraph.hpp>
#include "ops_cache.hpp"

namespace SubgraphsDumper {

struct ClonersMap {
    using clone_fn = std::function<const std::shared_ptr<ngraph::Node>(const std::shared_ptr<ngraph::Node> &,
                                                                       LayerTestsUtils::OPInfo &meta)>;
    using cloners_map_type = std::map<ngraph::NodeTypeInfo, clone_fn>;

    static float constant_size_threshold_mb;
    static const cloners_map_type cloners;
};

}  // namespace SubgraphsDumper
