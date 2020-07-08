// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/transformation_context.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

TransformationContext::TransformationContext(std::shared_ptr<Function> network)
    : network(network) {
#if 0   // TODO LPT-TO-NGRAPH
    auto it = details::CNNNetworkIterator(&network);
    auto end = details::CNNNetworkIterator();
    while (it != end) {
        _original_precisions_map[(*it)->name] = {};
        for (auto data : (*it)->outData) _original_precisions_map[(*it)->name][data->getName()] = data->getPrecision();
        it++;
    }
#endif
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
