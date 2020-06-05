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

#if 1   // TODO LPT-TO-NGRAPH: not needed?
void TransformationContext::removeLayer(std::shared_ptr<Node> layer) {
    std::cerr << "Deprecated function TransformationContext::removeLayer is called at " << __FILE__ << ':' << __LINE__ << '\n';
//    for (size_t i = 0lu; i < layers.size(); ++i) {
//        // FIXME: rely on node pointer or names, not on friendly names
//        if ((layers[i] != nullptr) && (layers[i]->get_friendly_name() == layer->get_friendly_name())) {
//            layers[i] = nullptr;
//            break;
//        }
//    }
}
#endif

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph