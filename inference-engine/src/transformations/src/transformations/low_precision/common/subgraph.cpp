// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/low_precision/common/subgraph.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cassert>

#include <ngraph/rt_info.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "transformations/low_precision/quantization_details.hpp"
#include "transformations/low_precision/common/ie_lpt_exception.hpp"

// #include <ie_common.h>
// #include <precision_utils.h>
// #include "cnn_network_impl.hpp"
// #include "ie_util_internal.hpp"
// #include "ie_parallel.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

// TODO: temporary commented : possible it doesn't need
// static const std::unordered_set<std::string> intermediateLayers{
//    "MaxPool",
//    "Resample"
// };

// TODO: Resample is skipped
bool isIntermediate(const ngraph::Node& node) {
    return is_type<ngraph::opset1::MaxPool>(&node);
}

bool Subgraph::fillSubgraphForQuantization(ngraph::opset1::FakeQuantize& fakeQuantize, std::unordered_set<std::string>& handledLayers) {
    // TODO: uncomment later
    //if (!ngraph::pass::low_precision::QuantizationDetails::outputLayoutIsSupported(fakeQuantize)) {
    //    return false;
    //}

    quantizationLayers.push_back(&fakeQuantize);
    handledLayers.insert(fakeQuantize.get_friendly_name());
    layers.emplace(fakeQuantize.get_friendly_name(), &fakeQuantize);

    for (size_t index = 0; index < fakeQuantize.get_output_size(); ++index) {
        const auto childInputs = fakeQuantize.get_output_target_inputs(index);
        for (const auto childInput : childInputs) {
            ngraph::Node& child = *childInput.get_node();
            if (handledLayers.find(child.get_friendly_name()) != handledLayers.end()) {
                continue;
            }

            ngraph::opset1::Concat* concatChild = ngraph::as_type<ngraph::opset1::Concat>(&child);
            if (concatChild != nullptr) {
                if (!fillSubgraphForConcat(*concatChild, handledLayers)) {
                    return false;
                }
            } else {
                ngraph::opset1::FakeQuantize* fakeQuantizeChild = ngraph::as_type<ngraph::opset1::FakeQuantize>(&child);
                if (fakeQuantizeChild != nullptr) {
                //
                } else {
                    return false;
                    // TODO: possible not neccessary
                    //if (intermediateLayers.find(child->type) != intermediateLayers.end()) {
                    //    if (!fillSubgraphForIntermediate(child, handledLayers)) {
                    //        return false;
                    //    }
                    //}
                }
            }
        }
    }

    return true;
}

bool Subgraph::fill(ngraph::Node& layer, std::unordered_set<std::string>& handledLayers) {
    for (size_t index = 0; index < layer.get_input_size(); ++index) {
        ngraph::Node& parent = *layer.get_input_node_ptr(index);
        if (handledLayers.find(parent.get_friendly_name()) != handledLayers.end()) {
            continue;
        }

        ngraph::opset1::Concat* concatParent = ngraph::as_type<ngraph::opset1::Concat>(&parent);
        if (concatParent != nullptr) {
            if (!fillSubgraphForConcat(*concatParent, handledLayers)) {
                return false;
            }
        } else {
            ngraph::opset1::FakeQuantize* fakeQuantizeParent = ngraph::as_type<ngraph::opset1::FakeQuantize>(&parent);
            if (fakeQuantizeParent != nullptr) {
                if (!fillSubgraphForQuantization(*fakeQuantizeParent, handledLayers)) {
                    return false;
                }
            } else {
                // TODO: if we move Concat transformation from branch specific to original map then we can remove it
                // TODO: temporary commented: possible it doesn't need
                // if (intermediateLayers.find(parent->type) != intermediateLayers.end()) {
                //    if (!fillSubgraphForIntermediate(parent, handledLayers)) {
                //        return false;
                //    }

                // } else {
                //    return false;
                // }
            }
        }
    }

    for (size_t index = 0; index < layer.get_output_size(); ++index) {
        const auto childInputs = layer.get_output_target_inputs(index);
        for (const auto childInput : childInputs) {
            ngraph::Node& child = *childInput.get_node();

            if (handledLayers.find(child.get_friendly_name()) != handledLayers.end()) {
                continue;
            }

            ngraph::opset1::Concat* concatChild = ngraph::as_type<ngraph::opset1::Concat>(&child);
            if (concatChild != nullptr) {
                if (!fillSubgraphForConcat(*concatChild, handledLayers)) {
                    return false;
                }
            } else {
                ngraph::opset1::FakeQuantize* fakeQuantizeChild = ngraph::as_type<ngraph::opset1::FakeQuantize>(&child);
                if (fakeQuantizeChild != nullptr) {
                    //
                } else if (isIntermediate(child)) {
                    if (!fillSubgraphForIntermediate(child, handledLayers)) {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

bool Subgraph::fillSubgraphForIntermediate(ngraph::Node& intermediate, std::unordered_set<std::string>& handledLayers) {
    handledLayers.insert(intermediate.get_friendly_name());
    layers.emplace(intermediate.get_friendly_name(), &intermediate);

    return fill(intermediate, handledLayers);
}

bool Subgraph::empty() const {
    return quantizationLayers.empty();
}

bool Subgraph::fillSubgraphForConcat(ngraph::opset1::Concat& concat, std::unordered_set<std::string>& handledLayers) {
    concatLayers.push_back(&concat);
    handledLayers.insert(concat.get_friendly_name());
    layers.emplace(concat.get_friendly_name(), &concat);

    return fill(concat, handledLayers);
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
