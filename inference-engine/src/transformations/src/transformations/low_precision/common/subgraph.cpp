// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/low_precision/common/subgraph.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ngraph/rt_info.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "transformations/low_precision/quantization_details.hpp"
#include "transformations/low_precision/common/ie_lpt_exception.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

Subgraph::Subgraph(ngraph::pass::ILayerTransformationsManager* layerTransformationsManager) : layerTransformationsManager(layerTransformationsManager) {
}

bool Subgraph::fillSubgraphForQuantization(ngraph::opset1::FakeQuantize& fakeQuantize, std::unordered_set<std::string>& handledLayers) {
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
                    if (layerTransformationsManager->isPrecisionPreserved(child.shared_from_this())) {
                        if (!fillSubgraphForIntermediate(child, handledLayers)) {
                            return false;
                        }
                    }
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
                if (layerTransformationsManager->isPrecisionPreserved(parent.shared_from_this())) {
                    if (!fillSubgraphForIntermediate(parent, handledLayers)) {
                        return false;
                    }
                } else {
                    return false;
                }
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
                if (fakeQuantizeChild == nullptr) {
                    //
                } else if (layerTransformationsManager->isPrecisionPreserved(child.shared_from_this())) {
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
