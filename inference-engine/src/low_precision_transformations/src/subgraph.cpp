// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ngraph/rt_info.hpp>
#include <ngraph/opsets/opset1.hpp>

#include <low_precision/common/subgraph.hpp>
#include "low_precision/quantization_details.hpp"
#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"


namespace ngraph {
namespace pass {
namespace low_precision {

bool isQuantizationPerChannel(const std::shared_ptr<ngraph::Node>& node) {
    if (node->outputs().size() > 1ul) {
        return false;
    }

    const auto inputs = ngraph::pass::low_precision::NetworkHelper::getInputs(node);
    for (const auto& input : inputs) {
        if (ngraph::is_type<opset1::Constant>(input.get_node())) {
            continue;
        }

        const Shape& in = input.get_shape();
        const Shape& out = node->output(0).get_shape();
        for (size_t i = 0; i < 2; ++i) {
            if ((i >= in.size()) || (i >= out.size())) {
                // all previous dimensions are equal
                return true;
            }
            if (in[i] != out[i]) {
                return false;
            }
        }
    }

    return true;
}

Subgraph::Subgraph(ngraph::pass::ILayerTransformationsManager* layerTransformationsManager) : layerTransformationsManager(layerTransformationsManager) {
}

bool Subgraph::fillSubgraphForQuantization(
    const std::shared_ptr<ngraph::opset1::FakeQuantize>& fakeQuantize,
    std::unordered_set<std::string>& handledLayers) {
    quantizationLayers.push_back(fakeQuantize);
    handledLayers.insert(fakeQuantize->get_friendly_name());
    layers.emplace(fakeQuantize->get_friendly_name(), fakeQuantize);

    for (size_t index = 0; index < fakeQuantize->get_output_size(); ++index) {
        const auto childInputs = fakeQuantize->get_output_target_inputs(index);
        for (const auto childInput : childInputs) {
            const std::shared_ptr<ngraph::Node> child = childInput.get_node()->shared_from_this();
            if (handledLayers.find(child->get_friendly_name()) != handledLayers.end()) {
                continue;
            }

            const std::shared_ptr<ngraph::opset1::Concat> concatChild = ngraph::as_type_ptr<ngraph::opset1::Concat>(child);
            if (concatChild != nullptr) {
                if (!fillSubgraphForConcat(concatChild, handledLayers)) {
                    return false;
                }
            } else {
                const std::shared_ptr<ngraph::opset1::FakeQuantize> fakeQuantizeChild = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(child);
                if (fakeQuantizeChild != nullptr) {
                    //
                } else {
                    if (layerTransformationsManager->isPrecisionPreserved(child) && isQuantizationPerChannel(child)) {
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

bool Subgraph::atLeastOneIsIntermediate(const std::shared_ptr<ngraph::Node>& node) const {
    for (size_t index = 0; index < node->get_output_size(); ++index) {
        const auto childInputs = node->get_output_target_inputs(index);
        for (const auto childInput : childInputs) {
            auto child = childInput.get_node()->shared_from_this();
            if (as_type_ptr<opset1::Concat>(child)) {
                return true;
            }

            if (!layerTransformationsManager->isPrecisionPreserved(child) || !isQuantizationPerChannel(child)) {
                // child branch is out of subgraph
                continue;
            }

            if (atLeastOneIsIntermediate(child)) {
                return true;
            }
        }
    }
    return false;
}

bool Subgraph::fill(const std::shared_ptr<ngraph::Node>& layer, std::unordered_set<std::string>& handledLayers) {
    // if at least one parent is handled incorrectly then subgraph is not in low precision
    for (size_t index = 0; index < layer->get_input_size(); ++index) {
        const std::shared_ptr<ngraph::Node> parent = layer->get_input_node_shared_ptr(index);
        if (handledLayers.find(parent->get_friendly_name()) != handledLayers.end()) {
            continue;
        }

        const std::shared_ptr<ngraph::opset1::Concat> concatParent = ngraph::as_type_ptr<ngraph::opset1::Concat>(parent);
        if (concatParent != nullptr) {
            if (!fillSubgraphForConcat(concatParent, handledLayers)) {
                return false;
            }
        } else {
            const std::shared_ptr<ngraph::opset1::FakeQuantize> fakeQuantizeParent = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(parent);
            if (fakeQuantizeParent != nullptr) {
                if (!fillSubgraphForQuantization(fakeQuantizeParent, handledLayers)) {
                    //
                }
            } else {
                const std::shared_ptr<ngraph::opset1::Constant> constant = ngraph::as_type_ptr<ngraph::opset1::Constant>(parent);
                if (constant != nullptr) {
                    //
                } else {
                    if (layerTransformationsManager->isPrecisionPreserved(parent) && isQuantizationPerChannel(parent)) {
                        if (!fillSubgraphForIntermediate(parent, handledLayers)) {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
            }
        }
    }

    // TODO: if at least one child was handled correctly then subgraph is low precision
    for (size_t index = 0; index < layer->get_output_size(); ++index) {
        const auto childInputs = layer->get_output_target_inputs(index);
        for (const auto childInput : childInputs) {
            const std::shared_ptr<ngraph::Node> child = childInput.get_node()->shared_from_this();

            if (handledLayers.find(child->get_friendly_name()) != handledLayers.end()) {
                continue;
            }

            const std::shared_ptr<ngraph::opset1::Concat> concatChild = ngraph::as_type_ptr<ngraph::opset1::Concat>(child);
            if (concatChild != nullptr) {
                if (!fillSubgraphForConcat(concatChild, handledLayers)) {
                    return false;
                }
            } else {
                // check if children branches between Concat operations
                if (!atLeastOneIsIntermediate(child)) {
                    continue;
                }

                const std::shared_ptr<ngraph::opset1::FakeQuantize> fakeQuantizeChild = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(child);
                if (fakeQuantizeChild != nullptr) {
                    //
                } else if (layerTransformationsManager->isPrecisionPreserved(child) && isQuantizationPerChannel(child)) {
                    if (!fillSubgraphForIntermediate(child, handledLayers)) {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

bool Subgraph::fillSubgraphForIntermediate(const std::shared_ptr<ngraph::Node>& intermediate, std::unordered_set<std::string>& handledLayers) {
    handledLayers.insert(intermediate->get_friendly_name());
    layers.emplace(intermediate->get_friendly_name(), intermediate);

    return fill(intermediate, handledLayers);
}

bool Subgraph::empty() const {
    return quantizationLayers.empty();
}

bool Subgraph::fillSubgraphForConcat(const std::shared_ptr<ngraph::opset1::Concat>& concat, std::unordered_set<std::string>& handledLayers) {
    concatLayers.push_back(concat);
    handledLayers.insert(concat->get_friendly_name());
    layers.emplace(concat->get_friendly_name(), concat);

    std::shared_ptr<ngraph::Node> node = concat;
    return fill(node, handledLayers);
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
