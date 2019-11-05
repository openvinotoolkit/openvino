#include "ie_net_layer.h"
#include "helpers.h"

void InferenceEngineBridge::IENetLayer::setAffinity(const std::string &target_affinity) {
    layer_ptr->affinity = target_affinity;
}

void InferenceEngineBridge::IENetLayer::setParams(const std::map<std::string, std::string> &params_map) {
    layer_ptr->params = params_map;
}

std::map<std::string, InferenceEngine::Blob::Ptr> InferenceEngineBridge::IENetLayer::getWeights() {
    auto w_layer = std::dynamic_pointer_cast<InferenceEngine::WeightableLayer>(layer_ptr);
    // IF current layer is weightable gather weights and biases from casted WeightableLayer and all other blobs
    // considered as custom and gathered from blobs field pf CNNLayer.
    std::map<std::string, InferenceEngine::Blob::Ptr> weights;
    if (w_layer != nullptr) {
        if (w_layer->_weights != nullptr) {
            weights["weights"] = w_layer->_weights;
        }
        if (w_layer->_biases != nullptr) {
            weights["biases"] = w_layer->_biases;
        }
        for (auto it : w_layer->blobs) {
            if (it.first == "weights" || it.first == "biases") {
                continue;
            }
            weights[it.first] = it.second;
        }
    } else {
        // Otherwise all layer's blobs are considered as custom and gathered from CNNLayer
        std::map<std::string, InferenceEngine::Blob::Ptr> map_placeholder;
        weights = map_placeholder;  // If layer has no blobs it should not be missed from weights map
        for (auto it : layer_ptr->blobs) {
            weights[it.first] = it.second;
        }
    }
    return weights;
}

void InferenceEngineBridge::IENetLayer::setPrecision(std::string precision) {
    layer_ptr->precision = precision_map[precision];
}
