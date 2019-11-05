#ifndef INFERENCEENGINE_BRIDGE_IE_NET_LAYER_H
#define INFERENCEENGINE_BRIDGE_IE_NET_LAYER_H

#include "inference_engine.hpp"

namespace InferenceEngineBridge {
    struct IENetLayer {
        InferenceEngine::CNNLayerPtr layer_ptr;
        InferenceEngine::CNNNetwork network_ptr;
        std::string name;
        std::string type;
        std::string precision;
        std::string shape;
        std::string layout;
        std::vector<std::string> children;
        std::vector<std::string> parents;
        std::string affinity;
        std::map<std::string, std::string> params;

        void setAffinity(const std::string &target_affinity);

        void setParams(const std::map<std::string, std::string> &params_map);

        std::map<std::string, InferenceEngine::Blob::Ptr> getWeights();

        void setPrecision(std::string precision);
    };
}

#endif //INFERENCEENGINE_BRIDGE_IE_NET_LAYER_H
