#ifndef INFERENCEENGINE_BRIDGE_HELPERS_H
#define INFERENCEENGINE_BRIDGE_HELPERS_H

#include <ie_iexecutable_network.hpp>

namespace InferenceEngineBridge {

    std::uint32_t getOptimalNumberOfRequests(const InferenceEngine::IExecutableNetwork::Ptr &actual);

    extern std::map<std::string, InferenceEngine::Precision> precision_map;

    extern std::map<std::string, InferenceEngine::Layout> layout_map;
}

#define IE_CHECK_CALL(expr) {                       \
    auto ret = (expr);                              \
    if (ret != InferenceEngine::StatusCode::OK) {   \
        THROW_IE_EXCEPTION << response.msg;         \
    }                                               \
}                                                   \



#endif //INFERENCEENGINE_BRIDGE_HELPERS_H
