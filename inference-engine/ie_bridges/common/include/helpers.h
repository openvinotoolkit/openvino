#ifndef INFERENCEENGINE_BRIDGE_HELPERS_H
#define INFERENCEENGINE_BRIDGE_HELPERS_H

#include <ie_iexecutable_network.hpp>

namespace InferenceEngineBridge {

    extern std::map<std::string, InferenceEngine::Precision> precision_map;

    extern std::map<std::string, InferenceEngine::Layout> layout_map;

    std::uint32_t getOptimalNumberOfRequests(const InferenceEngine::IExecutableNetwork::Ptr &actual);

    void latency_callback(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code);

    template<class T>
    T *get_buffer(InferenceEngine::Blob &blob) {
        return blob.buffer().as<T *>();
    }

    template<class T, class... Args>
    std::unique_ptr<T> make_unique(Args &&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }

    std::string get_version();

}

#define IE_CHECK_CALL(expr) {                       \
    auto ret = (expr);                              \
    if (ret != InferenceEngine::StatusCode::OK) {   \
        THROW_IE_EXCEPTION << response.msg;         \
    }                                               \
}                                                   \



#endif //INFERENCEENGINE_BRIDGE_HELPERS_H
