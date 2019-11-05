#ifndef INFERENCEENGINE_HELPERS_H
#define INFERENCEENGINE_HELPERS_H

uint32_t getOptimalNumberOfRequests(const InferenceEngine::IExecutableNetwork::Ptr actual);

#define stringify(name) # name
#define IE_CHECK_CALL(expr) {                       \
    auto ret = (expr);                              \
    if (ret != InferenceEngine::StatusCode::OK) {   \
        THROW_IE_EXCEPTION << response.msg;         \
    }                                               \
}                                                   \


#endif //INFERENCEENGINE_HELPERS_H
