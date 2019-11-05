#ifndef INFERENCEENGINE_BRIDGE_IE_EXEC_NETWORK_H
#define INFERENCEENGINE_BRIDGE_IE_EXEC_NETWORK_H

#include <ie_iexecutable_network.hpp>

#include "ie_network.h"
#include "infer_request_wrapper.h"

namespace InferenceEngineBridge {
    struct IEExecNetwork {

        IEExecNetwork(const std::string &name, std::size_t num_requests);

        InferenceEngineBridge::IENetwork GetExecGraphInfo();

        void infer();

        InferenceEngine::IExecutableNetwork::Ptr exec_network_ptr;
        std::vector<InferenceEngineBridge::InferRequestWrap> infer_requests;
        std::string name;

    };
}
#endif //INFERENCEENGINE_BRIDGE_IE_EXEC_NETWORK_H
