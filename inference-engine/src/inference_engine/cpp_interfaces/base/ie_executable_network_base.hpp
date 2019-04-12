// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief inference engine executanle network API wrapper, to be used by particular implementors
 * \file ie_executable_network_base.hpp
 */

#pragma once

#include <vector>
#include <memory>
#include <map>
#include <string>
#include <cpp_interfaces/interface/ie_imemory_state_internal.hpp>
#include <cpp_interfaces/base/ie_memory_state_base.hpp>
#include "cpp_interfaces/exception2status.hpp"

namespace InferenceEngine {

/**
 * @brief cpp interface for executable network, to avoid dll boundaries and simplify internal development
 * @tparam T Minimal CPP implementation of IExecutableNetwork (e.g. ExecutableNetworkInternal)
 */
template<class T>
class ExecutableNetworkBase : public IExecutableNetwork {
    std::shared_ptr<T> _impl;

public:
    typedef std::shared_ptr<ExecutableNetworkBase<T>> Ptr;

    explicit ExecutableNetworkBase(std::shared_ptr<T> impl) {
        if (impl.get() == nullptr) {
            THROW_IE_EXCEPTION << "implementation not defined";
        }
        _impl = impl;
    }

    StatusCode GetOutputsInfo(ConstOutputsDataMap &outs, ResponseDesc *resp) const noexcept override {
        TO_STATUS(outs = _impl->GetOutputsInfo());
    }

    StatusCode GetInputsInfo(ConstInputsDataMap &inputs, ResponseDesc *resp) const noexcept override {
        TO_STATUS(inputs = _impl->GetInputsInfo());
    }

    StatusCode CreateInferRequest(IInferRequest::Ptr &req, ResponseDesc *resp) noexcept override {
        TO_STATUS(_impl->CreateInferRequest(req));
    }

    StatusCode Export(const std::string &modelFileName, ResponseDesc *resp) noexcept override {
        TO_STATUS(_impl->Export(modelFileName));
    }

    StatusCode GetMappedTopology(std::map<std::string, std::vector<PrimitiveInfo::Ptr>> &deployedTopology,
                                 ResponseDesc *resp) noexcept override {
        TO_STATUS(_impl->GetMappedTopology(deployedTopology));
    }

    StatusCode GetExecGraphInfo(ICNNNetwork::Ptr &graphPtr, ResponseDesc *resp) noexcept override {
        TO_STATUS(_impl->GetExecGraphInfo(graphPtr));
    }

    StatusCode  QueryState(IMemoryState::Ptr & pState, size_t idx
        , ResponseDesc *resp) noexcept override {
        try {
            auto v = _impl->QueryState();
            if (idx >= v.size()) {
                return OUT_OF_BOUNDS;
            }
            pState = std::make_shared<MemoryStateBase<IMemoryStateInternal>>(v[idx]);
            return OK;
        } catch (const std::exception & ex) {\
            return InferenceEngine::DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();\
        } catch (...) {\
            return InferenceEngine::DescriptionBuffer(UNEXPECTED);\
        }
    }

    void Release() noexcept override {
        delete this;
    }

    // Need for unit tests only - TODO: unit tests should test using public API, non having details
    const std::shared_ptr<T> getImpl() const {
        return _impl;
    }

private:
    ~ExecutableNetworkBase() = default;
};

template <class T>
inline typename ExecutableNetworkBase<T>::Ptr make_executable_network(std::shared_ptr<T> impl) {
    typename ExecutableNetworkBase<T>::Ptr net(new ExecutableNetworkBase<T>(impl), [](IExecutableNetwork * p) {
        p->Release();
    });
    return net;
}

}  // namespace InferenceEngine
