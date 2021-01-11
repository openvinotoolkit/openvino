// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>
#include "nodes/list.hpp"
#include <ie_ngraph_utils.hpp>

#include <string>
#include <vector>
#include <cpp_interfaces/exception2status.hpp>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class ExtLayerBase: public ILayerExecImpl {
public:
    StatusCode getSupportedConfigurations(std::vector<LayerConfig>& conf, ResponseDesc *resp) noexcept override {
        if (!errorMsg.empty()) {
            if (resp) {
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
        conf = confs;
        return OK;
    }

    StatusCode init(LayerConfig& config, ResponseDesc *resp) noexcept override {
        for (auto& input : config.inConfs) {
            for (auto& offset : input.desc.getBlockingDesc().getOffsetPaddingToData()) {
                if (offset) {
                    return GENERAL_ERROR;
                }
            }
            if (input.desc.getBlockingDesc().getOffsetPadding()) {
                return GENERAL_ERROR;
            }
        }
        for (auto& output : config.outConfs) {
            for (auto& offset : output.desc.getBlockingDesc().getOffsetPaddingToData()) {
                if (offset) {
                    return GENERAL_ERROR;
                }
            }
            if (output.desc.getBlockingDesc().getOffsetPadding()) {
                return GENERAL_ERROR;
            }
        }
        return OK;
    }

protected:
    std::string errorMsg;
    std::vector<LayerConfig> confs;
};

template <class IMPL>
class ImplFactory : public ILayerImplFactory {
public:
    explicit ImplFactory(const std::shared_ptr<ngraph::Node>& op) : ngraphOp(op) {}

    // First implementation has more priority than next
    StatusCode getImplementations(std::vector<ILayerImpl::Ptr>& impls, ResponseDesc *resp) noexcept override {
        try {
            impls.push_back(ILayerImpl::Ptr(new IMPL(ngraphOp)));
        } catch (const InferenceEngine::Exception& ex) {
            return ex.getStatus();
        }
        return OK;
    }
protected:
    const std::shared_ptr<ngraph::Node> ngraphOp;
};

#define REG_FACTORY_FOR(__prim, __type) \
    void __prim ## __type(MKLDNNExtensions * extInstance) { \
        using namespace MKLDNNPlugin; \
        extInstance->layersFactory.registerNodeIfRequired(MKLDNNPlugin, __type, OV_PP_TOSTRING(__type), ImplFactory<__prim>); \
    }

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
