// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include <ie_device.hpp>
#include <details/ie_exception.hpp>
#include "description_buffer.hpp"

using namespace InferenceEngine;

FindPluginResponse InferenceEngine::findPlugin(const FindPluginRequest& req) {
    std::vector<std::string> pluginVec;
    switch (req.device) {
        case TargetDevice::eCPU:
#ifdef ENABLE_MKL_DNN
            pluginVec.push_back("MKLDNNPlugin");
#endif
#ifdef ENABLE_OPENVX_CVE
            pluginVec.push_back("OpenVXPluginCVE");
#elif defined ENABLE_OPENVX
            pluginVec.push_back("OpenVXPlugin");
#endif
            break;
        case TargetDevice::eGPU:
#ifdef ENABLE_CLDNN
            pluginVec.push_back("clDNNPlugin");
#endif
#ifdef ENABLE_OPENVX
            pluginVec.push_back("OpenVXPlugin");
#endif
            break;
        case TargetDevice::eFPGA:
#ifdef ENABLE_DLIA
            pluginVec.push_back("dliaPlugin");
#endif
#ifdef ENABLE_OPENVX
            pluginVec.push_back("OpenVXPlugin");
#endif
            break;
        case TargetDevice::eMYRIAD:
#ifdef ENABLE_MYRIAD
            pluginVec.push_back("myriadPlugin");
#endif
            break;
        case TargetDevice::eGNA:
#ifdef ENABLE_GNA
            pluginVec.push_back("GNAPlugin");
#endif
            break;
        case TargetDevice::eHETERO:
            pluginVec.push_back("HeteroPlugin");
            break;
        default:
            THROW_IE_EXCEPTION << "Cannot find plugin for device: " << getDeviceName(req.device);
    }
    std::for_each(pluginVec.begin(), pluginVec.end(), [](std::string &name){ name = name + IE_BUILD_POSTFIX;});
    return {pluginVec};
}

INFERENCE_ENGINE_API(StatusCode) InferenceEngine::findPlugin(
        const FindPluginRequest& req, FindPluginResponse& result, ResponseDesc* resp) noexcept {
    try {
        result = findPlugin(req);
    }
    catch (const std::exception& e) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << e.what();
    }
    return OK;
}
