// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_device.hpp>
#include <details/ie_exception.hpp>
#include "description_buffer.hpp"

using namespace InferenceEngine;

FindPluginResponse InferenceEngine::findPlugin(const FindPluginRequest& req) {
    switch (req.device) {
    case TargetDevice::eCPU:
        return { {
#ifdef ENABLE_MKL_DNN
                "MKLDNNPlugin",
#endif
#ifdef ENABLE_OPENVX_CVE
                "OpenVXPluginCVE",
#elif defined ENABLE_OPENVX
                "OpenVXPlugin",
#endif
            } };
    case TargetDevice::eGPU:
        return { {
#ifdef ENABLE_CLDNN
                "clDNNPlugin",
#endif
#ifdef ENABLE_OPENVX
                "OpenVXPlugin",
#endif
            } };
    case TargetDevice::eFPGA:
        return{ {
#ifdef ENABLE_DLIA
                "dliaPlugin",
#endif
#ifdef ENABLE_OPENVX
                "OpenVXPlugin",
#endif
            } };
    case TargetDevice::eMYRIAD:
        return{ {
#ifdef ENABLE_MYRIAD
                "myriadPlugin",
#endif
            } };
#ifdef ENABLE_HDDL
    case TargetDevice::eHDDL:
        return{ {
                "HDDLPlugin",
            } };
#endif
        case TargetDevice::eGNA:
            return{ {
#ifdef ENABLE_GNA
                        "GNAPlugin",
#endif
                    } };
    case TargetDevice::eHETERO:
        return{ {
                "HeteroPlugin",
            } };

    default:
        THROW_IE_EXCEPTION << "Cannot find plugin for device: " << getDeviceName(req.device);
    }
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
