// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This header file contains aspects of working on different devices like CPU, GEN, FPGA, etc.
 * @file ie_device.hpp
 */
#pragma once

#include <string>
#include <vector>
#include <map>
#include <ie_api.h>
#include <ie_common.h>

namespace InferenceEngine {

/**
 * @deprecated Deprecated since the enum is not scalable for 3rd party plugins / devices. All devices are managed by InferenceEngine::Core
 * @enum TargetDevice
 * @brief Describes known device types
 */
enum class TargetDevice : uint8_t {
    eDefault = 0,
    eBalanced = 1,
    eCPU = 2,
    eGPU = 3,
    eFPGA = 4,
    eMYRIAD = 5,
    eHDDL = 6,
    eGNA = 7,
    eHETERO = 8,
    eMULTI = 10,
};

/**
 * @deprecated Deprecated since InferenceEngine::TargetDevice is deprecated
 * @brief Describes the relationship between the enumerator type and the actual device's name
 */
class INFERENCE_ENGINE_DEPRECATED TargetDeviceInfo {
    struct Info {
        TargetDevice device;
        std::string name;
        Info(TargetDevice device, std::string name) : device(device), name(name){}
    };

    static const std::vector<Info> & getAll() {
#define DECL_DEVICE(device_type) {TargetDevice::e##device_type, #device_type}

        static std::vector<Info> g_allDeviceInfos = {
            DECL_DEVICE(Default),
            DECL_DEVICE(Balanced),
            DECL_DEVICE(CPU),
            DECL_DEVICE(GPU),
            DECL_DEVICE(FPGA),
            DECL_DEVICE(MYRIAD),
            DECL_DEVICE(HDDL),
            DECL_DEVICE(GNA),
            DECL_DEVICE(HETERO),
            DECL_DEVICE(MULTI)
        };
#undef DECLARE
        return g_allDeviceInfos;
    }

 public:
    /**
     * @deprecated Deprecated since InferenceEngine::TargetDevice is deprecated
     * @brief Converts string representation of device to InferenceEngine::TargetDevice enum value
     * @param deviceName A string representation of a device name
     * @return An instance of InferenceEngine::TargetDevice
     */
    INFERENCE_ENGINE_DEPRECATED
    static TargetDevice fromStr(const std::string &deviceName) {
        static std::map<std::string, InferenceEngine::TargetDevice> deviceFromNameMap = {
            { "CPU", InferenceEngine::TargetDevice::eCPU },
            { "GPU", InferenceEngine::TargetDevice::eGPU },
            { "FPGA", InferenceEngine::TargetDevice::eFPGA },
            { "MYRIAD", InferenceEngine::TargetDevice::eMYRIAD },
            { "HDDL", InferenceEngine::TargetDevice::eHDDL },
            { "GNA", InferenceEngine::TargetDevice::eGNA },
            { "BALANCED", InferenceEngine::TargetDevice::eBalanced },
            { "HETERO", InferenceEngine::TargetDevice::eHETERO },
            { "MULTI", InferenceEngine::TargetDevice::eMULTI}
        };
        auto val = deviceFromNameMap.find(deviceName);
        return val != deviceFromNameMap.end() ? val->second : InferenceEngine::TargetDevice::eDefault;
    }

    /**
     * @deprecated Deprecated since InferenceEngine::TargetDevice is deprecated
     * @brief Converts an instance of InferenceEngine::TargetDevice to string representation
     * @param device Instance of InferenceEngine::TargetDevice
     * @return A c-string with the name
     */
    INFERENCE_ENGINE_DEPRECATED
    static const char * name(TargetDevice device) {
        IE_SUPPRESS_DEPRECATED_START
        auto res = std::find_if(getAll().cbegin(), getAll().cend(), [&](const Info & info){
            return device == info.device;
        });
        if (res == getAll().cend()) {
            return "Unknown device";
        }
        IE_SUPPRESS_DEPRECATED_END
        return res->name.c_str();
    }
};

/**
 * @deprecated Deprecated since InferenceEngine::TargetDevice is deprecated
 * @brief Returns the device name
 * @param device Instance of InferenceEngine::TargetDevice
 * @return A c-string with the name
 */
INFERENCE_ENGINE_DEPRECATED
inline const char *getDeviceName(TargetDevice device) {
    IE_SUPPRESS_DEPRECATED_START
    return TargetDeviceInfo::name(device);
    IE_SUPPRESS_DEPRECATED_END
}

/**
 * @deprecated Deprecated since InferenceEngine::TargetDevice is deprecated
 * @struct FindPluginRequest
 * @brief Defines a message that contains the InferenceEngine::TargetDevice object to find a plugin for
 */
struct INFERENCE_ENGINE_DEPRECATED FindPluginRequest {
    /**
     * @brief object of InferenceEngine::TargetDevice to find a plugin for
     */
    TargetDevice device;
};

/**
 * @deprecated Deprecated since InferenceEngine::TargetDevice is deprecated
 * @struct FindPluginResponse
 * @brief Defines a message that contains a list of appropriate plugin names
 */
struct INFERENCE_ENGINE_DEPRECATED FindPluginResponse {
    /**
     * @brief A list of appropriate plugin names
     */
    std::vector<std::string> names;
};

IE_SUPPRESS_DEPRECATED_START

/**
 * @deprecated Deprecated since InferenceEngine::TargetDevice is deprecated
 * @brief Finds an appropriate plugin for requested target device
 * @param req A requested target device
 * @return A response object
 */
FindPluginResponse findPlugin(const FindPluginRequest &req);

/**
 * @deprecated Deprecated since InferenceEngine::TargetDevice is deprecated
 * @brief Finds an appropriate plugin for requested target device
 * @param req A requested target device
 * @param result The results of the request
 * @param resp The response message description
 * @return A status code
 */
INFERENCE_ENGINE_API(StatusCode) findPlugin(const FindPluginRequest &req, FindPluginResponse &result,
                                            ResponseDesc *resp) noexcept;

IE_SUPPRESS_DEPRECATED_END

}  // namespace InferenceEngine
