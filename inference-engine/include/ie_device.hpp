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
    eKMB = 9,
};

/**
 * @brief Describes the relationship between the enumerator type and the actual device's name
 */
class TargetDeviceInfo {
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
            DECL_DEVICE(KMB)
        };
#undef DECLARE
        return g_allDeviceInfos;
    }

 public:
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
            { "KMB", InferenceEngine::TargetDevice::eKMB }
        };
        auto val = deviceFromNameMap.find(deviceName);
        return val != deviceFromNameMap.end() ? val->second : InferenceEngine::TargetDevice::eDefault;
    }

    static const char * name(TargetDevice device) {
        auto res = std::find_if(getAll().cbegin(), getAll().cend(), [&](const Info & info){
            return device == info.device;
        });
        if (res == getAll().cend()) {
            return "Unknown device";
        }
        return res->name.c_str();
    }
};

/**
 * @brief Returns the device name
 * @param device Instance of TargetDevice
 * @return A c-string with the name
 */
inline const char *getDeviceName(TargetDevice device) {
    return TargetDeviceInfo::name(device);
}

/**
 * @struct FindPluginRequest
 * @brief Defines a message that contains the TargetDevice object to find a plugin for
 */
struct FindPluginRequest {
    /**
     * @brief object of TargetDevice to find a plugin for
     */
    TargetDevice device;
};

/**
 * @struct FindPluginResponse
 * @brief Defines a message that contains a list of appropriate plugin names
 */
struct FindPluginResponse {
    /**
     * @brief A list of appropriate plugin names
     */
    std::vector<std::string> names;
};

/**
 * @brief Finds an appropriate plugin for requested target device
 * @param req A requested target device
 * @param result The results of the request
 * @param resp The response message description
 * @return A response message
 */
FindPluginResponse findPlugin(const FindPluginRequest &req);

INFERENCE_ENGINE_API(StatusCode) findPlugin(const FindPluginRequest &req, FindPluginResponse &result,
                                            ResponseDesc *resp) noexcept;
}  // namespace InferenceEngine
