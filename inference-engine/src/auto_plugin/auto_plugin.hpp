// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>
#include <string>
#include <unordered_set>
#include <type_traits>

#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include "auto_exec_network.hpp"

namespace AutoPlugin {
namespace IE = InferenceEngine;
using ConfigType = std::map<std::string, std::string>;

class AutoInferencePlugin : public IE::InferencePluginInternal {
public:
    AutoInferencePlugin();
    ~AutoInferencePlugin() {
        // will discuss cancelable LoadNEtwork in the Arch Forum, but now have to wait
        for (auto& t : async_load_threads)
            t.join();
    }
    IE::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const IE::CNNNetwork& network, const ConfigType& config) override;
    IE::IExecutableNetworkInternal::Ptr LoadNetwork(const std::string& fileName, const ConfigType& config) override;
    IE::QueryNetworkResult QueryNetwork(const IE::CNNNetwork& network, const ConfigType& config) const override;
    IE::Parameter GetMetric(const std::string& name, const std::map<std::string, IE::Parameter>& options) const override;
    IE::Parameter GetConfig(const std::string& name, const std::map<std::string, IE::Parameter> & options) const override;
    void SetConfig(const ConfigType& config) override;

private:
    std::vector<AutoPlugin::DeviceInformation> GetDeviceChoice(const ConfigType&  config) const;
    std::vector<std::string> GetOptimizationCapabilities() const;
    DeviceInformation SelectDevice(const std::vector<DeviceInformation>& metaDevices, const std::string& networkPrecision = METRIC_VALUE(FP32));
    ConfigType GetSupportedConfig(const ConfigType& config, const AutoPlugin::DeviceName & deviceName) const;
    static ConfigType mergeConfigs(ConfigType config, const ConfigType& local);
    std::vector<std::thread> async_load_threads;
    template <typename T>
    std::shared_ptr<AutoExecutableNetwork> LoadNetworkImpl(const T &param, const ConfigType &config, const std::string &networkPrecision = METRIC_VALUE(FP32)) {
        if (GetCore() == nullptr) {
            IE_THROW() << "Please, work with AUTO device via InferencEngine::Core object";
        }
        auto fullConfig = mergeConfigs(_config, config);
        auto metaDevices = GetDeviceChoice(fullConfig);
        NetworkPromiseSharedPtr promiseFirstDevice = std::make_shared<NetworkPromise>();
        NetworkPromiseSharedPtr promiseActualDevice = std::make_shared<NetworkPromise>();
        auto LoadNetworkAsync = [this, param, promiseFirstDevice, promiseActualDevice](bool bIsAccel, const DeviceInformation device) {
            std::cout << "!!! DEBUG: Starting Async loading to the " << device.deviceName<<  " !!!" << std::endl;
            auto executableNetwork = GetCore()->LoadNetwork(param, device.deviceName, device.config);
            try {
                promiseFirstDevice->set_value(executableNetwork);
                std::cout << "!!! DEBUG: " << device.deviceName <<  " was loaded first !!!" << std::endl;
            }
            catch (const std::future_error &e) {
//                if (e.code() == std::future_errc::promise_already_satisfied) {
//                    std::cout << "!!! DEBUG: " << (bIsAccel? "Accelerator" : "CPU") <<
//                        " was already loaded to the promiseFirstDevice !!!" << std::endl;
//                }
            }
            if (bIsAccel) {
                promiseActualDevice->set_value(executableNetwork);
                std::cout << "!!! DEBUG: Accelerator" << device.deviceName <<  " was loaded !!!" << std::endl;
            }
        };

        const auto accelerator = SelectDevice(metaDevices, networkPrecision);
        // fixme (use CPU config, etc):
        const DeviceInformation CPU = {"CPU", {} };
        // start loading of the netwrok to the accel
        bool isAccelerator = accelerator.deviceName != "CPU";
        // FIXME: change to the default task-executor as in MULTI (but add an async Run() to that)
        async_load_threads.push_back(std::thread([=] {LoadNetworkAsync(isAccelerator, accelerator);}));
        // loading to the CPU in parallel, if it is not already an accelrator
        if (isAccelerator)
            async_load_threads.push_back(std::thread([=] { LoadNetworkAsync(false, CPU); }));
        else
            promiseActualDevice->set_value({});

        // FIXME: revert the exception handling logic back to gracefully handle LoadNetwork failures
        // DeviceInformation selectedDevice;
        // IE::SoExecutableNetworkInternal executableNetwork;
//            while (!metaDevices.empty()) {
//            selectedDevice = SelectDevice(metaDevices, networkPrecision);
//              try {
//                executableNetwork = GetCore()->LoadNetwork(param, selectedDevice.deviceName,
//                                                                    selectedDevice.config);
//                break;
//            } catch (...) {
//                auto eraseDevice = std::find_if(metaDevices.begin(), metaDevices.end(),
//                    [=](const DeviceInformation& d)->bool{return d.deviceName == selectedDevice.deviceName;});
//                if (eraseDevice == metaDevices.end()) {
//                    IE_THROW() << "Didn't find the selected device name";
//                }
//                metaDevices.erase(eraseDevice);
//                executableNetwork = {};
//            }
//        }
//        if (!executableNetwork) {
//            IE_THROW() << "Failed to load network by AUTO plugin";
//        }
        // AutoExecutableNetwork constructor blocks until any network is ready
        auto impl = std::make_shared<AutoExecutableNetwork>(promiseFirstDevice, promiseActualDevice);

//        if (std::is_same<std::string, T>::value) {
//            SetExeNetworkInfo(impl, executableNetwork->GetInputsInfo(),
//                                    executableNetwork->GetOutputsInfo());
//        }

        return impl;
    }
};

}  // namespace AutoPlugin
