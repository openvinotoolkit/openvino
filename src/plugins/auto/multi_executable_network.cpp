// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "multi_executable_network.hpp"
#include "plugin.hpp"
// ------------------------------MultiExecutableNetwork----------------------------
namespace MultiDevicePlugin {
MultiExecutableNetwork::MultiExecutableNetwork(MultiScheduleContext::Ptr& context, const MultiSchedule::Ptr& schedule)
    : ExecutableNetwork(schedule, context),
      _multiSContext(context) {
}

MultiExecutableNetwork::~MultiExecutableNetwork() {}
std::shared_ptr<IE::RemoteContext> MultiExecutableNetwork::GetContext() const {
    auto devices = [&] {
        std::lock_guard<std::mutex> lock(_multiSContext->_mutex);
        return _multiSContext->_devicePriorities;
    }();
    std::string devices_names;
    for (auto&& device : devices) {
        devices_names += device.deviceName + " ";
        const auto& n  = _multiSContext->_networksPerDevice.at(device.deviceName);
        try {
            return n->GetContext();
        } catch (const IE::NotImplemented&) {}
    }
    IE_THROW(NotImplemented) <<
        "None of the devices in the MULTI device has an associated remote context."
        << " Current list of devices allowed via the DEVICE_PRIORITIES config: " <<
        devices_names;
}


void MultiExecutableNetwork::SetConfig(const
    std::map<std::string, IE::Parameter>& config) {
    auto priorities = config.find(ov::device::priorities.name());
    if (priorities == config.end() || config.size() > 1) {
        IE_THROW() <<
            "The only config supported for the Network's SetConfig is MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES";
    } else {
        auto multiPlugin = std::dynamic_pointer_cast<MultiDeviceInferencePlugin>
            (this->_plugin);
        assert(multiPlugin != nullptr);
        auto metaDevices = multiPlugin->ParseMetaDevices(
                priorities->second.as<std::string>(), {});
        if (std::any_of(metaDevices.begin(),
        metaDevices.end(), [](const DeviceInformation & kvp) {
        return kvp.numRequestsPerDevices != -1;
    })) {
            IE_THROW() << "You can only change device priorities but not number of requests"
                << " with the Network's SetConfig(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES!";
        }
        {
            std::lock_guard<std::mutex> lock{_multiSContext->_mutex};
            for (auto&& device : metaDevices) {
                if (_multiSContext->_networksPerDevice.find(device.deviceName) ==
                    _multiSContext->_networksPerDevice.end()) {
                    IE_THROW(NotFound) <<
                        "You can only change device priorities but not add new devices with"
                        << " the Network's SetConfig(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES. "
                        << device.deviceName << " device was not in the original device list!";
                }
            }
            _multiSContext->_devicePriorities = metaDevices;
            // update value in config
            _multiSContext->_config[IE::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES]
                = priorities->second;
        }
    }
}

IE::Parameter MultiExecutableNetwork::GetConfig(const std::string& name) const {
    {
        auto it = _multiSContext->_config.find(name);
        if (it != _multiSContext->_config.end()) {
            return it->second;
        }
    }
    // find config key among networks config keys
    for (const auto& desc : _multiSContext->_networksPerDevice) {
        const auto& execNetwork = desc.second;
        auto param = execNetwork->GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        for (auto&& configKey : param.as<std::vector<std::string>>()) {
            if (configKey == name) {
                return execNetwork->GetConfig(configKey);
            }
        }
        IE_THROW() << "Unsupported ExecutableNetwork config key: " << name;
    }
    IE_THROW(NotFound) << name << " not found in the ExecutableNetwork config";
}

IE::Parameter MultiExecutableNetwork::GetMetric(const std::string& name) const {
    if (name == ov::supported_properties) {
        return decltype(ov::supported_properties)::value_type {
            // Metrics
            ov::PropertyName{ov::supported_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::model_name.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::optimal_number_of_infer_requests.name(), ov::PropertyMutability::RO},

            // Configs
            // device priority can be changed on-the-fly in MULTI
            ov::PropertyName{ov::device::priorities.name(), ov::PropertyMutability::RW},
            ov::PropertyName{ov::execution_devices.name(), ov::PropertyMutability::RO}
        };
    } else if (name == ov::optimal_number_of_infer_requests) {
        unsigned int res = 0u;
        for (auto n : _multiSContext->_networksPerDevice) {
            try {
                res += n.second->GetMetric(METRIC_KEY(
                            OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
            } catch (const IE::Exception& iie) {
                IE_THROW()
                        << "Every device used with the Multi-Device should "
                            << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                            << "Failed to query the metric for the " << n.first << " with error:" <<
                            iie.what();
            }
        }
        return decltype(ov::optimal_number_of_infer_requests)::value_type {res};
    } else if (name == ov::model_name) {
        auto it = _multiSContext->_networksPerDevice.begin();
        IE_ASSERT(it != _multiSContext->_networksPerDevice.end());
        return decltype(ov::model_name)::value_type {it->second->GetMetric(METRIC_KEY(NETWORK_NAME)).as<std::string>()};
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, {
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(NETWORK_NAME),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS)
        });
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys = {IE::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES};
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == ov::execution_devices) {
        std::vector<std::string> exeDevices = {};
        std::vector<std::string> exeDevicesUnsort = {};
        for (const auto & n : _multiSContext->_networksPerDevice) {
            exeDevicesUnsort.push_back(n.first);
        }
        for (const auto & n : _multiSContext->_devicePriorities) {
            if (std::find(exeDevicesUnsort.begin(), exeDevicesUnsort.end(), n.deviceName) != exeDevicesUnsort.end()) {
                exeDevices.push_back(n.deviceName);
            }
        }
        return decltype(ov::execution_devices)::value_type {exeDevices};
    } else {
        IE_THROW() << "Unsupported ExecutableNetwork metric key: " << name;
    }
}
}  // namespace MultiDevicePlugin
