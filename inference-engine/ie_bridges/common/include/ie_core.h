#ifndef INFERENCEENGINE_BRIDGE_IE_CORE_H
#define INFERENCEENGINE_BRIDGE_IE_CORE_H

#include <map>
#include <vector>

#include "ie_core.hpp"
#include "ie_network.hpp"
#include "ie_network.h"
#include "ie_exec_network.h"

namespace InferenceEngineBridge {
    struct IECore {
        explicit IECore(const std::string &xmlConfigFile = std::string());

        std::map <std::string, InferenceEngine::Version> getVersions(const std::string &deviceName);

        std::unique_ptr <InferenceEngineBridge::IEExecNetwork> loadNetwork(InferenceEngineBridge::IENetwork network,
                                                                           const std::string &deviceName,
                                                                           const std::map <std::string, std::string> &config,
                                                                           int &num_requests);

        std::map <std::string, std::string> queryNetwork(IENetwork network,
                                                         const std::string &deviceName,
                                                         const std::map <std::string, std::string> &config);

        void setConfig(const std::map <std::string, std::string> &config,
                       const std::string &deviceName = std::string());

        void registerPlugin(const std::string &pluginName, const std::string &deviceName);

        void unregisterPlugin(const std::string &deviceName);

        void registerPlugins(const std::string &xmlConfigFile);

        void addExtension(const std::string &ext_lib_path, const std::string &deviceName);

        std::vector <std::string> getAvailableDevices();

        InferenceEngine::Core actual;
    };
}

#endif //INFERENCEENGINE_BRIDGE_IE_CORE_H
