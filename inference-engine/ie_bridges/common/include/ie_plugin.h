#ifndef INFERENCEENGINE_BRIDGE_IE_PLUGIN_H
#define INFERENCEENGINE_BRIDGE_IE_PLUGIN_H

#include "ie_exec_network.h"
#include "ie_network.h"

namespace InferenceEngineBridge {
    struct IEPlugin {
/*
std::unique_ptr <InferenceEngineBridge::IEExecNetwork> load(const InferenceEngineBridge::IENetwork &net,
                                                            int num_requests,
                                                            const std::map <std::string, std::string> &config);
*/
        std::string device_name;
        std::string version;

        void setConfig(const std::map<std::string, std::string> &);

        void addCpuExtension(const std::string &extension_path);

        void setInitialAffinity(const InferenceEngineBridge::IENetwork &net);

        IEPlugin(const std::string &device, const std::vector<std::string> &plugin_dirs);

        IEPlugin() = default;

        std::set<std::string> queryNetwork(const InferenceEngineBridge::IENetwork &net);

        InferenceEngine::InferencePlugin actual;
    };
}
#endif //INFERENCEENGINE_BRIDGE_IE_PLUGIN_H
