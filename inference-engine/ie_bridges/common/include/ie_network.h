#ifndef INFERENCEENGINE_BRIDGE_IE_NETWORK_H
#define INFERENCEENGINE_BRIDGE_IE_NETWORK_H

#include "ie_core.hpp"

namespace InferenceEngineBridge {
    class IENetwork {
    public:

        IENetwork(const std::string &model, const std::string &weights, bool ngraph_compatibility);

        IENetwork(const InferenceEngine::CNNNetwork &cnn_network);

        IENetwork() = default;

        void setBatch(const std::size_t &size);

        void addOutput(const std::string &out_layer, size_t port_id);

        const std::vector <std::pair<std::string, InferenceEngineBridge::IENetLayer>> getLayers();

        const std::map <std::string, InferenceEngineBridge::InputInfo> getInputs();

        const std::map <std::string, InferenceEngineBridge::OutputInfo> getOutputs();

        void reshape(const std::map <std::string, std::vector<size_t>> &input_shapes);

        void serialize(const std::string &path_to_xml, const std::string &path_to_bin);

        void setStats(const std::map <std::string, std::map<std::string, std::vector < float>>

        > &stats);

        const std::map <std::string, std::map<std::string, std::vector < float>>>

        getStats();

        void load_from_buffer(const char *xml, size_t xml_size, uint8_t *bin, size_t bin_size);

    private:
        InferenceEngine::CNNNetwork actual;
        std::string name;
        std::size_t batch_size;
        std::string precision;
    };
}

#endif //INFERENCEENGINE_BRIDGE_IE_CORE_H
