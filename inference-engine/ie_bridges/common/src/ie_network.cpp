#include <cpp/ie_cnn_net_reader.h>

#include "ie_network.h"
#include "ie_core.h"
#include "../../../src/inference_engine/ie_ir_reader.hpp"

InferenceEngineBridge::IENetwork::IENetwork(const InferenceEngine::CNNNetwork& cnn_network) : actual(cnn_network) {
    name = actual.getName();
    batch_size = actual.getBatchSize();
    precision = actual.getPrecision().name();
}

InferenceEngineBridge::IENetwork::IENetwork(const std::string &model, const std::string &weights, const bool &ngraph_compatibility = false) {
    if (ngraph_compatibility){
        InferenceEngine::IRReader ir_reader;
        auto ngraph_function = ir_reader.read(model, weights);
        actual = InferenceEngine::CNNNetwork(InferenceEngine::convertFunctionToICNNNetwork(ngraph_function));
    } else {
        InferenceEngine::CNNNetReader net_reader;
        net_reader.ReadNetwork(model);
        net_reader.ReadWeights(weights);
        actual = net_reader.getNetwork();
    }
    name = actual.getName();
    batch_size = actual.getBatchSize();
    precision = actual.getPrecision().name();
}


void InferenceEngineBridge::IENetwork::load_from_buffer(const char *xml, std::size_t xml_size, uint8_t *bin, const std::size_t &bin_size) {
    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(xml, xml_size);
    InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::U8, {bin_size}, InferenceEngine::Layout::C);
    auto weights_blob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc, bin, bin_size);
    net_reader.SetWeights(weights_blob);
    name = net_reader.getName();
    actual = net_reader.getNetwork();
    batch_size = actual.getBatchSize();
    precision = actual.getPrecision().name();
}

void InferenceEngineBridge::IENetwork::serialize(const std::string &path_to_xml, const std::string &path_to_bin) {
    actual.serialize(path_to_xml, path_to_bin);
}
