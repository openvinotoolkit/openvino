// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/layer_test_utils/external_network_tool.hpp"
#include "common_test_utils/file_utils.hpp"

using namespace LayerTestsUtils;

#ifdef _WIN32
# define getpid _getpid
#endif

#define path_delimiter "/"
#ifdef _WIN32
#define path_delimiter "\\"
#endif

ExternalNetworkTool *ExternalNetworkTool::p_instance = nullptr;
ExternalNetworkMode ExternalNetworkTool::mode = ExternalNetworkMode::DISABLED;
const char *ExternalNetworkTool::modelsPath = "";
ExternalNetworkToolDestroyer ExternalNetworkTool::destroyer;

ExternalNetworkToolDestroyer::~ExternalNetworkToolDestroyer() {
    delete p_instance;
}

void ExternalNetworkToolDestroyer::initialize(ExternalNetworkTool *p) {
    p_instance = p;
}

ExternalNetworkTool &ExternalNetworkTool::getInstance() {
    if (!p_instance) {
        p_instance = new ExternalNetworkTool();
        destroyer.initialize(p_instance);
    }
    return *p_instance;
}

void ExternalNetworkTool::dumpNetworkToFile(const std::shared_ptr<ngraph::Function>& network,
                                            std::string network_name) const {
    auto exportPathString = std::string(modelsPath);
    auto hashed_network_name = "network_" + generateHashName(network_name);

    std::string out_xml_path = exportPathString
                                + (exportPathString.empty() ? "" : path_delimiter)
                                + hashed_network_name + ".xml";
    std::string out_bin_path = exportPathString
                                + (exportPathString.empty() ? "" : path_delimiter)
                                + hashed_network_name + ".bin";

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Serialize>(out_xml_path, out_bin_path);
    manager.run_passes(network);
    printf("Network dumped to %s\n", out_xml_path.c_str());
}

InferenceEngine::CNNNetwork ExternalNetworkTool::loadNetworkFromFile(const std::shared_ptr<InferenceEngine::Core> core,
                                                                     std::string network_name) const {
    auto importPathString = std::string(modelsPath);
    auto hashed_network_name = "network_" + generateHashName(network_name);

    std::string out_xml_path = importPathString
                                + (importPathString.empty() ? "" : path_delimiter)
                                + hashed_network_name + ".xml";
    std::string out_bin_path = importPathString
                                + (importPathString.empty() ? "" : path_delimiter)
                                + hashed_network_name + ".bin";

    auto network = core->ReadNetwork(out_xml_path, out_bin_path);
    printf("Network loaded from %s\n", out_xml_path.c_str());
    return network;
}
