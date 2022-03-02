// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/layer_test_utils/external_optimization_util.hpp"
#include "common_test_utils/file_utils.hpp"
#include "openvino/frontend/manager.hpp"

using namespace LayerTestsUtils;

ExternalOptimizationMode ExternalOptimizationUtil::mode = ExternalOptimizationMode::DISABLED;

std::string &ExternalOptimizationUtil::modelsPath() {
    static std::string models_path { "" };
    return models_path;
}

void ExternalOptimizationUtil::setModelsPath(std::string &val) {
    std::string &ref = modelsPath();
    ref = val;
}

bool ExternalOptimizationUtil::toDumpModel() {
    return isMode(ExtOptMode::DUMP) || isMode(ExtOptMode::DUMP_MODELS_ONLY) || isMode(ExtOptMode::DUMP_ALL);
}

bool ExternalOptimizationUtil::toDumpInput() {
    return isMode(ExtOptMode::DUMP) || isMode(ExtOptMode::DUMP_INPUTS_ONLY) || isMode(ExtOptMode::DUMP_ALL);
}

bool ExternalOptimizationUtil::toLoad() {
    return isMode(ExtOptMode::LOAD);
}

void ExternalOptimizationUtil::writeToHashMap(const std::string &network_name,
                                         const std::string &shorted_name) {
    std::ofstream hash_map_file;
    std::string file_path;
    const auto models_path = getModelsPath();
    if (!models_path.empty()) {
        file_path += models_path + path_delimiter + "hashMap.txt";
    } else {
        file_path += "hashMap.txt";
    }

    hash_map_file.open(file_path,  std::ios::out | std::ios::app);
    hash_map_file << "{\n";
    hash_map_file << "  \"test\": \"" << network_name << "\",\n";
    hash_map_file << "  \"short\": \"" << shorted_name << "\",\n";
    hash_map_file << "},\n";
    hash_map_file.close();
}

template <typename T>
std::vector<std::shared_ptr<ov::Node>> ExternalOptimizationUtil::topological_name_sort(T root_nodes) {
    std::vector<std::shared_ptr<ov::Node>> results = ngraph::topological_sort<T>(root_nodes);

    auto node_comaparator = [](std::shared_ptr<ov::Node> node_left,
                               std::shared_ptr<ov::Node> node_right) {
        std::string name_left = node_left->get_friendly_name();
        std::string name_right = node_right->get_friendly_name();
        bool res = std::strcmp(name_left.c_str(), name_right.c_str());
        return res;
    };

    std::sort(results.begin(), results.end(), node_comaparator);
    return results;
}

std::string ExternalOptimizationUtil::replaceInName(const std::string &network_name, const std::map<std::string, std::string> replace_map) {
    std::string new_network_name { network_name };

    for (auto &pair : replace_map) {
        auto &old_str = pair.first;
        auto &new_str = pair.second;
        auto index = new_network_name.find(old_str);
        while (index != std::string::npos) {
            new_network_name.replace(index,  old_str.length(), new_str);
            index = new_network_name.find(old_str, index + new_str.length());
        }
    }

    return new_network_name;
}

std::string ExternalOptimizationUtil::eraseInName(const std::string &network_name, const std::vector<std::string> patterns) {
    std::string new_network_name { network_name };

    for (auto &pattern : patterns) {
        auto index = new_network_name.find(pattern);
        while (index != std::string::npos) {
            new_network_name.erase(index,  pattern.length());
            index = new_network_name.find(pattern, index);
        }
    }

    return new_network_name;
}

std::string ExternalOptimizationUtil::eraseRepeatedInName(const std::string &network_name, const std::vector<char> target_symbols) {
    if (!network_name.length()) {
        return network_name;
    }
    char *buffer = new char[network_name.length()]();
    const char *data = network_name.c_str();
    size_t new_name_size = 1;
    char last_symbol = data[0];
    buffer[0] = data[0];

    auto check_symbol = [&target_symbols](const char symbol) {
        for (auto c : target_symbols) {
            if (c == symbol)
                return true;
        }
        return false;
    };

    for (size_t i = 1; i < network_name.length(); i++) {
        if (last_symbol == data[i] && check_symbol(last_symbol))
            continue;
        last_symbol = data[i];
        buffer[new_name_size] = last_symbol;
        ++new_name_size;
    }

    return std::string(buffer, new_name_size);
}

std::string ExternalOptimizationUtil::generateHashName(std::string value) {
    std::hash<std::string> hasher;
    size_t hash = hasher(value);
    std::string hash_string = std::to_string(hash);
    std::string hash_prefix = std::string(modelsHashPrefix);
    return hash_prefix + hash_string;
}

std::string ExternalOptimizationUtil::processTestName(const std::string &network_name, const size_t extension_len) {
    std::vector<std::string> erase_patterns = {
        "netPRC",
        "netPrecision",
        "targetDevice",
        "trgDev",
        "configItem",
        "targetConfig",
        "exportConfigItem",
        "importConfigItem",
        "inPRC",
        "outPRC",
        "GNA_DEVICE_MODE",
        "GNA_EXEC_TARGET",
        "inputShape",
        "oututShape",
    };
    std::map<std::string, std::string> replace_map = {
        { "GNA_SCALE_FACTOR", "gna_sf" },
        { "GNA_TARGET_1_0", "gna_10" },
        { "GNA_TARGET_2_0", "gna_20" },
        { "GNA_TARGET_3_0", "gna_30" },
        { "UNSPECIFIED", "UNSP" },
        // unification patterns
        { "GNA_SW_EXACT", "mode" },
        { "GNA_SW_FP32", "mode" },
        { "sw_exact", "mode" },
        { "sw_fp32", "mode" },
    };

    std::string new_network_name { network_name };
    new_network_name = eraseInName(new_network_name, erase_patterns);
    new_network_name = replaceInName(new_network_name, replace_map);
    new_network_name = eraseRepeatedInName(new_network_name, {'_'});

    auto prefix = std::string(modelsNamePrefix);
    auto prefix_len = prefix.length();
    auto max_name_len = MAX_FILE_NAME_SIZE - extension_len - prefix_len - 1;
    if (new_network_name.length() > max_name_len) {
        // Add hash if test name is still too long for a system
        auto hashed_network_name = generateHashName(network_name);
        auto fitted_size = max_name_len - SHORT_HASH_SIZE - 1;
        new_network_name = new_network_name.substr(0, fitted_size) + '_' + hashed_network_name.substr(0, SHORT_HASH_SIZE);
    }

    new_network_name = prefix + new_network_name;

    return new_network_name;
}

void ExternalOptimizationUtil::updateModelNames(std::shared_ptr<ov::Model> network) {
    auto rename = [](std::shared_ptr<ov::Node> node) {
        std::string id   {std::to_string(node->get_instance_id())};
        std::string type {node->get_type_name()};

        std::string new_name = id + "_" + type;

        node->set_friendly_name(new_name);
    };

    for (auto node : network->get_ordered_ops()) {
        rename(node);
    }
}

void ExternalOptimizationUtil::unifyModelNames(std::shared_ptr<ov::Model> network) {
    auto rename = [](std::shared_ptr<ov::Node> node, size_t index) {
        std::string id   {std::to_string(index)};
        std::string type {node->get_type_name()};

        std::string new_name = type + "_" + id;

        node->set_friendly_name(new_name);
    };

    size_t index = 0;
    for (auto node : network->get_ordered_ops()) {
        rename(node, index);
        ++index;
    }
}

template<>
void ExternalOptimizationUtil::writeToArkFile<float>(const std::string &fileName, const float *ptrMemory, uint32_t numRows, uint32_t numColumns) {
    std::ios_base::openmode mode = std::ios::binary;
    std::ofstream out_file(fileName.c_str(), mode);
    const std::string &token = "input ";
    if (out_file.good()) {
        out_file.write(token.c_str(), token.length());
        out_file.write("\0", 1);
        out_file.write("BFM ", 4);
        out_file.write("\4", 1);
        out_file.write(reinterpret_cast<char*>(&numRows), sizeof(uint32_t));
        out_file.write("\4", 1);
        out_file.write(reinterpret_cast<char*>(&numColumns), sizeof(uint32_t));
        out_file.write(reinterpret_cast<const char*>(ptrMemory), numRows * numColumns * sizeof(float));
        out_file.close();
    } else {
        throw std::runtime_error(std::string("Failed to open %s for writing in saveArkFile()!\n") + fileName);
    }
    printf("Input data dumped to ark file %s\n", fileName.c_str());
}

void ExternalOptimizationUtil::saveInputFile(const std::string &network_name,
                                        const InferenceEngine::InputInfo::CPtr &input_info,
                                        const InferenceEngine::Blob::CPtr &blob,
                                        uint32_t id,
                                        std::string extension) {
        const uint32_t utterance = 1;
        const auto models_path = getModelsPath();
        const std::string inputs_path = models_path.empty() ? "inputs" : models_path + path_delimiter + "inputs";
        if (!CommonTestUtils::directoryExists(inputs_path)) {
            CommonTestUtils::createDirectory(inputs_path);
        }

        const std::string new_network_name = processTestName(network_name);
        const std::string model_inputs_path = inputs_path + path_delimiter + new_network_name;
        if (!CommonTestUtils::directoryExists(model_inputs_path)) {
            CommonTestUtils::createDirectory(model_inputs_path);
        }

        const auto& dims = input_info->getTensorDesc().getDims();
        uint32_t elements_number = 1;
        for (const auto& dim : dims) {
            elements_number *= dim;
        }

        const std::string input_type_name = "Parameter";
        std::string input_file_name = input_type_name + "_"
                                    + std::to_string(id) + "_"
                                    + std::to_string(utterance) + "_"
                                    + std::to_string(elements_number) + "." + extension;

        std::string file_name = model_inputs_path + path_delimiter + input_file_name;

        const auto &precision = input_info->getPrecision();

        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
        const auto locked_memory = memory->rmap();

        // cast to precision
        switch (precision) {
        case InferenceEngine::Precision::FP32:
            writeDataToFile(file_name, locked_memory.as<const float *>(), utterance, elements_number, extension);
            break;
        case InferenceEngine::Precision::FP16:
            writeDataToFile(file_name, locked_memory.as<const ngraph::float16 *>(), utterance, elements_number, extension);
            break;
        case InferenceEngine::Precision::I16:
            writeDataToFile(file_name, locked_memory.as<const int16_t *>(), utterance, elements_number, extension);
            break;
        case InferenceEngine::Precision::U16:
            writeDataToFile(file_name, locked_memory.as<const uint16_t *>(), utterance, elements_number, extension);
            break;
        case InferenceEngine::Precision::I8:
            writeDataToFile(file_name, locked_memory.as<const int8_t *>(), utterance, elements_number, extension);
            break;
        case InferenceEngine::Precision::U8:
            writeDataToFile(file_name, locked_memory.as<const uint8_t *>(), utterance, elements_number, extension);
            break;
        default:
            printf("%s precision not supported\n", precision.name());
            return;
        }
    }

void ExternalOptimizationUtil::dumpNetworkToFile(const std::shared_ptr<ov::Model> network,
                                                 const std::string &network_name) {
    const auto exportPathString = getModelsPath();
    auto new_network_name = processTestName(network_name);

    std::string out_xml_path = exportPathString
                                + (exportPathString.empty() ? "" : path_delimiter)
                                + new_network_name + ".xml";
    std::string out_bin_path = exportPathString
                                + (exportPathString.empty() ? "" : path_delimiter)
                                + new_network_name + ".bin";

    auto network_copy = ov::clone_model(*network);
    unifyModelNames(network_copy);

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Serialize>(out_xml_path, out_bin_path, ngraph::pass::Serialize::Version::IR_V11);
    manager.run_passes(network_copy);
    printf("Network dumped to %s\n", out_xml_path.c_str());
    writeToHashMap(network_name, new_network_name);
}

static ov::frontend::FrontEndManager& get_frontend_manager() {
    static ov::frontend::FrontEndManager manager;
    return manager;
}

std::shared_ptr<ov::Model> ExternalOptimizationUtil::loadNetworkFromFile(const std::string &network_name) {
    const auto importPathString = getModelsPath();
    auto new_network_name = processTestName(network_name);

    std::string out_xml_path = importPathString
                                + (importPathString.empty() ? "" : path_delimiter)
                                + new_network_name + ".xml";
    std::string out_bin_path = importPathString
                                + (importPathString.empty() ? "" : path_delimiter)
                                + new_network_name + ".bin";

    auto& manager = get_frontend_manager();
    ov::frontend::FrontEnd::Ptr FE;
    ov::frontend::InputModel::Ptr inputModel;

    FE = manager.load_by_model(out_xml_path, out_bin_path);
    if (FE) {
        inputModel = FE->load(out_xml_path, out_bin_path);
    }
    if (!inputModel) {
        IE_THROW(NetworkNotRead) << "Unable to read the model " << out_xml_path;
    }
    auto model = FE->convert(inputModel);
    updateModelNames(model);
    printf("Network loaded from %s\n", out_xml_path.c_str());
    return model;
}
