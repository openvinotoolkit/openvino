// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <random>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/util/file_util.hpp"
#include "common_test_utils/file_utils.hpp"

namespace TestDataHelpers {

static const std::string model_bin_name = "test_model.bin";
static const std::string model_xml_name = "test_model.xml";
static const std::string model_exported_name = "test_exported_model.blob";

inline void generate_test_model() {
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(model_xml_name, model_bin_name);
    auto function = ngraph::builder::subgraph::makeConvPoolReluNoReshapes({1, 3, 227, 227});
    manager.run_passes(function);
}

inline std::string get_model_xml_file_name() {
    return model_xml_name;
}

inline std::string get_model_bin_file_name() {
    return model_bin_name;
}

inline std::string get_exported_blob_file_name() {
    return model_exported_name;
}

inline void release_test_model() {
    std::remove(model_xml_name.c_str());
    std::remove(model_bin_name.c_str());
}

inline void fill_random_input_nv12_data(uint8_t* data, const size_t w, const size_t h) {
    size_t size = w * h * 3 / 2;
    std::mt19937 gen(0);
    std::uniform_int_distribution<> distribution(0, 255);
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<uint8_t>(distribution(gen));
    }
    return;
}

inline std::string generate_test_xml_file() {
    // Create the file
    std::string plugin_xml = "plugin_test.xml";
    std::ofstream plugin_xml_file(plugin_xml);

    // Write to the file
    plugin_xml_file << "<!--\n";
    plugin_xml_file << "Copyright (C) 2023 Intel Corporation\n";
    plugin_xml_file << "SPDX-License-Identifier: Apache-2.0\n";
    plugin_xml_file << "-->\n";
    plugin_xml_file << "\n";

    plugin_xml_file << "<ie>\n";
    plugin_xml_file << "    <plugins>\n";
    plugin_xml_file << "        <plugin location=\"";
    plugin_xml_file << CommonTestUtils::getExecutableDirectory();
    plugin_xml_file << ov::util::FileTraits<char>::file_separator;
    plugin_xml_file << ov::util::FileTraits<char>::library_prefix();
    plugin_xml_file << "mock_engine";
    plugin_xml_file << IE_BUILD_POSTFIX;
    plugin_xml_file << ov::util::FileTraits<char>::dot_symbol;
    plugin_xml_file << ov::util::FileTraits<char>::library_ext();
    plugin_xml_file << "\" name=\"CUSTOM\">\n";
    plugin_xml_file << "        </plugin>\n";
    plugin_xml_file << "    </plugins>\n";
    plugin_xml_file << "</ie>\n";

    // Close the file
    plugin_xml_file.flush();
    plugin_xml_file.close();
    return plugin_xml;
}

inline void delete_test_xml_file() {
    std::remove("plugin_test.xml");
}
}  // namespace TestDataHelpers
