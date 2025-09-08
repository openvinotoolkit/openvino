// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_model_repo.hpp"

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu_no_reshapes.hpp"
#include "openvino/pass/serialize.hpp"

namespace TestDataHelpers {

const std::string model_bin_name = "test_model.bin";
const std::string model_xml_name = "test_model.xml";
const std::string model_exported_name = "test_exported_model.blob";

void generate_test_model() {
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(model_xml_name, model_bin_name);
    auto function = ov::test::utils::make_conv_pool_relu_no_reshapes({1, 3, 227, 227});
    manager.run_passes(function);
}

std::string generate_test_xml_file() {
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
    plugin_xml_file << ov::test::utils::getExecutableDirectory();
    plugin_xml_file << ov::util::FileTraits<char>::file_separator;
    plugin_xml_file << ov::util::FileTraits<char>::library_prefix();
    plugin_xml_file << "mock_engine";
    plugin_xml_file << OV_BUILD_POSTFIX;
    plugin_xml_file << ov::util::FileTraits<char>::dot_symbol;
    plugin_xml_file << ov::util::FileTraits<char>::library_ext();
    plugin_xml_file << "\" name=\"CUSTOM\">\n";
    plugin_xml_file << "        </plugin>\n";
    plugin_xml_file << "    </plugins>\n";
    plugin_xml_file << "</ie>\n";

    // Close the file
    plugin_xml_file.close();
    return plugin_xml;
}
}  // namespace TestDataHelpers
