// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <fstream>

namespace TestDataHelpers {

static const char kPathSeparator =
#if defined _WIN32 || defined __CYGWIN__
    '\\';
#else
    '/';
#endif

inline std::string getModelPathNonFatal() noexcept {
    if (const auto envVar = std::getenv("MODELS_PATH")) {
        return envVar;
    }

#ifdef MODELS_PATH
    return MODELS_PATH;
#else
    return "";
#endif
}

inline std::string get_models_path() {
    return getModelPathNonFatal() + kPathSeparator + std::string("models");
};

inline std::string get_data_path() {
    if (const auto envVar = std::getenv("DATA_PATH")) {
        return envVar;
    }

#ifdef DATA_PATH
    return DATA_PATH;
#else
    return "";
#endif
}

inline std::string generate_model_path(std::string dir, std::string filename) {
    return get_models_path() + kPathSeparator + dir + kPathSeparator + filename;
}

inline std::string generate_image_path(std::string dir, std::string filename) {
    return get_data_path() + kPathSeparator + "validation_set" + kPathSeparator + dir + kPathSeparator + filename;
}

inline std::string generate_test_xml_file() {
#ifdef _WIN32
#    ifdef __MINGW32__
    std::string libraryname = "libopenvino_intel_cpu_plugin.dll";
#    else
    std::string libraryname = "openvino_intel_cpu_plugin.dll";
#    endif
#elif defined __APPLE__
#    ifdef __aarch64__
    std::string libraryname = "libopenvino_arm_cpu_plugin.so";
#    else
    std::string libraryname = "libopenvino_intel_cpu_plugin.so";
#    endif
#else
    std::string libraryname = "libopenvino_intel_cpu_plugin.so";
#endif
    // Create the file
    std::string plugin_xml = "plugin_test.xml";
    std::ofstream plugin_xml_file(plugin_xml);

    // Write to the file
    plugin_xml_file << "<!--\n";
    plugin_xml_file << "Copyright (C) 2020 Intel Corporation\n";
    plugin_xml_file << "SPDX-License-Identifier: Apache-2.0\n";
    plugin_xml_file << "-->\n";
    plugin_xml_file << "\n";

    plugin_xml_file << "<ie>\n";
    plugin_xml_file << "    <plugins>\n";
    plugin_xml_file << "        <plugin location=\"" << libraryname << "\" name=\"CUSTOM\">\n";
    plugin_xml_file << "        </plugin>\n";
    plugin_xml_file << "    </plugins>\n";
    plugin_xml_file << "</ie>\n";

    // Close the file
    plugin_xml_file.close();
    return plugin_xml;
}

inline void delete_test_xml_file() {
    std::remove("plugin_test.xml");
}
}  // namespace TestDataHelpers
