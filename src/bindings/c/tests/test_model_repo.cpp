// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_model_repo.hpp"

#ifdef __APPLE__
#    include <mach-o/dyld.h>
#endif

#ifdef _WIN32
#    include <windows.h>
#else
#    include <limits.h>
#    include <unistd.h>
#endif

std::string getExecutableDirectory() {
    std::string path;
#ifdef _WIN32
    char buffer[MAX_PATH];
    int len = GetModuleFileNameA(NULL, buffer, MAX_PATH);
#elif defined(__APPLE__)
    Dl_info info;
    dladdr(reinterpret_cast<void*>(getExecutableDirectory), &info);
    const char* buffer = info.dli_fname;
    int len = std::strlen(buffer);
#else
    char buffer[PATH_MAX];
    int len = readlink("/proc/self/exe", buffer, PATH_MAX);
#endif
    if (len < 0) {
        throw "Can't get test executable path name";
    }
    path = std::string(buffer, len);
    return ov::util::get_directory(path);
}
namespace TestDataHelpers {

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
    plugin_xml_file << getExecutableDirectory();
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
    plugin_xml_file.close();
    return plugin_xml;
}
}  // namespace TestDataHelpers
