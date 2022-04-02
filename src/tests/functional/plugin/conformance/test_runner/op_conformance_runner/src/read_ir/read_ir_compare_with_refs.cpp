// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"

#include "read_ir_test/read_ir_compare_with_refs.hpp"
#include "conformance.hpp"

namespace ov {
namespace test {
namespace conformance {
namespace op {

using namespace ov::test::subgraph;

std::vector<std::string> readSkipTestConfigFiles(const std::vector<std::string>& filePaths) {
    std::vector<std::string> res;
    for (const auto& filePath : filePaths) {
        if (!CommonTestUtils::fileExists(filePath)) {
            std::string msg = "Input directory (" + filePath + ") doesn't not exist!";
            throw std::runtime_error(msg);
        }
        std::ifstream file(filePath);
        if (file.is_open()) {
            std::string buffer;
            while (getline(file, buffer)) {
                if (buffer.find("#") == std::string::npos && !buffer.empty()) {
                    res.emplace_back(buffer);
                }
            }
        } else {
            std::string msg = "Error in opening file: " + filePath;
            throw std::runtime_error(msg);
        }
        file.close();
    }
    return res;
}


namespace {
INSTANTIATE_TEST_SUITE_P(conformance,
                         ReadIRTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(readSkipTestConfigFiles(
                                         {"/media/nnlvdp-shared01/Share/conformance_ir/pull/11327/head/2022.2.0-7179-e51ebf48531-refs/"
                                          "pull/11327/head/filelist.lst"})),
//                                 ::testing::ValuesIn(CommonTestUtils::getFileListByPatternRecursive(IRFolderPaths, {std::regex(R"(.*\.xml)")})),
                                 ::testing::Values(targetDevice),
                                 ::testing::Values(pluginConfig)),
                         ReadIRTest::getTestCaseName);
}  // namespace

}  // namespace op
}  // namespace conformance
}  // namespace test
}  // namespace ov
