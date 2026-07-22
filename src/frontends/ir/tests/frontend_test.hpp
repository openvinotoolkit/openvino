// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/openvino.hpp"

class IRFrontendTestsImpl {
protected:
    ov::Core core;
    ov::frontend::FrontEndManager manager;

    std::filesystem::path prefix = ov::test::utils::generateTestFilePrefix();
    std::filesystem::path xmlFileName = std::filesystem::path(prefix).concat("_IrFrontendTestModel.xml");
    std::filesystem::path binFileName = std::filesystem::path(prefix).concat("_IrFrontendTestModel.bin");

    void createTemporalModelFile(std::string xmlFileContent,
                                 std::vector<unsigned char> binFileContent = std::vector<unsigned char>()) {
        ASSERT_TRUE(xmlFileContent.size() > 0);

        if (std::ofstream xml_file(xmlFileName); xml_file.is_open()) {
            xml_file << xmlFileContent;
        }

        if (std::ofstream bin_file(binFileName, std::ios::binary); bin_file.is_open()) {
            bin_file.write(reinterpret_cast<const char*>(binFileContent.data()), binFileContent.size());
        }
    }

    void RemoveTemporalFiles() {
        if (std::filesystem::exists(xmlFileName)) {
            std::filesystem::remove(xmlFileName);
        }
        if (std::filesystem::exists(binFileName)) {
            std::filesystem::remove(binFileName);
        }
    }

    std::shared_ptr<ov::Model> getWithIRFrontend(const std::string& model) {
        std::istringstream modelStringStream(model);
        std::istream& modelStream = modelStringStream;

        ov::frontend::FrontEnd::Ptr FE;
        ov::frontend::InputModel::Ptr inputModel;

        ov::AnyVector params{&modelStream};

        FE = manager.load_by_model(params);
        if (FE)
            inputModel = FE->load(params);

        if (inputModel)
            return FE->convert(inputModel);

        return nullptr;
    }
};
