// Copyright (C) 2018-2025 Intel Corporation
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

    std::string xmlFileName{};
    std::string binFileName{};

    void createTemporalModelFile(std::string xmlFileContent,
                                 std::vector<unsigned char> binFileContent = std::vector<unsigned char>()) {
        ASSERT_TRUE(xmlFileContent.size() > 0);
        auto filePrefix = ov::test::utils::generateTestFilePrefix();
        xmlFileName = filePrefix + "_IrFrontendTestModel.xml";
        binFileName = filePrefix + "_IrFrontendTestModel.bin";

        {
            std::ofstream xmlFile;
            xmlFile.open(xmlFileName);
            xmlFile << xmlFileContent;
            xmlFile.close();
        }

        if (binFileContent.size() > 0) {
            std::ofstream binFile;
            binFile.open(binFileName, std::ios::binary);
            binFile.write((const char*)binFileContent.data(), binFileContent.size());
            binFile.close();
        }
    }

    void RemoveTemporalFiles() {
        std::remove(xmlFileName.c_str());
        std::remove(binFileName.c_str());
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
