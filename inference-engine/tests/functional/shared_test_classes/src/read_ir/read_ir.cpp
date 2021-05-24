// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/core_config.hpp"
#include "functional_test_utils/layer_test_utils/op_info.hpp"

#include "shared_test_classes/read_ir/read_ir.hpp"
#include "shared_test_classes/read_ir/compare_results.hpp"
#include "shared_test_classes/read_ir/generate_inputs.hpp"

namespace LayerTestsDefinitions {
std::string ReadIRTest::getTestCaseName(const testing::TestParamInfo<std::tuple<std::string, std::string>> &obj) {
    std::string pathToModel, deviceName;
    std::tie(pathToModel, deviceName) = obj.param;

    std::ostringstream result;
    auto splittedFilename = CommonTestUtils::splitStringByDelimiter(pathToModel, CommonTestUtils::FileSeparator);
    if (splittedFilename.size() > 1) {
        result << "PRC=" << *std::next(splittedFilename.rbegin()) << "_";
    }
    result << "IR_name=" << splittedFilename.back() << "_";
    result << "TargetDevice=" << deviceName;
    return result.str();
}

void ReadIRTest::SetUp() {
    std::tie(pathToModel, targetDevice) = this->GetParam();
    auto net = getCore()->ReadNetwork(pathToModel);
    function = net.getFunction();
    const auto metaFile = CommonTestUtils::replaceExt(pathToModel, "meta");
    if (CommonTestUtils::fileExists(metaFile)) {
        pugi::xml_document doc;
        doc.load_file(metaFile.c_str());
        auto models = doc.child("meta_info").child("models");
        sourceModel = models.child("initial_model").attribute("name").as_string();
        for (const auto &model : models.children("model")) {
            ocuranceInModels.push_back({model.attribute("name").as_string(), model.attribute("count").as_uint()});
        }
        auto portsInfo = doc.child("meta_info").child("ports_info");
        auto getPortInfo = [&](size_t id) {
            LayerTestsUtils::PortInfo info;
            for (const auto &p : portsInfo.children()) {
                if (p.attribute("id").as_uint() == id) {
                    info.convert_to_const = p.attribute("convert_to_const").as_bool();
                    if (std::strcmp(p.attribute("min").as_string(), "undefined") != 0) {
                        info.min = p.attribute("min").as_double();
                    } else {
                        info.min = -10;
                    }
                    if (std::strcmp(p.attribute("max").as_string(), "undefined") != 0) {
                        info.max = p.attribute("max").as_double();
                    } else {
                        info.max = 10;
                    }
                    break;
                }
            }
            return info;
        };
        for (const auto &param : function->get_parameters()) {
            auto idx = function->get_parameter_index(param);
            auto info = getPortInfo(idx);
            if (info.convert_to_const) {
                const auto constant = ngraph::builder::makeConstant(param->get_element_type(),
                                                                    param->get_shape(),
                                                                    std::vector<double>{},
                                                                    true,
                                                                    info.max,
                                                                    info.min,
                                                                    1);
                ngraph::replace_node(param, constant);
                function->remove_parameter(param);
            }
        }
    }
}

void ReadIRTest::GenerateInputs() {
    auto inputMap = getInputMap();
    const auto &inputsInfo = executableNetwork.GetInputsInfo();
    for (const auto &param : function->get_parameters()) {
        const auto infoIt = inputsInfo.find(param->get_friendly_name());
        GTEST_ASSERT_NE(infoIt, inputsInfo.cend());

        const auto &info = infoIt->second;
        for (size_t i = 0; i < param->get_output_size(); i++) {
            for (const auto &node : param->get_output_target_inputs(i)) {
                const auto nodePtr = node.get_node()->shared_from_this();
                auto it = inputMap.find(nodePtr->get_type_info());
                for (size_t port = 0; port < nodePtr->get_input_size(); ++port) {
                    if (nodePtr->get_input_node_ptr(port)->shared_from_this() == param->shared_from_this()) {
                        inputs.push_back(it->second(nodePtr, *info, port));
                    }
                }
            }
        }
    }
}

void ReadIRTest::Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expected,
                         const std::vector<InferenceEngine::Blob::Ptr> &actual) {
    auto compareMap = getCompareMap();
    for (const auto &result : function->get_results()) {
        for (size_t i = 0; i < result->get_input_size(); ++i) {
            const auto inputNode = result->get_input_node_shared_ptr(i);
            auto it = compareMap.find(inputNode->get_type_info());
            it->second(inputNode, expected, actual, threshold);
        }
    }
}

std::vector<InferenceEngine::Blob::Ptr> ReadIRTest::GetOutputs() {
    std::vector<InferenceEngine::Blob::Ptr> outputs;
    for (const auto &result : function->get_results()) {
        for (size_t inPort = 0; inPort < result->get_input_size(); ++inPort) {
            const auto &inputNode = result->get_input_node_shared_ptr(inPort);
            for (size_t outPort = 0; outPort < inputNode->get_output_size(); ++outPort) {
                for (const auto &out : inputNode->get_output_target_inputs(outPort)) {
                    if (out.get_node()->shared_from_this() == result) {
                        std::string name = inputNode->get_friendly_name();
                        if (inputNode->get_output_size() > 1) {
                            name += "." + std::to_string(outPort);
                        }
                        outputs.push_back(inferRequest.GetBlob(name));
                        break;
                    }
                }
            }
        }
    }
    return outputs;
}
} // namespace LayerTestsDefinitions

