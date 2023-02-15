// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>
#ifdef _WIN32
#include <process.h>
#endif

#include <pugixml.hpp>

#include "ngraph_functions/builders.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/crash_handler.hpp"
#include "functional_test_utils/summary/op_info.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "conformance.hpp"
#include "read_ir_test/read_ir.hpp"

#include <setjmp.h>

namespace ov {
namespace test {
namespace conformance {
// It is used while files lookup
std::list<std::string> dirList;
}
namespace subgraph {

ShapeMode shapeMode = ShapeMode::BOTH;

std::string ReadIRTest::getTestCaseName(const testing::TestParamInfo<ReadIRParams> &obj) {
    using namespace CommonTestUtils;
    std::string pathToModel, deviceName;
    ov::AnyMap config;
    std::tie(pathToModel, deviceName, config) = obj.param;

    std::ostringstream result;
    auto splittedFilename = CommonTestUtils::splitStringByDelimiter(pathToModel, CommonTestUtils::FileSeparator);
    std::reverse(splittedFilename.begin(), splittedFilename.end());
    bool is_valid_path_format = true;

    // Check that op is valid
    if (splittedFilename.size() > 2) {
        auto pos = splittedFilename[2].find('-');
        std::string op_name = splittedFilename[2], op_version = "";
        if (pos != std::string::npos) {
            op_name = splittedFilename[2].substr(0, pos);
            op_version = splittedFilename[2].substr(pos + 1);
        }
        if (std::find(ov::test::conformance::unique_ops[op_name].begin(),
                      ov::test::conformance::unique_ops[op_name].end(), op_version) != ov::test::conformance::unique_ops[op_name].end() &&
            ov::test::conformance::unique_ops.find(op_name) != ov::test::conformance::unique_ops.end()) {
            std::string message = "Op=" + op_name;
            if (op_version != "") {
                message += "." + op_version;
            }
            message += "_";
            result << message;
        } else {
            is_valid_path_format = false;
        }
    }
    // Check the element_type
    if (splittedFilename.size() > 1) {
        if (std::find(ov::test::conformance::element_type_names.begin(),
                      ov::test::conformance::element_type_names.end(),
                      splittedFilename[1]) != ov::test::conformance::element_type_names.end()) {
            result << "Type=" << splittedFilename[1] << "_";
        } else {
            is_valid_path_format = false;
        }
    }
    result << "IR=" << (is_valid_path_format ? CommonTestUtils::replaceExt(splittedFilename[0], "") : pathToModel) << "_";
    result << "Device=" << deviceName << "_";
    result << "Config=(";
    auto configItem = config.begin();
    while (configItem != config.end()) {
        result << configItem->first << "=";
        configItem->second.print(result);
        if (++configItem != config.end()) {
            result << "_";
        }
    }
    result << ")";
    return result.str();
}

void ReadIRTest::query_model() {
    // in case of crash jump will be made and work will be continued
    auto crashHandler = std::unique_ptr<CommonTestUtils::CrashHandler>(new CommonTestUtils::CrashHandler());
    auto &s = ov::test::utils::OpSummary::getInstance();

    // place to jump in case of a crash
    int jmpRes = 0;
#ifdef _WIN32
    jmpRes = setjmp(CommonTestUtils::env);
#else
    jmpRes = sigsetjmp(CommonTestUtils::env, 1);
#endif
    if (jmpRes == CommonTestUtils::JMP_STATUS::ok) {
        crashHandler->StartTimer();
        if (functionRefs == nullptr) {
            functionRefs = ngraph::clone_function(*function);
            functionRefs->set_friendly_name("refFunction");
        }
        s.setDeviceName(targetDevice);

        if (FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()) {
            s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::SKIPPED);
            GTEST_SKIP() << "Disabled test due to configuration" << std::endl;
        } else {
            s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::CRASHED);
        }
        try {
            SubgraphBaseTest::query_model();
            s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::PASSED);
        } catch (...) {
            s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::FAILED);
        }
    } else if (jmpRes == CommonTestUtils::JMP_STATUS::alarmErr) {
        s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::HANGED);
        IE_THROW() << "Crash happens";
    } else if (jmpRes == CommonTestUtils::JMP_STATUS::anyError) {
        IE_THROW() << "Crash happens";
    }
}

uint64_t clip(uint64_t n, uint64_t lower, uint64_t upper) {
    return std::max(lower, std::min(n, upper));
}

void ReadIRTest::SetUp() {
    std::tie(pathToModel, targetDevice, configuration) = this->GetParam();
    function = core->read_model(pathToModel);
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

        auto params = function->get_parameters();
        for (const auto &param : params) {
            auto idx = -1;
            for (size_t i = 0; i < param->get_output_size(); i++) {
                for (const auto &node : param->get_output_target_inputs(i)) {
                    const auto nodePtr = node.get_node()->shared_from_this();
                    for (size_t port = 0; port < nodePtr->get_input_size(); ++port) {
                        if (nodePtr->get_input_node_ptr(port)->shared_from_this() == param->shared_from_this()) {
                            idx = port;
                            break;
                        }
                    }
                }
            }
            EXPECT_GE(idx, 0);

            auto info = getPortInfo(idx);
            if (info.convert_to_const) {
                const auto constant = ngraph::builder::makeConstant(param->get_element_type(),
                                                                    param->get_shape(),
                                                                    std::vector<double>{},
                                                                    true,
                                                                    info.max,
                                                                    info.min,
                                                                    1);
                ov::replace_node(param, constant);
                function->remove_parameter(param);
            }
        }
    }

    bool hasDynamic = false;
    for (const auto& param : function->get_parameters()) {
        if (param->get_partial_shape().is_dynamic()) {
            hasDynamic = true;
            break;
        }
    }
    if (!hasDynamic) {
        for (const auto& result : function->get_results()) {
            if (result->get_output_partial_shape(0).is_dynamic()) {
                hasDynamic = true;
                break;
            }
        }
    }
    if (hasDynamic && ov::test::subgraph::shapeMode == ov::test::subgraph::ShapeMode::STATIC) {
        GTEST_SKIP() << "Dynamic cases are skipped according `shape_mode`";
    } else if (!hasDynamic && ov::test::subgraph::shapeMode == ov::test::subgraph::ShapeMode::DYNAMIC) {
        GTEST_SKIP() << "Static cases are skipped according `shape_mode`";
    }

    std::vector<InputShape> inputShapes;
    for (const auto& param : function -> get_parameters()) {
        if (param->get_partial_shape().is_static()) {
            inputShapes.push_back(InputShape{{}, {param->get_shape()}});
        } else {
            std::vector<ov::Shape> staticShapes = { param->get_partial_shape().get_min_shape(),
                                                    param->get_partial_shape().get_min_shape(),
                                                    param->get_partial_shape().get_max_shape() };
            ov::Shape midShape;
            for (const auto s : param->get_partial_shape()) {
                int dimValue = 1;
                if (s.is_dynamic()) {
                    size_t range = s.get_max_length() - s.get_min_length();
                    if (range > std::numeric_limits<char>::max()) {
                        CommonTestUtils::fill_data_random(&range, 1, std::numeric_limits<char>::max(), s.get_min_length(), 1);
                    }
                    CommonTestUtils::fill_data_random(&dimValue, 1, range, s.get_min_length(), 1);
                } else {
                    dimValue = s.get_length();
                }
                midShape.push_back(dimValue);
            }
            staticShapes[1] = midShape;

            // Shape validation to avoid large values
            uint64_t dimMin = 1;
            uint64_t dimMax = std::numeric_limits<char>::max();
            for (int i = 0; i < staticShapes[0].size(); ++i) {
                auto& dim0 = staticShapes[0][i];
                auto& dim2 = staticShapes[2][i];
                if (dim0 != dim2) {
                    dim0 = clip(dim0, dimMin, dimMax);
                    dim2 = clip(dim2, dimMin, dimMax);
                }
            }
            inputShapes.push_back(InputShape{param->get_partial_shape(), staticShapes});
        }
    }
    if (inputShapes.empty()) {
        GTEST_SKIP() << "The graph is constant. The case is not applicable for Operation conformance scenario";
    }
    init_input_shapes(inputShapes);
    is_report_stages = true;
}

} // namespace subgraph
} // namespace test
} // namespace ov

