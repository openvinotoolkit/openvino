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
#include "common_test_utils/graph_comparator.hpp"
#include "functional_test_utils/crash_handler.hpp"
#include "functional_test_utils/summary/op_info.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "conformance.hpp"
#include "read_ir_test/read_ir.hpp"

#include "common_test_utils/postgres_link.hpp"

#include <setjmp.h>

namespace ov {
namespace test {
namespace conformance {
// It is used while files lookup, first value - path to model, second - amout of tests with this path
std::list<std::pair<std::string, int>> dirListInfo;
}

namespace subgraph {

std::string ReadIRTest::getTestCaseName(const testing::TestParamInfo<ReadIRParams> &obj) {
    using namespace ov::test::utils;
    std::pair<std::string, std::string> model_pair;
    std::string path_to_model, path_to_cache, deviceName;
    ov::AnyMap config;
    std::tie(model_pair, deviceName, config) = obj.param;
    std::tie(path_to_model, path_to_cache) = model_pair;

    std::ostringstream result;
    auto splittedFilename = ov::test::utils::splitStringByDelimiter(path_to_model, ov::test::utils::FileSeparator);
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
    result << "IR=" << (is_valid_path_format ? ov::test::utils::replaceExt(splittedFilename[0], "") : path_to_model) << "_";
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
    auto crashHandler = std::unique_ptr<ov::test::utils::CrashHandler>(new ov::test::utils::CrashHandler());
    auto &s = ov::test::utils::OpSummary::getInstance();

    // place to jump in case of a crash
    int jmpRes = 0;
#ifdef _WIN32
    jmpRes = setjmp(ov::test::utils::env);
#else
    jmpRes = sigsetjmp(ov::test::utils::env, 1);
#endif
    if (jmpRes == ov::test::utils::JMP_STATUS::ok) {
        crashHandler->StartTimer();
        if (functionRefs == nullptr) {
            functionRefs = ngraph::clone_function(*function);
            functionRefs->set_friendly_name("refFunction");
        }
        s.setDeviceName(targetDevice);

        if (FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()) {
            s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::SKIPPED, rel_influence_coef);
            GTEST_SKIP() << "Disabled test due to configuration" << std::endl;
        } else {
            s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::CRASHED, rel_influence_coef);
        }
        try {
            SubgraphBaseTest::query_model();
            s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::PASSED, rel_influence_coef);
        } catch (std::exception& err) {
            s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::FAILED, rel_influence_coef);
            GTEST_FAIL() << err.what();
        } catch (...) {
            s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::FAILED, rel_influence_coef);
            GTEST_FAIL() << "Something is wrong in Query model! Please check";
        }
    } else if (jmpRes == ov::test::utils::JMP_STATUS::alarmErr) {
        s.updateOPsStats(functionRefs, ov::test::utils::PassRate::Statuses::HANGED, rel_influence_coef);
        IE_THROW() << "Crash happens";
    } else if (jmpRes == ov::test::utils::JMP_STATUS::anyError) {
        IE_THROW() << "Crash happens";
    }
}

void ReadIRTest::import_export() {
    // in case of crash jump will be made and work will be continued
    auto crashHandler = std::unique_ptr<ov::test::utils::CrashHandler>(new ov::test::utils::CrashHandler());
    auto &summary = ov::test::utils::OpSummary::getInstance();

    // place to jump in case of a crash
    int jmpRes = 0;
#ifdef _WIN32
    jmpRes = setjmp(ov::test::utils::env);
#else
    jmpRes = sigsetjmp(ov::test::utils::env, 1);
#endif
    if (jmpRes == ov::test::utils::JMP_STATUS::ok) {
        crashHandler->StartTimer();
        summary.setDeviceName(targetDevice);
        try {
            ov::CompiledModel model = core->compile_model(function, targetDevice, configuration);

            std::stringstream strm;
            model.export_model(strm);

            ov::CompiledModel importedModel = core->import_model(strm, targetDevice, configuration);

            auto comparator = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::NAMES)
                        .enable(FunctionsComparator::CONST_VALUES);

            auto importedFunction = importedModel.get_runtime_model()->clone();
            auto res = comparator.compare(importedFunction, function);
            EXPECT_TRUE(res.valid) << res.message;

            summary.updateOPsImplStatus(function, true);
        } catch (const std::exception &e) {
            summary.updateOPsImplStatus(function, false);
            GTEST_FAIL() << "Exception in the Core::compile_model() method call: " << e.what();
        } catch (...) {
            summary.updateOPsImplStatus(function, false);
            GTEST_FAIL() << "Error in the Core::query_model() method call!";
        }
    } else if (jmpRes == ov::test::utils::JMP_STATUS::anyError) {
        summary.updateOPsImplStatus(function, false);
        GTEST_FAIL() << "Crash happens";
    } else if (jmpRes == ov::test::utils::JMP_STATUS::alarmErr) {
        summary.updateOPsImplStatus(function, false);
        GTEST_FAIL() << "Hang happens";
    }
}

uint64_t clip(uint64_t n, uint64_t lower, uint64_t upper) {
    return std::max(lower, std::min(n, upper));
}

void ReadIRTest::SetUp() {
    std::pair<std::string, std::string> model_pair;
    std::tie(model_pair, targetDevice, configuration) = this->GetParam();
    std::tie(path_to_model, path_to_cache) = model_pair;
    function = core->read_model(path_to_model);
    const auto metaFile = ov::test::utils::replaceExt(path_to_model, "meta");
    if (ov::test::utils::fileExists(metaFile)) {
        pugi::xml_document doc;
        doc.load_file(metaFile.c_str());
        auto models = doc.child("meta_info").child("models");
        size_t model_len = 0, occurance = 0;
        for (const auto &model : models.children("model")) {
            ocurance_in_models.push_back({model.attribute("name").as_string(), model.attribute("count").as_uint()});
            model_len++;
            occurance += model.attribute("count").as_uint();
        }
        rel_influence_coef = doc.child("meta_info").child("graph_priority").attribute("value").as_double();
        // TODO: remove after cache update w/a
        if (rel_influence_coef == 0) {
            rel_influence_coef = 1.f;
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

#ifdef ENABLE_CONFORMANCE_PGQL
    // Updating data in runtime. Should be set before possible call of a first GTEST status
    auto pgLink = this->GetPGLink();
    if (pgLink) {
        auto devNameProperty = core->get_property(this->targetDevice, "FULL_DEVICE_NAME");
        auto devName = devNameProperty.is<std::string>() ?  devNameProperty.as<std::string>() : "";
        pgLink->set_custom_field("targetDeviceName", devName, true);
        if (this->targetDevice == "CPU") {
            pgLink->set_custom_field("targetDevice", this->targetDevice, true);
            pgLink->set_custom_field("targetDeviceArch", devName.find("ARM") != std::string::npos ? "arm" : "", true);
        } else if (this->targetDevice == "GPU") {
            if (devName.find("dGPU") != std::string::npos) {
                pgLink->set_custom_field("targetDevice", "DGPU", true);
            } else {
                pgLink->set_custom_field("targetDevice", this->targetDevice, true);
            }
        } else {
            pgLink->set_custom_field("targetDevice", this->targetDevice, true);
        }
        pgLink->set_custom_field("caseType", hasDynamic ? "dynamic" : "static");
        pgLink->set_custom_field("irWeight", std::to_string(rel_influence_coef), true);

        // Do not store waste results
        if (hasDynamic && ov::test::conformance::shapeMode == ov::test::conformance::ShapeMode::STATIC) {
            pgLink->set_refuse_result();
        } else if (!hasDynamic && ov::test::conformance::shapeMode == ov::test::conformance::ShapeMode::DYNAMIC) {
            pgLink->set_refuse_result();
        }

        auto splittedFilename = ov::test::utils::splitStringByDelimiter(path_to_model, ov::test::utils::FileSeparator);
        std::reverse(splittedFilename.begin(), splittedFilename.end());

        // Try to resolve missing info
        if (splittedFilename.size() > 2) {
            auto pos = splittedFilename[2].find('-');
            std::string op_name = "", op_version = "opset";
            if (pos != std::string::npos) {
                op_name = splittedFilename[2].substr(0, pos);
                op_version += splittedFilename[2].substr(pos + 1);
                if (ov::test::conformance::unique_ops.find(op_name) != ov::test::conformance::unique_ops.end() &&
                    std::find(ov::test::conformance::unique_ops[op_name].begin(),
                              ov::test::conformance::unique_ops[op_name].end(),
                              op_version) != ov::test::conformance::unique_ops[op_name].end()) {
                    pgLink->set_custom_field("opName", op_name, true);
                    pgLink->set_custom_field("opSet", op_version, true);
                }
            } else {
                for (const auto& path_part : splittedFilename) {
                    if (ov::test::conformance::unique_ops.find(path_part) != ov::test::conformance::unique_ops.end()) {
                        op_name = path_part;
                        break;
                    }
                }
                if (op_name.length() > 0) {
                    for (const auto& node : function->get_ordered_ops()) {
                        if (node->get_type_name() == op_name) {
                            op_version = node->get_type_info().version_id;
                            pgLink->set_custom_field("opSet", op_version, true);
                        }
                    }
                }
            }
        }
        pgLink->manual_start();
    }
#endif

    if (hasDynamic && ov::test::conformance::shapeMode == ov::test::conformance::ShapeMode::STATIC) {
        GTEST_SKIP() << "Dynamic cases are skipped according `shape_mode`";
    } else if (!hasDynamic && ov::test::conformance::shapeMode == ov::test::conformance::ShapeMode::DYNAMIC) {
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
                        ov::test::utils::fill_data_random(&range, 1, std::numeric_limits<char>::max(), s.get_min_length(), 1);
                    }
                    ov::test::utils::fill_data_random(&dimValue, 1, range, s.get_min_length(), 1);
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
    std::cout << "[ CONFORMANCE ] Influence coefficient: " << rel_influence_coef << std::endl;
    init_input_shapes(inputShapes);
    is_report_stages = true;
}

std::vector<ov::Tensor> ReadIRTest::calculate_refs() {
    auto start_time = std::chrono::system_clock::now();
    if (is_report_stages) {
        std::cout << "[ REFERENCE   ] `SubgraphBaseTest::calculate_refs()` is started"<< std::endl;
    }
    ov::TensorVector output_tensors;
    if (!ov::test::utils::fileExists(path_to_cache)) {
        std::cout << "[ REFERENCE   ] Calculate reference in runtime" << std::endl;
        output_tensors = SubgraphBaseTest::calculate_refs();
        if (path_to_cache != "") {
            std::ofstream ofstream_tensor(path_to_cache, std::ios::out | std::ios::binary);
            for (const auto& out_tensor : output_tensors) {
                ofstream_tensor.write(reinterpret_cast<const char*>(out_tensor.data()), out_tensor.get_byte_size());
            }
            ofstream_tensor.close();
        }
    } else {
        std::cout << "[ REFERENCE   ] Read reference from file: " << path_to_cache << std::endl;
        // Because of functionRefs is a static function
        std::ifstream ref_data_ifstream(path_to_cache, std::ifstream::binary);
        ref_data_ifstream.open(path_to_cache, std::ios::binary);
        if (!ref_data_ifstream.is_open())
            IE_THROW() << "Weights file " << path_to_cache << " cannot be opened!";

        size_t buf_size = 0;
        for (const auto& output : functionRefs->outputs()) {
            buf_size += (sizeof output.get_element_type() * ov::shape_size(output.get_partial_shape().get_shape()));
        }
        char* ref_buffer = nullptr;
        ref_data_ifstream.read(ref_buffer, buf_size);

        size_t pos = 0;
        for (const auto& output : functionRefs->outputs()) {
            auto out_tensor = ov::runtime::Tensor(output.get_element_type(), output.get_shape(), &ref_buffer[pos]);
            pos += out_tensor.get_byte_size();
        }
    }
    if (is_report_stages) {
        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        std::cout << "[ REFERENCE   ] `SubgraphBaseTest::calculate_refs()` is finished successfully. Duration is " << duration.count() << "s" << std::endl;
    }
    return output_tensors;
}

} // namespace subgraph
} // namespace test
} // namespace ov

