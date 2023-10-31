// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>
#ifdef _WIN32
#include <process.h>
#endif

#include <pugixml.hpp>

#include "shared_test_classes/base/utils/ranges.hpp"
#include "shared_test_classes/base/utils/generate_inputs.hpp"
#include "ov_models/builders.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "functional_test_utils/crash_handler.hpp"
#include "functional_test_utils/summary/op_info.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "dynamism.hpp"
#include "input_info.hpp"
#include "conformance.hpp"
#include "read_ir_test/read_ir.hpp"

#include "common_test_utils/postgres_link.hpp"

#include <setjmp.h>

#include "openvino/pass/manager.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/runtime/common.hpp"

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

    std::string subgrapth_dir = "subgraph";
    std::vector<std::string> graphConvertLogicTypes = { "fused_names", "repeat_pattern" };

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
        } else if (splittedFilename[2] != subgrapth_dir) {
            is_valid_path_format = false;
        }
    }

    // Check the element_type
    if (splittedFilename.size() > 1) {
        if (std::find(ov::test::conformance::element_type_names.begin(),
                      ov::test::conformance::element_type_names.end(),
                      splittedFilename[1]) != ov::test::conformance::element_type_names.end()) {
            result << "Type=" << splittedFilename[1] << "_";
        } else if (std::find(graphConvertLogicTypes.begin(),
                             graphConvertLogicTypes.end(),
                             splittedFilename[1]) != graphConvertLogicTypes.end()) {
            result << "ConvertLogic=" << splittedFilename[1] << "_";
        } else {
            is_valid_path_format = false;
        }
    }
    result << "IR=" << (is_valid_path_format ? ov::test::utils::replaceExt(splittedFilename[0], "") : path_to_model) << "_";
    result << "Device=" << deviceName << "_";

    std::vector<std::string> shapeModes = { "static", "dynamic" };
    // Check the shape type
    if (splittedFilename.size() > 3 &&
        std::find(shapeModes.begin(), shapeModes.end(), splittedFilename[3]) != shapeModes.end()) {
        result << "Shape=" << splittedFilename[3] << "_";
    }
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

uint64_t clip(uint64_t n, uint64_t lower, uint64_t upper) {
    return std::max(lower, std::min(n, upper));
}

void ReadIRTest::SetUp() {
    // todo: find the optimal way to find TEST_P instances
    // inference + query_model + import_export
    summary.setDowngradeCoefficient(3);
    std::pair<std::string, std::string> model_pair;
    std::tie(model_pair, targetDevice, configuration) = this->GetParam();
    std::tie(path_to_model, path_to_cache) = model_pair;
    function = core->read_model(path_to_model);
    const auto metaFile = ov::test::utils::replaceExt(path_to_model, "meta");
    if (ov::test::utils::fileExists(metaFile)) {
        pugi::xml_document doc;
        doc.load_file(metaFile.c_str());
        rel_influence_coef = doc.child("meta_info").child("graph_priority").attribute("value").as_double();
        // TODO: remove after cache update w/a
        if (rel_influence_coef == 0) {
            rel_influence_coef = 1.f;
        }
        auto input_info_xml = doc.child("meta_info").child("input_info");
        std::map<std::string, ov::tools::subgraph_dumper::InputInfo> input_info;
        for (const auto &input : input_info_xml.children()) {
            auto in_name = std::string(input.attribute("id").value());
            ov::tools::subgraph_dumper::InputInfo in_info;
            in_info.is_const = input.attribute("convert_to_const").as_bool();
            if (std::string(input.attribute("min").value()) != "undefined") {
                in_info.ranges.min = input.attribute("min").as_double();
            }
            if (std::string(input.attribute("max").value()) != "undefined") {
                in_info.ranges.max = input.attribute("max").as_double();
            }
            input_info.insert({in_name, in_info});
        }
        auto inputMap = utils::getInputMap();
        std::vector<std::shared_ptr<ov::op::v0::Parameter>> parameter_to_remove;
        for (const auto& param : function->get_parameters()) {
            auto in_info = input_info.find(param->get_friendly_name())->second;
            if (!in_info.is_const) {
                continue;
            }
            utils::ConstRanges::set(in_info.ranges.min, in_info.ranges.max);
            // auto next_node = param->get_default_output().get_node_shared_ptr();
            auto next_node = param->get_default_output().get_target_inputs().begin()->get_node()->shared_from_this();
            auto it = inputMap.find(next_node->get_type_info());
            auto tensor = it->second(next_node, function->get_parameter_index(param), param->get_element_type(), param->get_shape());
            auto const_node = std::make_shared<ov::op::v0::Constant>(tensor);
            const_node->set_friendly_name(param->get_friendly_name());
            ov::replace_node(param, const_node);
            parameter_to_remove.push_back(param);
            utils::ConstRanges::reset();
        }
        for (const auto& param : parameter_to_remove) {
            function->remove_parameter(param);
        }
    }

    bool hasDynamic = tools::subgraph_dumper::is_dynamic_model(function);

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

