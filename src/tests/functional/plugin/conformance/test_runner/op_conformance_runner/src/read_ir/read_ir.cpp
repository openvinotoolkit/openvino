// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>
#ifdef _WIN32
#include <process.h>
#endif

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/postgres_link.hpp"

#include "shared_test_classes/base/utils/generate_inputs.hpp"

#include "op_conformance_utils/utils/dynamism.hpp"
#include "op_conformance_utils/meta_info/meta_info.hpp"
#include "conformance.hpp"

#include "utils/models.hpp"
#include "utils/types.hpp"

#include "read_ir_test/read_ir.hpp"

namespace ov {
namespace test {
namespace op_conformance {

std::string ReadIRTest::getTestCaseName(const testing::TestParamInfo<ReadIRParams> &obj) {
    using namespace ov::test::utils;
    std::pair<std::string, std::string> model_pair;
    std::string path_to_model, path_to_ref_tensor, deviceName;
    ov::AnyMap config;
    std::tie(model_pair, deviceName, config) = obj.param;
    std::tie(path_to_model, path_to_ref_tensor) = model_pair;

    std::ostringstream result;

    enum class IR_TYPE {
        OP,
        SUBGRAPH,
        OTHER
    };

    auto ir_type = IR_TYPE::OTHER;
    std::map<std::string, std::string> filled_info = {
        { "hash", "" },
        { "element_type", "" },
        { "shape_type", "" },
        { "extractor_name", ""},
        { "op_name", "" },
    };

    {
        auto splitted_filename = ov::test::utils::splitStringByDelimiter(path_to_model, ov::test::utils::FileSeparator);
        std::set<std::string> shape_type = {"static", "dynamic"},
                              graph_extractors = {"fused_names", "repeat_pattern"};
        std::map<std::string, IR_TYPE> graph_type = {{"operation", IR_TYPE::OP}, {"subgraph", IR_TYPE::SUBGRAPH}};
        for (const auto& item : splitted_filename) {
            auto ir_name = ov::util::replace_extension(item, "");
            if (ir_name != item) {
                filled_info["hash"] = ir_name;
                continue;
            }
            if (std::find(element_type_names.begin(),
                          element_type_names.end(),
                          item) != element_type_names.end()) {
                filled_info["element_type"] = item;
                continue;
            }
            if (shape_type.find(item) != shape_type.end()) {
                filled_info["shape_type"] = item;
                continue;
            }
            if (graph_extractors.find(item) != graph_extractors.end()) {
                filled_info["extractor_name"] = item;
                continue;
            }

            auto pos = item.find('-');
            if (pos != std::string::npos) {
                std::string op_name = item, op_version = "";
                if (pos != std::string::npos) {
                    op_version = op_name.substr(pos + 1);
                    op_name = op_name.substr(0, pos);
                }
                if (std::find(unique_ops[op_name].begin(),
                              unique_ops[op_name].end(), op_version) != unique_ops[op_name].end() &&
                    unique_ops.find(op_name) != unique_ops.end()) {
                    filled_info["op_name"] = op_name + "." + op_version;
                    continue;
                }
            }
            if (graph_type.find(item) != graph_type.end()) {
                ir_type = graph_type.at(item);
                continue;
            }
        }
    }

    switch (ir_type) {
    case IR_TYPE::OP: {
        result << "Op=" << filled_info["op_name"] << "_";
        result << "Type=" << filled_info["element_type"] << "_";
        result << "Shape=" << filled_info["shape_type"] << "_";
        result << "IR=" << filled_info["hash"] << "_";
        break;
    }
    case IR_TYPE::SUBGRAPH: {
        result << "Extractor=" << filled_info["extractor_name"] << "_";
        result << "Shape=" << filled_info["shape_type"] << "_";
        result << "IR=" << filled_info["hash"] << "_";
        break;
    }
    default: {
        result << "IR=" << path_to_model << "_";
        break;
    }
    }
    result << "Device=" << deviceName << "_";
    result << "Config=" << config;
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
    std::tie(path_to_model, path_to_ref_tensor) = model_pair;
    function = core->read_model(path_to_model);
    const auto metaFile = ov::util::replace_extension(path_to_model, "meta");
    if (ov::util::file_exists(metaFile)) {
        auto meta_info = ov::conformance::MetaInfo::read_meta_from_file(metaFile, true);
        auto input_info = meta_info.get_input_info();
        rel_influence_coef = meta_info.get_graph_priority();

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

    bool hasDynamic = ov::util::is_dynamic_model(function);

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
            for (const auto& dimension : param->get_partial_shape()) {
                int dimValue = 1;
                if (dimension.is_dynamic()) {
                    size_t range = dimension.get_max_length() - dimension.get_min_length();
                    if (range > std::numeric_limits<char>::max()) {
                        ov::test::utils::fill_data_random(&range, 1, std::numeric_limits<char>::max(), dimension.get_min_length(), 1);
                    } else {
                        ov::test::utils::fill_data_random(&dimValue, 1, range, dimension.get_min_length(), 1);
                    }
                } else {
                    dimValue = dimension.get_length();
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
    if (!ov::test::utils::fileExists(path_to_ref_tensor)) {
        std::cout << "[ REFERENCE   ] Calculate reference in runtime" << std::endl;
        output_tensors = SubgraphBaseTest::calculate_refs();
        if (path_to_ref_tensor != "") {
            std::ofstream ofstream_tensor(path_to_ref_tensor, std::ios::out | std::ios::binary);
            for (const auto& out_tensor : output_tensors) {
                ofstream_tensor.write(reinterpret_cast<const char*>(out_tensor.data()), out_tensor.get_byte_size());
            }
            ofstream_tensor.close();
        }
    } else {
        std::cout << "[ REFERENCE   ] Read reference from file: " << path_to_ref_tensor << std::endl;
        // Because of functionRefs is a static function
        std::ifstream ref_data_ifstream(path_to_ref_tensor, std::ifstream::binary);
        ref_data_ifstream.open(path_to_ref_tensor, std::ios::binary);
        if (!ref_data_ifstream.is_open())
            IE_THROW() << "Weights file " << path_to_ref_tensor << " cannot be opened!";

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

TEST_P(ReadIRTest, Inference) {
    run();
}

TEST_P(ReadIRTest, QueryModel) {
    query_model();
}

TEST_P(ReadIRTest, ImportExport) {
    import_export();
}

namespace {

#define _OPENVINO_OP_REG(NAME, NAMESPACE)                                                                  \
    INSTANTIATE_TEST_SUITE_P(conformance_##NAME,                                                           \
                             ReadIRTest,                                                                   \
                             ::testing::Combine(::testing::ValuesIn(get_model_paths(conformance::IRFolderPaths, #NAME)),  \
                                                ::testing::Values(conformance::targetDevice),                           \
                                                ::testing::Values(conformance::pluginConfig)),                          \
                             ReadIRTest::getTestCaseName); \

// It should point on latest opset which contains biggest list of operations
#include "openvino/opsets/opset13_tbl.hpp"
#undef _OPENVINO_OP_REG

INSTANTIATE_TEST_SUITE_P(conformance_subgraph,
                        ReadIRTest,
                        ::testing::Combine(::testing::ValuesIn(get_model_paths(conformance::IRFolderPaths)),
                                           ::testing::Values(conformance::targetDevice),
                                           ::testing::Values(conformance::pluginConfig)),
                        ReadIRTest::getTestCaseName);

}  // namespace

} // namespace op_conformance
} // namespace test
} // namespace ov
