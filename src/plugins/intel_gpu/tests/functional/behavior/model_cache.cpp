// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sys/stat.h>
#include <sys/types.h>

#include <cstdio>

#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/subgraph_builders/read_concat_split_assign.hpp"
#include "common_test_utils/subgraph_builders/single_concat_with_constant.hpp"
#include "common_test_utils/subgraph_builders/ti_with_lstm_cell.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/codec_xor.hpp"
#include "shared_test_classes/subgraph/weights_decompression_builders.hpp"
#include "openvino/op/matmul.hpp"
#ifndef WIN32
#    include <unistd.h>
#endif

#ifdef WIN32
#    define stat _stat
#endif

namespace {

enum class Import_API {
    IMPORT_EXPORT,
    COMPILE_FILEPATH,
    COMPILE_MODEL,
    COMPILE_MODEL_WITH_MODEL_STR
};

std::string import_api_to_string(Import_API api) {
    switch (api) {
    case Import_API::IMPORT_EXPORT:
        return "import_export";
    case Import_API::COMPILE_FILEPATH:
        return "compile_filepath";
    case Import_API::COMPILE_MODEL:
        return "compile_model";
    case Import_API::COMPILE_MODEL_WITH_MODEL_STR:
        return "compile_model_with_model_str";
    default:
        return "";
    }
}

typedef std::tuple<Import_API, bool, ov::element::Type, ov::element::Type> testParams;

class CheckWeightlessCacheAccuracy : public ::testing::Test, public ::testing::WithParamInterface<testParams> {
public:
    static std::string get_test_case_name(::testing::TestParamInfo<testParams> obj) {
        const auto& [import_api_, do_encryption_, inference_mode_, model_dtype_] = obj.param;

        std::ostringstream result;
        const char separator = '_';
        result << "import_api=" << import_api_to_string(import_api_) << separator;
        result << "do_encryption=" << do_encryption_ << separator;
        result << "inference_mode=" << inference_mode_ << separator;
        result << "model_dtype=" << model_dtype_;
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> model;
    std::string xml_path;
    std::string bin_path;
    std::string cache_path;
    std::string cache_dir;
    Import_API import_api;
    bool do_encryption;
    ov::element::Type inference_mode;
    ov::element::Type model_dtype;

    void SetUp() override;
    void TearDown() override;
    void run();
};

void CheckWeightlessCacheAccuracy::SetUp() {
    std::string filePrefix = ov::test::utils::generateTestFilePrefix();
    xml_path = filePrefix + ".xml";
    bin_path = filePrefix + ".bin";
    cache_path = filePrefix + ".blob";
    cache_dir = filePrefix + "_cache_dir";

    std::tie(import_api, do_encryption, inference_mode, model_dtype) = GetParam();
}

void CheckWeightlessCacheAccuracy::TearDown() {
    std::remove(xml_path.c_str());
    std::remove(bin_path.c_str());
    std::remove(cache_path.c_str());

    ov::test::utils::removeFilesWithExt(cache_dir, "blob");
    ov::test::utils::removeFilesWithExt(cache_dir, "cl_cache");
    ov::test::utils::removeDir(cache_dir);
}

void CheckWeightlessCacheAccuracy::run() {
    ov::AnyMap config = {ov::cache_dir(cache_dir),
                         ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE),
                         ov::hint::inference_precision(inference_mode)};
    ov::AnyMap config_with_weights_path = {ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE),
                                           ov::weights_path(bin_path),
                                           ov::hint::inference_precision(inference_mode)};

    if (do_encryption) {
        ov::EncryptionCallbacks encryption_callbacks;
        encryption_callbacks.encrypt = ov::util::codec_xor;
        encryption_callbacks.decrypt = ov::util::codec_xor;
        config.insert(ov::cache_encryption_callbacks(encryption_callbacks));
        config_with_weights_path.insert(ov::cache_encryption_callbacks(encryption_callbacks));
    }
    auto core = ov::test::utils::PluginCache::get().core();
    ov::pass::Serialize(xml_path, bin_path).run_on_model(model);

    ov::CompiledModel compiled_model;
    if (import_api == Import_API::IMPORT_EXPORT) {
        compiled_model = core->compile_model(xml_path, ov::test::utils::DEVICE_GPU, config);
        auto ofstr = std::ofstream(cache_path, std::ofstream::binary);
        compiled_model.export_model(ofstr);
        ofstr.close();
    } else if (import_api == Import_API::COMPILE_FILEPATH) {
        compiled_model = core->compile_model(xml_path, ov::test::utils::DEVICE_GPU, config);
    } else if (import_api == Import_API::COMPILE_MODEL) {
        auto model = core->read_model(xml_path);
        compiled_model = core->compile_model(model, ov::test::utils::DEVICE_GPU, config);
    } else if (import_api == Import_API::COMPILE_MODEL_WITH_MODEL_STR) {
        std::ifstream model_s(xml_path);
        std::string model_str((std::istreambuf_iterator<char>(model_s)), std::istreambuf_iterator<char>());
        auto model_weight = ov::read_tensor_data(bin_path);
        auto model = core->read_model(model_str, model_weight);
        compiled_model = core->compile_model(model, ov::test::utils::DEVICE_GPU, config);
    } else {
        OPENVINO_THROW("Unknown import API");
    }

    auto get_cache_path = [&]() {
        std::string path;
        if (import_api == Import_API::COMPILE_FILEPATH || import_api == Import_API::COMPILE_MODEL || import_api == Import_API::COMPILE_MODEL_WITH_MODEL_STR) {
            auto blobs = ov::test::utils::listFilesWithExt(cache_dir, "blob");
            EXPECT_EQ(blobs.size(), 1);
            path = blobs[0];
        } else {
            path = cache_path;
        }
        return path;
    };

    auto get_mod_time = [&](const std::string& path) {
        struct stat result;
        if (stat(path.c_str(), &result) == 0) {
            return result.st_mtime;
        }
        return static_cast<time_t>(0);
    };

    auto first_cache_path = get_cache_path();
    auto first_mod_time = get_mod_time(first_cache_path);
    ASSERT_NE(first_mod_time, static_cast<time_t>(0));

    ov::CompiledModel imported_model;
    if (import_api == Import_API::COMPILE_FILEPATH) {
        imported_model = core->compile_model(xml_path, ov::test::utils::DEVICE_GPU, config);
    } else if (import_api == Import_API::COMPILE_MODEL) {
        auto model = core->read_model(xml_path);
        imported_model = core->compile_model(model, ov::test::utils::DEVICE_GPU, config);
    } else if (import_api == Import_API::COMPILE_MODEL_WITH_MODEL_STR) {
        std::ifstream model_s(xml_path);
        std::string model_str((std::istreambuf_iterator<char>(model_s)), std::istreambuf_iterator<char>());
        auto model_weight = ov::read_tensor_data(bin_path);
        auto model = core->read_model(model_str, model_weight);
        imported_model = core->compile_model(model, ov::test::utils::DEVICE_GPU, config);
    } else if (import_api == Import_API::IMPORT_EXPORT) {
        auto ifstr = std::ifstream(cache_path, std::ifstream::binary);
        imported_model = core->import_model(ifstr, ov::test::utils::DEVICE_GPU, config_with_weights_path);
        ifstr.close();
    } else {
        OPENVINO_THROW("Unknown import API");
    }

    auto second_cache_path = get_cache_path();
    auto second_mod_time = get_mod_time(second_cache_path);

    // Something went wrong if a new cache is created during the second run.
    ASSERT_EQ(first_mod_time, second_mod_time);

    auto orig_req = compiled_model.create_infer_request();
    auto new_req = imported_model.create_infer_request();

    for (size_t param_idx = 0; param_idx < model->get_parameters().size(); ++param_idx) {
        auto input = model->get_parameters().at(param_idx);
        auto tensor = ov::test::utils::create_and_fill_tensor_real_distribution(input->get_element_type(),
                                                                                input->get_shape(),
                                                                                -100,
                                                                                100,
                                                                                param_idx);
        orig_req.set_tensor(input, tensor);
        new_req.set_tensor(input, tensor);
    }

    orig_req.infer();
    new_req.infer();

    auto result_vector = model->get_results();
    for (auto& res : result_vector) {
        auto orig_out = orig_req.get_tensor(res);
        auto new_out = new_req.get_tensor(res);
        ov::test::utils::compare(orig_out, new_out, inference_mode);
    }
}

TEST_P(CheckWeightlessCacheAccuracy, ReadConcatSplitAssign) {
    OV_ASSERT_NO_THROW(model = ov::test::utils::make_read_concat_split_assign({1, 1, 2, 4}, model_dtype));
    OV_ASSERT_NO_THROW(run());
}

TEST_P(CheckWeightlessCacheAccuracy, SingleConcatWithConstant) {
    OV_ASSERT_NO_THROW(model = ov::test::utils::make_single_concat_with_constant({1, 1, 2, 4}, model_dtype));
    OV_ASSERT_NO_THROW(run());
}

TEST_P(CheckWeightlessCacheAccuracy, TiWithLstmCell) {
    OV_ASSERT_NO_THROW(model = ov::test::utils::make_ti_with_lstm_cell(model_dtype));
    OV_ASSERT_NO_THROW(run());
}

class CheckWeightlessCacheAccuracyLowPrecision : public CheckWeightlessCacheAccuracy {};

TEST_P(CheckWeightlessCacheAccuracyLowPrecision, MatmulWeightsDecompression) {
    ov::test::MatMulDecompressionShapeParams shape_params{{{}, {{1, 4, 16}}}, {1, 16, 32}};
    auto dynShape = shape_params.data_shape.first;
    if (dynShape.rank() == 0) {
        dynShape = shape_params.data_shape.second.front();
    }
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, dynShape)};
    const auto weights_subgraph = ov::test::initMatMulDecompressionSubgraph(shape_params.weights_shape,
                                                                            shape_params.decompression_group_size,
                                                                            ov::element::f32,
                                                                            model_dtype,
                                                                            ov::element::f32,
                                                                            ov::element::dynamic,
                                                                            true,
                                                                            ov::test::DecompressionType::full,
                                                                            ov::test::DecompressionType::full,
                                                                            false);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(params[0], weights_subgraph);

    ov::ResultVector results;
    for (const auto& output : matmul->outputs()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(output));
    }
    model = std::make_shared<ov::Model>(results, params, "MatmulWeightsDecompression");
    OV_ASSERT_NO_THROW(run());
}

const std::vector<Import_API> import_api_types = {
    Import_API::IMPORT_EXPORT,
    Import_API::COMPILE_FILEPATH,
    Import_API::COMPILE_MODEL,
    Import_API::COMPILE_MODEL_WITH_MODEL_STR,
};

const std::vector<ov::element::Type> inference_modes = {
    ov::element::f32,
    ov::element::f16,
};

const std::vector<ov::element::Type> model_dtypes = {
    ov::element::f32,
    ov::element::f16,
    ov::element::bf16,
};

const std::vector<ov::element::Type> low_precision_dtypes = {
    ov::element::u8,
    ov::element::u4,
    ov::element::i4,
};

INSTANTIATE_TEST_SUITE_P(smoke_CheckWeightlessCacheAccuracy,
                         CheckWeightlessCacheAccuracy,
                         ::testing::Combine(::testing::ValuesIn(import_api_types),
                                            ::testing::Bool(),
                                            ::testing::ValuesIn(inference_modes),
                                            ::testing::ValuesIn(model_dtypes)),
                         CheckWeightlessCacheAccuracy::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_CheckWeightlessCacheAccuracyLowPrecision,
                         CheckWeightlessCacheAccuracyLowPrecision,
                         ::testing::Combine(::testing::ValuesIn(import_api_types),
                                            ::testing::Bool(),
                                            ::testing::ValuesIn(inference_modes),
                                            ::testing::ValuesIn(low_precision_dtypes)),
                         CheckWeightlessCacheAccuracy::get_test_case_name);

TEST(smoke_CheckWeightlessCacheAccuracy, ConstantFoldingAttrPropagation) {
    const auto num_elements = 4;
    std::vector<ov::float16> data(num_elements, 0.f);
    auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{num_elements}, data);
    auto attr = ov::WeightlessCacheAttribute(num_elements * sizeof(ov::float16), 0, ov::element::f16);
    constant->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] = attr;

    ov::ParameterVector inputParams;
    ov::ResultVector results;
    ov::pass::Manager manager("ConstantFoldingAttrPropagationTest");
    std::shared_ptr<ov::Model> model = nullptr;

    auto convert_op = std::make_shared<ov::op::v0::Convert>(constant, ov::element::f32);
    results.push_back(std::make_shared<ov::op::v0::Result>(convert_op->output(0)));
    model = std::make_shared<ov::Model>(results, inputParams, "aux");
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.run_passes(model);

    const auto& ops = model->get_ops();
    auto constant_it = std::find_if(ops.begin(), ops.end(), [](const std::shared_ptr<ov::Node>& node) {
        return ov::op::util::is_constant(node);
    });
    ASSERT_NE(constant_it, ops.end());
    ASSERT_NE(constant, *constant_it);
    auto transformed_constant_rt_info = (*constant_it)->get_rt_info();
    auto attr_it = transformed_constant_rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());
    ASSERT_NE(attr_it, transformed_constant_rt_info.end());
    ASSERT_EQ(attr_it->second.as<ov::WeightlessCacheAttribute>().original_size, num_elements * sizeof(ov::float16));
}

}  // namespace
