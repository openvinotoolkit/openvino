// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/model_cache.hpp"

#include "common_test_utils/subgraph_builders/read_concat_split_assign.hpp"
#include "common_test_utils/subgraph_builders/single_concat_with_constant.hpp"
#include "common_test_utils/subgraph_builders/ti_with_lstm_cell.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/util/codec_xor.hpp"
#include "shared_test_classes/subgraph/weights_decompression_builders.hpp"

namespace ov {
namespace test {
namespace behavior {

std::string WeightlessCacheAccuracy::get_test_case_name(const ::testing::TestParamInfo<WeightlessCacheAccuracyTestParams>& obj) {
    std::ostringstream result;

    result << "use_compile_model_api=" << utils::bool2str(std::get<0>(obj.param));
    result << "_do_encryption="        << utils::bool2str(std::get<1>(obj.param));
    result << "_inference_mode="       << std::get<2>(obj.param);
    result << "_model_dtype="          << std::get<3>(obj.param);
    result << "_device="               << std::get<4>(obj.param);

    return result.str();
}

void WeightlessCacheAccuracy::SetUp() {
    std::string filePrefix = ov::test::utils::generateTestFilePrefix();
    m_xml_path = filePrefix + ".xml";
    m_bin_path = filePrefix + ".bin";
    m_cache_path = filePrefix + ".blob";
    m_cache_dir = filePrefix + "_cache_dir";

    std::tie(m_use_compile_model_api, m_do_encryption, m_inference_mode, m_model_dtype, m_target_device) = GetParam();
}

void WeightlessCacheAccuracy::TearDown() {
    std::remove(m_xml_path.c_str());
    std::remove(m_bin_path.c_str());
    std::remove(m_cache_path.c_str());

    ov::test::utils::removeFilesWithExt(m_cache_dir, "blob");
    ov::test::utils::removeFilesWithExt(m_cache_dir, "cl_cache");
    ov::test::utils::removeDir(m_cache_dir);
}

void WeightlessCacheAccuracy::run() {
    ov::AnyMap config = {ov::cache_dir(m_cache_dir),
                         ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE),
                         ov::hint::inference_precision(m_inference_mode)};
    ov::AnyMap config_with_weights_path = {ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE),
                                           ov::weights_path(m_bin_path),
                                           ov::hint::inference_precision(m_inference_mode)};

    if (m_do_encryption) {
        ov::EncryptionCallbacks encryption_callbacks;
        encryption_callbacks.encrypt = ov::util::codec_xor;
        encryption_callbacks.decrypt = ov::util::codec_xor;
        config.insert(ov::cache_encryption_callbacks(encryption_callbacks));
        config_with_weights_path.insert(ov::cache_encryption_callbacks(encryption_callbacks));
    }
    auto core = ov::test::utils::PluginCache::get().core();
    ov::pass::Serialize(m_xml_path, m_bin_path).run_on_model(m_model);

    auto compiled_model = core->compile_model(m_xml_path, m_target_device, config);

    if (!m_use_compile_model_api) {
        auto ofstr = std::ofstream(m_cache_path, std::ofstream::binary);
        compiled_model.export_model(ofstr);
        ofstr.close();
    }

    auto get_cache_path = [&]() {
        std::string path;
        if (m_use_compile_model_api) {
            auto blobs = ov::test::utils::listFilesWithExt(m_cache_dir, "blob");
            EXPECT_EQ(blobs.size(), 1);
            path = blobs[0];
        } else {
            path = m_cache_path;
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
    if (m_use_compile_model_api) {
        imported_model = core->compile_model(m_xml_path, m_target_device, config);
    } else {
        auto ifstr = std::ifstream(m_cache_path, std::ifstream::binary);
        imported_model = core->import_model(ifstr, m_target_device, config_with_weights_path);
        ifstr.close();
    }

    auto second_cache_path = get_cache_path();
    auto second_mod_time = get_mod_time(second_cache_path);

    // Something went wrong if a new cache is created during the second run.
    ASSERT_EQ(first_mod_time, second_mod_time);

    auto orig_req = compiled_model.create_infer_request();
    auto new_req = imported_model.create_infer_request();

    for (size_t param_idx = 0; param_idx < m_model->get_parameters().size(); ++param_idx) {
        auto input = m_model->get_parameters().at(param_idx);
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

    auto result_vector = m_model->get_results();
    for (auto& res : result_vector) {
        auto orig_out = orig_req.get_tensor(res);
        auto new_out = new_req.get_tensor(res);
        ov::test::utils::compare(orig_out, new_out, m_inference_mode);
    }
}

TEST_P(WeightlessCacheAccuracy, ReadConcatSplitAssign) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    OV_ASSERT_NO_THROW(m_model = ov::test::utils::make_read_concat_split_assign({1, 1, 2, 4}, m_model_dtype));
    OV_ASSERT_NO_THROW(run());
}

TEST_P(WeightlessCacheAccuracy, SingleConcatWithConstant) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    OV_ASSERT_NO_THROW(m_model = ov::test::utils::make_single_concat_with_constant({1, 1, 2, 4}, m_model_dtype));
    OV_ASSERT_NO_THROW(run());
}

TEST_P(WeightlessCacheAccuracy, TiWithLstmCell) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    OV_ASSERT_NO_THROW(m_model = ov::test::utils::make_ti_with_lstm_cell(m_model_dtype));
    OV_ASSERT_NO_THROW(run());
}

TEST_P(WeightlessCacheAccuracyLowPrecision, MatmulWeightsDecompression) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::test::MatMulDecompressionShapeParams shape_params{{{}, {{1, 4, 16}}}, {1, 16, 32}};
    auto dynShape = shape_params.data_shape.first;
    if (dynShape.rank() == 0) {
        dynShape = shape_params.data_shape.second.front();
    }
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, dynShape)};
    const auto weights_subgraph = ov::test::initMatMulDecompressionSubgraph(shape_params.weights_shape,
                                                                            shape_params.decompression_group_size,
                                                                            ov::element::f32,
                                                                            m_model_dtype,
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
    m_model = std::make_shared<ov::Model>(results, params, "MatmulWeightsDecompression");
    OV_ASSERT_NO_THROW(run());
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
