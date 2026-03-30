// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/model_cache.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/subgraph_builders/read_concat_split_assign.hpp"
#include "common_test_utils/subgraph_builders/single_concat_with_constant.hpp"
#include "common_test_utils/subgraph_builders/ti_with_lstm_cell.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/runtime/weightless_properties_utils.hpp"
#include "openvino/util/codec_xor.hpp"
#include "shared_test_classes/subgraph/weights_decompression_params.hpp"
#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"

namespace ov {
namespace test {
namespace behavior {

std::string WeightlessCacheAccuracy::get_test_case_name(const ::testing::TestParamInfo<WeightlessCacheAccuracyTestParams>& obj) {
    std::ostringstream result;

    result << "do_encryption="   << utils::bool2str(std::get<0>(obj.param));
    result << "_inference_mode=" << std::get<1>(obj.param);
    result << "_model_dtype="    << std::get<2>(obj.param);
    result << "_config=";
    for (const auto& [name, value] : std::get<3>(obj.param)) {
        result << name << "[" << value.as<std::string>() << "]|";
    }
    result << "_device=" << std::get<4>(obj.param);

    return result.str();
}

void WeightlessCacheAccuracy::SetUp() {
    const std::string file_prefix = ov::test::utils::generateTestFilePrefix();
    m_xml_path   = file_prefix + ".xml";
    m_bin_path   = file_prefix + ".bin";
    m_cache_path = file_prefix + ".blob";
    m_cache_dir_ir = file_prefix + "_cache_dir_ir";
    m_cache_dir_model = file_prefix + "_cache_dir_model";

    std::tie(m_do_encryption, m_inference_mode, m_model_dtype, std::ignore, m_target_device) =
        GetParam();
}

void WeightlessCacheAccuracy::TearDown() {
    std::filesystem::remove(m_xml_path);
    std::filesystem::remove(m_bin_path);
    std::filesystem::remove(m_cache_path);

    std::filesystem::remove_all(m_cache_dir_ir);
    std::filesystem::remove_all(m_cache_dir_model);
}

void WeightlessCacheAccuracy::run() {
    ov::AnyMap config = {hint::inference_precision(m_inference_mode)};
    for (const auto& property : std::get<3>(GetParam())) {
        config.insert(property);
    }
    if (m_do_encryption) {
        config.insert(cache_encryption_callbacks(EncryptionCallbacks{ov::util::codec_xor, ov::util::codec_xor}));
    }

    auto core = ov::test::utils::PluginCache::get().core();
    ov::pass::Serialize(m_xml_path, m_bin_path).run_on_model(m_model);

    auto get_modification_time = [](const std::filesystem::path& cache_dir) {
        const static std::filesystem::path blob_ext{".blob"};
        std::filesystem::file_time_type result;
        uint64_t counter{0};
        for (auto const& dir_entry : std::filesystem::directory_iterator{cache_dir}) {
            if (dir_entry.path().extension() == blob_ext) {
                result = dir_entry.last_write_time();
                counter++;
            }
        }
        EXPECT_EQ(counter, 1);
        return result;
    };

    std::unordered_map<std::shared_ptr<ov::Node>, ov::Tensor> inputs;
    for (size_t param_idx = 0; param_idx < m_model->get_parameters().size(); param_idx++) {
        auto input = m_model->get_parameters().at(param_idx);
        auto tensor = ov::test::utils::create_and_fill_tensor_real_distribution(input->get_element_type(),
                                                                                input->get_shape(),
                                                                                -100,
                                                                                100,
                                                                                param_idx);
        inputs.insert({input, tensor});
    }

    auto result_vector = m_model->get_results();
    std::vector<ov::Tensor> outputs_ref, outputs_ir, outputs_model, output_imported_strm;

    using CompileModelFn = std::function<ov::CompiledModel(const ov::AnyMap&)>;
    auto compile_model_and_infer = [&](std::vector<ov::Tensor>& outputs, const std::filesystem::path& cache_dir, CompileModelFn core_compile_model) {
        auto config_new = config;
        config_new.insert(ov::cache_dir(cache_dir.string()));
        // Compile model and create cache.
        auto compiled_model_1 = core_compile_model(config_new);
        auto first_mod_time = get_modification_time(cache_dir);

        // Load cached model
        auto compiled_model_2 = core_compile_model(config_new);

        auto second_mod_time = get_modification_time(cache_dir);

        // Something went wrong if a new cache is created during the second run.
        ASSERT_EQ(first_mod_time, second_mod_time);

        // Inference
        auto infer_request = compiled_model_2.create_infer_request();
        for (const auto& input : inputs) {
            infer_request.set_tensor(input.first, input.second);
        }
        infer_request.infer();
        for (const auto& output : result_vector) {
            outputs.push_back(infer_request.get_tensor(output));
        }
    };

    std::exception_ptr reference_error, ir_error, model_error, import_error;
    /////////////////////////////////////////////
    std::thread t_reference([&] {
        try {
            auto compiled_model = core->compile_model(m_xml_path, m_target_device, config);
            auto infer_request = compiled_model.create_infer_request();
            for (const auto& input : inputs) {
                infer_request.set_tensor(input.first, input.second);
            }
            infer_request.infer();
            for (const auto& output : result_vector) {
                outputs_ref.push_back(infer_request.get_tensor(output));
            }
        } catch (...) {
            reference_error = std::current_exception();
        }
    });
    /////////////////////////////////////////////
    std::thread t_ir([&] {
        try {
            compile_model_and_infer(outputs_ir, m_cache_dir_ir, [&](const ov::AnyMap& config){
                return core->compile_model(m_xml_path, m_target_device, config);
            });
        } catch (...) {
            ir_error = std::current_exception();
        }
    });
    /////////////////////////////////////////////
    std::thread t_model([&] {
        try {
            compile_model_and_infer(outputs_model, m_cache_dir_model, [&](const ov::AnyMap& config) {
                auto readed_model = core->read_model(m_xml_path);
                return core->compile_model(readed_model, m_target_device, config);
            });
        } catch (...) {
            model_error = std::current_exception();
        }
    });
    /////////////////////////////////////////////
    std::thread t_import([&] {
        try {
            {
                auto config_new = config;
                auto compiled_model = core->compile_model(m_xml_path, m_target_device, config_new);
                auto ofstr = std::ofstream(m_cache_path, std::ofstream::binary);
                compiled_model.export_model(ofstr);
                ofstr.close();
            }
            auto first_mod_time = std::filesystem::last_write_time(m_cache_path);

            ov::CompiledModel compiled_model_2;
            {
                auto config_with_weights_path = config;
                if (ov::util::is_weightless_enabled(config).value_or(false)) {
                    config_with_weights_path.insert(ov::weights_path(m_bin_path.string()));
                }
                config_with_weights_path.erase(ov::cache_dir.name());

                auto ifstr = std::ifstream(m_cache_path, std::ifstream::binary);
                compiled_model_2 = core->import_model(ifstr, m_target_device, config_with_weights_path);
                ifstr.close();
            }

            auto second_mod_time = std::filesystem::last_write_time(m_cache_path);

            // Something went wrong if a new cache is created during the second run.
            ASSERT_EQ(first_mod_time, second_mod_time);

            auto infer_request = compiled_model_2.create_infer_request();
            for (const auto& input : inputs) {
                infer_request.set_tensor(input.first, input.second);
            }
            infer_request.infer();
            for (const auto& output : result_vector) {
                output_imported_strm.push_back(infer_request.get_tensor(output));
            }
        } catch (...) {
            import_error = std::current_exception();
        }
    });
    /////////////////////////////////////////////
    
    t_reference.join();
    t_ir.join();
    t_model.join();
    t_import.join();

    if (reference_error) {
        std::rethrow_exception(reference_error);
    }
    if (ir_error) {
        std::rethrow_exception(ir_error);
    }
    if (model_error) {
        std::rethrow_exception(model_error);
    }
    if (import_error) {
        std::rethrow_exception(import_error);
    }

    for (size_t i = 0UL; i < outputs_ref.size(); i++) {
        ov::test::utils::compare(outputs_ref[i], outputs_ir[i], m_inference_mode);
        ov::test::utils::compare(outputs_ref[i], outputs_model[i], m_inference_mode);
        ov::test::utils::compare(outputs_ref[i], output_imported_strm[i], m_inference_mode);
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
    const auto weights_subgraph =
        ov::test::utils::initMatMulDecompressionSubgraph(shape_params.weights_shape,
                                                         shape_params.decompression_group_size,
                                                         ov::element::f32,
                                                         m_model_dtype,
                                                         ov::element::f32,
                                                         ov::element::dynamic,
                                                         true,
                                                         ov::test::utils::DecompressionType::full,
                                                         ov::test::utils::DecompressionType::full,
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
