// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>

#include "base/ov_behavior_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/subgraph_builders/read_concat_split_assign.hpp"
#include "common_test_utils/subgraph_builders/single_concat_with_constant.hpp"
#include "common_test_utils/subgraph_builders/ti_with_lstm_cell.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/pass/serialize.hpp"

namespace {
typedef std::tuple<bool, ov::element::Type, ov::element::Type> testParams;

class CheckWeightlessCacheAccuracy : public ::testing::Test, public ::testing::WithParamInterface<testParams> {
public:
    static std::string get_test_case_name(::testing::TestParamInfo<testParams> obj) {
        bool use_compile_model_api_;
        ov::element::Type inference_mode_;
        ov::element::Type model_dtype_;
        std::tie(use_compile_model_api_, inference_mode_, model_dtype_) = obj.param;

        std::ostringstream result;
        const char separator = '_';
        result << "use_compile_model_api=" << use_compile_model_api_ << separator;
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
    bool use_compile_model_api;  // for loading from cache
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

    std::tie(use_compile_model_api, inference_mode, model_dtype) = GetParam();
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
    auto core = ov::test::utils::PluginCache::get().core();
    ov::pass::Serialize(xml_path, bin_path).run_on_model(model);

    ov::CompiledModel compiled_model;
    compiled_model = core->compile_model(xml_path, ov::test::utils::DEVICE_GPU, config);

    if (!use_compile_model_api) {
        auto ofstr = std::ofstream(cache_path, std::ofstream::binary);
        compiled_model.export_model(ofstr);
        ofstr.close();
    }

    auto ifstr = std::ifstream(cache_path, std::ifstream::binary);
    ov::CompiledModel imported_model;
    if (use_compile_model_api) {
        imported_model = core->compile_model(xml_path, ov::test::utils::DEVICE_GPU, config);
    } else {
        imported_model = core->import_model(ifstr, ov::test::utils::DEVICE_GPU, config_with_weights_path);
    }
    ifstr.close();

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

const std::vector<ov::element::Type> inference_modes = {
    ov::element::f32,
    ov::element::f16,
};

const std::vector<ov::element::Type> model_dtypes = {
    ov::element::f32,
    ov::element::f16,
    ov::element::bf16,
};

INSTANTIATE_TEST_SUITE_P(smoke_CheckWeightlessCacheAccuracy,
                         CheckWeightlessCacheAccuracy,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::ValuesIn(inference_modes),
                                            ::testing::ValuesIn(model_dtypes)),
                         CheckWeightlessCacheAccuracy::get_test_case_name);

}  // namespace
