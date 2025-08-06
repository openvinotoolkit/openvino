// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <exception>
#include <memory>

#include "behavior/ov_infer_request/inference.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model_util.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/op.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "overload/overload_test_utils_npu.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

using testing::_;
using ::testing::AllOf;
using ::testing::HasSubstr;

namespace ov {
namespace test {
namespace behavior {

class WeightsSeparationTests : public ov::test::behavior::OVPluginTestBase,
                               public testing::WithParamInterface<CompilationParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CompilationParams> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
        targetDevice = ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();

        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();
    }

    std::string generateCacheDirName(const std::string& test_name) {
        using namespace std::chrono;
        // Generate unique file names based on test name, thread id and timestamp
        // This allows execution of tests in parallel (stress mode)
        auto hash = std::to_string(std::hash<std::string>()(test_name));
        std::stringstream ss;
        auto ts = duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch());
        ss << hash << "_" << "_" << ts.count();
        return ss.str();
    }

    void TearDown() override {
        if (!m_cache_dir.empty()) {
            core->set_property({ov::cache_dir()});
            core.reset();
            ov::test::utils::PluginCache::get().reset();
            ov::test::utils::removeFilesWithExt(m_cache_dir, "blob");
            ov::test::utils::removeDir(m_cache_dir);
        }

        if (!model_path.empty()) {
            utils::removeFile(model_path + ".xml");
            utils::removeFile(model_path + ".bin");
        }

        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }

        APIBaseTest::TearDown();
    }

    std::shared_ptr<ov::Model> createTestModel(const bool addWeightlessCacheAttribute = true) {
        constexpr auto precision = element::f32;

        auto weights = std::make_shared<op::v0::Constant>(element::f32, Shape{5}, std::vector<float>{1.0f});
        auto input = std::make_shared<op::v0::Parameter>(precision, Shape{1});
        auto add = std::make_shared<op::v1::Add>(input, weights);

        weights->set_friendly_name("weights");
        input->set_friendly_name("input");
        add->set_friendly_name("add");

        if (addWeightlessCacheAttribute) {
            weights->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
                ov::WeightlessCacheAttribute(weights->get_byte_size(), 0, weights->get_element_type());
        }

        auto model = std::make_shared<Model>(OutputVector{add}, ParameterVector{input}, "Simple with weights");
        ov::util::set_tensors_names(AUTO, *model, {}, {{0, {"add"}}});
        return model;
    }

    Tensor from_stream(std::istream& stream, const size_t size) {
        ov::Tensor t(element::from<char>(), Shape{size});
        stream.read(t.data<char>(), size);
        return t;
    }

    void create_infer_request_and_check_result() {
        const ov::Tensor expected =
            utils::create_tensor(element::f32, Shape{5}, std::vector<float>{3.0f, 5.0f, 3.0f, 3.0f, 3.0f});
        const ov::Tensor input = utils::create_tensor(element::f32, Shape{1}, std::vector<float>{2.0f});
        OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
        OV_ASSERT_NO_THROW(inference_request.set_tensor("input", input));
        OV_ASSERT_NO_THROW(inference_request.infer());

        const ov::Tensor output = inference_request.get_tensor("add");
        OV_ASSERT_NO_THROW(utils::compare(expected, output));
    }

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest inference_request;
    std::string m_cache_dir;
    std::string model_path;
};

TEST_P(WeightsSeparationTests, CheckOneShotVersionThrows) {
    model = createTestModel();
    configuration.insert(ov::intel_npu::weightless_blob(true));
    configuration.insert(ov::intel_npu::separate_weights_version(ov::intel_npu::WSVersion::ONE_SHOT));
    OV_EXPECT_THROW(compiled_model = core->compile_model(model, target_device, configuration), ov::Exception, _);
}

TEST_P(WeightsSeparationTests, CheckForFailureNoWeightlessCacheAttribute) {
    model = createTestModel(false);
    configuration.insert(ov::intel_npu::weightless_blob(true));
    configuration.insert(ov::intel_npu::separate_weights_version(ov::intel_npu::WSVersion::ITERATIVE));
    OV_EXPECT_THROW(compiled_model = core->compile_model(model, target_device, configuration), ov::Exception, _);
}

TEST_P(WeightsSeparationTests, CorrectInferenceResultNoImport) {
    model = createTestModel();
    configuration.insert(ov::intel_npu::weightless_blob(true));
    configuration.insert(ov::intel_npu::separate_weights_version(ov::intel_npu::WSVersion::ITERATIVE));

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(model, target_device, configuration));
    ASSERT_TRUE(compiled_model);
    EXPECT_FALSE(compiled_model.get_property(ov::loaded_from_cache));

    create_infer_request_and_check_result();
}

TEST_P(WeightsSeparationTests, CorrectInferenceResultIfCachingUsed) {
    m_cache_dir = generateCacheDirName(GetTestName());
    core->set_property({ov::cache_dir(m_cache_dir)});

    model = createTestModel();
    configuration.emplace(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
    configuration.insert(ov::intel_npu::separate_weights_version(ov::intel_npu::WSVersion::ITERATIVE));

    compiled_model = core->compile_model(model, target_device, configuration);
    ASSERT_TRUE(compiled_model);
    EXPECT_FALSE(compiled_model.get_property(ov::loaded_from_cache));

    compiled_model = core->compile_model(model, target_device, configuration);
    ASSERT_TRUE(compiled_model);
    EXPECT_TRUE(compiled_model.get_property(ov::loaded_from_cache));

    create_infer_request_and_check_result();
}

TEST_P(WeightsSeparationTests, CorrectInferenceResultIfModelHintUsed) {
    model = createTestModel();
    configuration.insert(ov::intel_npu::weightless_blob(true));
    configuration.insert(ov::intel_npu::separate_weights_version(ov::intel_npu::WSVersion::ITERATIVE));

    compiled_model = core->compile_model(model, target_device, configuration);
    ASSERT_TRUE(compiled_model);

    std::stringstream export_stream;
    compiled_model.export_model(export_stream);

    configuration.insert(ov::hint::model(model));
    compiled_model = core->import_model(export_stream, target_device, configuration);
    ASSERT_TRUE(compiled_model);

    create_infer_request_and_check_result();
}

TEST_P(WeightsSeparationTests, CorrectInferenceResultIfWeightsPathUsed) {
    model = createTestModel();

    model_path = ov::util::path_join({utils::getCurrentWorkingDir(), utils::generateTestFilePrefix()}).string();
    ov::serialize(model, model_path + ".xml", model_path + ".bin");

    configuration.insert(ov::intel_npu::weightless_blob(true));
    configuration.insert(ov::intel_npu::separate_weights_version(ov::intel_npu::WSVersion::ITERATIVE));
    compiled_model = core->compile_model(model, target_device, configuration);
    ASSERT_TRUE(compiled_model);

    std::stringstream export_stream;
    compiled_model.export_model(export_stream);

    configuration.insert(ov::weights_path(model_path + ".bin"));
    compiled_model = core->import_model(export_stream, target_device, configuration);
    ASSERT_TRUE(compiled_model);

    create_infer_request_and_check_result();
}

TEST_P(WeightsSeparationTests, CorrectInferenceResultIfTensorImported) {
    model = createTestModel();

    configuration.insert(ov::intel_npu::weightless_blob(true));
    configuration.insert(ov::intel_npu::separate_weights_version(ov::intel_npu::WSVersion::ITERATIVE));
    compiled_model = core->compile_model(model, target_device, configuration);
    ASSERT_TRUE(compiled_model);

    std::stringstream export_stream;
    compiled_model.export_model(export_stream);
    ov::Tensor exported_model = from_stream(export_stream, export_stream.str().size());

    configuration.insert(ov::hint::model(model));
    compiled_model = core->import_model(exported_model, target_device, configuration);
    ASSERT_TRUE(compiled_model);

    create_infer_request_and_check_result();
}

TEST_P(WeightsSeparationTests, ImportFailureIfNoWeightsProvided) {
    model = createTestModel();

    configuration.insert(ov::intel_npu::weightless_blob(true));
    configuration.insert(ov::intel_npu::separate_weights_version(ov::intel_npu::WSVersion::ITERATIVE));
    compiled_model = core->compile_model(model, target_device, configuration);
    ASSERT_TRUE(compiled_model);

    std::stringstream export_stream;
    compiled_model.export_model(export_stream);

    OV_EXPECT_THROW(core->import_model(export_stream, target_device, configuration), ov::Exception, _);
}

TEST_P(WeightsSeparationTests, ModelHintHasPriorityOverWeightsPath) {
    model = createTestModel();

    configuration.insert(ov::intel_npu::weightless_blob(true));
    configuration.insert(ov::intel_npu::separate_weights_version(ov::intel_npu::WSVersion::ITERATIVE));
    compiled_model = core->compile_model(model, target_device, configuration);
    ASSERT_TRUE(compiled_model);

    std::stringstream export_stream;
    compiled_model.export_model(export_stream);

    configuration.insert(ov::hint::model(model));
    configuration.insert(ov::weights_path("this_file_does_not_exist"));
    compiled_model = core->import_model(export_stream, target_device, configuration);
    ASSERT_TRUE(compiled_model);

    create_infer_request_and_check_result();
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
