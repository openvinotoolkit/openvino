// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/compiled_model_base.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/model_util.hpp"

namespace ov::test::behavior {

using testing::_;

namespace utils = ::ov::test::utils;

const std::vector<ov::AnyMap> configs = {
    {},
};
const std::vector<ov::AnyMap> swPluginConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledModelBaseTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(configs)),
                         OVCompiledModelBaseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledModelBaseTestOptional,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(configs)),
                         OVCompiledModelBaseTestOptional::getTestCaseName);

namespace {
using op::v0::Parameter, op::v0::Constant, op::v1::Add;

std::shared_ptr<Model> make_model_with_weights(const ov::Tensor& w) {
    constexpr auto precision = element::f32;

    auto weights = std::make_shared<Constant>(w);
    auto input = std::make_shared<Parameter>(precision, Shape{1});
    auto add = std::make_shared<Add>(input, weights);

    weights->set_friendly_name("weights");
    input->set_friendly_name("input");
    add->set_friendly_name("add");

    auto model = std::make_shared<Model>(OutputVector{add}, ParameterVector{input}, "Simple with weights");
    util::set_tensors_names(AUTO, *model, {}, {{0, {"add"}}});
    return model;
}

std::shared_ptr<Model> make_model_with_weights() {
    constexpr auto precision = element::f32;

    auto weights = std::make_shared<Constant>(element::f32, Shape{5}, std::vector<float>{1.0f});
    auto input = std::make_shared<Parameter>(precision, Shape{1});
    auto add = std::make_shared<Add>(input, weights);

    weights->set_friendly_name("weights");
    input->set_friendly_name("input");
    add->set_friendly_name("add");

    auto model = std::make_shared<Model>(OutputVector{add}, ParameterVector{input}, "Simple with weights");
    util::set_tensors_names(AUTO, *model, {}, {{0, {"add"}}});
    return model;
}
}  // namespace

TEST_P(OVCompiledModelBaseTest, import_from_weightless_blob) {
    const auto w_file_path =
        ov::util::path_join({utils::getCurrentWorkingDir(), utils::generateTestFilePrefix() + "_weights.bin"});
    // TEMPLATE will export model as weightless blob.
    configuration.emplace(ov::weights_path(w_file_path.string()));

    std::stringstream export_stream;
    {
        auto model = make_model_with_weights();
        auto compiled_model = core->compile_model(model, target_device, configuration);
        ASSERT_FALSE(!compiled_model);
        compiled_model.export_model(export_stream);
    }

    OV_EXPECT_THROW(core->import_model(export_stream, target_device), ov::Exception, _);

    utils::removeFile(w_file_path.string());
}

TEST_P(OVCompiledModelBaseTest, compile_from_regular_blob) {
    std::stringstream export_stream;
    auto compiled_model_ref = core->compile_model(make_model_with_weights(), target_device, configuration);
    ASSERT_FALSE(!compiled_model_ref);
    compiled_model_ref.export_model(export_stream);

    ov::Tensor expected, output;
    auto input = utils::create_tensor(element::f32, Shape{1}, std::vector<float>{2.0f});
    // Infer reference model
    {
        auto infer_request = compiled_model_ref.create_infer_request();
        infer_request.set_tensor("input", input);
        infer_request.infer();
        expected = infer_request.get_tensor("add");
    }
    // Infer compiled from stream
    {
        configuration.emplace(ov::blob_stream(export_stream));
        auto empty_model = std::make_shared<Model>(OutputVector{}, ParameterVector{}, "Empty model");
        auto model_from_stream = core->compile_model(empty_model, target_device, configuration);
        auto infer_request_import = model_from_stream.create_infer_request();
        infer_request_import.set_tensor("input", input);
        infer_request_import.infer();
        output = infer_request_import.get_tensor("add");
    }

    utils::compare(expected, output);
}

TEST_P(OVCompiledModelBaseTest, compile_from_weightless_blob) {
    auto weights = utils::create_tensor(element::f32, Shape{5}, std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    std::stringstream export_stream;
    auto w_file_path =
        ov::util::path_join({utils::getCurrentWorkingDir(), utils::generateTestFilePrefix() + "_weights.bin"});

    // add weights the TEMPLATE will export model as weightless blob.
    configuration.emplace(ov::weights_path(w_file_path.string()));
    {
        auto model = make_model_with_weights(weights);
        auto compiled_model_ref = core->compile_model(model, target_device, configuration);
        ASSERT_FALSE(!compiled_model_ref);
        compiled_model_ref.export_model(export_stream);
    }

    auto expected = utils::create_tensor(element::f32, Shape{5}, std::vector<float>{3.0f, 5.0f, 3.0f, 3.0f, 3.0f});
    {
        // store weights in file, not same as in orignal model model
        auto w = utils::create_tensor(element::f32, Shape{5}, std::vector<float>{1.0f, 3.0f, 1.0f, 1.0f, 1.0f});
        auto w_file = std::ofstream(w_file_path, std::ios::binary);
        w_file.write(reinterpret_cast<const char*>(w.data()), w.get_byte_size());
    }

    ov::Tensor output;
    auto input = utils::create_tensor(element::f32, Shape{1}, std::vector<float>{2.0f});
    // Infer compiled from weightless stream
    {
        configuration.emplace(ov::blob_stream(export_stream));
        auto empty_model = std::make_shared<Model>(OutputVector{}, ParameterVector{}, "Empty model");
        auto import_model = core->compile_model(empty_model, target_device, configuration);
        auto infer_request_import = import_model.create_infer_request();
        infer_request_import.set_tensor("input", input);
        infer_request_import.infer();
        output = infer_request_import.get_tensor("add");
    }

    utils::compare(expected, output);
    utils::removeFile(w_file_path.string());
}

TEST_P(OVCompiledModelBaseTest, compile_from_weightless_blob_but_no_weights) {
    std::stringstream export_stream;
    auto w_file_path =
        ov::util::path_join({utils::getCurrentWorkingDir(), utils::generateTestFilePrefix() + "_weights.bin"});

    // add weights the TEMPLATE will export model as weightless blob.
    configuration.emplace(ov::weights_path(w_file_path.string()));
    auto model = make_model_with_weights();
    auto compiled_model_ref = core->compile_model(model, target_device, configuration);
    ASSERT_FALSE(!compiled_model_ref);
    compiled_model_ref.export_model(export_stream);

    auto expected = utils::create_tensor(element::f32, Shape{5}, std::vector<float>{3.0f, 3.0f, 3.0f, 3.0f, 3.0f});
    ov::Tensor output;
    auto input = utils::create_tensor(element::f32, Shape{1}, std::vector<float>{2.0f});
    // Infer compiled from weightless stream and no weights use original model
    {
        configuration.emplace(ov::blob_stream(export_stream));
        auto import_model = core->compile_model(model, target_device, configuration);
        auto infer_request_import = import_model.create_infer_request();
        infer_request_import.set_tensor("input", input);
        infer_request_import.infer();
        output = infer_request_import.get_tensor("add");
    }

    utils::compare(expected, output);
    utils::removeFile(w_file_path.string());
}

TEST_P(OVCompiledModelBaseTest, compile_from_cached_weightless_blob_use_hint) {
    auto cache_dir = ov::util::path_join({utils::getCurrentWorkingDir(), "cache"});
    auto w_file_path = ov::util::path_join({cache_dir, utils::generateTestFilePrefix() + "_weights.bin"});
    {
        // store weights in file, not same as in orignal model model
        utils::createDirectory(cache_dir);
        auto w = utils::create_tensor(element::f32, Shape{5}, std::vector<float>{1.0f, 3.0f, 1.0f, 1.0f, 1.0f});
        auto w_file = std::ofstream(w_file_path, std::ios::binary);
        w_file.write(reinterpret_cast<const char*>(w.data()), w.get_byte_size());
    }

    // add weights the TEMPLATE will export model as weightless blob.
    configuration.emplace(ov::weights_path(w_file_path.string()));
    configuration.emplace(ov::cache_dir(cache_dir));
    auto model = make_model_with_weights();

    {
        auto compiled_model = core->compile_model(model, target_device, configuration);
        ASSERT_FALSE(!compiled_model);
        EXPECT_FALSE(compiled_model.get_property(ov::loaded_from_cache.name()).as<bool>());
    }
    {
        auto compiled_model = core->compile_model(model, target_device, configuration);
        ASSERT_FALSE(!compiled_model);
        EXPECT_TRUE(compiled_model.get_property(ov::loaded_from_cache.name()).as<bool>());
    }

    std::filesystem::remove_all(cache_dir);
}

TEST_P(OVCompiledModelBaseTest, compile_from_cached_weightless_blob_no_hint) {
    auto cache_dir = ov::util::path_join({utils::getCurrentWorkingDir(), "cache"});
    auto w_file_path = ov::util::path_join({cache_dir, utils::generateTestFilePrefix() + "_weights.bin"});
    {
        // store weights in file, not same as in orignal model model
        utils::createDirectory(cache_dir);
        auto w = utils::create_tensor(element::f32, Shape{5}, std::vector<float>{1.0f, 3.0f, 1.0f, 1.0f, 1.0f});
        auto w_file = std::ofstream(w_file_path, std::ios::binary);
        w_file.write(reinterpret_cast<const char*>(w.data()), w.get_byte_size());
    }

    // add weights the TEMPLATE will export model as weightless blob.
    configuration.emplace(ov::cache_dir(cache_dir));
    auto model = make_model_with_weights();

    {
        auto cfg_with_hint = configuration;
        cfg_with_hint.emplace(ov::weights_path(w_file_path.string()));
        auto compiled_model = core->compile_model(model, target_device, configuration);
        ASSERT_FALSE(!compiled_model);
        EXPECT_FALSE(compiled_model.get_property(ov::loaded_from_cache.name()).as<bool>());
    }
    {
        auto compiled_model = core->compile_model(model, target_device, configuration);
        ASSERT_FALSE(!compiled_model);
        EXPECT_TRUE(compiled_model.get_property(ov::loaded_from_cache.name()).as<bool>());
    }

    std::filesystem::remove_all(cache_dir);
}

TEST_P(OVCompiledModelBaseTest, compile_from_cached_weightless_blob_but_no_weights) {
    auto cache_dir = ov::util::Path{utils::getCurrentWorkingDir()} / "cache";
    auto w_file_path = cache_dir / (utils::generateTestFilePrefix() + "_weights.bin");

    // add weights the TEMPLATE will export model as weightless blob.
    configuration.emplace(ov::weights_path(w_file_path.string()));
    configuration.emplace(ov::cache_dir(cache_dir));
    auto model = make_model_with_weights();

    {
        auto compiled_model = core->compile_model(model, target_device, configuration);
        ASSERT_FALSE(!compiled_model);
        EXPECT_FALSE(compiled_model.get_property(ov::loaded_from_cache.name()).as<bool>());
    }
    {
        // model not loaded from cache as no weights on path
        auto compiled_model = core->compile_model(model, target_device, configuration);
        ASSERT_FALSE(!compiled_model);
        EXPECT_FALSE(compiled_model.get_property(ov::loaded_from_cache.name()).as<bool>());
    }

    std::filesystem::remove_all(cache_dir);
}
}  // namespace ov::test::behavior
