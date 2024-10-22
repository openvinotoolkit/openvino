// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <future>

#include "auto_func_test.hpp"
#include "common_test_utils/include/common_test_utils/file_utils.hpp"
#include "openvino/pass/serialize.hpp"

using namespace ov::auto_plugin::tests;

TEST_F(AutoFuncTests, can_load_stateful_model_and_syncinfer_single_requests) {
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(
        compiled_model = core.compile_model(model_stateful, "AUTO", {ov::device::priorities("MOCK_GPU", "MOCK_CPU")}));
    ov::InferRequest req1;
    OV_ASSERT_NO_THROW(req1 = compiled_model.create_infer_request());
    auto f1 = std::async(std::launch::async, [&] {
        req1.infer();
    });

    f1.wait();

    OV_ASSERT_NO_THROW(f1.get());
}

TEST_F(AutoFuncTests, can_load_stateful_model_path_and_syncinfer_single_requests) {
    ov::CompiledModel compiled_model;
    std::string filePrefix = ov::test::utils::generateTestFilePrefix();
    auto m_xml_path = filePrefix + ".xml";
    auto m_bin_path = filePrefix + ".bin";
    ov::pass::Serialize(m_xml_path, m_bin_path).run_on_model(model_stateful);
    OV_ASSERT_NO_THROW(compiled_model =
                           core.compile_model(m_xml_path, "AUTO", {ov::device::priorities("MOCK_GPU", "MOCK_CPU")}));
    ov::InferRequest req1;
    OV_ASSERT_NO_THROW(req1 = compiled_model.create_infer_request());
    auto f1 = std::async(std::launch::async, [&] {
        req1.infer();
    });

    f1.wait();

    OV_ASSERT_NO_THROW(f1.get());
    ov::test::utils::removeIRFiles(m_xml_path, m_bin_path);
}

TEST_F(AutoFuncTests, can_load_stateful_model_and_syncinfer_multi_requests) {
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(
        compiled_model = core.compile_model(model_stateful, "AUTO", {ov::device::priorities("MOCK_GPU", "MOCK_CPU")}));
    ov::InferRequest req1, req2, req3;
    OV_ASSERT_NO_THROW(req1 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req2 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req3 = compiled_model.create_infer_request());
    auto f1 = std::async(std::launch::async, [&] {
        req1.infer();
    });
    auto f2 = std::async(std::launch::async, [&] {
        req2.infer();
    });
    auto f3 = std::async(std::launch::async, [&] {
        req3.infer();
    });

    f1.wait();
    f2.wait();
    f3.wait();

    OV_ASSERT_NO_THROW(f1.get());
    OV_ASSERT_NO_THROW(f2.get());
    OV_ASSERT_NO_THROW(f3.get());
}

TEST_F(AutoFuncTests, can_load_stateful_model_path_and_syncinfer_multi_requests) {
    ov::CompiledModel compiled_model;
    std::string filePrefix = ov::test::utils::generateTestFilePrefix();
    auto m_xml_path = filePrefix + ".xml";
    auto m_bin_path = filePrefix + ".bin";
    ov::pass::Serialize(m_xml_path, m_bin_path).run_on_model(model_stateful);
    OV_ASSERT_NO_THROW(compiled_model =
                           core.compile_model(m_xml_path, "AUTO", {ov::device::priorities("MOCK_GPU", "MOCK_CPU")}));
    ov::InferRequest req1, req2, req3;
    OV_ASSERT_NO_THROW(req1 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req2 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req3 = compiled_model.create_infer_request());
    auto f1 = std::async(std::launch::async, [&] {
        req1.infer();
    });
    auto f2 = std::async(std::launch::async, [&] {
        req2.infer();
    });
    auto f3 = std::async(std::launch::async, [&] {
        req3.infer();
    });

    f1.wait();
    f2.wait();
    f3.wait();

    OV_ASSERT_NO_THROW(f1.get());
    OV_ASSERT_NO_THROW(f2.get());
    OV_ASSERT_NO_THROW(f3.get());
    ov::test::utils::removeIRFiles(m_xml_path, m_bin_path);
}

TEST_F(AutoFuncTests, can_load_stateful_model_and_asyncinfer_multi_requests) {
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(
        compiled_model = core.compile_model(model_stateful, "AUTO", {ov::device::priorities("MOCK_GPU", "MOCK_CPU")}));
    ov::InferRequest req1, req2, req3;
    OV_ASSERT_NO_THROW(req1 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req2 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req3 = compiled_model.create_infer_request());
    req1.start_async();
    OV_ASSERT_NO_THROW(req1.wait());

    req2.start_async();
    OV_ASSERT_NO_THROW(req2.wait());

    req3.start_async();
    OV_ASSERT_NO_THROW(req3.wait());
}

TEST_F(AutoFuncTests, can_load_stateful_model_path_and_asyncinfer_multi_requests) {
    ov::CompiledModel compiled_model;
    std::string filePrefix = ov::test::utils::generateTestFilePrefix();
    auto m_xml_path = filePrefix + ".xml";
    auto m_bin_path = filePrefix + ".bin";
    ov::pass::Serialize(m_xml_path, m_bin_path).run_on_model(model_stateful);
    OV_ASSERT_NO_THROW(compiled_model =
                           core.compile_model(m_xml_path, "AUTO", {ov::device::priorities("MOCK_GPU", "MOCK_CPU")}));
    ov::InferRequest req1, req2, req3;
    OV_ASSERT_NO_THROW(req1 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req2 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req3 = compiled_model.create_infer_request());
    req1.start_async();
    OV_ASSERT_NO_THROW(req1.wait());

    req2.start_async();
    OV_ASSERT_NO_THROW(req2.wait());

    req3.start_async();
    OV_ASSERT_NO_THROW(req3.wait());
    ov::test::utils::removeIRFiles(m_xml_path, m_bin_path);
}

TEST_F(AutoFuncTests, can_load_stateful_model_and_asyncinfer_multi_requests_with_cache_enabled) {
    core.set_property(ov::cache_dir(cache_path));
    ov::CompiledModel compiled_model;
    {
        ov::CompiledModel test_compiled_model;
        OV_ASSERT_NO_THROW(
            test_compiled_model =
                core.compile_model(model_stateful, "AUTO", {ov::device::priorities("MOCK_GPU", "MOCK_CPU")}));
    }
    // will only cache model for actual device
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 1);
    OV_ASSERT_NO_THROW(
        compiled_model = core.compile_model(model_stateful, "AUTO", {ov::device::priorities("MOCK_GPU", "MOCK_CPU")}));
    ov::InferRequest req1, req2, req3;
    OV_ASSERT_NO_THROW(req1 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req2 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req3 = compiled_model.create_infer_request());
    req1.start_async();
    OV_ASSERT_NO_THROW(req1.wait());

    req2.start_async();
    OV_ASSERT_NO_THROW(req2.wait());

    req3.start_async();
    OV_ASSERT_NO_THROW(req3.wait());
    // cached model exists for actual device
    // will reuse cached model for actual device without CPU accelerating(No cached model for CPU)
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 1);
    ov::test::utils::removeFilesWithExt(cache_path, "blob");
    core.set_property(ov::cache_dir(""));
}

TEST_F(AutoFuncTests, can_load_stateful_model_path_and_asyncinfer_multi_requests_with_cache_enabled) {
    ov::CompiledModel compiled_model;
    core.set_property(ov::cache_dir(cache_path));
    std::string filePrefix = ov::test::utils::generateTestFilePrefix();
    auto m_xml_path = filePrefix + ".xml";
    auto m_bin_path = filePrefix + ".bin";
    ov::pass::Serialize(m_xml_path, m_bin_path).run_on_model(model_stateful);
    {
        ov::CompiledModel test_compiled_model;
        OV_ASSERT_NO_THROW(
            test_compiled_model =
                core.compile_model(m_xml_path, "AUTO", {ov::device::priorities("MOCK_GPU", "MOCK_CPU")}));
    }
    // will only cache model for actual device
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 1);
    OV_ASSERT_NO_THROW(compiled_model =
                           core.compile_model(m_xml_path, "AUTO", {ov::device::priorities("MOCK_GPU", "MOCK_CPU")}));
    ov::InferRequest req1, req2, req3;
    OV_ASSERT_NO_THROW(req1 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req2 = compiled_model.create_infer_request());
    OV_ASSERT_NO_THROW(req3 = compiled_model.create_infer_request());
    req1.start_async();
    OV_ASSERT_NO_THROW(req1.wait());

    req2.start_async();
    OV_ASSERT_NO_THROW(req2.wait());

    req3.start_async();
    OV_ASSERT_NO_THROW(req3.wait());
    ov::test::utils::removeIRFiles(m_xml_path, m_bin_path);
    // cached model exists for actual device
    // will reuse cached model for actual device without CPU accelerating(No cached model for CPU)
    ASSERT_EQ(ov::test::utils::listFilesWithExt(cache_path, "blob").size(), 1);
    ov::test::utils::removeFilesWithExt(cache_path, "blob");
    core.set_property(ov::cache_dir(""));
}