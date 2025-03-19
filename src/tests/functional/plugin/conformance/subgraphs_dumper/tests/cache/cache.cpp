// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "openvino/op/ops.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/openvino.hpp"

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"

#include "cache/cache.hpp"
#include "op_conformance_utils/meta_info/meta_info.hpp"
#include "op_conformance_utils/utils/file.hpp"
#include "utils/cache.hpp"

#include "base_test.hpp"

namespace {

class ICacheUnitTest : public SubgraphsDumperBaseTest,
                       public virtual ov::tools::subgraph_dumper::ICache {
protected:
    std::shared_ptr<ov::Model> test_model;
    ov::conformance::MetaInfo test_meta;
    std::string test_model_path, model_name;
    std::string test_artifacts_dir;

    void SetUp() override {
        SubgraphsDumperBaseTest::SetUp();
        model_name = "test_model";
        test_artifacts_dir = "test_artifacts";
        test_model_path = ov::util::path_join({ test_artifacts_dir, model_name + ".xml" }).string();
        ov::util::create_directory_recursive(test_artifacts_dir);
        {
            auto params = ov::ParameterVector {
                std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::PartialShape{1, 1, 1, 1}),
            };
            // params->begin()->set_friendly_name("in_0");
            auto convert = std::make_shared<ov::op::v0::Convert>(params.front(), ov::element::f16);
            convert->set_friendly_name("convert_0");
            test_model = std::make_shared<ov::Model>(convert, params);
            test_model->set_friendly_name(model_name);
        }
        test_meta = ov::conformance::MetaInfo(test_model_path, {{"in_0", ov::conformance::InputInfo({1, 2}, 0, 1, true)}});
    }

    void TearDown() override {
        ov::test::utils::removeDir(test_artifacts_dir);
    }
};

TEST_F(ICacheUnitTest, set_serialization_dir) {
    OV_ASSERT_NO_THROW(this->set_serialization_dir(test_artifacts_dir));
    ASSERT_EQ(test_artifacts_dir, this->m_serialization_dir);
}

TEST_F(ICacheUnitTest, update_cache) {
    OV_ASSERT_NO_THROW(this->update_cache(test_model, test_model_path));
    OV_ASSERT_NO_THROW(this->update_cache(test_model, test_model_path, true));
    OV_ASSERT_NO_THROW(this->update_cache(test_model, test_model_path, false));
}

TEST_F(ICacheUnitTest, serialize_cache) {
    OV_ASSERT_NO_THROW(this->serialize_cache());
}

TEST_F(ICacheUnitTest, serialize_model) {
    std::pair<std::shared_ptr<ov::Model>, ov::conformance::MetaInfo> graph_info({ test_model, test_meta });
    ASSERT_TRUE(this->serialize_model(graph_info, test_artifacts_dir));
    auto xml_path = test_model_path;
    auto bin_path = ov::util::replace_extension(test_model_path, "bin");
    auto meta_path = ov::util::replace_extension(test_model_path, "meta");
    try {
        if (!ov::util::file_exists(xml_path) ||
            !ov::util::file_exists(bin_path)) {
            throw std::runtime_error("Model was not serilized!");
        }
        if (!ov::util::file_exists(meta_path)) {
            throw std::runtime_error("Meta was not serilized!");
        }
        auto serialized_model = ov::util::core->read_model(xml_path, bin_path);
        auto res = compare_functions(test_model, serialized_model, true, true, true, true, true, true);
        if (!res.first) {
            throw std::runtime_error("Serialized and runtime model are not equal!");
        }
    } catch(std::exception& e) {
        ov::test::utils::removeFile(xml_path);
        ov::test::utils::removeFile(bin_path);
        ov::test::utils::removeFile(meta_path);
        GTEST_FAIL() << e.what() << std::endl;
    }
}

TEST_F(ICacheUnitTest, is_model_large_to_read) {
    this->mem_size = 0;
    OV_ASSERT_NO_THROW(this->is_model_large_to_read(test_model, test_model_path));
    ASSERT_TRUE(this->is_model_large_to_read(test_model, test_model_path));
    this->mem_size = 1 << 30;
    OV_ASSERT_NO_THROW(this->is_model_large_to_read(test_model, test_model_path));
    ASSERT_FALSE(this->is_model_large_to_read(test_model, test_model_path));
}

TEST_F(ICacheUnitTest, is_model_large_to_store_const) {
    this->mem_size = 0;
    OV_ASSERT_NO_THROW(this->is_model_large_to_store_const(test_model));
    ASSERT_TRUE(this->is_model_large_to_store_const(test_model));
    this->mem_size = 1 << 30;
    OV_ASSERT_NO_THROW(this->is_model_large_to_store_const(test_model));
    ASSERT_FALSE(this->is_model_large_to_store_const(test_model));
}

}  // namespace
