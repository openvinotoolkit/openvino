// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "pugixml.hpp"

#include "openvino/openvino.hpp"
#include "openvino/util/file_util.hpp"

#include "common_test_utils/file_utils.hpp"

#include "op_conformance_utils/meta_info/meta_info.hpp"
#include "op_conformance_utils/utils/file.hpp"

#include "base_test.hpp"

namespace {

using namespace ov::conformance;

// ======================== Input Info Unit tests =============================================

class InputInfoUnitTest : public SubgraphsDumperBaseTest {};

TEST_F(InputInfoUnitTest, constructor) {
    OV_ASSERT_NO_THROW(auto in_info = InputInfo());
    OV_ASSERT_NO_THROW(auto in_info = InputInfo({10}));
    OV_ASSERT_NO_THROW(auto in_info = InputInfo({}, 0));
    OV_ASSERT_NO_THROW(auto in_info = InputInfo({}, 0, 1));
    OV_ASSERT_NO_THROW(auto in_info = InputInfo({}, 0, 1, true));
}

TEST_F(InputInfoUnitTest, update_ranges) {
    auto in_info_0 = InputInfo();
    auto in_info_1 = InputInfo({}, 0);
    in_info_0 = in_info_1;
    ASSERT_EQ(in_info_0.ranges.min, in_info_1.ranges.min);
    ASSERT_EQ(in_info_0.ranges.max, in_info_1.ranges.max);
    ASSERT_EQ(in_info_0.is_const, in_info_1.is_const);

    auto in_info_2 = InputInfo({}, 1, 2);
    auto ref_in_info = InputInfo({}, 0, 2);
    in_info_0 = in_info_2;
    ASSERT_EQ(in_info_0.ranges.min, ref_in_info.ranges.min);
    ASSERT_EQ(in_info_0.ranges.max, ref_in_info.ranges.max);
    ASSERT_EQ(in_info_0.is_const, ref_in_info.is_const);
}

TEST_F(InputInfoUnitTest, update_shapes) {
    auto in_info_0 = InputInfo({10});
    ASSERT_EQ(in_info_0.min_shape, ov::PartialShape({10}));
    ASSERT_EQ(in_info_0.max_shape, ov::PartialShape({10}));
    auto in_info_1 = InputInfo({20});
    in_info_0 = in_info_1;
    ASSERT_EQ(in_info_0.min_shape, ov::PartialShape({10}));
    ASSERT_EQ(in_info_1.max_shape, ov::PartialShape({20}));
}

// ======================== Model Info Func tests =============================================

class ModelInfoFuncTest : public ::testing::Test {};

TEST_F(ModelInfoFuncTest, constructor) {
    OV_ASSERT_NO_THROW(auto model_info = ModelInfo());
    OV_ASSERT_NO_THROW(auto model_info = ModelInfo("model.xml"));
    OV_ASSERT_NO_THROW(auto model_info = ModelInfo("model.xml", 1));
    OV_ASSERT_NO_THROW(auto model_info = ModelInfo("model.xml", 1, 2));
    OV_ASSERT_NO_THROW(auto model_info = ModelInfo("model.xml", 1, 2, 3));
}

// ======================== Meta Info Functional tests =============================================

class MetaInfoFuncTest : public SubgraphsDumperBaseTest {
protected:
    std::string test_model_path, test_model_name;
    std::map<std::string, InputInfo> test_in_info;
    std::map<std::string, ModelInfo> test_model_info;
    std::string test_artifacts_dir, test_extractor_name;

    void SetUp() override {
        SubgraphsDumperBaseTest::SetUp();
        test_model_path = "test_model_path.xml";
        test_extractor_name = "test_extractor";
        test_model_name = ov::util::replace_extension(test_model_path, "");
        test_in_info = {{ "test_in_0", InputInfo({10}, DEFAULT_MIN_VALUE, 1, true) }};
        test_model_info = {{ test_model_name, ModelInfo(test_model_path, 5) }};
        test_artifacts_dir = ov::util::path_join({ov::test::utils::getCurrentWorkingDir(), "test_artifacts"});
        ov::util::create_directory_recursive(test_artifacts_dir);
    }

    void TearDown() override {
        ov::test::utils::removeDir(test_artifacts_dir);
    }
};

TEST_F(MetaInfoFuncTest, constructor) {
    OV_ASSERT_NO_THROW(auto meta = MetaInfo());
    OV_ASSERT_NO_THROW(auto meta = MetaInfo(test_model_name));
    OV_ASSERT_NO_THROW(auto meta = MetaInfo(test_model_name, test_in_info));
    OV_ASSERT_NO_THROW(auto meta = MetaInfo(test_model_name, test_in_info, 2));
    OV_ASSERT_NO_THROW(auto meta = MetaInfo(test_model_name, test_in_info, 3, 1, test_extractor_name));
    OV_ASSERT_NO_THROW(auto meta = MetaInfo(test_model_name, test_in_info, 3, 5, test_extractor_name, 5));
}

TEST_F(MetaInfoFuncTest, get_input_info) {
    auto test_meta = MetaInfo(test_model_name, test_in_info);
    OV_ASSERT_NO_THROW(test_meta.get_input_info());
    ASSERT_EQ(test_meta.get_input_info(), test_in_info);
}

TEST_F(MetaInfoFuncTest, get_model_info) {
    auto test_meta = MetaInfo(test_model_path, test_in_info, 5);
    OV_ASSERT_NO_THROW(test_meta.get_model_info());
    ASSERT_EQ(test_meta.get_model_info(), test_model_info);
}

TEST_F(MetaInfoFuncTest, get_any_extractor) {
    auto test_meta = MetaInfo(test_model_path, test_in_info, 5, 3, test_extractor_name);
    OV_ASSERT_NO_THROW(test_meta.get_any_extractor());
    ASSERT_EQ(test_meta.get_any_extractor(), test_extractor_name);
}

TEST_F(MetaInfoFuncTest, update) {
    std::map<std::string, InputInfo> test_in_info = {{ "test_in_0", InputInfo({10}, DEFAULT_MIN_VALUE, 1, true) }};
    auto test_meta = MetaInfo(test_model_name, test_in_info, 1, 1, test_extractor_name);
    ASSERT_EQ(test_meta.get_input_info().at("test_in_0").min_shape, ov::PartialShape({10}));
    ASSERT_EQ(test_meta.get_input_info().at("test_in_0").max_shape, ov::PartialShape({10}));
    std::map<std::string, InputInfo> test_input_info_1 = {{ "test_in_0", InputInfo({50}, 0, 1, true) }};
    std::string test_model_1 = "test_model_1";
    std::string test_model_path_1 = ov::util::path_join({ "path", "to",  test_model_1 + ".xml"});
    ASSERT_ANY_THROW(test_meta.update(test_model_path_1, {}));
    ASSERT_ANY_THROW(test_meta.update(test_model_path_1, {{ "test_in_1", InputInfo({10}) }}));
    ASSERT_ANY_THROW(test_meta.update(test_model_path_1, {{ "test_in_0", InputInfo({10}, 0, 1, false) }}));
    OV_ASSERT_NO_THROW(test_meta.update(test_model_path_1, test_input_info_1));
    ASSERT_EQ(test_meta.get_input_info().at("test_in_0").min_shape, ov::PartialShape({10}));
    ASSERT_EQ(test_meta.get_input_info().at("test_in_0").max_shape, ov::PartialShape({50}));
    OV_ASSERT_NO_THROW(test_meta.update(test_model_path_1, test_input_info_1, 1, 2, "test_extractor_1"));
    OV_ASSERT_NO_THROW(test_meta.update(test_model_path_1, test_input_info_1, 2));
    OV_ASSERT_NO_THROW(test_meta.update(test_model_path_1, test_input_info_1, 2, 4, "test"));
}

TEST_F(MetaInfoFuncTest, serialize) {
    auto test_meta = MetaInfo(test_model_name, test_in_info);
    std::string seriliazation_path(ov::util::path_join({test_artifacts_dir, "test_meta.meta"}));
    test_meta.serialize(seriliazation_path);
    ASSERT_TRUE(ov::util::file_exists(seriliazation_path));
}

// ======================== Meta Info Unit tests =============================================

class MetaInfoUnitTest : public MetaInfoFuncTest,
                         public virtual MetaInfo {
protected:
    void SetUp() override {
        MetaInfoFuncTest::SetUp();
        this->input_info = test_in_info;
        this->model_info = test_model_info;
    }
};

TEST_F(MetaInfoUnitTest, serialize) {
    std::string seriliazation_path(ov::util::path_join({test_artifacts_dir, "test_meta.meta"}));
    this->extractors = { "extractor_0", "extractor_1" };
    this->serialize(seriliazation_path);
    ASSERT_TRUE(ov::util::file_exists(seriliazation_path));

    pugi::xml_document doc;
    doc.load_file(seriliazation_path.c_str());
    {
        auto models_xml = doc.child("meta_info").child("models");
        for (const auto model_xml : models_xml.children()) {
            auto model_name_xml = std::string(model_xml.attribute("name").value());
            ASSERT_NE(model_info.find(model_name_xml), model_info.end());
            ASSERT_EQ(model_info[model_name_xml].this_op_cnt, model_xml.attribute("this_op_count").as_uint());
            ASSERT_EQ(model_info[model_name_xml].total_op_cnt, model_xml.attribute("total_op_count").as_uint());
            auto paths = model_info[model_name_xml].model_paths;
            for (const auto& path_xml : model_xml.child("path")) {
                auto path_xml_value = std::string(path_xml.attribute("path").value());
                ASSERT_NE(std::find(paths.begin(), paths.end(), path_xml_value), paths.end());
            }
        }
    }
    {
        auto graph_priority_xml = doc.child("meta_info").child("graph_priority").attribute("value").as_double();
        ASSERT_EQ(graph_priority_xml, this->get_graph_priority());
    }
    {
        auto input_info_xml = doc.child("meta_info").child("input_info");
        for (const auto& in_info_xml : input_info_xml.children()) {
            auto in_xml = std::string(in_info_xml.attribute("id").value());
            ASSERT_NE(input_info.find(in_xml), input_info.end());
            ASSERT_EQ(input_info[in_xml].is_const, in_info_xml.attribute("convert_to_const").as_bool());
            auto min_xml = std::string(in_info_xml.attribute("min").value()) == "undefined" ? DEFAULT_MIN_VALUE : in_info_xml.attribute("min").as_double();
            ASSERT_EQ(input_info[in_xml].ranges.min, min_xml);
            auto max_xml = std::string(in_info_xml.attribute("max").value()) == "undefined" ? DEFAULT_MAX_VALUE : in_info_xml.attribute("max").as_double();
            ASSERT_EQ(input_info[in_xml].ranges.max, max_xml);
            auto max_shape_str = std::string(in_info_xml.attribute("max_shape").value());
            auto max_shape_ref = ov::test::utils::partialShape2str({this->get_input_info().begin()->second.max_shape});
            ASSERT_EQ(max_shape_str, max_shape_ref);
            auto min_shape_str = std::string(in_info_xml.attribute("min_shape").value());
            auto min_shape_ref = ov::test::utils::partialShape2str({this->get_input_info().begin()->second.min_shape});
            ASSERT_EQ(min_shape_str, min_shape_ref);
        }
    }
    {
        auto extractors_node = doc.child("meta_info").child("extractors");
        std::unordered_set<std::string> xml_extractors;
        for (const auto& in_info_xml : extractors_node.children()) {
            xml_extractors.insert(std::string(in_info_xml.attribute("name").value()));
        }
        ASSERT_EQ(xml_extractors, this->extractors);
    }
}

TEST_F(MetaInfoUnitTest, read_meta_from_file) {
    std::string seriliazation_path(ov::util::path_join({test_artifacts_dir, "test_meta.meta"}));
    this->extractors = { "extractor_0", "extractor_1" };
    this->serialize(seriliazation_path);
    auto new_meta = MetaInfo::read_meta_from_file(seriliazation_path);
    ASSERT_TRUE(this->extractors.count(new_meta.get_any_extractor()));
    ASSERT_EQ(new_meta.get_input_info(), this->input_info);
    ASSERT_EQ(new_meta.get_model_info(), this->model_info);
}

TEST_F(MetaInfoUnitTest, update) {
    auto test_meta = MetaInfo(test_model_name, test_in_info);
    std::map<std::string, InputInfo> test_meta_1 = {{ "test_in_0", InputInfo({20}, 0, 1, true) }};
    std::string test_model_1 = "test_model_1";
    std::string test_model_path_1 = ov::util::path_join({ "path", "to",  test_model_1 + ".xml"});
    OV_ASSERT_NO_THROW(this->update(test_model_path_1, test_meta_1));
    ASSERT_NE(this->model_info.find(test_model_1), this->model_info.end());
    ASSERT_EQ(*this->model_info[test_model_1].model_paths.begin(), test_model_path_1);
    ASSERT_EQ(this->model_info[test_model_1].this_op_cnt, 1);
    ASSERT_EQ(this->input_info.begin()->second.min_shape, ov::PartialShape({10}));
    ASSERT_EQ(this->input_info.begin()->second.max_shape, ov::PartialShape({20}));
    OV_ASSERT_NO_THROW(this->update(test_model_path_1, test_meta_1));
    ASSERT_EQ(this->model_info[test_model_1].model_paths.size(), 1);
    ASSERT_EQ(this->model_info[test_model_1].this_op_cnt, 2);
    ASSERT_EQ(this->input_info.begin()->second.min_shape, ov::PartialShape({10}));
    ASSERT_EQ(this->input_info.begin()->second.max_shape, ov::PartialShape({20}));
    test_model_path_1 = ov::util::path_join({ "path", "to", "test", test_model_1 + ".xml"});
    OV_ASSERT_NO_THROW(this->update(test_model_path_1, test_meta_1, 0, 1, "test_extractor"));
    ASSERT_EQ(this->model_info[test_model_1].model_paths.size(), 2);
    ASSERT_EQ(this->model_info[test_model_1].this_op_cnt, 3);
    ASSERT_EQ(this->model_info[test_model_1].this_op_cnt, 3);
    ASSERT_EQ(this->extractors, std::unordered_set<std::string>({"test_extractor"}));
}

TEST_F(MetaInfoUnitTest, get_model_name_by_path) {
    OV_ASSERT_NO_THROW(this->get_model_name_by_path(test_model_path));
    auto name = this->get_model_name_by_path(test_model_path);
    ASSERT_EQ(name, test_model_name);
}

TEST_F(MetaInfoUnitTest, get_graph_priority) {
    auto meta = MetaInfo(test_model_name, test_in_info);
    this->update(test_model_name, meta.get_input_info());
    OV_ASSERT_NO_THROW(this->get_abs_graph_priority());
    OV_ASSERT_NO_THROW(this->get_graph_priority());
    ASSERT_TRUE(this->get_graph_priority() >= 0 && this->get_graph_priority() <= 1);
}

TEST_F(MetaInfoUnitTest, get_any_extractor) {
    auto meta = MetaInfo(test_model_name, test_in_info, 1, 1, "test_extractor");
    OV_ASSERT_NO_THROW(meta.get_any_extractor());
    ASSERT_EQ(meta.get_any_extractor(), "test_extractor");
}

}  // namespace