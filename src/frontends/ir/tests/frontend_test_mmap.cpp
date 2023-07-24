// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"
#include "frontend_test.hpp"
#include "openvino/opsets/opset1.hpp"

class IRFrontendMMapTestsAdvanced : public ::testing::TestWithParam<bool>, public IRFrontendTestsImpl {
protected:
    size_t binsize;

    void SetUp() override {
        size_t CONST_SIZE = 10000000; /*~ 77 MB for f64 to guarantee size is more than size of mapped libraries*/
        auto parameter = std::make_shared<ov::opset1::Parameter>(ov::element::f64, ov::Shape{CONST_SIZE});
        parameter->set_friendly_name("input");
        auto constant = std::make_shared<ov::opset1::Constant>(ov::element::f64,
                                                               ov::Shape{CONST_SIZE},
                                                               std::vector<double>(CONST_SIZE, 0));
        constant->set_friendly_name("value1");
        auto add = std::make_shared<ov::opset1::Add>(parameter, constant);
        add->set_friendly_name("Add");
        auto result = std::make_shared<ov::opset1::Result>(add);
        result->set_friendly_name("output");
        auto model = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{parameter});

        auto filePrefix = ov::test::utils::generateTestFilePrefix();
        xmlFileName = filePrefix + "_IrFrontendTestModel.xml";
        binFileName = filePrefix + "_IrFrontendTestModel.bin";
        ov::serialize(model, xmlFileName, binFileName);
        binsize = ov::test::utils::fileSize(binFileName) / 1024;
    }

    void TearDown() override {
        RemoveTemporalFiles();
    }
};

TEST_P(IRFrontendMMapTestsAdvanced, core_read_and_compile_model) {
    // Test checks that with `mmap` enabled .bin file
    // isn't read into RAM on `read_model` stage and
    // maps into RAM on `compile_model` stage

    bool is_mmap = GetParam();

    // trigger plugin loading in order to map libraries and not affect memory values during compilation
    core.get_versions("CPU");
    core.set_property(ov::enable_mmap(GetParam()));

    auto rss_init = ov::test::utils::getVmRSSInKB();
    auto model = core.read_model(xmlFileName, binFileName);
    auto rss_read = ov::test::utils::getVmRSSInKB();

    // RAM should increase on at least 90% of .bin size. 10% is a proposed error that cover file system cache
    size_t bin_in_RAM = binsize * 0.9;
    bool is_weights_read = (rss_read - rss_init) > bin_in_RAM;
    EXPECT_TRUE(is_mmap != is_weights_read);

    auto rss_mapped_read = ov::test::utils::getRssFileInKB();
    auto compiled_model = core.compile_model(model, "CPU");
    auto rss_mapped_compiled = ov::test::utils::getRssFileInKB();

    // Mappings size (RssFile) should increase at least on .bin size
    bool is_weights_mapped = (rss_mapped_compiled - rss_mapped_read) > binsize;
    EXPECT_TRUE(is_mmap == is_weights_mapped);
}

TEST_P(IRFrontendMMapTestsAdvanced, fe_read_and_compile_model) {
    // Test checks that with `mmap` enabled .bin file
    // isn't read into RAM on `read_model` stage and
    // maps into RAM on `compile_model` stage

    bool is_mmap = GetParam();

    ov::frontend::InputModel::Ptr input_model;
    std::shared_ptr<ov::Model> model;

    // trigger plugin loading in order to map libraries and not affect memory values during compilation
    core.get_versions("CPU");
    core.set_property(ov::enable_mmap(GetParam()));

    auto rss_init = ov::test::utils::getVmRSSInKB();
    ov::AnyVector params{xmlFileName, binFileName, is_mmap};
    auto FE = manager.load_by_model(params);
    if (FE)
        input_model = FE->load(params);
    if (input_model)
        model = FE->convert(input_model);
    auto rss_read = ov::test::utils::getVmRSSInKB();

    // RAM should (or not if `mmap` enabled) increase at least on 90% of .bin size.
    // 10% is a proposed error that cover file system cache
    size_t bin_in_RAM = binsize * 0.9;
    bool is_weights_read = (rss_read - rss_init) > bin_in_RAM;
    EXPECT_TRUE(is_mmap != is_weights_read);

    auto rss_mapped_read = ov::test::utils::getRssFileInKB();
    auto compiled_model = core.compile_model(model, "CPU");
    auto rss_mapped_compiled = ov::test::utils::getRssFileInKB();

    // Mappings size (RssFile) should (or not if `mmap` disabled) increase at least on .bin size
    bool is_weights_mapped = (rss_mapped_compiled - rss_mapped_read) > binsize;
    EXPECT_TRUE(is_mmap == is_weights_mapped);
}

INSTANTIATE_TEST_SUITE_P(EnableMMapPropery, IRFrontendMMapTestsAdvanced, ::testing::Bool());
