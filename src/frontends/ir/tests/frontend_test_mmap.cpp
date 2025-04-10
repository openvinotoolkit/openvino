// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"
#include "frontend_test.hpp"
#include "openvino/opsets/opset1.hpp"

#ifndef __APPLE__  // TODO: add getVmRSSInKB() for Apple platform

class IRFrontendMMapTestsAdvanced : public ::testing::Test, public IRFrontendTestsImpl {
protected:
    size_t binsize, REF_RSS;

    void SetUp() override {
        size_t SIZE_MB = 32;
        size_t CONST_SIZE = SIZE_MB * 1024 * 1024 / sizeof(ov::element::f32);
        auto parameter = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{CONST_SIZE});
        auto constant = std::make_shared<ov::opset1::Constant>(ov::element::f32,
                                                               ov::Shape{CONST_SIZE},
                                                               std::vector<float>(CONST_SIZE, 0));
        auto add = std::make_shared<ov::opset1::Add>(parameter, constant);
        auto result = std::make_shared<ov::opset1::Result>(add);
        auto model = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{parameter});

        auto filePrefix = ov::test::utils::generateTestFilePrefix();
        xmlFileName = filePrefix + "_IrFrontendTestModel.xml";
        binFileName = filePrefix + "_IrFrontendTestModel.bin";
        ov::serialize(model, xmlFileName);
        binsize = ov::test::utils::fileSize(binFileName) / 1024;

        // In case of enabled `mmap` RAM should not increase more than 50% of .bin size
        // Otherwise RAM should increase on at least 50% of .bin size
        REF_RSS = binsize / 2;
    }

    void TearDown() override {
        RemoveTemporalFiles();
    }
};

TEST_F(IRFrontendMMapTestsAdvanced, core_enable_mmap_property) {
    // Test checks that with  enabled `mmap` .bin file
    // isn't read into RAM on `read_model` stage.
    // Otherwise, with disabled `mmap` .bin file should
    // be in RAM

    auto test = [&](const bool& is_mmap) {
        core.set_property(ov::enable_mmap(is_mmap));

        auto rss_init = ov::test::utils::getVmRSSInKB();
        auto model = core.read_model(xmlFileName);
        auto rss_read = ov::test::utils::getVmRSSInKB();

        if (is_mmap != core.get_property("", ov::enable_mmap)) {
            std::cout << "Test failed: core property is not set correctly" << std::endl;
            exit(1);
        }

        bool is_weights_read = (rss_read - rss_init) > REF_RSS;
        if (is_mmap == is_weights_read) {
            std::cerr << "Test failed: mmap is " << (is_mmap ? "enabled" : "disabled") << ", but weights are "
                      << (is_weights_read ? "read" : "not read") << " in RAM" << std::endl;
            exit(1);
        }
        std::cerr << "Test passed" << std::endl;
        exit(0);
    };

    for (const auto is_mmap : {true, false})
        // Run test in a separate process to not affect RAM values by previous tests
        EXPECT_EXIT(test(is_mmap), ::testing::ExitedWithCode(0), "Test passed");
}

TEST_F(IRFrontendMMapTestsAdvanced, core_enable_mmap_property_user_config) {
    // Test checks that with  enabled `mmap` .bin file
    // isn't read into RAM on `read_model` stage.
    // Otherwise, with disabled `mmap` .bin file should
    // be in RAM

    auto test = [&](const bool& is_mmap) {
        auto rss_init = ov::test::utils::getVmRSSInKB();
        auto model = core.read_model(xmlFileName, {}, {{ov::enable_mmap(is_mmap)}});
        auto rss_read = ov::test::utils::getVmRSSInKB();

        if (true != core.get_property("", ov::enable_mmap)) {
            std::cout << "Test failed: core property changed by user configuration" << std::endl;
            exit(1);
        }

        bool is_weights_read = (rss_read - rss_init) > REF_RSS;
        if (is_mmap == is_weights_read) {
            std::cerr << "Test failed: mmap is " << (is_mmap ? "enabled" : "disabled") << ", but weights are "
                      << (is_weights_read ? "read" : "not read") << " in RAM" << std::endl;
            exit(1);
        }
        std::cerr << "Test passed" << std::endl;
        exit(0);
    };

    for (const auto is_mmap : {true, false})
        // Run test in a separate process to not affect RAM values by previous tests
        EXPECT_EXIT(test(is_mmap), ::testing::ExitedWithCode(0), "Test passed");
}

TEST_F(IRFrontendMMapTestsAdvanced, fe_read_ir_by_default) {
    // Test checks that IR FE `read` IR by default,
    // so .bin file should be loaded to RAM

    auto test = [&]() {
        ov::frontend::InputModel::Ptr input_model;
        std::shared_ptr<ov::Model> model;

        auto rss_init = ov::test::utils::getVmRSSInKB();
        auto FE = manager.load_by_model(xmlFileName);
        if (FE)
            input_model = FE->load(xmlFileName);
        if (input_model)
            model = FE->convert(input_model);
        auto rss_read = ov::test::utils::getVmRSSInKB();

        bool is_weights_read = (rss_read - rss_init) > REF_RSS;
        if (!is_weights_read) {
            std::cerr << "Test failed: weights are not read; RAM consumption is less than expected" << std::endl;
            exit(1);
        }
        std::cerr << "Test passed" << std::endl;
        exit(0);
    };

    // Run test in a separate process to not affect RAM values by previous tests
    ASSERT_EXIT(test(), ::testing::ExitedWithCode(0), "Test passed");
}

TEST_F(IRFrontendMMapTestsAdvanced, core_mmap_ir_by_default) {
    // Test checks that Core uses `mmap` by default,
    // so .bin file should not be loaded to RAM

    auto test = [&]() {
        auto rss_init = ov::test::utils::getVmRSSInKB();
        auto model = core.read_model(xmlFileName, binFileName);
        auto rss_read = ov::test::utils::getVmRSSInKB();

        bool is_weights_mapped = (rss_read - rss_init) < REF_RSS;
        if (!is_weights_mapped) {
            std::cerr << "Test failed: weights are not mapped; RAM consumption is more than expected" << std::endl;
            exit(1);
        }
        std::cerr << "Test passed" << std::endl;
        exit(0);
    };

    // Run test in a separate process to not affect RAM values by previous tests
    ASSERT_EXIT(test(), ::testing::ExitedWithCode(0), "Test passed");
}

#endif
