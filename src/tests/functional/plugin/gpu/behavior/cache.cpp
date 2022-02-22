// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include <ie_core.hpp>
#include <ie_plugin_config.hpp>

class CompiledKernelsCacheTest : public CommonTestUtils::TestsCommon {
protected:
    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::shared_ptr<ngraph::Function> function;
    std::string cache_path;

    void SetUp() override {
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        cache_path = test_name + "_cache";
    }
};

TEST_F(CompiledKernelsCacheTest, CanCreateCacheDirAndDumpBinaries) {
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    // Create CNNNetwork from ngraph::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    std::map<std::string, std::string> config = {{ CONFIG_KEY(CACHE_DIR), cache_path }};
    try {
        // Load CNNNetwork to target plugins
        auto execNet = ie->LoadNetwork(cnnNet, "GPU", config);

        // Check that directory with cached kernels exists after loading network
        ASSERT_TRUE(CommonTestUtils::directoryExists(cache_path)) << "Directory with cached kernels doesn't exist";
        // Check that folder contains cache files and remove them
        ASSERT_GT(CommonTestUtils::removeFilesWithExt(cache_path, "cl_cache"), 0);
        // Remove directory and check that it doesn't exist anymore
        ASSERT_EQ(CommonTestUtils::removeDir(cache_path), 0);
        ASSERT_FALSE(CommonTestUtils::directoryExists(cache_path));
    } catch (std::exception& ex) {
        // Cleanup in case of any exception
        if (CommonTestUtils::directoryExists(cache_path)) {
            ASSERT_GE(CommonTestUtils::removeFilesWithExt(cache_path, "cl_cache"), 0);
            ASSERT_EQ(CommonTestUtils::removeDir(cache_path), 0);
        }
        FAIL() << ex.what() << std::endl;
    }
}

TEST_F(CompiledKernelsCacheTest, TwoNetworksWithSameModelCreatesSameCache) {
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    // Create two CNNNetwork from same ngraph::Function
    InferenceEngine::CNNNetwork cnnNet1(function);
    InferenceEngine::CNNNetwork cnnNet2(function);
    std::map<std::string, std::string> config = {{ CONFIG_KEY(CACHE_DIR), cache_path }};
    try {
        // Load 1st CNNNetwork
        auto execNet1 = ie->LoadNetwork(cnnNet1, "GPU", config);
        auto n_cache_files = CommonTestUtils::listFilesWithExt(cache_path, "cl_cache").size();

        // Check that directory with cached kernels exists after loading network
        ASSERT_TRUE(CommonTestUtils::directoryExists(cache_path)) << "Directory with cached kernels doesn't exist";
        // Load 2nd CNNNetwork
        auto execNet2 = ie->LoadNetwork(cnnNet2, "GPU", config);

        // Check that two loaded networks with same function creates same caches
        ASSERT_EQ(CommonTestUtils::removeFilesWithExt(cache_path, "cl_cache"), n_cache_files);

        // Remove directory and check that it doesn't exist anymore
        ASSERT_EQ(CommonTestUtils::removeDir(cache_path), 0);
        ASSERT_FALSE(CommonTestUtils::directoryExists(cache_path));
    } catch (std::exception& ex) {
        // Cleanup in case of any exception
        if (CommonTestUtils::directoryExists(cache_path)) {
            ASSERT_GE(CommonTestUtils::removeFilesWithExt(cache_path, "cl_cache"), 0);
            ASSERT_EQ(CommonTestUtils::removeDir(cache_path), 0);
        }
        FAIL() << ex.what() << std::endl;
    }
}


#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

TEST_F(CompiledKernelsCacheTest, CanCreateCacheDirAndDumpBinariesUnicodePath) {
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    // Create CNNNetwork from ngraph::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    for (std::size_t testIndex = 0; testIndex < CommonTestUtils::test_unicode_postfix_vector.size(); testIndex++) {
        std::wstring postfix  = L"_" + CommonTestUtils::test_unicode_postfix_vector[testIndex];
        std::wstring cache_path_w = CommonTestUtils::addUnicodePostfixToPath(cache_path, postfix);

        try {
            auto cache_path_mb = ov::util::wstring_to_string(cache_path_w);
            std::map<std::string, std::string> config = {{ CONFIG_KEY(CACHE_DIR), cache_path_mb }};
            // Load CNNNetwork to target plugins
            auto execNet = ie->LoadNetwork(cnnNet, "GPU", config);

            // Check that directory with cached kernels exists after loading network
            ASSERT_TRUE(CommonTestUtils::directoryExists(cache_path_w)) << "Directory with cached kernels doesn't exist";
            // Check that folder contains cache files and remove them
            ASSERT_GT(CommonTestUtils::removeFilesWithExt(cache_path_w, L"cl_cache"), 0);
            // Remove directory and check that it doesn't exist anymore
            ASSERT_EQ(CommonTestUtils::removeDir(cache_path_w), 0);
            ASSERT_FALSE(CommonTestUtils::directoryExists(cache_path_w));
        } catch (std::exception& ex) {
            // Cleanup in case of any exception
            if (CommonTestUtils::directoryExists(cache_path_w)) {
                ASSERT_GE(CommonTestUtils::removeFilesWithExt(cache_path_w, L"cl_cache"), 0);
                ASSERT_EQ(CommonTestUtils::removeDir(cache_path_w), 0);
            }
            FAIL() << ex.what() << std::endl;
        }
    }
}

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
