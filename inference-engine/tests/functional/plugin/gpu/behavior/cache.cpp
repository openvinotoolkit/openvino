// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_common.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include <ie_core.hpp>
#include <cldnn/cldnn_config.hpp>

#include "w_dirent.h"

class CompiledKernelsCacheTest : public CommonTestUtils::TestsCommon {
protected:
    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::shared_ptr<ngraph::Function> function;
    std::string cache_path;

    void SetUp() override {
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        cache_path = test_name + "_cache/";
    }

    void TearDown() override {
        std::remove(cache_path.c_str());
    }
};

inline std::string fileExt(const std::string &filename) {
    auto pos = filename.rfind('.');
    if (pos == std::string::npos)
        return {};
    return filename.substr(pos + 1);
}

TEST_F(CompiledKernelsCacheTest, CanCreateCacheDirAndDumpBinaries) {
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    std::map<std::string, std::string> config = {{ CLDNN_CONFIG_KEY(KERNELS_CACHE_DIR), cache_path }};
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, "GPU", config);

    // Check that directory with cached kernels exists after loading network
    auto cache_fp = testing::internal::FilePath(cache_path.c_str());
    ASSERT_TRUE(cache_fp.DirectoryExists()) << "Directory with cached kernels doesn't exist";

    // Try to open directory and get all files from it
    DIR *dir;
    struct dirent *ent;
    ASSERT_TRUE((dir = opendir(cache_path.c_str())) != NULL);
    bool compiledKernelsAreFound = false;
    while ((ent = readdir(dir)) != NULL) {
        auto file_fp = testing::internal::FilePath::ConcatPaths(cache_fp, testing::internal::FilePath(ent->d_name));
        struct stat stat_path;
        stat(file_fp.c_str(), &stat_path);
        if (!S_ISDIR(stat_path.st_mode)) {
            bool validExtension = fileExt(file_fp.string()) == "cl_cache";
            // Check that this folder contains at least 1 binary file with cl cache
            compiledKernelsAreFound |= validExtension;
            // Cleanup. Remove cache files
            if (validExtension)
                ASSERT_EQ(std::remove(file_fp.c_str()), 0) << errno;
        }
    }
    closedir(dir);

    ASSERT_TRUE(compiledKernelsAreFound);

    // Remove directory and check that it doesn't exist anymore
    ASSERT_EQ(testing::internal::posix::RmDir(cache_path.c_str()), 0);
    ASSERT_FALSE(cache_fp.DirectoryExists());
}
