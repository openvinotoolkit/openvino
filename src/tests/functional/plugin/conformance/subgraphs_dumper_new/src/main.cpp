// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gflag_config.hpp"
#include "cache/op_cache.hpp"
#include "utils/model_utils.hpp"

// todo:: iefode: remove using
using namespace ov::tools::subgraph_dumper;

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return 0;
    }
    // SubgraphsDumper::ClonersMap::constant_size_threshold_mb = FLAGS_constants_size_threshold;

    std::vector<std::string> local_cache_dirs = CommonTestUtils::splitStringByDelimiter(FLAGS_local_cache);
    std::vector<std::string> dirs = CommonTestUtils::splitStringByDelimiter(FLAGS_input_folders);

    std::vector<std::string> models;

    if (!CommonTestUtils::directoryExists(FLAGS_output_folder)) {
        std::string msg = "Output directory (" + FLAGS_output_folder + ") doesn't not exist! The directory will be created.";
        std::cout << msg << std::endl;
        CommonTestUtils::createDirectoryRecursive(FLAGS_output_folder);
    }
    try {
        models = find_models(dirs, FLAGS_path_regex);
    } catch (std::runtime_error& e) {
        std::cout << "Try 'subgraphdumper -h' for more information. \nException: " << e.what() << std::endl;
        return 1;
    }

    std::shared_ptr<OpCache> cache = OpCache::get();
    cache->set_serialization_dir(FLAGS_output_folder);
    std::map<ModelCacheStatus, std::vector<std::string>> cache_model_status;
    // Upload previously cached graphs to cache
    if (!FLAGS_local_cache.empty()) {
        auto cachedOps = find_models(local_cache_dirs);
        cache_model_status = cache_models(cache, cachedOps, FLAGS_extract_body);
    }
    {
        auto tmp_cache_model_status = cache_models(cache, models, FLAGS_extract_body);
        cache_model_status.insert(tmp_cache_model_status.begin(), tmp_cache_model_status.end());
    }
    cache->serialize_cache();
    save_model_status_to_file(cache_model_status, FLAGS_output_folder);
    return cache_model_status[ModelCacheStatus::NOT_FULLY_CACHED].empty() && cache_model_status[ModelCacheStatus::NOT_READ].empty();
}
