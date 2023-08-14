// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gflag_config.hpp"
#include "cache/op_cache.hpp"
#include "cache/graph_cache.hpp"
#include "utils/model.hpp"

using namespace ov::tools::subgraph_dumper;

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return 0;
    }

    std::vector<std::string> local_cache_dirs = ov::test::utils::splitStringByDelimiter(FLAGS_local_cache);
    std::vector<std::string> dirs = ov::test::utils::splitStringByDelimiter(FLAGS_input_folders);

    std::vector<std::string> models;
    std::map<ModelCacheStatus, std::vector<std::string>> cache_model_status;

    if (!ov::test::utils::directoryExists(FLAGS_output_folder)) {
        std::string msg = "Output directory (" + FLAGS_output_folder + ") doesn't not exist! The directory will be created.";
        std::cout << msg << std::endl;
        ov::test::utils::createDirectoryRecursive(FLAGS_output_folder);
    }
    try {
        auto all_models = find_models(dirs, FLAGS_path_regex);
        models = all_models.first;
        cache_model_status.insert(all_models.second);
    } catch (std::runtime_error& e) {
        std::cout << "[ INFO ] Try 'subgraphsDumper -h' for more information. \nException: " << e.what() << std::endl;
        return 1;
    }

    std::vector<std::shared_ptr<ICache>> caches;
    if (FLAGS_cache_type == "OP" || FLAGS_cache_type.empty()) {
        std::cout << "[ INFO ] OpCache is enabled!" << std::endl;
        caches.push_back(OpCache::get());
    }
    if (FLAGS_cache_type == "GRAPH" || FLAGS_cache_type.empty()) {
        std::cout << "[ INFO ] GraphCache is enabled!" << std::endl;
        caches.push_back(GraphCache::get());
    }

    for (auto& cache : caches) {
        cache->set_serialization_dir(FLAGS_output_folder);
    }
    // Upload previously cached graphs to cache
    if (!FLAGS_local_cache.empty()) {
        auto cached_ops = find_models(local_cache_dirs);
        // todo: add normal caching with meta info reading
        auto this_cache_model_status = cache_models(caches, cached_ops.first, FLAGS_extract_body, true);
        auto not_read_model = cached_ops.second;
        for (auto& model_status : cache_model_status) {
            auto& key = model_status.first;
            auto& value = model_status.second;
            if (not_read_model.first == key) {
                value.insert(value.end(), not_read_model.second.begin(), not_read_model.second.end());
            }
            if (this_cache_model_status.count(key)) {
                value.insert(value.end(), this_cache_model_status[key].begin(), this_cache_model_status[key].end());
            }
        }
    }
    {
        auto this_cache_model_status = cache_models(caches, models, FLAGS_extract_body);
        for (auto& model_status : cache_model_status) {
            auto& key = model_status.first;
            auto& value = model_status.second;
            if (this_cache_model_status.count(key)) {
                value.insert(value.end(), this_cache_model_status[key].begin(), this_cache_model_status[key].end());
            }
        }
    }

    // for (auto& cache : caches) {
    //     cache->serialize_cache();
    //     cache->reset_cache();
    // }

    save_model_status_to_file(cache_model_status, FLAGS_output_folder);
    return cache_model_status[ModelCacheStatus::NOT_FULLY_CACHED].empty() && cache_model_status[ModelCacheStatus::NOT_READ].empty();
}
