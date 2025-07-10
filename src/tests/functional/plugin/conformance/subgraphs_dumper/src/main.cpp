// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gflag_config.hpp"
#include "cache/op_cache.hpp"
#include "cache/graph_cache.hpp"
#include "utils/model.hpp"

#include "openvino/util/file_util.hpp"
#include "op_conformance_utils/utils/file.hpp"

#include "utils/memory.hpp"
#include "utils/cache.hpp"


using namespace ov::tools::subgraph_dumper;

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return 0;
    }

    if (!FLAGS_device.empty() && !FLAGS_plugin_lib_name.empty()) {
        try {
            auto plugin_path = ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), std::string(FLAGS_plugin_lib_name) + OV_BUILD_POSTFIX);
            if (!ov::util::file_exists(plugin_path)) {
                throw std::runtime_error("[ WARNING ][ GRAPH CACHE ] Plugin: " + plugin_path + " does not exists!");
            }
            ov::util::core->register_plugin(plugin_path, FLAGS_device);
            std::cout << "[ INFO ] Device: " << FLAGS_device << " is registred in OV core with " << FLAGS_plugin_lib_name << " lib" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "[ ERROR ] Impossible to register device " << FLAGS_device << " with lib " << FLAGS_plugin_lib_name <<
            std::endl << e.what() << std::endl;
        }
    }

    std::vector<std::string> local_cache_dirs = ov::util::split_str(FLAGS_local_cache);
    std::vector<std::string> dirs = ov::util::split_str(FLAGS_input_folders);

    std::vector<std::string> models;
    std::map<ov::util::ModelCacheStatus, std::vector<std::string>> cache_model_status;

    if (!ov::util::directory_exists(FLAGS_output_folder)) {
        std::string msg = "Output directory (" + FLAGS_output_folder + ") doesn't not exist! The directory will be created.";
        std::cout << msg << std::endl;
        ov::util::create_directory_recursive(FLAGS_output_folder);
    }
    try {
        auto all_models = ov::util::find_models(dirs, FLAGS_path_regex);
        models = all_models.first;
        cache_model_status.insert(all_models.second);
    } catch (std::runtime_error& e) {
        std::cout << "[ INFO ] Try 'subgraphsDumper -h' for more information. \nException: " << e.what() << std::endl;
        return 1;
    }
    size_t ram_size_gb = ov::util::get_ram_size();
    ram_size_gb >>= 30;
    std::cout << "[ INFO ] RAM size is " << ram_size_gb << "GB" << std::endl;

    std::vector<std::shared_ptr<ICache>> caches;
    if (FLAGS_cache_type == "OP" || FLAGS_cache_type.empty()) {
        std::cout << "[ INFO ] OpCache is enabled!" << std::endl;
        caches.push_back(OpCache::get());
    }
    if (FLAGS_cache_type == "GRAPH" || FLAGS_cache_type.empty()) {
        std::cout << "[ INFO ] GraphCache is enabled!" << std::endl;
        caches.push_back(GraphCache::get(FLAGS_device));
    }

    for (auto& cache : caches) {
        cache->set_serialization_dir(FLAGS_output_folder);
        // Upload previously cached graphs to cache
        if (!FLAGS_local_cache.empty()) {
            std::vector<std::string> tmp_paths;
            for (auto& dir : local_cache_dirs) {
                tmp_paths.push_back(ov::util::path_join({dir, cache->m_cache_subdir}).string());
            }
            auto cached_ops = ov::util::find_models(tmp_paths, FLAGS_path_regex);
            auto this_cache_model_status = ov::util::cache_models(cache, cached_ops.first, FLAGS_extract_body, true);
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
            auto this_cache_model_status = ov::util::cache_models(cache, models, FLAGS_extract_body);
            for (auto& model_status : cache_model_status) {
                auto& key = model_status.first;
                auto& value = model_status.second;
                if (this_cache_model_status.count(key)) {
                    value.insert(value.end(), this_cache_model_status[key].begin(), this_cache_model_status[key].end());
                }
            }
        }

        cache->serialize_cache();
        cache->reset_cache();
    }

    save_model_status_to_file(cache_model_status, FLAGS_output_folder);
    return cache_model_status[ov::util::ModelCacheStatus::NOT_FULLY_CACHED].empty() &&
           cache_model_status[ov::util::ModelCacheStatus::NOT_READ].empty();
}
