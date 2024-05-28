// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/debug_configuration.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <vector>

namespace cldnn {
const char *debug_configuration::prefix = "GPU_Debug: ";

// Default policy is that dump_configuration will override other configuration from IE.

#ifdef GPU_DEBUG_CONFIG

#define GPU_DEBUG_COUT_ std::cout << cldnn::debug_configuration::prefix

template<typename T>
void print_option(std::string option_name, T option_value) {
    GPU_DEBUG_COUT_ << "Config " << option_name << " = " << option_value << std::endl;
}

static std::string to_upper_case(const std::string& var) {
    std::stringstream s;

    for (size_t i = 0; i < var.size(); i++) {
        if (std::isupper(var[i])) {
            if (i != 0) {
                s << "_";
            }
            s << var[i];
        } else {
            s << static_cast<char>(std::toupper(var[i]));
        }
    }

    return s.str();
}

static std::vector<std::string> get_possible_option_names(const std::string& var, std::vector<std::string> allowed_option_prefixes) {
    std::vector<std::string> result;

    for (auto& prefix : allowed_option_prefixes) {
        result.push_back(prefix + var);
        result.push_back(prefix + to_upper_case(var));
    }

    return result;
}

template <typename T>
T convert_to(const std::string &str) {
    std::istringstream ss(str);
    T res;
    ss >> res;
    return res;
}

template <>
std::string convert_to(const std::string &str) {
    return str;
}

static std::set<int64_t> parse_int_set(std::string& str) {
    std::set<int64_t> int_array;
    // eliminate '"' from string to avoid parsing error
    str.erase(std::remove_if(str.begin(), str.end(), [](char c) {
                return c == '\"'; }), str.end());
    if (str.size() > 0) {
        str = " " + str + " ";
        std::istringstream ss(str);
        std::string token;
        while (ss >> token) {
            try {
                int_array.insert(static_cast<int64_t>(std::stol(token)));
            } catch(const std::exception& ex) {
                int_array.clear();
                GPU_DEBUG_COUT << "OV_GPU_DumpMemoryPoolIters was ignored. It cannot be parsed to integer array." << std::endl;
                break;
            }
        }
    }
    return int_array;
}

template<typename T>
void get_debug_env_var(const std::string &var, T &val, std::vector<std::string> allowed_option_prefixes) {
    bool found = false;
    for (auto o : get_possible_option_names(var, allowed_option_prefixes)) {
        if (const auto env_var = std::getenv(o.c_str())) {
            val = convert_to<T>(env_var);
            found = true;
        }
    }

    if (found) {
        print_option(var, val);
    }
}

template<typename T>
void get_gpu_debug_env_var(const std::string &var, T &val) {
    return get_debug_env_var(var, val, {"OV_GPU_"});
}

template<typename T>
void get_common_debug_env_var(const std::string &var, T &val) {
    // The list below should be prioritized from lowest to highest prefix priority
    // If an option is set several times with different prefixes, version with the highest priority will be actually used.
    // This may allow to enable global option with some value and override this value for GPU plugin
    // For example: OV_GPU_Verbose=2 OV_Verbose=1 ./my_app => this->verbose == 2
    // In that case we enable Verbose (with level = 1) for all OV components that support this option, but for GPU plugin we increase verbose level to 2
    std::vector<std::string> allowed_option_prefixes = {
        "OV_",
        "OV_GPU_"
    };

    return get_debug_env_var(var, val, allowed_option_prefixes);
}

static void print_help_messages() {
    std::vector<std::pair<std::string, std::string>> message_list;
    message_list.emplace_back("OV_GPU_Help", "Print help messages");
    message_list.emplace_back("OV_GPU_Verbose", "Verbose execution");
    message_list.emplace_back("OV_GPU_VerboseColor", "Print verbose color");
    message_list.emplace_back("OV_GPU_ListLayers", "Print layers names");
    message_list.emplace_back("OV_GPU_PrintMultiKernelPerf", "Print execution time of each kernel in multi-kernel primitimive");
    message_list.emplace_back("OV_GPU_PrintInputDataShapes",  "Print data_shapes of input layers for benchmark_app.");
    message_list.emplace_back("OV_GPU_DisableUsm", "Disable usm usage");
    message_list.emplace_back("OV_GPU_DisableOnednn", "Disable onednn for discrete GPU (no effect for integrated GPU)");
    message_list.emplace_back("OV_GPU_DisableOnednnPermuteFusion", "Disable permute fusion for onednn gemm (no effect for integrated GPU)");
    message_list.emplace_back("OV_GPU_DisableOnednnOptPostOps", "Disable onednn optimize post operators");
    message_list.emplace_back("OV_GPU_DumpProfilingData", "Enables dump of extended profiling information to specified directory."
                              " Please use OV_GPU_DumpProfilingDataPerIter=1 env variable to collect performance per iteration."
                              " Note: Performance impact may be significant as this option enforces host side sync after each primitive");
    message_list.emplace_back("OV_GPU_DumpProfilingDataIteration", "Enable collecting profiling data only at iterations with requested range. "
                              "For example for dump profiling data only when iteration is from 10 to 20, you can use "
                              "OV_GPU_DumpProfilingDataIteration='10..20'. Additionally, you can dump profiling data only "
                              "from one specific iteration by giving the same values for the start and end, and the open "
                              "ended range is also available by range from given start to the last iteration as -1. e.g. "
                              "OV_GPU_DumpProfilingDataIteration='10..-1'");
    message_list.emplace_back("OV_GPU_DumpGraphs", "1) dump ngraph before and after transformation. 2) dump graph in model compiling."
                              "3) dump graph in execution.");
    message_list.emplace_back("OV_GPU_DumpSources", "Dump opencl sources");
    message_list.emplace_back("OV_GPU_DumpLayersPath", "Enable dumping intermediate buffers and set the dest path");
    message_list.emplace_back("OV_GPU_DumpLayers", "Dump intermediate buffers of specified layers only, separated by space."
                               " Support case-insensitive and regular expression. For example .*conv.*");
    message_list.emplace_back("OV_GPU_DumpLayersResult", "Dump output buffers of result layers only");
    message_list.emplace_back("OV_GPU_DumpLayersInput",  "Dump intermediate buffers of input layers only");
    message_list.emplace_back("OV_GPU_DumpLayersDstOnly", "Dump only output of layers");
    message_list.emplace_back("OV_GPU_DumpLayersLimitBatch", "Limit the size of batch to dump");
    message_list.emplace_back("OV_GPU_DumpLayersRaw", "If true, dump data is stored in raw memory format.");
    message_list.emplace_back("OV_GPU_DumpLayersRawBinary", "If true, dump data is stored in binary format.");
    message_list.emplace_back("OV_GPU_DryRunPath", "Dry run and serialize execution graph into the specified path");
    message_list.emplace_back("OV_GPU_BaseBatchForMemEstimation", "Base batch size to be used in memory estimation");
    message_list.emplace_back("OV_GPU_AfterProc", "Run inference after the specified process PIDs are finished, separated by space."
                              " Supported on only on linux.");
    message_list.emplace_back("OV_GPU_SerialCompile", "Serialize creating primitives and compiling kernels");
    message_list.emplace_back("OV_GPU_ForceImplTypes", "Force implementation type of a target primitive or layer. [primitive or layer_name]:[impl_type]"
                              " For example fc:onednn gemm:onednn reduce:ocl do:cpu"
                              " For primitives fc, gemm, do, reduce, concat are supported. Separated by space.");
    message_list.emplace_back("OV_GPU_MaxKernelsPerBatch", "Maximum number of kernels in a batch during compiling kernels");
    message_list.emplace_back("OV_GPU_ImplsCacheCapacity", "The maximum number of entries in the kernel impl cache");
    message_list.emplace_back("OV_GPU_DisableAsyncCompilation", "Disable async compilation");
    message_list.emplace_back("OV_GPU_DisableWinogradConv", "Disable Winograd convolution");
    message_list.emplace_back("OV_GPU_DisableDynamicImpl", "Disable dynamic implementation");
    message_list.emplace_back("OV_GPU_DisableRuntimeBufferFusing", "Disable runtime buffer fusing");
    message_list.emplace_back("OV_GPU_DisableMemoryReuse", "Disable memory reuse");
    message_list.emplace_back("OV_GPU_EnableSDPA", "This allows the enforcement of SDPA decomposition logic: 0 completely disables SDPA kernel usage, "
                              "and 1 enables it for all the cases.");
    message_list.emplace_back("OV_GPU_DumpMemoryPool", "Dump memory pool contents of each iteration");
    message_list.emplace_back("OV_GPU_DumpMemoryPoolIters", "List of iterations to dump memory pool status, separated by space.");
    message_list.emplace_back("OV_GPU_DumpMemoryPoolPath", "Enable dumping memory pool status to csv file and set the dest path");
    message_list.emplace_back("OV_GPU_DisableBuildTimeWeightReorderForDynamicNodes", "Disable build time weight reorder for dynmaic nodes.");
    message_list.emplace_back("OV_GPU_DisableRuntimeSkipReorder", "Disable runtime skip reorder.");
    message_list.emplace_back("OV_GPU_DisablePrimitiveFusing", "Disable primitive fusing");
    message_list.emplace_back("OV_GPU_DisableFakeAlignment", "Disable fake alignment");
    message_list.emplace_back("OV_GPU_DumpIteration", "Dump n-th execution of network, separated by space.");
    message_list.emplace_back("OV_GPU_MemPreallocationOptions", "Controls buffer pre-allocation feature. Expects 4 values separated by space in "
                              "the following order: number of iterations for pre-allocation(int), max size of single iteration in bytes(int), "
                              "max per-dim allowed diff(int), unconditional buffers preallocation ratio(float). For example for disabling memory "
                              "preallocation at all, you can use OV_GPU_MemPreallocationOptions='0 0 0 1.0'");
    message_list.emplace_back("OV_GPU_LoadDumpRawBinary",
                               "Specified layers which are loading dumped binary files generated by OV_GPU_DumpLayersRawBinary debug-config."
                               " Currently, other layers except input-layer('parameter' type) are loading binaries for only input."
                               " Different input or output tensors are seperated by ','. Different layers are separated by space. For example, "
                               " \"[input_layer_name1]:[binary_dumped_file1],[binary_dump_file2] [input_layer_name2]:[binary_dump_1],[binary_dump_2]\"");

    auto max_name_length_item = std::max_element(message_list.begin(), message_list.end(),
        [](std::pair<std::string, std::string>& a, std::pair<std::string, std::string>& b){
            return a.first.size() < b.first.size();
    });
    int name_width = static_cast<int>(max_name_length_item->first.size()) + 2;

    GPU_DEBUG_COUT_ << "Supported environment variables for debugging" << std::endl;
    for (auto& p : message_list) {
        GPU_DEBUG_COUT_ << " - " << std::left << std::setw(name_width) << p.first + "  " << p.second << std::endl;
    }
}

#endif

debug_configuration::debug_configuration()
        : help(0)
        , verbose(0)
        , verbose_color(0)
        , list_layers(0)
        , print_multi_kernel_perf(0)
        , print_input_data_shapes(0)
        , disable_usm(0)
        , disable_onednn(0)
        , disable_onednn_permute_fusion(0)
        , disable_onednn_opt_post_ops(0)
        , dump_profiling_data(std::string(""))
        , dump_profiling_data_per_iter(0)
        , dump_graphs(std::string())
        , dump_sources(std::string())
        , dump_layers_path(std::string())
        , dry_run_path(std::string())
        , dump_layers_dst_only(0)
        , dump_layers_result(0)
        , dump_layers_input(0)
        , dump_layers_limit_batch(std::numeric_limits<int>::max())
        , dump_layers_raw(0)
        , dump_layers_binary(0)
        , dump_memory_pool(0)
        , dump_memory_pool_path(std::string())
        , base_batch_for_memory_estimation(-1)
        , serialize_compile(0)
        , max_kernels_per_batch(0)
        , impls_cache_capacity(-1)
        , enable_sdpa(-1)
        , disable_async_compilation(0)
        , disable_winograd_conv(0)
        , disable_dynamic_impl(0)
        , disable_runtime_buffer_fusing(0)
        , disable_memory_reuse(0)
        , disable_build_time_weight_reorder_for_dynamic_nodes(0)
        , disable_runtime_skip_reorder(0)
        , disable_primitive_fusing(0)
        , disable_fake_alignment(0) {
#ifdef GPU_DEBUG_CONFIG
    get_gpu_debug_env_var("Help", help);
    get_common_debug_env_var("Verbose", verbose);
    get_gpu_debug_env_var("VerboseColor", verbose_color);
    get_gpu_debug_env_var("ListLayers", list_layers);
    get_gpu_debug_env_var("PrintMultiKernelPerf", print_multi_kernel_perf);
    get_gpu_debug_env_var("PrintInputDataShapes", print_input_data_shapes);
    get_gpu_debug_env_var("DisableUsm", disable_usm);
    get_gpu_debug_env_var("DumpGraphs", dump_graphs);
    get_gpu_debug_env_var("DumpSources", dump_sources);
    get_gpu_debug_env_var("DumpLayersPath", dump_layers_path);
    get_gpu_debug_env_var("DumpLayersLimitBatch", dump_layers_limit_batch);
    get_gpu_debug_env_var("DumpLayersRaw", dump_layers_raw);
    get_gpu_debug_env_var("DumpLayersRawBinary", dump_layers_binary);
    get_gpu_debug_env_var("DumpLayersDstOnly", dump_layers_dst_only);
    get_gpu_debug_env_var("DumpLayersResult", dump_layers_result);
    get_gpu_debug_env_var("DumpLayersInput", dump_layers_input);
    get_gpu_debug_env_var("DisableOnednn", disable_onednn);
    get_gpu_debug_env_var("DisableOnednnPermuteFusion", disable_onednn_permute_fusion);
    get_gpu_debug_env_var("DisableOnednnOptPostOps", disable_onednn_opt_post_ops);
    get_gpu_debug_env_var("DumpProfilingData", dump_profiling_data);
    get_gpu_debug_env_var("DumpProfilingDataPerIter", dump_profiling_data_per_iter);
    std::string dump_prof_data_iter_str;
    get_gpu_debug_env_var("DumpProfilingDataIteration", dump_prof_data_iter_str);
    get_gpu_debug_env_var("DryRunPath", dry_run_path);
    get_gpu_debug_env_var("DumpMemoryPool", dump_memory_pool);
    std::string dump_runtime_memory_pool_iters_str;
    get_gpu_debug_env_var("DumpMemoryPoolIters", dump_runtime_memory_pool_iters_str);
    get_gpu_debug_env_var("DumpMemoryPoolPath", dump_memory_pool_path);
    get_gpu_debug_env_var("BaseBatchForMemEstimation", base_batch_for_memory_estimation);
    std::string dump_layers_str;
    get_gpu_debug_env_var("DumpLayers", dump_layers_str);
    std::string after_proc_str;
    get_gpu_debug_env_var("AfterProc", after_proc_str);
    get_gpu_debug_env_var("SerialCompile", serialize_compile);
    std::string forced_impl_types_str;
    get_gpu_debug_env_var("ForceImplTypes", forced_impl_types_str);
    get_gpu_debug_env_var("MaxKernelsPerBatch", max_kernels_per_batch);
    get_gpu_debug_env_var("ImplsCacheCapacity", impls_cache_capacity);
    get_gpu_debug_env_var("EnableSDPA", enable_sdpa);
    get_gpu_debug_env_var("DisableAsyncCompilation", disable_async_compilation);
    get_gpu_debug_env_var("DisableWinogradConv", disable_winograd_conv);
    get_gpu_debug_env_var("DisableDynamicImpl", disable_dynamic_impl);
    get_gpu_debug_env_var("DisableRuntimeBufferFusing", disable_runtime_buffer_fusing);
    get_gpu_debug_env_var("DisableMemoryReuse", disable_memory_reuse);
    get_gpu_debug_env_var("DisableBuildTimeWeightReorderForDynamicNodes", disable_build_time_weight_reorder_for_dynamic_nodes);
    get_gpu_debug_env_var("DisableRuntimeSkipReorder", disable_runtime_skip_reorder);
    get_gpu_debug_env_var("DisablePrimitiveFusing", disable_primitive_fusing);
    get_gpu_debug_env_var("DisableFakeAlignment", disable_fake_alignment);
    std::string dump_iteration_str;
    get_gpu_debug_env_var("DumpIteration", dump_iteration_str);
    std::string mem_preallocation_params_str;
    get_gpu_debug_env_var("MemPreallocationOptions", mem_preallocation_params_str);
    std::string load_dump_raw_bin_str;
    get_gpu_debug_env_var("LoadDumpRawBinary", load_dump_raw_bin_str);

    if (help > 0) {
        print_help_messages();
        exit(0);
    }

    if (dump_prof_data_iter_str.length() > 0) {
        dump_prof_data_iter_str = " " + dump_prof_data_iter_str + " ";
        std::istringstream iss(dump_prof_data_iter_str);
        char dot;
        int64_t start, end;
        bool is_valid_range = false;
        if (iss >> start >> dot >> dot >> end) {
            if (start <= end || end == -1) {
                try {
                    is_valid_range = true;
                    dump_prof_data_iter_params.start = start;
                    dump_prof_data_iter_params.end = end;
                } catch(const std::exception &) {
                    is_valid_range = false;
                }
            }
        }
        if (!is_valid_range)
            std::cout << "OV_GPU_DumpProfilingDataIteration was ignored. It cannot be parsed to valid iteration range." << std::endl;
        dump_prof_data_iter_params.is_enabled = is_valid_range;
    }

    if (dump_layers_str.length() > 0) {
        // Insert delimiter for easier parsing when used
        dump_layers_str = " " + dump_layers_str + " ";
        std::stringstream ss(dump_layers_str);
        std::string layer;
        while (ss >> layer) {
            dump_layers.push_back(layer);
        }
    }

    if (forced_impl_types_str.length() > 0) {
        forced_impl_types_str = " " + forced_impl_types_str + " ";
        std::stringstream ss(forced_impl_types_str);
        std::string type;
        while (ss >> type) {
            forced_impl_types.push_back(type);
        }
    }

    // Parsing for loading binary files
    if (load_dump_raw_bin_str.length() > 0) {
        load_dump_raw_bin_str = " " + load_dump_raw_bin_str + " ";
        std::stringstream ss(load_dump_raw_bin_str);
        std::string type;
        while (ss >> type) {
            load_layers_raw_dump.push_back(type);
        }
    }

    if (dump_iteration_str.size() > 0) {
        dump_iteration = parse_int_set(dump_iteration_str);
    }

    if (dump_runtime_memory_pool_iters_str.size() > 0) {
        dump_memory_pool_iters = parse_int_set(dump_runtime_memory_pool_iters_str);
    }

    if (mem_preallocation_params_str.size() > 0) {
        mem_preallocation_params_str = " " + mem_preallocation_params_str + " ";
        std::istringstream ss(mem_preallocation_params_str);
        std::vector<std::string> params;
        std::string param;
        while (ss >> param)
            params.push_back(param);

        bool correct_params = params.size() == 4;
        if (correct_params) {
            try {
                mem_preallocation_params.next_iters_preallocation_count = std::stol(params[0]);
                mem_preallocation_params.max_per_iter_size = std::stol(params[1]);
                mem_preallocation_params.max_per_dim_diff = std::stol(params[2]);
                mem_preallocation_params.buffers_preallocation_ratio = std::stof(params[3]);
            } catch(const std::exception &) {
                correct_params = false;
            }
        }

        if (!correct_params)
            GPU_DEBUG_COUT_ << "OV_GPU_MemPreallocationOptions were ignored, because they cannot be parsed.\n";

        mem_preallocation_params.is_initialized = correct_params;
    }

    if (after_proc_str.length() > 0) {
#ifdef _WIN32
        GPU_DEBUG_COUT_ << "Warning: OV_GPU_AfterProc is supported only on linux" << std::endl;
#else
        after_proc_str = " " + after_proc_str + " "; // Insert delimiter for easier parsing when used
        std::stringstream ss(after_proc_str);
        std::string pid;
        while (ss >> pid) {
            after_proc.push_back(pid);
        }
#endif
    }
#endif
}

const debug_configuration *debug_configuration::get_instance() {
    static std::unique_ptr<debug_configuration> instance(nullptr);
#ifdef GPU_DEBUG_CONFIG
    static std::mutex _m;
    std::lock_guard<std::mutex> lock(_m);
    if (nullptr == instance)
        instance.reset(new debug_configuration());
    return instance.get();
#else
    return nullptr;
#endif
}

bool debug_configuration::is_target_dump_prof_data_iteration(int64_t iteration) const {
#ifdef GPU_DEBUG_CONFIG
    if (iteration < 0)
        return true;

    if (dump_prof_data_iter_params.start > iteration)
        return false;

    if (dump_prof_data_iter_params.start <= dump_prof_data_iter_params.end &&
        dump_prof_data_iter_params.end < iteration)
        return false;

    return true;
#else
    return false;
#endif
}

std::vector<std::string> debug_configuration::get_filenames_for_matched_layer_loading_binaries(const std::string& id) const {
    std::vector<std::string> file_names;
#ifdef GPU_DEBUG_CONFIG
    if (load_layers_raw_dump.empty())
        return file_names;

    for (const auto& load_layer : load_layers_raw_dump) {
        size_t file = load_layer.rfind(":");
        if (file != std::string::npos) {
            if (id == load_layer.substr(0, file)) {
                auto file_name_str = load_layer.substr(file + 1);
                size_t head = 0;
                size_t found = 0;
                do {
                    found = file_name_str.find(",", head);
                    if (found != std::string::npos)
                        file_names.push_back(file_name_str.substr(head, (found - head)));
                    else
                        file_names.push_back(file_name_str.substr(head));

                    head = found+1;
                    GPU_DEBUG_LOG << " Layer name loading raw dump : " << load_layer.substr(0, file) << " / the dump file : "
                                << file_names.back() << std::endl;
                } while (found != std::string::npos);

                return file_names;
            }
        }
    }
#endif

    return file_names;
}

std::string debug_configuration::get_matched_from_filelist(const std::vector<std::string>& file_names, std::string pattern) const {
#ifdef GPU_DEBUG_CONFIG
    for (const auto& file : file_names) {
        auto found = file.find(pattern);
        if (found != std::string::npos) {
            return file;
        }
    }
#endif
    return std::string();
}

std::string debug_configuration::get_name_for_dump(const std::string& file_name) const {
    std::string filename = file_name;
#ifdef GPU_DEBUG_CONFIG
    std::replace(filename.begin(), filename.end(), '\\', '_');
    std::replace(filename.begin(), filename.end(), '/', '_');
    std::replace(filename.begin(), filename.end(), ' ', '_');
    std::replace(filename.begin(), filename.end(), ':', '_');
#endif
    return filename;
}

bool debug_configuration::is_layer_for_dumping(const std::string& layer_name, bool is_output, bool is_input) const {
#ifdef GPU_DEBUG_CONFIG
    // Dump result layer
    if (is_output == true && dump_layers_result == 1 &&
        (layer_name.find("constant:") == std::string::npos))
        return true;
    // Dump all layers
    if (dump_layers.empty() && dump_layers_result == 0 && dump_layers_input == 0)
        return true;

    // Dump input layers
    size_t pos = layer_name.find(':');
    auto type = layer_name.substr(0, pos);
    if (is_input == true && type == "parameter" && dump_layers_input == 1)
        return true;

    auto is_match = [](const std::string& layer_name, const std::string& pattern) -> bool {
        auto upper_layer_name = std::string(layer_name.length(), '\0');
        std::transform(layer_name.begin(), layer_name.end(), upper_layer_name.begin(), ::toupper);
        auto upper_pattern = std::string(pattern.length(), '\0');
        std::transform(pattern.begin(), pattern.end(), upper_pattern.begin(), ::toupper);
        // Check pattern from exec_graph
        size_t pos = upper_layer_name.find(':');
        auto upper_exec_graph_name = upper_layer_name.substr(pos + 1, upper_layer_name.size());
        if (upper_exec_graph_name.compare(upper_pattern) == 0) {
            return true;
        }
        // Check pattern with regular expression
        std::regex re(upper_pattern);
        return std::regex_match(upper_layer_name, re);
    };

    auto iter = std::find_if(dump_layers.begin(), dump_layers.end(), [&](const std::string& dl){
        return is_match(layer_name, dl);
    });
    return (iter != dump_layers.end());
#else
    return false;
#endif
}

bool debug_configuration::is_target_iteration(int64_t iteration) const {
#ifdef GPU_DEBUG_CONFIG
    if (iteration < 0)
        return true;

    if (dump_iteration.empty())
        return true;

    if (dump_iteration.find(iteration) == std::end(dump_iteration))
        return false;

    return true;
#else
    return false;
#endif
}
} // namespace cldnn
