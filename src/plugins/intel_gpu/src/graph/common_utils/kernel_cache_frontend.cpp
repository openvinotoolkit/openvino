// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_cache_frontend.hpp"

#include <algorithm>
#include <functional>
#include <list>
#include <regex>
#include <set>
#include <sstream>
#include <string_view>
#include <tuple>
#include <utility>

#ifdef ENABLE_CM_FOR_GPU
#    include "impls/cm/utils/kernels_db.hpp"
#endif
#if defined(OV_GPU_WITH_OCL_RT) || defined(OV_GPU_WITH_ZE_RT) || defined(OV_GPU_WITH_SYCL_RT)
#    include "impls/ocl_v2/utils/kernels_db.hpp"
#endif

#include "openvino/core/except.hpp"

namespace cldnn {
namespace {

std::string reorder_source_options(const std::string& original_options) {
    std::stringstream stream(original_options);
    std::set<std::string> sorted_options;

    while (stream.good()) {
        std::string option;
        stream >> option;
        sorted_options.insert(option);
    }

    std::string result;
    for (const auto& option : sorted_options) {
        result += option + " ";
    }
    return result;
}

class source_kernel_frontend final {
public:
    static void append_batch_headers(kernels_cache::batch_program& program, const std::map<std::string, std::string>& batch_headers) {
        if (program.language != kernel_language::OCLC) {
            return;
        }
        static const std::vector<std::string> micro_kernel_include_names{
            "generic_vector_ops",
            "tile_ops",
            "sdpa_utils",
        };
        for (const auto& header : batch_headers) {
            if (std::find(micro_kernel_include_names.begin(), micro_kernel_include_names.end(), header.first) == micro_kernel_include_names.end()) {
                program.source.push_back(header.second);
            } else {
                program.micro_headers.push_back(header.second);
            }
        }
    }

    static std::string normalize_options(const kernel_string& kernel) {
        if (kernel.batch_compilation && kernel.language != kernel_language::CM) {
            return reorder_source_options(kernel.options);
        }
        return kernel.options;
    }

    static void expand_includes(kernels_cache::batch_program& program) {
        const auto find_and_remove_includes = [](const std::string& code, std::vector<std::string>& required_headers) {
            const std::regex include_regex(R"(#include\s+\"([^\"]+)\")");
            std::string processed_kernel;
            std::sregex_iterator iterator(code.begin(), code.end(), include_regex);
            const std::sregex_iterator end;

            size_t last_position = 0;
            for (; iterator != end; ++iterator) {
                auto header_name = (*iterator)[1].str();
                header_name = header_name.substr(header_name.find_last_of("/") + 1);
                header_name = header_name.substr(0, header_name.find_last_of("."));
                required_headers.push_back(header_name);
                processed_kernel += code.substr(last_position, iterator->position() - last_position);
                last_position = iterator->position() + iterator->length();
            }
            processed_kernel += code.substr(last_position);
            return processed_kernel;
        };

        std::list<std::string> sources_to_process(program.source.begin(), program.source.end());
        program.source.clear();
        std::list<std::string> all_headers;
        while (!sources_to_process.empty()) {
            std::vector<std::string> new_headers;
            auto source = sources_to_process.front();
            sources_to_process.pop_front();
            program.source.insert(program.source.begin(), find_and_remove_includes(source, new_headers));
            for (auto& header : new_headers) {
                if (std::find(all_headers.begin(), all_headers.end(), header) != all_headers.end()) {
                    continue;
                }
                all_headers.push_front(header);
                std::string_view header_code;
#if defined(OV_GPU_WITH_OCL_RT) || defined(OV_GPU_WITH_ZE_RT) || defined(OV_GPU_WITH_SYCL_RT)
#    ifdef ENABLE_CM_FOR_GPU
                header_code = program.language == kernel_language::OCLC_V2 ? ov::intel_gpu::ocl::SourcesDB::get_kernel_header(header)
                                                                           : ov::intel_gpu::cm::SourcesDB::get_kernel_header(header);
#    else
                header_code = ov::intel_gpu::ocl::SourcesDB::get_kernel_header(header);
#    endif
#else
                OPENVINO_THROW("Source kernel headers are unavailable without a source runtime backend");
#endif
                sources_to_process.push_back(std::string(header_code) + "\n");
            }
        }

        if (program.language == kernel_language::CM) {
            program.source.insert(program.source.begin(), "#include <cm/cm.h>\n#include <cm/cmtl.h>\n");
        }
    }

    static void apply_dump_options(kernels_cache::batch_program& batch, const kernel_cache_frontend_context& context) {
        if (context.dump_sources_path.empty() || batch.language == kernel_language::CM) {
            return;
        }

        auto dump_file = context.dump_sources_path;
        if (!dump_file.empty() && dump_file.back() != '/') {
            dump_file += '/';
        }
        dump_file += "clDNN_program_" + std::to_string(context.program_id) + "_bucket_" + std::to_string(batch.bucket_id) + "_part_" +
                     std::to_string(batch.batch_id) + "_" + std::to_string(batch.hash_value) + ".cl";
        batch.options += " -g -s " + dump_file;
    }
};

class precompiled_spirv_frontend final {
public:
    static std::string normalize_options(const kernel_string& kernel) {
        return kernel.options;
    }
};

}  // namespace

void kernel_cache_frontend::prepare(const kernels_cache::kernels_code& pending,
                                    const kernel_cache_frontend_context& context,
                                    std::vector<kernels_cache::batch_program>& batches) {
    OPENVINO_ASSERT(context.batch_headers != nullptr, "[GPU] Kernel cache frontend requires a batch-header registry");
    std::map<std::string, std::tuple<int32_t, std::vector<kernels_cache::batch_program>>> program_buckets;
    for (const auto& item : pending) {
        const auto& code = item.second;
        for (size_t kernel_part_index = 0; kernel_part_index < code.kernel_strings.size(); ++kernel_part_index) {
            const auto& kernel = code.kernel_strings[kernel_part_index];
            const bool is_spirv = kernel->language == kernel_language::SPIRV;
            std::string full_code = kernel->jit + kernel->str + kernel->undefs;
            const std::string entry_point = kernel->entry_point;
            std::string options = is_spirv ? precompiled_spirv_frontend::normalize_options(*kernel) : source_kernel_frontend::normalize_options(*kernel);

            std::string key = options;
            if (!kernel->batch_compilation) {
                key += " __PROGRAM__" + std::to_string(program_buckets.size());
            }
            if (code.dump_custom_program) {
                key += " __DUMP_CUSTOM_PROGRAM__";
            }
            key += " __LANG__" + std::to_string(static_cast<size_t>(kernel->language));

            auto& bucket_id = std::get<0>(program_buckets[key]);
            auto& current_bucket = std::get<1>(program_buckets[key]);
            if (current_bucket.empty()) {
                bucket_id = static_cast<int32_t>(program_buckets.size() - 1);
                current_bucket.emplace_back(bucket_id, 0, options, kernel->language);
                if (!is_spirv) {
                    source_kernel_frontend::append_batch_headers(current_bucket.back(), *context.batch_headers);
                }
            }

            if (current_bucket.back().kernels_counter >= context.max_kernels_per_batch ||
                current_bucket.back().entry_point_to_id.find(entry_point) != current_bucket.back().entry_point_to_id.end()) {
                const auto batch_id = static_cast<int32_t>(current_bucket.size());
                current_bucket.emplace_back(bucket_id, batch_id, options, kernel->language);
                if (!is_spirv) {
                    source_kernel_frontend::append_batch_headers(current_bucket.back(), *context.batch_headers);
                }
            }

            auto& current_batch = current_bucket.back();
            current_batch.dump_custom_program = code.dump_custom_program;
            current_batch.entry_point_to_id.emplace(entry_point, std::make_pair(code.params, kernel_part_index));
            current_batch.has_microkernels |= kernel->has_microkernels;
            current_batch.source.push_back(std::move(full_code));
            ++current_batch.kernels_counter;
        }
    }

    for (auto& program_bucket : program_buckets) {
        const auto options = program_bucket.first;
        auto& bucket_batches = std::get<1>(program_bucket.second);
        for (auto& batch : bucket_batches) {
            const bool source_requires_includes = batch.language == kernel_language::OCLC_V2 || batch.language == kernel_language::CM;
            if (source_requires_includes) {
                source_kernel_frontend::expand_includes(batch);
            }

            std::string full_code = options + " " + context.driver_version + context.device_name;
            for (const auto& source : batch.source) {
                full_code += source;
            }
            batch.hash_value = std::hash<std::string>()(full_code);

            if (batch.language != kernel_language::SPIRV) {
                source_kernel_frontend::apply_dump_options(batch, context);
            }
            batches.push_back(batch);
        }
    }
}

}  // namespace cldnn
