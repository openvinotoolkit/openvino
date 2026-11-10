// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels_cache.hpp"
#include <regex>

#ifdef ENABLE_CM_FOR_GPU
#include "impls/cm/utils/kernels_db.hpp"
#endif
#include "impls/ocl_v2/utils/kernels_db.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "intel_gpu/runtime/engine_configuration.hpp"
#include "openvino/util/pp.hpp"
#include "intel_gpu/graph/serialization/set_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "intel_gpu/graph/serialization/map_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/program.hpp"

#include "intel_gpu/runtime/utils.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/file_util.hpp"
#include "intel_gpu/runtime/device.hpp"

#ifdef _WIN32
#include <sdkddkver.h>
#ifdef NTDDI_WIN10_RS5
#include <appmodel.h>
#endif
#endif

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iterator>
#include <functional>
#include <sstream>
#include <fstream>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <memory>
#include <utility>

#if defined(__unix__) && !defined(__ANDROID__)
#include <malloc.h>
#endif

#ifdef OV_GPU_WITH_ONEDNN_MICROKERNELS
#ifndef NOMINMAX
# define NOMINMAX
#endif
#include "gpu/intel/gemm/jit/include/gemmstone/microkernel/fuser.hpp"
#endif

namespace {
std::mutex cacheAccessMutex;

std::string join_strings(const std::vector<std::string> &strings) {
    size_t total_size = 0;
    for (const auto& str : strings) {
        total_size += str.size();
    }
    std::string acc_str;
    acc_str.reserve(total_size);
    for (const auto& str : strings) {
        acc_str.append(str);
    }
    return acc_str;
}

std::string reorder_options(const std::string& org_options) {
    std::stringstream ss(org_options);
    std::set<std::string> sorted_options;

    while (ss.good()) {
        std::string word;
        ss >> word;
        sorted_options.insert(word);
    }

    std::string options;

    for (const auto& o : sorted_options) {
        options += o + " ";
    }

    return options;
}

}  // namespace

namespace cldnn {
std::mutex kernels_cache::_mutex;

std::string kernels_cache::get_cache_path() const {
    auto path = ov::util::path_to_string(_config.get_cache_dir());
    if (path.empty()) {
        return {};
    }

    if (path.back() != '/' && path.back() != '\\') {
        path += "/";
    }
    return path;
}

bool kernels_cache::is_cache_enabled() const {
    if (!_config.get_allow_new_shape_infer() &&
        (_config.get_cache_mode() == ov::CacheMode::OPTIMIZE_SPEED)) {
        return false;
    }

    return !_config.get_cache_dir().empty();
}

namespace {

// A REQD_SUB_GROUP_SIZE / intel_reqd_sub_group_size occurrence on a #define line is the macro being defined
// (common.cl), not a kernel asking for a sub-group size.
bool on_define_line(const std::string& src, size_t pos) {
    size_t line_start = src.rfind('\n', pos);
    line_start = (line_start == std::string::npos) ? 0 : line_start + 1;
    // Look only between the start of the line and the occurrence, not on to the end of the source.
    return std::string_view(src).substr(line_start, pos - line_start).find("#define") != std::string_view::npos;
}

// The [begin, end) byte ranges of the comments in src, in order: a '//' comment to the end of its line, a '/* */'
// comment to its '*/'. String and character literals are stepped over, so a '//' or '/*' inside one starts nothing.
// An occurrence in a comment is not a kernel's request - custom kernels mention them in comments - and rewriting it
// would end a '//' comment at the inserted directive and turn the rest of the line into code.
std::vector<std::pair<size_t, size_t>> comment_ranges(const std::string& src) {
    std::vector<std::pair<size_t, size_t>> ranges;
    for (size_t i = 0; i < src.size(); ++i) {
        const char c = src[i];
        if (c == '"' || c == '\'') {
            for (++i; i < src.size() && src[i] != c && src[i] != '\n'; ++i) {
                if (src[i] == '\\') {
                    ++i;
                }
            }
        } else if (c == '/' && i + 1 < src.size() && src[i + 1] == '/') {
            const size_t end = std::min(src.find('\n', i + 2), src.size());
            ranges.emplace_back(i, end);
            i = end;
        } else if (c == '/' && i + 1 < src.size() && src[i + 1] == '*') {
            const size_t close = src.find("*/", i + 2);
            const size_t end = close == std::string::npos ? src.size() : close + 2;
            ranges.emplace_back(i, end);
            i = end - 1;
        }
    }
    return ranges;
}

bool in_ranges(const std::vector<std::pair<size_t, size_t>>& ranges, size_t pos) {
    auto it = std::upper_bound(ranges.begin(), ranges.end(), pos, [](size_t p, const std::pair<size_t, size_t>& r) {
        return p < r.first;
    });
    return it != ranges.begin() && pos < std::prev(it)->second;
}

std::string redefine_directive(const std::string& arg) {
    return "\n#undef _NV_SUB_GROUP_SIZE\n#define _NV_SUB_GROUP_SIZE (" + arg + ")\n";
}

// Sub-group size for a kernel that states none, derived from the (x, y, z) arguments of its reqd_work_group_size.
//
// Without a declaration the emulation would run the kernel at the 32 default. When the whole work-group fits in 8 or
// 16 lanes that leaves an inconsistent sub-group: get_sub_group_size() reports the lanes that exist, while block
// reads/writes stride and get_max_sub_group_size() report 32. Kernels written for such a work-group - the b8 family,
// convolution_gpu_yxfb_yxio_b8 among them - count on the work-group being exactly one sub-group, which is what the
// Intel compiler hands them in practice. Taking the work-group size as the sub-group size is the one consistent
// choice, and it only changes what those strided accesses see: shuffles and reductions already stop at the lanes
// that exist. Only the widths the emulation supports below a warp are mapped; anything else keeps 32.
//
// The arguments are usually jit constants whose value is not known here, so the choice is left to the preprocessor
// as a constant expression. An empty result means the arguments were not recognised and nothing should change.
std::string sub_group_size_from_work_group_size(const std::string& args) {
    std::vector<std::string> dims;
    std::stringstream ss(args);
    for (std::string dim; std::getline(ss, dim, ',');) {
        dims.push_back(dim);
    }
    if (dims.size() != 3) {
        return {};
    }
    const std::string wg = "(" + dims[0] + ")*(" + dims[1] + ")*(" + dims[2] + ")";
    return "(" + wg + ") == 8u ? 8u : ((" + wg + ") == 16u ? 16u : 32u)";
}

}  // namespace

std::string nvidia_replace_reqd_sub_group_size(const std::string& src) {
    // Kernels state their sub-group size either through common.cl's REQD_SUB_GROUP_SIZE(x) or, as a number of ocl_v2
    // kernels do, by writing __attribute__((intel_reqd_sub_group_size(x))) directly. Neither survives on NVIDIA:
    // cl_intel_required_subgroup_size is not advertised so the macro expands to nothing, and the raw attribute is
    // unknown to the NVIDIA compiler. The emulation keys off _NV_SUB_GROUP_SIZE instead, which every sub-group helper
    // in nvidia/compat_sg.cl reads at its call site.
    // One pattern for both spellings, so the occurrences are visited in source order and the first request - the one
    // repeated at the top of the entry - is the first in the source whichever spelling states it. Group 1 is the raw
    // attribute's argument, group 2 the macro's.
    static const std::regex request_re(
        R"(__attribute__\s*\(\s*\(\s*intel_reqd_sub_group_size\s*\(([^()]*)\)\s*\)\s*\))"
        R"(|REQD_SUB_GROUP_SIZE\s*\(([^()]*)\))");

    std::string first_arg;

    // Rewrite each occurrence into a redefinition at the point it appears, so a batch entry holding several kernels
    // with different sub-group sizes still gets the right value for each of them. to_size gives the size an
    // occurrence asks for and whether the occurrence itself stays in the source.
    auto rewrite = [&](const std::string& text,
                       const std::regex& re,
                       const std::function<std::pair<std::string, bool>(const std::smatch&)>& to_size) {
        std::string out;
        size_t last = 0;
        const auto comments = comment_ranges(text);
        for (std::sregex_iterator it(text.begin(), text.end(), re), end; it != end; ++it) {
            const size_t pos = static_cast<size_t>(it->position());
            if (on_define_line(text, pos) || in_ranges(comments, pos)) {
                continue;
            }
            const auto [arg, keep_match] = to_size(*it);
            if (arg.empty()) {
                continue;
            }
            if (first_arg.empty()) {
                first_arg = arg;
            }
            out.append(text, last, pos - last);
            out += redefine_directive(arg);
            if (keep_match) {
                out += it->str();
            }
            last = pos + static_cast<size_t>(it->length());
        }
        out.append(text, last, std::string::npos);
        return out;
    };

    const auto stated_size = [](const std::smatch& m) {
        // The macro form expands to nothing on this target, so leaving it in place keeps the source closer to the
        // original. The raw attribute has to go - the NVIDIA compiler does not know it.
        return m[1].matched ? std::make_pair(m[1].str(), false) : std::make_pair(m[2].str(), true);
    };
    std::string modified = rewrite(src, request_re, stated_size);

    // No stated sub-group size: fall back to the one the work-group size implies, if any. An explicit declaration
    // always wins - softmax_gpu_bf asks for 16 and launches a work-group of 4. The whole attribute is matched so the
    // redefinition lands ahead of it rather than between its parentheses; reqd_work_group_size is standard OpenCL,
    // so the attribute itself stays.
    if (first_arg.empty()) {
        static const std::regex wg_re(
            R"(__attribute__\s*\(\s*\(\s*reqd_work_group_size\s*\(([^()]*)\)\s*\)\s*\))");
        modified = rewrite(src, wg_re, [](const std::smatch& m) {
            return std::make_pair(sub_group_size_from_work_group_size(m[1].str()), true);
        });
    }

    if (first_arg.empty()) {
        return src;
    }

    // Repeat the redefinition at the very top of the entry. An entry holds one kernel's jit + source + undefs, and the
    // source commonly defines helper functions ahead of the kernel that use sub-group built-ins as well; those sit
    // before the attribute and would otherwise compile against the previous entry's value. Macro bodies expand
    // lazily, so naming a jit constant that is only defined further down (SUBGROUP_SIZE, SIMD, ...) is fine.
    modified = redefine_directive(first_arg) + modified;

    // Restore the default at the end of the entry so nothing leaks into the next kernel of the batch.
    modified += "\n#undef _NV_SUB_GROUP_SIZE\n#define _NV_SUB_GROUP_SIZE 32u\n";
    return modified;
}

void kernels_cache::get_program_source(const kernels_code& kernels_source_code, std::vector<kernels_cache::batch_program>* all_batches) const {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "KernelsCache::BuildAll::GetProgramSource");
    std::map<std::string, std::tuple<int32_t, std::vector<batch_program>>> program_buckets;

    for (const auto& k : kernels_source_code) {
        const auto& code = k.second;
        bool dump_custom_program = code.dump_custom_program;

        for (size_t kernel_part_idx = 0; kernel_part_idx < code.kernel_strings.size(); kernel_part_idx++) {
            const auto& kernel_string = code.kernel_strings[kernel_part_idx];
            std::string full_code = kernel_string->jit + kernel_string->str + kernel_string->undefs;
            std::string entry_point = kernel_string->entry_point;
            std::string options = kernel_string->options;
            bool batch_compilation = kernel_string->batch_compilation;
            bool is_cm = kernel_string->language == kernel_language::CM;

            // Order matters for cm options
            if (batch_compilation && !is_cm) {
                options = reorder_options(options);
            }

            std::string key = options;

            if (!batch_compilation) {
                key += " __PROGRAM__" + std::to_string(program_buckets.size());
            }

            if (dump_custom_program) {
                key += " __DUMP_CUSTOM_PROGRAM__";  // Adding label to key so it would be separated from other programs
            }
            key += " __LANG__" + std::to_string(static_cast<size_t>(kernel_string->language));

            auto& bucket_id = std::get<0>(program_buckets[key]);
            auto& current_bucket = std::get<1>(program_buckets[key]);
            if (current_bucket.empty()) { // new bucket
                const auto& batch_id = 0;
                // increase bucket id if and only if new bucket comes
                bucket_id = static_cast<int32_t>(program_buckets.size() - 1);
                current_bucket.push_back(batch_program(bucket_id, batch_id, options, batch_headers, kernel_string->language));
            }

            // Create new kernels batch when the limit is reached
            // and current kernel's entry_point is duplicated in this kernels batch
            if (current_bucket.back().kernels_counter >= _config.get_max_kernels_per_batch()
                || current_bucket.back().entry_point_to_id.find(entry_point) != current_bucket.back().entry_point_to_id.end()) {
                const auto& batch_id = static_cast<int32_t>(current_bucket.size());
                current_bucket.push_back(batch_program(bucket_id, batch_id, options, batch_headers, kernel_string->language));
            }

            auto& current_batch = current_bucket.back();
            current_batch.dump_custom_program = dump_custom_program;
            current_batch.entry_point_to_id.emplace(entry_point, std::make_pair(code.params, kernel_part_idx));

            current_batch.has_microkernels |= kernel_string->has_microkernels;

            current_batch.source.push_back(std::move(full_code));
            current_batch.kernels_counter++;
        }
    }

    // Compute hash value for each batch
    // Hash calculation might require additional optimizations, but currently execution time of this part is much smaller than loading
    // of the precompiled binaries or get_undef_jit calls
    // Hash is computed for string that contains compilation options + driver version +
    // full source code (jit + template + undef sections) of all kernels in the batches
    for (auto& c : program_buckets) {
        auto options = c.first;
        auto& batches = std::get<1>(c.second);
        for (auto& b : batches) {
            auto find_and_remove_includes = [](const std::string& code, std::vector<std::string>& required_headers) {
                std::regex include_regex(R"(#include\s+\"([^\"]+)\")");
                std::string processed_kernel;
                std::sregex_iterator it(code.begin(), code.end(), include_regex);
                std::sregex_iterator end;

                size_t last_pos = 0;
                for (; it != end; ++it) {
                    auto header_name = (*it)[1].str();
                    header_name = header_name.substr(header_name.find_last_of("/") + 1);
                    header_name = header_name.substr(0, header_name.find_last_of("."));
                    required_headers.push_back(header_name);
                    processed_kernel += code.substr(last_pos, it->position() - last_pos);
                    last_pos = it->position() + it->length();
                }
                processed_kernel += code.substr(last_pos);
                return processed_kernel;
            };

            auto process_batch_includes = [find_and_remove_includes](kernels_cache::batch_program& prog) {
                std::list<std::string> sources_to_process(prog.source.begin(), prog.source.end());
                const size_t kernel_entries = prog.source.size();

                prog.source.clear();
                std::list<std::string> all_headers;
                while (!sources_to_process.empty()) {
                    std::vector<std::string> new_headers;
                    auto source = sources_to_process.front();
                    sources_to_process.pop_front();
                    prog.source.insert(prog.source.begin(), find_and_remove_includes(source, new_headers));
                    for (auto& header : new_headers) {
                        if (std::find(all_headers.begin(), all_headers.end(), header) == all_headers.end()) {
                            all_headers.push_front(header);
                            std::string_view header_code =
#ifdef ENABLE_CM_FOR_GPU
                                prog.language == kernel_language::OCLC_V2
                                ? ov::intel_gpu::ocl::SourcesDB::get_kernel_header(header)
                                : ov::intel_gpu::cm::SourcesDB::get_kernel_header(header);
#else
                                ov::intel_gpu::ocl::SourcesDB::get_kernel_header(header);
#endif
                            sources_to_process.push_back(std::string(header_code) + "\n");
                        }
                    }
                }

                if (prog.language == kernel_language::CM) {
                    prog.source.insert(prog.source.begin(), "#include <cm/cm.h>\n#include <cm/cmtl.h>\n");
                }

                // Every kernel is processed before any header, and each processed entry is inserted at the front,
                // so the headers end up ahead of all the kernels: everything the kernels did not contribute is a header.
                prog.header_count = prog.source.size() - kernel_entries;
            };

            if (b.language == kernel_language::OCLC_V2 || b.language == kernel_language::CM)
                process_batch_includes(b);

            // For the NVIDIA SYCL backend, add the compatibility headers (fp16 support, math overloads,
            // sub-group built-ins, work-group collectives, etc.) that the native NVIDIA OpenCL runtime lacks. They go
            // into b.source like any other header, so the hash below covers them and the dumped program shows them.
            // The program is laid out as
            //     prelude | batch headers | sub-group overrides | kernels
            // The overrides must sit between the headers and the kernels: they shadow names that sub_group_block_read.cl
            // and friends declare as functions, so emitting them any earlier would macro-expand those names inside their
            // own declarations. See nvidia/compat_sg_overrides.cl.
            if (_use_nvidia_compat) {
                const auto compat = get_nvidia_compat_headers();
                const size_t header_count = std::min(b.header_count, b.source.size());
                b.source.insert(b.source.begin() + static_cast<std::ptrdiff_t>(header_count), compat.sg_overrides);
                b.source.insert(b.source.begin(), compat.prelude);
                b.header_count = header_count + 2;
            }

            std::string full_code = options + " " + _device->get_info().driver_version;
            full_code += _device->get_info().dev_name;
            // Partition the kernel cache per runtime: OCL and ZE store incompatible native
            // binaries, so their .cl_cache files must never share a hash (wrong-format load).
            full_code += get_runtime_cache_tag();
            for (auto& ss : b.source)
                full_code += ss;

            b.hash_value = std::hash<std::string>()(full_code);

            std::string dump_sources_dir = GPU_DEBUG_VALUE_OR(_config.get_dump_sources_path(), "");

            // Add -g -s to build options to allow IGC assembly dumper to associate assembler sources with corresponding OpenCL kernel code lines
            // Should be used with the IGC_ShaderDump option
            // Note: Skip adding -g -s for CM kernels as these options are not supported by CM compiler, nor for the NVIDIA OpenCL compiler.
            if (!dump_sources_dir.empty() && b.language != kernel_language::CM && !_use_nvidia_compat) {
                std::string current_dump_file_name = std::move(dump_sources_dir);
                if (!current_dump_file_name.empty() && current_dump_file_name.back() != '/')
                    current_dump_file_name += '/';

                current_dump_file_name += "clDNN_program_" + std::to_string(_prog_id) + "_bucket_" + std::to_string(b.bucket_id)
                                        + "_part_" + std::to_string(b.batch_id) + "_" + std::to_string(b.hash_value) + ".cl";

                b.options += " -g -s " + current_dump_file_name;
            }

            all_batches->push_back(b);
        }
    }
}

kernels_cache::kernels_cache(engine& engine,
                             const ExecutionConfig& config,
                             uint32_t prog_id,
                             std::shared_ptr<ov::threading::ITaskExecutor> task_executor,
                             const std::map<std::string, std::string>& batch_headers)
    : _device(engine.get_device())
    , _builder(engine.create_kernel_builder())
    , _task_executor(task_executor)
    , _config(config)
    , _prog_id(prog_id)
    , batch_headers(std::move(batch_headers)) {
    // The NVIDIA SYCL backend always compiles with clBuildProgram on the NVIDIA OpenCL runtime and launches the PTX
    // through CUDA, so it always needs the compatibility headers.
    _use_nvidia_compat = engine.runtime_type() == runtime_types::sycl && _device->get_info().vendor_id == NVIDIA_VENDOR_ID;
}

kernels_cache::nvidia_compat_headers kernels_cache::get_nvidia_compat_headers() const {
    nvidia_compat_headers result;

    // A kernels_cache built without batch headers (e.g. for a standalone probe kernel) compiles its kernels
    // without the compatibility headers. Otherwise the batch headers come from the primitive DB, which lacks them
    // when it was generated without primitive_db_gen.py -nvidia, e.g. a stale codegen output in a build directory
    // reconfigured for NVIDIA.
    if (batch_headers.empty()) {
        return result;
    }

    // Every batch gets all five, so each one is required here.
    auto get_header = [this](const char* name) -> const std::string& {
        auto it = batch_headers.find(name);
        OPENVINO_ASSERT(it != batch_headers.end(), "[GPU] NVIDIA compatibility header '", name,
                        "' is missing from the primitive DB. It must be generated with GPU_VENDOR=NVIDIA "
                        "(primitive_db_gen.py -nvidia); rebuild the kernel_selector codegen output.");
        return it->second;
    };

    // The NVIDIA compatibility prelude, split by feature. Emitted in this order: each header relies only
    // on the ones listed before it (see the "Depends on" notes at the top of each file).
    // compat_wg is added to every batch, not only to those calling a work-group collective: ptxas allocates the
    // shared buffer it declares only to the kernels that reference it, so the other kernels carry none of it.
    static const char* const nvidia_prelude_names[] = {
        "nvidia/compat_common",  // extension pragmas, feature macros, linear ids
        "nvidia/compat_half",    // half limits, math, conversions, reinterprets, relational
        "nvidia/compat_sg",      // sub-group emulation
        "nvidia/compat_wg",      // work-group collectives
    };
    for (const char* name : nvidia_prelude_names) {
        result.prelude += get_header(name);
    }

    result.sg_overrides = get_header("nvidia/compat_sg_overrides");

    return result;
}

std::string kernels_cache::get_combined_source(const batch_program& batch) const {
    if (!_use_nvidia_compat) {
        return join_strings(batch.source);
    }

    // The NVIDIA compatibility headers are already in batch.source (see get_program_source()). What is left is
    // re-expressing each kernel's sub-group size. That is done here rather than before hashing because it runs a few
    // regexes over every kernel, which is only worth paying when the batch actually gets compiled (or dumped); the
    // result follows from the kernel source alone, which the hash already covers.
    std::string combined_source;
    const size_t header_count = std::min(batch.header_count, batch.source.size());
    for (size_t i = 0; i < header_count; ++i) {
        combined_source += batch.source[i];
    }
    for (size_t i = header_count; i < batch.source.size(); ++i) {
        combined_source += nvidia_replace_reqd_sub_group_size(batch.source[i]);
    }
    return combined_source;
}

void kernels_cache::build_batch(const batch_program& batch, compiled_kernels& compiled_kernels) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "KernelsCache::build_batch");
    bool dump_sources = batch.dump_custom_program;
    std::string dump_sources_dir = GPU_DEBUG_VALUE_OR(_config.get_dump_sources_path(), "");
    GPU_DEBUG_IF(!dump_sources_dir.empty()) {
        dump_sources = true;
    }
    // Skip batches that do not contain kernels
    if (batch.kernels_counter == 0) {
        return;
    }

    std::string current_dump_file_name;
    if (dump_sources) {
        current_dump_file_name = std::move(dump_sources_dir);
        if (!current_dump_file_name.empty() && current_dump_file_name.back() != '/')
            current_dump_file_name += '/';

        // Use .cm extension for CM kernels, .cl for OpenCL kernels
        std::string ext = (batch.language == kernel_language::CM) ? ".cm" : ".cl";
        current_dump_file_name += "clDNN_program_" + std::to_string(_prog_id) + "_bucket_" + std::to_string(batch.bucket_id)
                               + "_part_" + std::to_string(batch.batch_id) + "_" + std::to_string(batch.hash_value) + ext;
    }

    // The source the builder compiles. It is built ahead of the cache lookup only when it is dumped, so that the dump
    // shows exactly what the compiler gets; otherwise it is built only on a cache miss.
    std::string combined_source;
    std::ofstream dump_file;
    if (dump_sources) {
        combined_source = get_combined_source(batch);
        dump_file.open(current_dump_file_name);
        if (dump_file.good()) {
            dump_file << combined_source;
        }
    }
    std::string cached_bin_name = get_cache_path() + std::to_string(batch.hash_value) + ".cl_cache";
    ///////////////////////////////////////////////////////////////////////////////////
    std::vector<uint8_t> precompiled;
    if (is_cache_enabled()) {
        std::lock_guard<std::mutex> lock(cacheAccessMutex);
        precompiled = ov::util::load_binary(ov::util::make_path(cached_bin_name));
    }
    std::vector<kernel::ptr> kernels;
    if (!precompiled.empty()) {
        _builder->build_kernels(precompiled.data(), precompiled.size(), KernelFormat::NATIVE_BIN, "", kernels);
    } else {
        if (combined_source.empty()) {
            combined_source = get_combined_source(batch);
        }

        _builder->build_kernels(combined_source.data(), combined_source.size(), KernelFormat::SOURCE, batch.options, kernels);
        OPENVINO_ASSERT(!kernels.empty(), "[GPU] Expected to compile more than 0 kernels in the batch");
        OPENVINO_ASSERT(kernels.size() == batch.kernels_counter, "[GPU] Number of compiled kernels is different than kernel batch size");
        if (dump_sources && dump_file.good()) {
            dump_file << "\n/* Build Log:\n";
            // Retreive build log from the first kernel only
            // It should be the same for all kernels in batch
            dump_file << kernels[0]->get_build_log();
            dump_file << "\n*/\n";
        }
        if (batch.has_microkernels) {
#ifdef OV_GPU_WITH_ONEDNN_MICROKERNELS
            OPENVINO_ASSERT(batch.kernels_counter == 1 && kernels.size() == 1);
            std::vector<uint8_t> binary = kernels[0]->get_binary();
            kernels.clear();
            // Update binary and rebuild kernel
            gemmstone::microkernel::fuse(binary, combined_source.c_str());
            _builder->build_kernels(binary.data(), binary.size(), KernelFormat::NATIVE_BIN, "", kernels);
#else  // OV_GPU_WITH_ONEDNN_MICROKERNELS
            OPENVINO_THROW("[GPU] Can't compile kernel w/ microkernels as onednn is not available");
#endif  // OV_GPU_WITH_ONEDNN_MICROKERNELS
        }
        if (is_cache_enabled()) {
                // If kernels caching is enabled, then we save compiled bucket to binary file with name ${code_hash_value}.cl_cache
                // Note: Bin file contains full bucket, not separate kernels, so kernels reuse across different models is quite limited
                // Bucket size can be changed by max_kernels_per_batch config option, but forcing it to 1 will lead to much longer
                // compile time.
                std::vector<uint8_t> binary = kernels[0]->get_binary();
                std::lock_guard<std::mutex> lock(cacheAccessMutex);
                ov::intel_gpu::save_binary(cached_bin_name, binary);
        }
    }
    {
        std::lock_guard<std::mutex> lock(_mutex);
        for (auto& k : kernels) {
            auto entry_point = k->get_id();
            const auto& iter = batch.entry_point_to_id.find(entry_point);
            if (iter != batch.entry_point_to_id.end()) {
                const auto& params = iter->second.first;
                auto kernel_part_idx = iter->second.second;
                if (compiled_kernels.find(params) != compiled_kernels.end()) {
                    compiled_kernels[params].push_back(std::make_pair(k, kernel_part_idx));
                } else {
                    compiled_kernels[params] = { std::make_pair(k, kernel_part_idx) };
                }
                if (_kernel_batch_hash.find(params) == _kernel_batch_hash.end()) {
                   _kernel_batch_hash[params] = batch.hash_value;
                }
            } else {
                throw std::runtime_error("Could not find entry point");
            }
        }
    }
}

kernel::ptr kernels_cache::get_kernel_from_cached_kernels(std::string id) const {
    auto res = _cached_kernels.find(id);
    OPENVINO_ASSERT(_cached_kernels.end() != res, "[GPU] Kernel " + id + " not found in the cached kernel cache!");

    return res->second->clone(_reuse_kernels);
}

std::vector<kernel::ptr> kernels_cache::get_kernels(const kernel_impl_params& params) const {
    OPENVINO_ASSERT((_pending_compilation == false), "Kernel cache is not compiled, call build_all() first!");

    std::string current_node_id;
    if (params.desc) {
        current_node_id = params.desc->id;
    }
    auto res = _kernels.find(params);
    OPENVINO_ASSERT(_kernels.end() != res, "Kernel for {" + current_node_id + "} is not found in the kernel cache!");
    OPENVINO_ASSERT(!res->second.empty(), "Number of kernels should not be zero for " + current_node_id);

    std::vector<kernel::ptr> kernels(res->second.size());
    for (const auto& k : res->second) {
        const auto& kernel_ptr = k.first;
        auto kernel_part_idx = k.second;
        kernels[kernel_part_idx] = kernel_ptr->clone(_reuse_kernels);
    }
    return kernels;
}

void kernels_cache::build_all() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "KernelsCache::BuildAll");
    if (!_pending_compilation)
        return;

    std::vector<batch_program> batches;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        get_program_source(_kernels_code, &batches);
    }

    // build_batch crashes randomly when threaded while running from a Microsoft Store app
    // it seems to be a bug in Intel's graphics driver, disabling threading is a work around
    auto use_threads{true};
#if defined(_WIN32) && defined(NTDDI_WIN10_RS5)
    UINT32 length{0};
    auto error_code{GetCurrentPackageFullName(&length, nullptr)};
    // If we get this error, it means we're a regular desktop application, and we can use threads
    use_threads = error_code == APPMODEL_ERROR_NO_PACKAGE;
#endif

    if (_task_executor && use_threads) {
        std::exception_ptr exception;
        std::vector<ov::threading::Task> tasks;
        for (size_t idx = 0; idx < batches.size(); idx++) {
            auto& batch = batches[idx];
            tasks.push_back([this, &batch, &exception] {
                try {
                    build_batch(batch, _kernels);
                } catch (...) {
                    exception = std::current_exception();
                }
            });
        }
        _task_executor->run_and_wait(tasks);
        tasks.clear();

        if (exception) {
            std::rethrow_exception(exception);
        }
    } else {
        for (size_t idx = 0; idx < batches.size(); idx++) {
            build_batch(batches[idx], _kernels);
        }
    }

    {
        std::lock_guard<std::mutex> lock(_mutex);
        _kernels_code.clear();
        _pending_compilation = false;
#if defined(OPENVINO_GNU_LIBC) && !defined(__ANDROID__)
    //  NOTE: In linux, without malloc_trim, an amount of the memory used by compilation is not being returned to system thought they are freed.
    //  (It is at least 500 MB when we perform parallel compilation)
    //  It is observed that freeing the memory manually with malloc_trim saves significant amount of the memory.
    //  Also, this is not happening in Windows.
    //  So, added malloc_trim for linux build until we figure out a better solution.
        malloc_trim(0);
#endif
    }
}

void kernels_cache::reset() {
    _kernels.clear();
    _kernels_code.clear();
    _kernel_batch_hash.clear();
    _pending_compilation = false;
}

void kernels_cache::add_kernels_source(const kernel_impl_params& params,
                                        const std::vector<std::shared_ptr<kernel_string>>& kernel_sources,
                                        bool dump_custom_program) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (!kernel_sources.empty() && (_kernels_code.find(params) == _kernels_code.end())) {
        auto res = _kernels_code.insert({params, {kernel_sources, params, dump_custom_program}});

        assert(_kernels.find(params) == _kernels.end());
        if (res.second) {
            _pending_compilation = true;
        }
    }
}

std::string kernels_cache::get_cached_kernel_id(kernel::ptr kernel) const {
    auto program_binaries = kernel->get_binary();

    auto iter = _cached_binaries.find(program_binaries);
    OPENVINO_ASSERT(iter != _cached_binaries.end(), "[GPU] Not found cached kernel binaries");

    return kernel->get_id() + "@" + std::to_string(iter->second);
}

std::vector<std::string> kernels_cache::get_cached_kernel_ids(const std::vector<kernel::ptr>& kernels) const {
    std::vector<std::string> kernel_ids;

    for (const auto& kernel : kernels) {
        auto key = get_cached_kernel_id(kernel);
        kernel_ids.emplace_back(key);
    }

    return kernel_ids;
}

void kernels_cache::add_to_cached_kernels(const std::vector<kernel::ptr>& kernels) {
    static std::atomic<uint32_t> id_gen{0};

    for (const auto& kernel : kernels) {
        auto program_binaries = kernel->get_binary();

        std::lock_guard<std::mutex> lock(_mutex);
        auto iter = _cached_binaries.find(program_binaries);
        if (iter == _cached_binaries.end()) {
            _cached_binaries[program_binaries] = id_gen++;
        }
        auto key = get_cached_kernel_id(kernel);

        if (_cached_kernels.find(key) == _cached_kernels.end()) {
            _cached_kernels[key] = kernel;
        }
    }
}

void kernels_cache::save(BinaryOutputBuffer& ob) const {
    ob << _cached_binaries.size();

    // zebin (ELF format) or SPIR-V.
    auto is_driver_agnostic = [](const std::vector<unsigned char>& bin) {
        constexpr uint32_t ELF_MAGIC = 0x464C457F;
        constexpr uint32_t SPIRV_MAGIC = 0x07230203;

        if (bin.size() < sizeof(uint32_t)) {
            return false;
        }
        uint32_t magic;
        std::memcpy(&magic, bin.data(), sizeof(magic));
        return magic == ELF_MAGIC || magic == SPIRV_MAGIC;
    };

    for (const auto& cached_binary : _cached_binaries) {
        auto is_driver_agnostic_binary = is_driver_agnostic(cached_binary.first);

        ob << cached_binary.second;
        ob << cached_binary.first;
        ob << is_driver_agnostic_binary;
        if (!is_driver_agnostic_binary) {
            auto driver_version = _device->get_info().driver_version;
            ob << driver_version;
        }
    }
}

void kernels_cache::load(BinaryInputBuffer& ib) {
    std::unordered_map<uint32_t, std::vector<unsigned char>> precompiled_kernels;

    size_t num_cached_binaries;
    ib >> num_cached_binaries;
    for (size_t i = 0; i < num_cached_binaries; ++i) {
        bool is_driver_agnostic_binary = true;

        uint32_t id;
        ib >> id;
        ib >> precompiled_kernels[id];
        ib >> is_driver_agnostic_binary;
        if (!is_driver_agnostic_binary) {
            // Legacy patchtoken path
            std::string driver_version, current_driver_version;
            ib >> driver_version;
            current_driver_version = _device->get_info().driver_version;

            if (driver_version != current_driver_version) {
                OPENVINO_THROW("Driver version mismatch in cached patchtoken kernels");
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(_mutex);
        _cached_kernels.clear();

        for (auto& precompiled_kernel : precompiled_kernels) {
            std::vector<kernel::ptr> kernels;
            _builder->build_kernels(precompiled_kernel.second.data(), precompiled_kernel.second.size(), KernelFormat::NATIVE_BIN, "", kernels);
            for (auto& k : kernels) {
                const auto& entry_point = k->get_id();
                std::string cached_kernel_id = entry_point + "@" + std::to_string(precompiled_kernel.first);
                const auto& iter = _cached_kernels.find(cached_kernel_id);
                if (iter == _cached_kernels.end()) {
                    _cached_kernels[cached_kernel_id] = k;
                }
            }
        }
    }
}

kernels_cache::compiled_kernels kernels_cache::compile(const kernel_impl_params& params,
                                            const std::vector<std::shared_ptr<kernel_string>>& kernel_sources,
                                            bool dump_custom_program) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "KernelsCache::compile");
    if (kernel_sources.empty())
        return {};

    kernels_code t_kernels_code;

    // Get kernels code from kernel sources
    for (size_t k = 0; k < kernel_sources.size(); ++k) {
        t_kernels_code.insert({params, {kernel_sources, params, dump_custom_program}});
    }

    // Create batches
    std::vector<batch_program> batches;
    get_program_source(t_kernels_code, &batches);

    compiled_kernels output_kernels;
    // Build batches
    for (size_t idx = 0; idx < batches.size(); ++idx) {
        build_batch(batches[idx], output_kernels);
    }

    OPENVINO_ASSERT(output_kernels.size() == 1, "Only the kernels of the single primitive should be compiled.");

    t_kernels_code.clear();
#if defined(OPENVINO_GNU_LIBC) && !defined(__ANDROID__)
    //  NOTE: In linux, without malloc_trim, an amount of the memory used by compilation is not being returned to system thought they are freed.
    //  (It is at least 500 MB when we perform parallel compilation)
    //  It is observed that freeing the memory manually with malloc_trim saves significant amount of the memory.
    //  Also, this is not happening in Windows.
    //  So, added malloc_trim for linux build until we figure out a better solution.
        malloc_trim(0);
#endif

    return output_kernels;
}
}  // namespace cldnn
