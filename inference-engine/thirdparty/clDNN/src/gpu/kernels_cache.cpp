/*
// Copyright (c) 2016-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "kernels_cache.h"
#include "ocl_toolkit.h"
#include <algorithm>
#include <cassert>
#include <sstream>
#include <fstream>
#include <set>
#include <string>
#include <memory>
#include <utility>

#include "kernel_selector_helper.h"
#include "cldnn_itt.h"

#ifndef ENABLE_UNICODE_PATH_SUPPORT
# ifdef _WIN32
#  if defined __INTEL_COMPILER || defined _MSC_VER
#   define ENABLE_UNICODE_PATH_SUPPORT
#  endif
# elif defined(__GNUC__) && (__GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 2)) || defined(__clang__)
#  define ENABLE_UNICODE_PATH_SUPPORT
# endif
#endif

#ifndef _WIN32
#ifdef ENABLE_UNICODE_PATH_SUPPORT
#include <locale>
#include <codecvt>
#endif
#else
#include <Windows.h>
#endif

namespace {

std::mutex cacheAccessMutex;

#ifdef ENABLE_UNICODE_PATH_SUPPORT
std::wstring multiByteCharToWString(const char* str) {
#ifdef _WIN32
    int strSize = static_cast<int>(std::strlen(str));
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str, strSize, NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str, strSize, &wstrTo[0], size_needed);
    return wstrTo;
#else
    std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_encoder;
    std::wstring result = wstring_encoder.from_bytes(str);
    return result;
#endif  // _WIN32
}
#endif  // ENABLE_UNICODE_PATH_SUPPORT

static std::vector<unsigned char> loadBinaryFromFile(std::string path) {
    std::lock_guard<std::mutex> lock(cacheAccessMutex);

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring widefilename = multiByteCharToWString(path.c_str());
    const wchar_t* filename = widefilename.c_str();
    FILE *fp = _wfopen(filename, L"rb");
#else
    const char* filename = path.c_str();
    FILE *fp = fopen(filename, "rb");
#endif

    if (fp) {
        fseek(fp, 0, SEEK_END);
        size_t nsize = (size_t)ftell(fp);

        fseek(fp, 0, SEEK_SET);

        std::vector<unsigned char> ret(nsize);

        auto res = fread(ret.data(), sizeof(unsigned char), nsize, fp);
        (void)res;
        fclose(fp);
        return ret;
    }

    return {};
}

static void saveBinaryToFile(std::string path, const std::vector<unsigned char> buffer) {
    std::lock_guard<std::mutex> lock(cacheAccessMutex);
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring widefilename = multiByteCharToWString(path.c_str());
    const wchar_t* filename = widefilename.c_str();
#else
    const char* filename = path.c_str();
#endif
    std::ofstream out_file(filename, std::ios::out | std::ios::binary);
    if (out_file.is_open()) {
        out_file.write((char*)&buffer[0], buffer.size());
    }
}

std::string get_undef_jit(cldnn::gpu::kernels_cache::source_code org_source_code) {
    const std::string white_space_with_new_lines = " \t\r\n";
    const std::string white_space = " \t";

    size_t current_pos = 0;

    const std::string define = "define";

    std::set<std::string> to_undef;
    for (const auto& source : org_source_code) {
        do {
            size_t index_to_hash = source.find_first_not_of(white_space_with_new_lines, current_pos);
            if (index_to_hash != std::string::npos && source[index_to_hash] == '#') {
                size_t index_define = source.find_first_not_of(white_space, index_to_hash + 1);

                if (index_define != std::string::npos && !source.compare(index_define, define.size(), define)) {
                    size_t index_to_name = source.find_first_not_of(white_space, index_define + define.size());
                    if (index_to_name != std::string::npos) {
                        size_t index_to_end_name =
                            source.find_first_of(white_space_with_new_lines + "(", index_to_name);
                        if (index_to_end_name == std::string::npos) {
                            index_to_end_name = source.size();
                        }
                        std::string name = source.substr(index_to_name, index_to_end_name - index_to_name);
                        to_undef.insert(name);
                    }
                }
            }

            current_pos = source.find_first_of('\n', current_pos + 1);
        } while (current_pos != std::string::npos);
    }

    std::string undefs;
    for (const auto& name : to_undef) {
        undefs += "#ifdef " + name + "\n";
        undefs += "#undef " + name + "\n";
        undefs += "#endif\n";
    }

    return undefs;
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

inline bool does_options_support_batch_compilation(const std::string& options) {
    return options.find("-D") == std::string::npos && options.find("-I") == std::string::npos;
}

}  // namespace

namespace cldnn {
namespace gpu {

std::string kernels_cache::get_cache_path() const {
    auto path = _context.get_configuration().kernels_cache_path;
    if (path.empty()) {
        return {};
    }

    if (path.back() != '/' && path.back() != '\\') {
        path += "/";
    }
    return path;
}

bool kernels_cache::is_cache_enabled() const {
    return !_context.get_configuration().kernels_cache_path.empty();
}

size_t kernels_cache::get_max_kernels_per_batch() const {
    return 10;
}

kernels_cache::sorted_code kernels_cache::get_program_source(const kernels_code& kernels_source_code) const {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "KernelsCache::BuildAll::GetProgramSource");
    sorted_code scode;

    for (const auto& code : kernels_source_code) {
        std::string full_code = code.kernel_strings->jit + code.kernel_strings->str;
        full_code += get_undef_jit({full_code});
        const source_code org_source_code = { full_code };
        std::string entry_point = code.kernel_strings->entry_point;
        std::string options = code.kernel_strings->options;
        bool batch_compilation = code.kernel_strings->batch_compilation;
        bool dump_custom_program = code.dump_custom_program;
        bool one_time_kernel = code.one_time_kernel;

        batch_compilation &= does_options_support_batch_compilation(options);

        if (batch_compilation) {
            options = reorder_options(options);
        }

        std::string key = options;

        if (batch_compilation == false) {
            key += " __PROGRAM__" + std::to_string(scode.size());
        }

        if (dump_custom_program) {
            key += " __DUMP_CUSTOM_PROGRAM__";  // Adding label to key so it would be separated from other programs
        }

        if (one_time_kernel) {
            key += " __ONE_TIME__";
        }

        auto& current_bucket = scode[key];
        current_bucket.dump_custom_program = dump_custom_program;
        current_bucket.one_time = one_time_kernel;

        if (current_bucket.source.empty()) {
            current_bucket.options = options;
        }

        // Create new kernels bucket when the limit is reached
        if ((current_bucket.kernels_counter % get_max_kernels_per_batch()) == 0) {
            current_bucket.source.push_back({});
        }

        current_bucket.entry_point_to_id[entry_point] = code.id;
        assert(org_source_code.size() == 1);

        current_bucket.source.back().push_back(std::move(org_source_code.front()));

        current_bucket.kernels_counter++;
    }

    // Compute hash value for each bucket
    // Hash calculation might require additional optimizations, but currently execution time of this part is much smaller than loading
    // of the precompiled binaries or get_undef_jit calls
    // Hash is computed for string that contains compilation options + driver version +
    // full source code (jit + template + undef sections) of all kernels in the bucket
    for (auto& c : scode) {
        program_code& code = c.second;
        auto options = c.first;
        for (size_t i = 0; i < code.source.size(); i++) {
            std::string full_code = options + " " + _context.get_device_info().driver_version;
            for (auto& ss : code.source[i])
                full_code += ss;
            code.hash_values.push_back(std::hash<std::string>()(full_code));
        }
    }

    return scode;
}

kernels_cache::kernels_cache(gpu_toolkit& context, uint32_t prog_id) : _context(context), _prog_id(prog_id) {}

kernels_cache::kernel_id kernels_cache::set_kernel_source(
    const std::shared_ptr<kernel_selector::kernel_string>& kernel_string,
    bool dump_custom_program,
    bool one_time_kernel) {
    std::lock_guard<std::mutex> lock(_context.get_cache_mutex());

    // we need unique id in order to avoid conflict across topologies.
    const auto kernel_num = _kernels.size() + _kernels_code.size();
    kernels_cache::kernel_id id = kernel_string->entry_point + "_" + std::to_string(kernel_num);

    auto res = _kernels_code.emplace( kernel_string, id, dump_custom_program, one_time_kernel );

    assert(_kernels.find(id) == _kernels.end());
    if (res.second) {
        _pending_compilation = true;
    }
    return id;
}

static std::vector<unsigned char> getProgramBinaries(cl::Program program) {
    // Get the size of the program binary in bytes.
    std::vector<size_t> binary_sizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();

    if (binary_sizes.size() != 1)
        throw std::runtime_error("Invalid binaries count");

    size_t binary_size = binary_sizes.front();
    // Binary is not available for the device.
    if (binary_size == 0)
        throw std::runtime_error("Binary is not avaliable after program build");

    // Get program binary.
    return program.getInfo<CL_PROGRAM_BINARIES>().front();
}

kernels_cache::kernels_map kernels_cache::build_program(const program_code& program_source) const {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "KernelsCache::BuildProgram");
    static uint32_t current_file_index = 0;

    bool dump_sources = !_context.get_configuration().ocl_sources_dumps_dir.empty() || program_source.dump_custom_program;

    std::string dump_file_name = "";
    if (dump_sources) {
        dump_file_name = _context.get_configuration().ocl_sources_dumps_dir;
        if (!dump_file_name.empty() && dump_file_name.back() != '/')
            dump_file_name += '/';

        dump_file_name += "clDNN_program_" + std::to_string(current_file_index++) + "_part_";
    }

    try {
        kernels_map kmap;
        std::string err_log;  // accumulated build log from all program's parts (only contains messages from parts which
                              // failed to compile)

        uint32_t part_idx = 0;
        for (size_t i = 0; i < program_source.source.size(); i++) {
            auto sources_bucket_to_compile = program_source.source[i];
            const auto& hash_value = program_source.hash_values[i];
            std::string cached_bin_name = get_cache_path() + std::to_string(hash_value) + ".cl_cache";
            cl::Program::Binaries precompiled_kernels = {};
            if (is_cache_enabled()) {
                // Try to load file with name ${hash_value}.cl_cache which contains precompiled kernels for current bucket
                // If read is successful, then remove kernels from compilation bucket
                auto bin = loadBinaryFromFile(cached_bin_name);
                if (!bin.empty()) {
                    precompiled_kernels.push_back(bin);
                }
            }
            auto current_dump_file_name = dump_file_name + std::to_string(part_idx++) + ".cl";
            std::ofstream dump_file;

            if (dump_sources) {
                dump_file.open(current_dump_file_name);

                if (dump_file.good()) {
                    for (auto& s : sources_bucket_to_compile)
                        dump_file << s;
                }
            }

            try {
                cl::vector<cl::Kernel> kernels;
                // Run compilation
                if (precompiled_kernels.empty()) {
                    cl::Program program(_context.context(), sources_bucket_to_compile);
                    {
                        OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "KernelsCache::BuildProgram::RunCompilation");
                        program.build(_context.device(), program_source.options.c_str());
                    }

                    if (dump_sources && dump_file.good()) {
                        dump_file << "\n/* Build Log:\n";
                        for (auto& p : program.getBuildInfo<CL_PROGRAM_BUILD_LOG>())
                            dump_file << p.second << "\n";

                        dump_file << "*/\n";
                    }

                    program.createKernels(&kernels);
                    if (is_cache_enabled()) {
                        // If kernels caching is enabled, then we save compiled bucket to binary file with name ${code_hash_value}.cl_cache
                        // Note: Bin file contains full bucket, not separate kernels, so kernels reuse across different models is quite limited
                        // Bucket size can be changed in get_max_kernels_per_batch() method, but forcing it to 1 will lead to much longer
                        // compile time.
                        saveBinaryToFile(cached_bin_name, getProgramBinaries(program));
                    }
                } else {
                    cl::Program program(_context.context(), {_context.device()}, precompiled_kernels);
                    program.build(_context.device(), program_source.options.c_str());
                    program.createKernels(&kernels);
                }

                for (auto& k : kernels) {
                    auto kernel_name = k.getInfo<CL_KERNEL_FUNCTION_NAME>();
                    kmap.emplace(kernel_name, kernels_cache::kernel_type(k, _context.get_device_info().supports_usm));
                }
            } catch (const cl::BuildError& err) {
                if (dump_sources && dump_file.good())
                    dump_file << "\n/* Build Log:\n";

                for (auto& p : err.getBuildLog()) {
                    if (dump_sources && dump_file.good())
                        dump_file << p.second << "\n";

                    err_log += p.second + '\n';
                }

                if (dump_sources && dump_file.good())
                    dump_file << "*/\n";
            }
        }

        if (!err_log.empty()) {
            static const size_t max_msg_length = 128;
            std::string short_err_log(err_log, 0, std::min(err_log.length(), max_msg_length));
            throw std::runtime_error("Program build failed:\n" + std::move(short_err_log));
        }

        return kmap;
    } catch (const cl::Error& err) {
        throw ocl_error(err);
    }
}

kernels_cache::kernel_type kernels_cache::get_kernel(kernel_id id, bool one_time_kernel) {
    build_all();
    if (one_time_kernel) {
        return _one_time_kernels.at(id);
    } else {
        return _kernels.at(id);
    }
}

void kernels_cache::build_all() {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "KernelsCache::BuildAll");
    if (!_pending_compilation)
        return;

    std::lock_guard<std::mutex> lock(_context.get_cache_mutex());

    auto sorted_program_code = get_program_source(_kernels_code);

    _one_time_kernels.clear();
    for (auto& program : sorted_program_code) {
        auto kernels = build_program(program.second);

        for (auto& k : kernels) {
            const auto& entry_point = k.first;
            const auto& k_id = program.second.entry_point_to_id[entry_point];
            if (program.second.one_time) {
                _one_time_kernels[k_id] = k.second;
            } else {
                _kernels[k_id] = k.second;
            }
        }
    }

    _kernels_code.clear();
    _pending_compilation = false;
}

void kernels_cache::reset() {
    _kernels.clear();
    _one_time_kernels.clear();
    _kernels_code.clear();
    _pending_compilation = false;
}

}  // namespace gpu
}  // namespace cldnn
