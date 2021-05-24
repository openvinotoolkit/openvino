// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
#if (CLDNN_THREADING == CLDNN_THREADING_TBB)
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#elif(CLDNN_THREADING == CLDNN_THREADING_THREADPOOL)
#include <thread>
#include <future>
#include <queue>
#include <condition_variable>
#endif
#if defined(__unix__) && !defined(__ANDROID__)
#include <malloc.h>
#endif

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

#if (CLDNN_THREADING != CLDNN_THREADING_SEQ)
#define DEFAULT_NUM_THREADS 2
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
        out_file.write(reinterpret_cast<const char*>(&buffer[0]), buffer.size());
    }
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


void kernels_cache::get_program_source(const kernels_code& kernels_source_code, std::vector<kernels_cache::batch_program>* all_batches) const {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "KernelsCache::BuildAll::GetProgramSource");
    std::map<std::string, std::vector<batch_program>> program_buckets;

    for (const auto& code : kernels_source_code) {
        std::string full_code = code.kernel_strings->jit + code.kernel_strings->str + code.kernel_strings->undefs;
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
            key += " __PROGRAM__" + std::to_string(program_buckets.size());
        }

        if (dump_custom_program) {
            key += " __DUMP_CUSTOM_PROGRAM__";  // Adding label to key so it would be separated from other programs
        }

        if (one_time_kernel) {
            key += " __ONE_TIME__";
        }
        auto& current_bucket = program_buckets[key];
        if (current_bucket.empty()) { // new bucket
            const auto& bucket_id = program_buckets.size() - 1;
            current_bucket.push_back(batch_program());
            current_bucket.back().bucket_id = static_cast<int32_t>(bucket_id);
            current_bucket.back().batch_id = 0;
            current_bucket.back().options = options;
        }

        // Create new kernels batch when the limit is reached
        if (current_bucket.back().kernels_counter >= get_max_kernels_per_batch()) {
            const auto& batch_id = current_bucket.size();
            current_bucket.push_back(batch_program());
            current_bucket.back().bucket_id = static_cast<int32_t>(program_buckets.size());
            current_bucket.back().batch_id = static_cast<int32_t>(batch_id);
            current_bucket.back().options = options;
        }

        auto& current_batch = current_bucket.back();
        current_batch.dump_custom_program = dump_custom_program;
        current_batch.one_time = one_time_kernel;
        current_batch.entry_point_to_id[entry_point] = code.id;

        assert(org_source_code.size() == 1);

        current_batch.source.push_back(std::move(org_source_code.front()));
        current_batch.kernels_counter++;
    }

    // Compute hash value for each batch
    // Hash calculation might require additional optimizations, but currently execution time of this part is much smaller than loading
    // of the precompiled binaries or get_undef_jit calls
    // Hash is computed for string that contains compilation options + driver version +
    // full source code (jit + template + undef sections) of all kernels in the batches
    for (auto& c : program_buckets) {
        auto options = c.first;
        auto& batches = c.second;
        for (auto& b : batches) {
            std::string full_code = options + " " + _context.get_device_info().driver_version;
            for (auto& ss : b.source)
                full_code += ss;
            b.hash_value = std::hash<std::string>()(full_code);
            all_batches->push_back(b);
        }
    }
}

kernels_cache::kernels_cache(gpu_toolkit& context, uint32_t prog_id) : _context(context), _prog_id(prog_id) {
}

kernels_cache::kernel_id kernels_cache::set_kernel_source(
    const std::shared_ptr<kernel_selector::kernel_string>& kernel_string,
    bool dump_custom_program,
    bool one_time_kernel) {
    std::lock_guard<std::mutex> lock(_context.get_cache_mutex());

    // we need unique id in order to avoid conflict across topologies.
    const auto kernel_num = _kernels.size() + _kernels_code.size();
    kernels_cache::kernel_id id = kernel_string->entry_point + "_" + std::to_string(kernel_num);

    auto res = _kernels_code.emplace(kernel_string, id, dump_custom_program, one_time_kernel);

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

void kernels_cache::build_batch(const batch_program& batch) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "KernelsCache::BuildProgram");

    bool dump_sources = !_context.get_configuration().ocl_sources_dumps_dir.empty() || batch.dump_custom_program;

    std::string err_log;  // accumulated build log from all program's parts (only contains messages from parts which

    std::string current_dump_file_name = "";
    if (dump_sources) {
        current_dump_file_name = _context.get_configuration().ocl_sources_dumps_dir;
        if (!current_dump_file_name.empty() && current_dump_file_name.back() != '/')
            current_dump_file_name += '/';

        current_dump_file_name += "clDNN_program_" + std::to_string(batch.bucket_id) + "_part_" + std::to_string(batch.batch_id) + ".cl";
    }

    std::ofstream dump_file;
    if (dump_sources) {
        dump_file.open(current_dump_file_name);
        if (dump_file.good()) {
            for (auto& s : batch.source)
                dump_file << s;
        }
    }

    std::string cached_bin_name = get_cache_path() + std::to_string(batch.hash_value) + ".cl_cache";
    cl::Program::Binaries precompiled_kernels = {};

    if (is_cache_enabled()) {
        // Try to load file with name ${hash_value}.cl_cache which contains precompiled kernels for current bucket
        // If read is successful, then remove kernels from compilation bucket
        auto bin = loadBinaryFromFile(cached_bin_name);
        if (!bin.empty()) {
            precompiled_kernels.push_back(bin);
        }
    }
    try {
        cl::vector<cl::Kernel> kernels;

        // Run compilation
        if (precompiled_kernels.empty()) {
            cl::Program program(_context.context(), batch.source);
            {
                OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "KernelsCache::BuildProgram::RunCompilation");
                program.build(_context.device(), batch.options.c_str());
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
            program.build(_context.device(), batch.options.c_str());
            program.createKernels(&kernels);
        }
        {
            std::lock_guard<std::mutex> lock(_context.get_cache_mutex());
            for (auto& k : kernels) {
                const auto& entry_point = k.getInfo<CL_KERNEL_FUNCTION_NAME>();
                const auto& k_id = batch.entry_point_to_id.find(entry_point);
                const auto& k_type = kernel_type(k, _context.get_device_info().supports_usm);
                if (k_id != batch.entry_point_to_id.end()) {
                    const auto& kmap = std::make_pair(k_id->second, k_type);
                    if (batch.one_time) {
                        _one_time_kernels.insert(kmap);
                    } else {
                        _kernels.insert(kmap);
                    }
                } else {
                    throw std::runtime_error("Could not find entry point");
                }
            }
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
    if (!err_log.empty()) {
        throw std::runtime_error("Program build failed. You may enable OCL source dump to see the error log.\n");
    }
}

kernels_cache::kernel_type kernels_cache::get_kernel(kernel_id id, bool one_time_kernel) const {
    if (_pending_compilation)
        throw std::runtime_error("Kernel cache is not compiled, call build_all() first!");

    const auto& kernels = one_time_kernel ?  _one_time_kernels : _kernels;
    auto res = kernels.find(id);
    if (kernels.end() == res)
        throw std::runtime_error("Kernel " + id + " not found in the kernel cache!");
    return res->second;
}

void kernels_cache::build_all() {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "KernelsCache::BuildAll");
    if (!_pending_compilation)
        return;
    std::vector<batch_program> batches;
    {
        std::lock_guard<std::mutex> lock(_context.get_cache_mutex());
        get_program_source(_kernels_code, &batches);
        _one_time_kernels.clear();
#if (CLDNN_THREADING == CLDNN_THREADING_TBB)
        int n_threads = _context.get_configuration().n_threads;
        arena = std::unique_ptr<tbb::task_arena>(new tbb::task_arena());
        arena->initialize(n_threads);
#elif(CLDNN_THREADING == CLDNN_THREADING_THREADPOOL)
        int n_threads = _context.get_configuration().n_threads;
        pool = std::unique_ptr<thread_pool>(new thread_pool(n_threads));
#endif
    }

#if (CLDNN_THREADING == CLDNN_THREADING_TBB)
    arena->execute([this, &batches] {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, batches.size()), [this, &batches](const tbb::blocked_range<size_t>& r) {
            for (auto i = r.begin(); i != r.end(); ++i) {
                build_batch(batches[i]);
            }
        });
    });
#elif(CLDNN_THREADING == CLDNN_THREADING_THREADPOOL)
    std::vector<std::future<void>> builds;
    for (size_t i = 0; i < batches.size(); ++i) {
        builds.push_back(pool->enqueue([this, &batches, i] () {
            build_batch(batches[i]);
        }));
    }
    std::for_each(builds.begin(), builds.end(), [] (std::future<void>& f) { f.wait(); });
#else
    // no parallel build
    for (const auto& batch : batches) {
        build_batch(batch);
    }
#endif

    {
        std::lock_guard<std::mutex> lock(_context.get_cache_mutex());
        _kernels_code.clear();
        _pending_compilation = false;
#if (CLDNN_THREADING == CLDNN_THREADING_TBB)
        arena.reset();
#if defined(__unix__) && !defined(__ANDROID__)
    //  NOTE: In linux, without malloc_trim, an amount of the memory used by compilation is not being returned to system thought they are freed.
    //  (It is at least 500 MB when we perform parallel compilation)
    //  It is observed that freeing the memory manually with malloc_trim saves significant amount of the memory.
    //  Also, this is not happening in Windows.
    //  So, added malloc_trim for linux build until we figure out a better solution.
        malloc_trim(0);
#endif
#elif(CLDNN_THREADING == CLDNN_THREADING_THREADPOOL)
        pool.reset();
#if defined(__unix__) && !defined(__ANDROID__)
        malloc_trim(0);
#endif
#endif
    }
}

void kernels_cache::reset() {
    _kernels.clear();
    _one_time_kernels.clear();
    _kernels_code.clear();
    _pending_compilation = false;
}
}  // namespace gpu
}  // namespace cldnn
