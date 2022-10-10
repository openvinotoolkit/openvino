// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels_factory.hpp"
#include "kernels_cache.hpp"
#include "ocl/ocl_engine.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/util/file_util.hpp"

#include <algorithm>
#include <cassert>
#include <sstream>
#include <fstream>
#include <set>
#include <string>
#include <tuple>
#include <memory>
#include <utility>

#include "cldnn_itt.hpp"
#if defined(__unix__) && !defined(__ANDROID__)
#include <malloc.h>
#endif

namespace {
std::mutex cacheAccessMutex;

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
    auto path = _engine.configuration().kernels_cache_path;
    if (path.empty()) {
        return {};
    }

    if (path.back() != '/' && path.back() != '\\') {
        path += "/";
    }
    return path;
}

bool kernels_cache::is_cache_enabled() const {
    return !_engine.configuration().kernels_cache_path.empty();
}

size_t kernels_cache::get_max_kernels_per_batch() const {
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->max_kernels_per_batch >= 1) {
        return static_cast<size_t>(debug_config->max_kernels_per_batch);
    }
    return 8;
}


void kernels_cache::get_program_source(const kernels_code& kernels_source_code, std::vector<kernels_cache::batch_program>* all_batches) const {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "KernelsCache::BuildAll::GetProgramSource");
    std::map<std::string, std::tuple<int32_t, std::vector<batch_program>>> program_buckets;

    for (const auto& code : kernels_source_code) {
        std::string full_code = code.kernel_strings->jit + code.kernel_strings->str + code.kernel_strings->undefs;
        std::string entry_point = code.kernel_strings->entry_point;
        std::string options = code.kernel_strings->options;
        bool batch_compilation = code.kernel_strings->batch_compilation;
        bool dump_custom_program = code.dump_custom_program;

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

        auto& bucket_id = std::get<0>(program_buckets[key]);
        auto& current_bucket = std::get<1>(program_buckets[key]);
        if (current_bucket.empty()) { // new bucket
            const auto& batch_id = 0;
            // increase bucket id if and only if new bucket comes
            bucket_id = static_cast<int32_t>(program_buckets.size() - 1);
            current_bucket.push_back(batch_program(bucket_id, batch_id, options, batch_header_str));
        }

        // Create new kernels batch when the limit is reached
        if (current_bucket.back().kernels_counter >= get_max_kernels_per_batch()) {
            const auto& batch_id = static_cast<int32_t>(current_bucket.size());
            current_bucket.push_back(batch_program(bucket_id, batch_id, options, batch_header_str));
        }

        auto& current_batch = current_bucket.back();
        current_batch.dump_custom_program = dump_custom_program;
        current_batch.entry_point_to_id[entry_point] = code.id;

        current_batch.source.push_back(std::move(full_code));
        current_batch.kernels_counter++;
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
            std::string full_code = options + " " + _engine.get_device_info().driver_version;
            for (auto& ss : b.source)
                full_code += ss;
            b.hash_value = std::hash<std::string>()(full_code);
            all_batches->push_back(b);
        }
    }
}

kernels_cache::kernels_cache(engine& engine, uint32_t prog_id, const std::vector<std::string>& batch_header_str)
                                : _engine(engine), _prog_id(prog_id), batch_header_str(std::move(batch_header_str)) { }

kernel_id kernels_cache::set_kernel_source(
    const std::shared_ptr<kernel_string>& kernel_string,
    bool dump_custom_program) {
    std::lock_guard<std::mutex> lock(_mutex);
    // we need unique id in order to avoid conflict across topologies.
    const auto kernel_num = _kernels.size() + (_kernel_idx++);
    kernel_id id = kernel_string->entry_point + "_" + std::to_string(kernel_num);

    auto res = _kernels_code.emplace(kernel_string, id, dump_custom_program);

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

// TODO: This build_batch method should be backend specific
void kernels_cache::build_batch(const engine& build_engine, const batch_program& batch) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "KernelsCache::build_batch");

    auto& cl_build_engine = dynamic_cast<const ocl::ocl_engine&>(build_engine);

    bool dump_sources = !_engine.configuration().sources_dumps_dir.empty() || batch.dump_custom_program;
    std::string dump_sources_dir = _engine.configuration().sources_dumps_dir;
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(!debug_config->dump_sources.empty()) {
        dump_sources = true;
        dump_sources_dir = debug_config->dump_sources;
    }

    std::string err_log;  // accumulated build log from all program's parts (only contains messages from parts which

    std::string current_dump_file_name = "";
    if (dump_sources) {
        current_dump_file_name = dump_sources_dir;
        if (!current_dump_file_name.empty() && current_dump_file_name.back() != '/')
            current_dump_file_name += '/';

        current_dump_file_name += "clDNN_program_" + std::to_string(_prog_id) + "_bucket_" + std::to_string(batch.bucket_id)
                               + "_part_" + std::to_string(batch.batch_id) + ".cl";
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
        std::vector<uint8_t> bin;
        {
            std::lock_guard<std::mutex> lock(cacheAccessMutex);
            bin = ov::util::load_binary(cached_bin_name);
        }
        if (!bin.empty()) {
            precompiled_kernels.push_back(bin);
        }
    }
    try {
        cl::vector<cl::Kernel> kernels;

        // Run compilation
        if (precompiled_kernels.empty()) {
            cl::Program program(cl_build_engine.get_cl_context(), batch.source);
            {
                OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "KernelsCache::BuildProgram::RunCompilation");
                program.build(cl_build_engine.get_cl_device(), batch.options.c_str());
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
                std::lock_guard<std::mutex> lock(cacheAccessMutex);
                ov::util::save_binary(cached_bin_name, getProgramBinaries(program));
            }
        } else {
            cl::Program program(cl_build_engine.get_cl_context(), {cl_build_engine.get_cl_device()}, precompiled_kernels);
            program.build(cl_build_engine.get_cl_device(), batch.options.c_str());
            program.createKernels(&kernels);
        }
        {
            std::lock_guard<std::mutex> lock(_mutex);
            for (auto& k : kernels) {
                const auto& entry_point = k.getInfo<CL_KERNEL_FUNCTION_NAME>();
                const auto& k_id = batch.entry_point_to_id.find(entry_point);
                if (k_id != batch.entry_point_to_id.end()) {
                    cl_kernel kern = k.get();
                    cl_context context = cl_build_engine.get_cl_context().get();
                    kernel::ptr kernel = kernels_factory::create(_engine, context, kern, entry_point);
                    const auto& kmap = std::make_pair(k_id->second, kernel);
                    _kernels.insert(kmap);
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
        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(debug_config->verbose) {
            std::cout << "-------- OpenCL build error" << std::endl;
            std::cout << err_log << std::endl;
            std::cout << "-------- End of OpenCL build error" << std::endl;
        }
        std::stringstream err_ss(err_log);
        std::string line;
        int cnt = 0;

        while (std::getline(err_ss, line, '\n')) {
            if (line.find("error") != std::string::npos)
                cnt = 5;
            cnt--;
            if (cnt > 0)
                std::cout << line << std::endl;
            else if (cnt == 0)
                std::cout << "...." << std::endl;
        }

        throw std::runtime_error("Program build failed(" + std::to_string(batch.bucket_id) + + "_part_"
                                 + std::to_string(batch.batch_id)
                                 + "): You may enable OCL source dump to see the error log.\n");
    }
}

kernel::ptr kernels_cache::get_kernel(kernel_id id) const {
    if (_pending_compilation)
        throw std::runtime_error("Kernel cache is not compiled, call build_all() first!");

    auto res = _kernels.find(id);
    if (_kernels.end() == res)
        throw std::runtime_error("Kernel " + id + " not found in the kernel cache!");
    return res->second;
}

void kernels_cache::build_all() {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "KernelsCache::BuildAll");
    if (!_pending_compilation)
        return;

    std::unique_ptr<ocl::ocl_engine> _build_engine = nullptr;
    if (_engine.type() == engine_types::ocl) {
        _build_engine = std::unique_ptr<ocl::ocl_engine>(new ocl::ocl_engine(_engine.get_device(), runtime_types::ocl,
                                                                    _engine.configuration(), _engine.get_task_executor()));
    }
    std::vector<batch_program> batches;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        get_program_source(_kernels_code, &batches);
    }

    auto _task_executor = _engine.get_task_executor();
    std::exception_ptr exception;
    std::vector<InferenceEngine::Task> tasks;
    for (size_t idx = 0; idx < batches.size(); idx++) {
        auto& batch = batches[idx];
        tasks.push_back([this, &_build_engine, &batch, &exception] {
            try {
                build_batch(*_build_engine, batch);
            } catch(...) {
                exception = std::current_exception();
            }
        });
    }
    _task_executor->runAndWait(tasks);
    tasks.clear();

    if (exception) {
        std::rethrow_exception(exception);
    }

    {
        std::lock_guard<std::mutex> lock(_mutex);
        _kernels_code.clear();
        _pending_compilation = false;
#if defined(__unix__) && !defined(__ANDROID__)
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
    _pending_compilation = false;
}

}  // namespace cldnn
