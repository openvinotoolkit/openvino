// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels_factory.hpp"
#include "kernels_cache.hpp"
#include "ocl/ocl_kernel.hpp"
#include "ocl/ocl_engine.hpp"
#include "ocl/ocl_common.hpp"
#include "intel_gpu/graph/serialization/set_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "intel_gpu/graph/serialization/map_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/file_util.hpp"
#include "openvino/util/pp.hpp"

#ifdef WIN32
#include <sdkddkver.h>
#ifdef NTDDI_WIN10_RS5
#include <appmodel.h>
#endif
#endif

#include <algorithm>
#include <cassert>
#include <sstream>
#include <fstream>
#include <set>
#include <string>
#include <tuple>
#include <memory>
#include <utility>

#if defined(__unix__) && !defined(__ANDROID__)
#include <malloc.h>
#endif

#ifdef ENABLE_ONEDNN_FOR_GPU
#ifndef NOMINMAX
# define NOMINMAX
#endif
#include "gpu/intel/microkernels/fuser.hpp"
#endif

namespace {
std::mutex cacheAccessMutex;

#ifdef ENABLE_ONEDNN_FOR_GPU
cl::Program fuse_microkernels(const cl::Context& context, const cl::Device& device, cl::Program& program, const std::string& code) {
    using namespace dnnl::impl::gpu::intel;
    std::vector<std::vector<uint8_t>> binaries = program.getInfo<CL_PROGRAM_BINARIES>();
    OPENVINO_ASSERT(binaries.size() == 1);
    std::vector<uint8_t> binary = binaries[0];
    micro::fuseMicrokernels(binary, code.c_str());

    cl::Program::Binaries fused_binary = { binary };
    cl::Program fused_program(context, {device}, fused_binary);
    fused_program.build({device});

    return fused_program;
}
#endif  // ENABLE_ONEDNN_FOR_GPU

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
    auto path = _config.get_property(ov::cache_dir);
    if (path.empty()) {
        return {};
    }

    if (path.back() != '/' && path.back() != '\\') {
        path += "/";
    }
    return path;
}

bool kernels_cache::is_cache_enabled() const {
    if (!_config.get_property(ov::intel_gpu::allow_new_shape_infer) &&
        (_config.get_property(ov::cache_mode) == ov::CacheMode::OPTIMIZE_SPEED)) {
        return false;
    }

    return !_config.get_property(ov::cache_dir).empty();
}

size_t kernels_cache::get_max_kernels_per_batch() const {
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->max_kernels_per_batch >= 1) {
        return static_cast<size_t>(debug_config->max_kernels_per_batch);
    }
    return _config.get_property(ov::intel_gpu::max_kernels_per_batch);
}

void kernels_cache::get_program_source(const kernels_code& kernels_source_code, std::vector<kernels_cache::batch_program>* all_batches) const {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "KernelsCache::BuildAll::GetProgramSource");
    std::map<std::string, std::tuple<int32_t, std::vector<batch_program>>> program_buckets;

    for (const auto& k : kernels_source_code) {
        auto& code = k.second;
        bool dump_custom_program = code.dump_custom_program;

        for (size_t kernel_part_idx = 0; kernel_part_idx < code.kernel_strings.size(); kernel_part_idx++) {
            auto& kernel_string = code.kernel_strings[kernel_part_idx];
            std::string full_code = kernel_string->jit + kernel_string->str + kernel_string->undefs;
            std::string entry_point = kernel_string->entry_point;
            std::string options = kernel_string->options;
            bool batch_compilation = kernel_string->batch_compilation;

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
                current_bucket.push_back(batch_program(bucket_id, batch_id, options, batch_headers));
            }

            // This is a temporary walk-around to avoid severe performance drop.
            // It will be removed after OpenCL compiler is updated.
            auto need_separate_batch = [&](std::string& unique_kernel_name) -> bool {
                const std::vector<std::string> special_kernels = {"gemm_tiled_opt"};

                // check if the current kernel name is in special_kernels
                for (auto& special_kernel : special_kernels) {
                    if (entry_point.find(special_kernel) != std::string::npos)
                        return true;
                }

                // check if the current_batch has one of special_kernels
                if (current_bucket.back().kernels_counter == 1) {
                    auto& kernel_in_current_batch = current_bucket.back().entry_point_to_id.begin()->first;
                    for (auto& special_kernel : special_kernels) {
                        if (kernel_in_current_batch.find(special_kernel) != std::string::npos)
                            return true;
                    }
                }
                return false;
            };

            // Create new kernels batch when the limit is reached
            // and current kernel's entry_point is duplicated in this kernels batch
            if (current_bucket.back().kernels_counter >= get_max_kernels_per_batch()
                || current_bucket.back().entry_point_to_id.find(entry_point) != current_bucket.back().entry_point_to_id.end()
                || need_separate_batch(entry_point)) {
                const auto& batch_id = static_cast<int32_t>(current_bucket.size());
                current_bucket.push_back(batch_program(bucket_id, batch_id, options, batch_headers));
            }

            auto& current_batch = current_bucket.back();
            current_batch.dump_custom_program = dump_custom_program;
            current_batch.entry_point_to_id.emplace(entry_point, std::make_pair(code.params, kernel_part_idx));

            current_batch.has_microkernels |= kernel_string->has_microkernels;

            // TODO: Technically, microkernels doesn't require specific headers, but we don't want to include
            // some headers to all batches as it may lead to compilation error on some driver versions.
            // Need to generalize work with headers to include only necessary parts
            if (current_batch.has_microkernels) {
                current_batch.source.insert(current_batch.source.begin(), current_batch.micro_headers.begin(), current_batch.micro_headers.end());
            }

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
            std::string full_code = options + " " + _engine.get_device_info().driver_version;
            full_code += _engine.get_device_info().dev_name;
            for (auto& ss : b.source)
                full_code += ss;

            b.hash_value = std::hash<std::string>()(full_code);

            std::string dump_sources_dir = "";
            GPU_DEBUG_GET_INSTANCE(debug_config);
            GPU_DEBUG_IF(!debug_config->dump_sources.empty()) {
                dump_sources_dir = debug_config->dump_sources;
            }

            // Add -g -s to build options to allow IGC assembly dumper to associate assembler sources with corresponding OpenCL kernel code lines
            // Should be used with the IGC_ShaderDump option
            if (!dump_sources_dir.empty()) {
                std::string current_dump_file_name = dump_sources_dir;
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
    : _engine(engine)
    , _task_executor(task_executor)
    , _config(config)
    , _prog_id(prog_id)
    , batch_headers(std::move(batch_headers)) { }

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
void kernels_cache::build_batch(const engine& build_engine, const batch_program& batch, compiled_kernels& compiled_kernels) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "KernelsCache::build_batch");

    auto& cl_build_engine = dynamic_cast<const ocl::ocl_engine&>(build_engine);

    bool dump_sources = batch.dump_custom_program;
    std::string dump_sources_dir = "";
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
                               + "_part_" + std::to_string(batch.batch_id) + "_" + std::to_string(batch.hash_value) + ".cl";
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
                OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "KernelsCache::BuildProgram::RunCompilation");
                if (program.build({cl_build_engine.get_cl_device()}, batch.options.c_str()) != CL_SUCCESS)
                    throw std::runtime_error("Failed in building program.");
            }

            if (dump_sources && dump_file.good()) {
                dump_file << "\n/* Build Log:\n";
                for (auto& p : program.getBuildInfo<CL_PROGRAM_BUILD_LOG>())
                    dump_file << p.second << "\n";

                dump_file << "*/\n";
            }

            if (batch.has_microkernels) {
#ifdef ENABLE_ONEDNN_FOR_GPU
                OPENVINO_ASSERT(batch.kernels_counter == 1);
                // Do we need full source code here (with batch headers)?
                program = fuse_microkernels(cl_build_engine.get_cl_context(), cl_build_engine.get_cl_device(), program, batch.source.back());
#else  // ENABLE_ONEDNN_FOR_GPU
                OPENVINO_THROW("[GPU] Can't compile kernel w/ microkernels as onednn is not available");
#endif  // ENABLE_ONEDNN_FOR_GPU
            }


            program.createKernels(&kernels);

            if (is_cache_enabled()) {
                // If kernels caching is enabled, then we save compiled bucket to binary file with name ${code_hash_value}.cl_cache
                // Note: Bin file contains full bucket, not separate kernels, so kernels reuse across different models is quite limited
                // Bucket size can be changed in get_max_kernels_per_batch() method, but forcing it to 1 will lead to much longer
                // compile time.
                std::lock_guard<std::mutex> lock(cacheAccessMutex);
                ov::intel_gpu::save_binary(cached_bin_name, getProgramBinaries(program));
            }
        } else {
            cl::Program program(cl_build_engine.get_cl_context(), {cl_build_engine.get_cl_device()}, precompiled_kernels);
            if (program.build({cl_build_engine.get_cl_device()}, batch.options.c_str()) != CL_SUCCESS)
                throw std::runtime_error("Failed in building program with a precompiled kernel.");

            program.createKernels(&kernels);
        }

        {
            std::lock_guard<std::mutex> lock(_mutex);
            for (auto& k : kernels) {
                const auto& entry_point = k.getInfo<CL_KERNEL_FUNCTION_NAME>();
                const auto& iter = batch.entry_point_to_id.find(entry_point);
                if (iter != batch.entry_point_to_id.end()) {
                    cl_kernel kern = k.get();
                    cl_context context = cl_build_engine.get_cl_context().get();
                    kernel::ptr kernel = kernels_factory::create(_engine, context, kern, entry_point);

#ifdef GPU_DEBUG_CONFIG
                    GPU_DEBUG_IF(cldnn::debug_configuration::get_instance()->check_kernels_properties >= 1) {
                        auto kernel_properties = kernel->get_properties();

                        if (kernel_properties.spill_mem_size > 0 || kernel_properties.private_mem_size > 0) {
                            GPU_DEBUG_COUT << "WARNING: Detected extra private memory usage or spill for " << entry_point << " "
                                           << "kernel with properties: " << kernel_properties.to_string() << "\n";
                        } else {
                            GPU_DEBUG_TRACE_DETAIL << "Create kernel " << entry_point << " with properties: " << kernel_properties.to_string() << "\n";
                        }
                    }
#endif

                    auto& params = iter->second.first;
                    auto kernel_part_idx = iter->second.second;
                    if (compiled_kernels.find(params) != compiled_kernels.end()) {
                        compiled_kernels[params].push_back(std::make_pair(kernel, kernel_part_idx));
                    } else {
                        compiled_kernels[params] = { std::make_pair(kernel, kernel_part_idx) };
                    }
                    if (_kernel_batch_hash.find(params) == _kernel_batch_hash.end()) {
                       _kernel_batch_hash[params] = batch.hash_value;
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
        GPU_DEBUG_INFO << "-------- OpenCL build error" << std::endl;
        GPU_DEBUG_INFO << err_log << std::endl;
        GPU_DEBUG_INFO << "-------- End of OpenCL build error" << std::endl;
        std::stringstream err_ss(err_log);
        std::string line;
        std::stringstream err;
        int cnt = 0;

        while (std::getline(err_ss, line, '\n')) {
            if (line.find("error") != std::string::npos)
                cnt = 5;
            cnt--;
            if (cnt > 0)
                err << line << std::endl;
            else if (cnt == 0)
                err << "...." << std::endl;
        }

        throw std::runtime_error("Program build failed(" + std::to_string(batch.bucket_id) + + "_part_"
                                 + std::to_string(batch.batch_id)
                                 + "):\n" + err.str());
    }
}

kernel::ptr kernels_cache::get_kernel_from_cached_kernels(std::string id) const {
    auto res = _cached_kernels.find(id);
    OPENVINO_ASSERT(_cached_kernels.end() != res, "[GPU] Kernel " + id + " not found in the cached kernel cache!");
    return res->second->clone();
}

std::vector<kernel::ptr> kernels_cache::get_kernels(kernel_impl_params params) const {
    OPENVINO_ASSERT((_pending_compilation == false), "Kernel cache is not compiled, call build_all() first!");

    std::string current_node_id;
    if (params.desc) {
        current_node_id = params.desc->id;
    }
    auto res = _kernels.find(params);
    OPENVINO_ASSERT(_kernels.end() != res, "Kernel for {" + current_node_id + "} is not found in the kernel cache!");
    OPENVINO_ASSERT(res->second.size() != 0, "Number of kernels should not be zero for " + current_node_id);

    std::vector<kernel::ptr> kernels(res->second.size());
    for (auto& k : res->second) {
        auto& kernel_ptr = k.first;
        auto kernel_part_idx = k.second;
        kernels[kernel_part_idx] = kernel_ptr->clone();
    }
    return kernels;
}

bool kernels_cache::validate_simple_kernel_execution(kernel::ptr krl) {
    auto casted = downcast<ocl::ocl_kernel>(krl.get());
    auto kernel = casted->get_handle();
    try {
        auto casted_dev = dynamic_cast<ocl::ocl_device*>(_engine.get_device().get());
        auto device = casted_dev->get_device();
        cl::Context ctx(device);

        cl::Buffer buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * 8);
        if (kernel.setArg(0, buffer) != CL_SUCCESS)
            return false;

        cl::Event ev;
        cl::CommandQueue queue(ctx, device);
        if (queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(8), cl::NDRange(8), nullptr, &ev) != CL_SUCCESS)
            return false;

        uint8_t result[8];
        uint8_t expected[8] = { 1, 3, 5, 7, 9, 11, 13, 15 };
        if (queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(uint8_t) * 8, &result) != CL_SUCCESS)
            return false;

        for (int i = 0; i < 8; ++i) {
            if (result[i] != expected[i])
                return false;
        }

        ev.wait();
        return true;
    } catch (...) {
        return false;
    }
}

void kernels_cache::build_all() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "KernelsCache::BuildAll");
    if (!_pending_compilation)
        return;

    ocl::ocl_engine& _build_engine = downcast<ocl::ocl_engine>(_engine);
    std::vector<batch_program> batches;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        get_program_source(_kernels_code, &batches);
    }

    // build_batch crashes randomly when threaded while running from a Microsoft Store app
    // it seems to be a bug in Intel's graphics driver, disabling threading is a work around
    auto use_threads{true};
#if defined(WIN32) && defined(NTDDI_WIN10_RS5)
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
            tasks.push_back([this, &_build_engine, &batch, &exception] {
                try {
                    build_batch(_build_engine, batch, _kernels);
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
            build_batch(_build_engine, batches[idx], _kernels);
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
    auto ocl_kernel = std::static_pointer_cast<cldnn::ocl::ocl_kernel>(kernel);
    const auto& entry_point = ocl_kernel->get_handle().getInfo<CL_KERNEL_FUNCTION_NAME>();
    auto program = ocl_kernel->get_handle().getInfo<CL_KERNEL_PROGRAM>();
    cl::vector<unsigned char> program_binaries = getProgramBinaries(program);

    auto iter = _cached_binaries.find(program_binaries);
    OPENVINO_ASSERT(iter != _cached_binaries.end(), "[GPU] Not found cached kernel binaries");

    return entry_point + "@" + std::to_string(iter->second);
}

std::vector<std::string> kernels_cache::get_cached_kernel_ids(const std::vector<kernel::ptr>& kernels) const {
    std::vector<std::string> kernel_ids;

    for (auto& kernel : kernels) {
        auto key = get_cached_kernel_id(kernel);
        kernel_ids.emplace_back(key);
    }

    return kernel_ids;
}

void kernels_cache::add_to_cached_kernels(const std::vector<kernel::ptr>& kernels) {
    static std::atomic<uint32_t> id_gen{0};

    for (auto& kernel : kernels) {
        auto ocl_kernel = std::static_pointer_cast<cldnn::ocl::ocl_kernel>(kernel);
        auto program = ocl_kernel->get_handle().getInfo<CL_KERNEL_PROGRAM>();
        cl::vector<unsigned char> program_binaries = getProgramBinaries(program);

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
    OPENVINO_ASSERT(_engine.type() == engine_types::ocl, "[GPU] Not supported engine type");

    ob << _cached_binaries.size();
    for (auto& cached_binary : _cached_binaries) {
        ob << cached_binary.second;
        ob << cached_binary.first;
    }
}

void kernels_cache::load(BinaryInputBuffer& ib) {
    OPENVINO_ASSERT(_engine.type() == engine_types::ocl, "[GPU] Not supported engine type");

    std::unordered_map<uint32_t, std::vector<unsigned char>> precompiled_kernels;

    size_t num_cached_binaries;
    ib >> num_cached_binaries;
    for (size_t i = 0; i < num_cached_binaries; ++i) {
        uint32_t id;
        ib >> id;
        ib >> precompiled_kernels[id];
    }

    std::unique_ptr<ocl::ocl_engine> build_engine =
        cldnn::make_unique<ocl::ocl_engine>(_engine.get_device(), runtime_types::ocl);

    try {
        std::lock_guard<std::mutex> lock(_mutex);
        _cached_kernels.clear();

        for (auto& precompiled_kernel : precompiled_kernels) {
            cl::vector<cl::Kernel> kernels;
            cl::Program program(build_engine->get_cl_context(), {build_engine->get_cl_device()}, {precompiled_kernel.second});
            program.build({build_engine->get_cl_device()});
            program.createKernels(&kernels);

            for (auto& k : kernels) {
                const auto& entry_point = k.getInfo<CL_KERNEL_FUNCTION_NAME>();
                std::string cached_kernel_id = entry_point + "@" + std::to_string(precompiled_kernel.first);
                const auto& iter = _cached_kernels.find(cached_kernel_id);
                if (iter == _cached_kernels.end()) {
                    cl_kernel cl_kernel = k.get();
                    cl_context cl_context = build_engine->get_cl_context().get();
                    kernel::ptr kernel = kernels_factory::create(_engine, cl_context, cl_kernel, entry_point);
                    _cached_kernels[cached_kernel_id] = kernel;
                }
            }
        }
    } catch (const cl::BuildError& err) {
        std::string err_log = "";
        for (auto& p : err.getBuildLog()) {
            err_log += p.second + '\n';
        }
        OPENVINO_THROW(err_log);
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

    ocl::ocl_engine& _build_engine = downcast<ocl::ocl_engine>(_engine);

    // Create batches
    std::vector<batch_program> batches;
    get_program_source(t_kernels_code, &batches);

    compiled_kernels output_kernels;
    // Build batches
    for (size_t idx = 0; idx < batches.size(); ++idx) {
        build_batch(_build_engine, batches[idx], output_kernels);
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
