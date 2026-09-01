// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels_cache.hpp"

#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/serialization/map_serializer.hpp"
#include "intel_gpu/graph/serialization/set_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/file_util.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "intel_gpu/runtime/runtime_backend_registry.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "kernel_cache_frontend.hpp"
#include "openvino/util/pp.hpp"

#ifdef WIN32
#    include <sdkddkver.h>
#    ifdef NTDDI_WIN10_RS5
#        include <appmodel.h>
#    endif
#endif

#include <cassert>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#if defined(__unix__) && !defined(__ANDROID__)
#    include <malloc.h>
#endif

#ifdef ENABLE_ONEDNN_FOR_GPU
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include "gpu/intel/gemm/jit/include/gemmstone/microkernel/fuser.hpp"
#endif

namespace {
std::mutex cacheAccessMutex;

std::string join_strings(const std::vector<std::string>& strings) {
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

cldnn::KernelFormat to_kernel_format(cldnn::gpu_cached_kernel_artifact artifact) {
    switch (artifact) {
    case cldnn::gpu_cached_kernel_artifact::native_device_binary:
        return cldnn::KernelFormat::NATIVE_BIN;
    case cldnn::gpu_cached_kernel_artifact::spirv:
        return cldnn::KernelFormat::SPIRV;
    }
    OPENVINO_THROW("[GPU] Unsupported cached kernel artifact format");
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
    if (!_config.get_allow_new_shape_infer() && (_config.get_cache_mode() == ov::CacheMode::OPTIMIZE_SPEED)) {
        return false;
    }

    return !_config.get_cache_dir().empty();
}

void kernels_cache::get_program_source(const kernels_code& kernels_source_code, std::vector<kernels_cache::batch_program>* all_batches) const {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "KernelsCache::BuildAll::GetProgramSource");
    kernel_cache_frontend_context context;
    context.max_kernels_per_batch = _config.get_max_kernels_per_batch();
    context.program_id = _prog_id;
    context.device_name = _device->get_info().dev_name;
    context.driver_version = _device->get_info().driver_version;
    context.dump_sources_path = GPU_DEBUG_VALUE_OR(_config.get_dump_sources_path(), "");
    context.batch_headers = &batch_headers;
    kernel_cache_frontend::prepare(kernels_source_code, context, *all_batches);
}

kernels_cache::kernels_cache(engine& engine,
                             const ExecutionConfig& config,
                             uint32_t prog_id,
                             std::shared_ptr<ov::threading::ITaskExecutor> task_executor,
                             const std::map<std::string, std::string>& batch_headers)
    : _device(engine.get_device()),
      _builder(engine.create_kernel_builder()),
      _task_executor(task_executor),
      _config(config),
      _prog_id(prog_id),
      batch_headers(std::move(batch_headers)) {}

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
        std::string ext = batch.language == kernel_language::CM ? ".cm" : (batch.language == kernel_language::SPIRV ? ".spv" : ".cl");
        current_dump_file_name += "clDNN_program_" + std::to_string(_prog_id) + "_bucket_" + std::to_string(batch.bucket_id) + "_part_" +
                                  std::to_string(batch.batch_id) + "_" + std::to_string(batch.hash_value) + ext;
    }

    std::ofstream dump_file;
    if (dump_sources) {
        dump_file.open(current_dump_file_name);
        if (dump_file.good()) {
            for (const auto& s : batch.source)
                dump_file << s;
        }
    }
    const auto* const cache_extension = batch.language == kernel_language::SPIRV ? ".spv_cache" : ".cl_cache";
    std::string cached_bin_name = get_cache_path() + std::to_string(batch.hash_value) + cache_extension;
    ///////////////////////////////////////////////////////////////////////////////////
    std::vector<uint8_t> precompiled;
    if (is_cache_enabled()) {
        std::lock_guard<std::mutex> lock(cacheAccessMutex);
        precompiled = ov::util::load_binary(ov::util::make_path(cached_bin_name));
    }
    std::vector<kernel::ptr> kernels;
    const auto source_format = batch.language == kernel_language::SPIRV ? KernelFormat::SPIRV : KernelFormat::SOURCE;
    const auto cached_format = batch.language == kernel_language::SPIRV ? KernelFormat::SPIRV : KernelFormat::NATIVE_BIN;
    const auto entry_point = batch.entry_point_to_id.size() == 1 ? batch.entry_point_to_id.begin()->first : std::string{};
    const auto make_artifact = [&](const void* payload, size_t payload_size, KernelFormat format, const std::string& build_options) {
        kernel_artifact artifact;
        artifact.payload = payload;
        artifact.payload_size = payload_size;
        artifact.format = format;
        artifact.entry_point = entry_point;
        artifact.build_options = build_options;
        return artifact;
    };
    if (!precompiled.empty()) {
        const auto artifact = make_artifact(precompiled.data(), precompiled.size(), cached_format, "");
        _builder->build_kernels(artifact, kernels);
    } else {
        auto combined_source = join_strings(batch.source);
        const auto artifact = make_artifact(combined_source.data(), combined_source.size(), source_format, batch.options);
        _builder->build_kernels(artifact, kernels);
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
#ifdef ENABLE_ONEDNN_FOR_GPU
            OPENVINO_ASSERT(batch.kernels_counter == 1 && kernels.size() == 1);
            std::vector<uint8_t> binary = kernels[0]->get_binary();
            kernels.clear();
            // Update binary and rebuild kernel
            gemmstone::microkernel::fuse(binary, combined_source.c_str());
            const auto native_artifact = make_artifact(binary.data(), binary.size(), KernelFormat::NATIVE_BIN, "");
            _builder->build_kernels(native_artifact, kernels);
#else   // ENABLE_ONEDNN_FOR_GPU
            OPENVINO_THROW("[GPU] Can't compile kernel w/ microkernels as onednn is not available");
#endif  // ENABLE_ONEDNN_FOR_GPU
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
                    compiled_kernels[params] = {std::make_pair(k, kernel_part_idx)};
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
    const auto cached_kernel_format = to_kernel_format(runtime_backend_registry::get(_device->get_runtime_type()).kernel_cache.artifact);

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
            _builder->build_kernels(precompiled_kernel.second.data(), precompiled_kernel.second.size(), cached_kernel_format, "", kernels);
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
