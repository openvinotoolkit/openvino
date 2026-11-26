// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_otd_runtime.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "runtime/ocl/ocl_stream.hpp"

namespace ov::intel_gpu::ocl::moe_otd {

cldnn::stream::ptr create_copy_stream(cldnn::engine& engine, const cldnn::ExecutionConfig& config) {
    auto* native = dynamic_cast<cldnn::ocl::ocl_engine*>(&engine);
    OPENVINO_ASSERT(native && native->extension_supported("cl_intel_command_queue_families"),
                    "LES copy engine requires OpenCL command queue families");
    const auto families = native->get_cl_device().getInfo<CL_DEVICE_QUEUE_FAMILY_PROPERTIES_INTEL>();
    for (size_t family = 0; family < families.size(); ++family) {
        if (families[family].count == 0 || std::strncmp(families[family].name, "bcs", 3) != 0) {
            continue;
        }
        const cl_queue_properties properties[] = {
            CL_QUEUE_FAMILY_INTEL, static_cast<cl_queue_properties>(family),
            CL_QUEUE_INDEX_INTEL, 0,
            CL_QUEUE_PROPERTIES, static_cast<cl_queue_properties>(config.get_enable_profiling() ? CL_QUEUE_PROFILING_ENABLE : 0),
            0};
        cl_int status = CL_SUCCESS;
        cl::CommandQueue queue(clCreateCommandQueueWithProperties(native->get_cl_context().get(),
                                                                 native->get_cl_device().get(), properties, &status));
        OPENVINO_ASSERT(status == CL_SUCCESS, "LES BCS queue creation failed: ", status);
        return engine.create_stream(config, queue.get());
    }
    OPENVINO_THROW("LES copy engine requires an exposed BCS queue family");
}

bool les_trace_enabled() {
    static const bool enabled = [] {
        const auto* value = std::getenv("MOE_LES_TRACE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

bool les_prediction_compare_enabled() {
    static const bool enabled = [] {
        const auto* value = std::getenv("OV_GPU_LES_PREDICT_COMPARE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

uint64_t les_device_time(cldnn::stream& stream) {
    auto queue = dynamic_cast<cldnn::ocl::ocl_stream&>(stream).get_cl_queue().get();
    cl_device_id device = nullptr;
    OPENVINO_ASSERT(clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device), &device, nullptr) == CL_SUCCESS,
                    "LES trace cannot query queue device");
    cl_ulong device_time = 0;
    cl_ulong host_time = 0;
    OPENVINO_ASSERT(clGetDeviceAndHostTimer(device, &device_time, &host_time) == CL_SUCCESS,
                    "LES trace requires device/host timer support");
    return device_time;
}

void les_trace_interval(const char* kind, uint64_t start, uint64_t end) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    GPU_DEBUG_INFO << "[LES_TRACE] {\"kind\":\"" << kind << "\",\"start_ns\":" << start
                   << ",\"end_ns\":" << end << "}" << std::endl;
}

void OtdPerfCounters::dump() const {
    const auto* enabled = std::getenv("MOE_OTD_PERF_LOG");
    if (!enabled || std::strcmp(enabled, "1") != 0) {
        return;
    }
    const auto hits = gpu_hits.load(std::memory_order_relaxed);
    const auto misses = gpu_misses.load(std::memory_order_relaxed);
    const auto total = hits + misses;
    const auto loads = tensor_load_count.load(std::memory_order_relaxed);
    GPU_DEBUG_INFO << "[OTD_PERF] gpu_hits=" << hits << ", gpu_misses=" << misses << ", gpu_hit_rate=" << (total > 0 ? 100.0 * hits / total : 0.0) << "%"
                   << ", tensor_loads=" << loads << ", avg_disk_io_us=" << (loads > 0 ? disk_io_ns.load(std::memory_order_relaxed) / 1000 / loads : 0)
                   << ", avg_gpu_copy_us=" << (loads > 0 ? gpu_copy_ns.load(std::memory_order_relaxed) / 1000 / loads : 0)
                   << ", total_disk_io_ms=" << disk_io_ns.load(std::memory_order_relaxed) / 1000000
                   << ", total_gpu_copy_ms=" << gpu_copy_ns.load(std::memory_order_relaxed) / 1000000
                   << ", batched_fallbacks=" << batched_fallbacks.load(std::memory_order_relaxed)
                   << ", grouped_fallbacks=" << grouped_fallbacks.load(std::memory_order_relaxed)
                   << ", created_onednn_kernels=" << created_onednn_kernels.load(std::memory_order_relaxed)
                   << ", les_worker_tasks=" << les_worker_tasks.load(std::memory_order_relaxed)
                   << ", les_upload_chunks=" << les_upload_chunks.load(std::memory_order_relaxed)
                   << ", les_upload_bytes=" << les_upload_bytes.load(std::memory_order_relaxed)
                   << ", les_waves=" << les_waves.load(std::memory_order_relaxed)
                   << ", les_ready_pairs=" << les_ready_pairs.load(std::memory_order_relaxed)
                   << ", les_missing_pairs=" << les_missing_pairs.load(std::memory_order_relaxed)
                   << ", les_shared_pairs=" << les_shared_pairs.load(std::memory_order_relaxed)
                   << ", les_prediction_matches=" << les_prediction_matches.load(std::memory_order_relaxed)
                   << ", les_prediction_pairs=" << les_prediction_pairs.load(std::memory_order_relaxed)
                   << ", les_proxy_weight_initializations=" << les_proxy_weight_initializations.load(std::memory_order_relaxed)
                   << ", les_proxy_fc_ns=" << les_proxy_fc_ns.load(std::memory_order_relaxed)
                   << ", les_proxy_topk_ns=" << les_proxy_topk_ns.load(std::memory_order_relaxed)
                   << ", les_proxy_submit_ns=" << les_proxy_submit_ns.load(std::memory_order_relaxed)
                   << ", les_prefetch_experts=" << les_prefetch_experts.load(std::memory_order_relaxed)
                   << ", les_prefetch_failures=" << les_prefetch_failures.load(std::memory_order_relaxed)
                   << ", les_prefetch_useful_bytes=" << les_prefetch_useful_bytes.load(std::memory_order_relaxed)
                   << ", les_prefetch_useless_bytes=" << les_prefetch_useless_bytes.load(std::memory_order_relaxed)
                   << ", les_prefetch_late_bytes=" << les_prefetch_late_bytes.load(std::memory_order_relaxed)
                   << ", les_prefetch_extra_misses=" << les_prefetch_extra_misses.load(std::memory_order_relaxed)
                   << ", les_prefetch_deduplicated=" << les_prefetch_deduplicated.load(std::memory_order_relaxed)
                   << ", les_prefetch_promoted=" << les_prefetch_promoted.load(std::memory_order_relaxed)
                   << ", les_prefetch_skipped=" << les_prefetch_skipped.load(std::memory_order_relaxed)
                   << ", les_early_predictions=" << les_early_predictions.load(std::memory_order_relaxed) << std::endl;
}

OtdPerfCounters* get_perf_counters() {
    static const bool enabled = [] {
        const auto* value = std::getenv("MOE_OTD_PERF_LOG");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (!enabled) {
        return nullptr;
    }

    static OtdPerfCounters counters;
    static bool registered = [] {
        std::atexit([] {
            counters.dump();
        });
        return true;
    }();
    (void)registered;
    return &counters;
}

void fill_weights_memory(cldnn::stream& exec_stream, ParallelWeightReader& weight_reader, const std::vector<SlicedTensor>& tensors, size_t index, size_t slot) {
    auto* perf = get_perf_counters();
    // Sources and slots outlive the device copies
    const bool blocking = exec_stream.get_queue_type() != cldnn::QueueTypes::in_order;
    std::vector<uint8_t> payload;

    for (const auto& tensor : tensors) {
        if (tensor.source) {
            tensor.slots->copy_from(exec_stream, *tensor.source, index * tensor.slice_size, slot * tensor.slice_size, tensor.slice_size, blocking);
            continue;
        }
        const size_t weight_file_size = weight_reader.file_size();
        const size_t src_offset = tensor.file_offset + (index * tensor.slice_size);
        OPENVINO_ASSERT(src_offset <= weight_file_size && tensor.slice_size <= weight_file_size - src_offset,
                        "Read range out of file: src_offset=",
                        src_offset,
                        ", slice_size=",
                        tensor.slice_size,
                        ", file_size=",
                        weight_file_size,
                        ", index=",
                        index);
        payload.resize(tensor.slice_size);
        const auto t0 = std::chrono::steady_clock::now();
        weight_reader.read(reinterpret_cast<char*>(payload.data()), tensor.slice_size, src_offset);
        const auto t1 = std::chrono::steady_clock::now();
        tensor.slots->copy_from(exec_stream, payload.data(), 0, slot * tensor.slice_size, tensor.slice_size, true);
        if (perf) {
            const auto t2 = std::chrono::steady_clock::now();
            perf->disk_io_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()), std::memory_order_relaxed);
            perf->gpu_copy_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()),
                                        std::memory_order_relaxed);
            perf->tensor_load_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

std::vector<cldnn::event::ptr> fill_weights_memory_async(cldnn::stream& transfer_stream,
                                                      ParallelWeightReader& weight_reader,
                                                      const std::vector<SlicedTensor>& tensors,
                                                      const std::vector<uint32_t>& experts,
                                                      const std::vector<size_t>& slots,
                                                      uint64_t* uploaded_bytes,
                                                      LookaheadStagingPool* staging_pool) {
    constexpr size_t chunk_bytes = LookaheadStagingPool::chunk_bytes;
    struct ReadPlan {
        cldnn::memory::ptr memory;
        cldnn::memory::ptr source_memory;
        size_t source;
        size_t destination;
        size_t bytes;
    };
    OPENVINO_ASSERT(experts.size() == slots.size(), "Invalid LES upload plan");
    std::vector<ReadPlan> plans;
    for (size_t index = 0; index < experts.size(); ++index) {
        for (const auto& tensor : tensors) {
            const auto& memory = tensor.slots;
            const auto bytes = tensor.slice_size;
            if (bytes == 0) {
                continue;
            }
            const auto file_size = tensor.source ? tensor.source->size() : weight_reader.file_size();
            const auto base = tensor.source ? 0 : tensor.file_offset;
            OPENVINO_ASSERT(base <= file_size && experts[index] < (file_size - base) / bytes, "LES source offset overflow");
            const auto source = base + experts[index] * bytes;
            OPENVINO_ASSERT(slots[index] < memory->size() / bytes, "LES upload outside physical slot buffer");
            plans.push_back({memory, tensor.source, source, slots[index] * bytes, bytes});
        }
    }
    LookaheadStagingPool local_pool;
    auto& pool = staging_pool ? *staging_pool : local_pool;
    auto& staging = pool.chunks;
    std::vector<cldnn::event::ptr> events;
    std::vector<cldnn::event::ptr> device_uploads;
    auto& cursor = pool.cursor;
    std::exception_ptr error;
    auto* perf = get_perf_counters();
    try {
        for (const auto& plan : plans) {
            if (plan.source_memory) {
                auto& upload = device_uploads.emplace_back();
                upload = plan.memory->copy_from(transfer_stream, *plan.source_memory, plan.source,
                                                plan.destination, plan.bytes, false);
                events.push_back(upload);
                if (uploaded_bytes) {
                    *uploaded_bytes += plan.bytes;
                }
                transfer_stream.flush();
                continue;
            }
            for (size_t offset = 0; offset < plan.bytes;) {
                auto& buffer = staging[cursor++ % staging.size()];
                if (buffer.upload) {
                    buffer.upload->wait();
                    buffer.upload.reset();
                }
                auto bytes = std::min(chunk_bytes, plan.bytes - offset);
                if (!buffer.memory) {
                    buffer.memory = plan.memory->get_engine()->allocate_memory(
                        cldnn::layout{cldnn::data_types::u8, cldnn::format::bfyx,
                                      {1, 1, 1, static_cast<int32_t>(chunk_bytes)}},
                        cldnn::allocation_type::usm_host, false);
                }
                cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::write> payload(buffer.memory, transfer_stream);
                const auto read_start = les_trace_enabled() ? les_device_time(transfer_stream) : 0;
                weight_reader.read(reinterpret_cast<char*>(payload.data()), bytes, plan.source + offset);
                if (les_trace_enabled()) {
                    les_trace_interval("read", read_start, les_device_time(transfer_stream));
                }
                buffer.upload = plan.memory->copy_from(transfer_stream, payload.data(), 0, plan.destination + offset, bytes, false);
                OPENVINO_ASSERT(buffer.upload, "LES upload did not return a completion event");
                events.push_back(buffer.upload);
                if (uploaded_bytes) {
                    *uploaded_bytes += bytes;
                }
                transfer_stream.flush();
                if (perf) {
                    perf->les_upload_chunks.fetch_add(1, std::memory_order_relaxed);
                    perf->les_upload_bytes.fetch_add(bytes, std::memory_order_relaxed);
                }
                offset += bytes;
            }
        }
    } catch (...) {
        error = std::current_exception();
    }
    for (auto& buffer : staging) {
        if (buffer.upload) {
            try {
                buffer.upload->wait();
                buffer.upload.reset();
            } catch (...) {
                if (!error) {
                    error = std::current_exception();
                }
            }
        }
    }
    for (const auto& event : device_uploads) {
        try {
            if (event) {
                event->wait();
            }
        } catch (...) {
            if (!error) {
                error = std::current_exception();
            }
        }
    }
    if (error) {
        std::rethrow_exception(error);
    }
    return events;
}

}  // namespace ov::intel_gpu::ocl::moe_otd
