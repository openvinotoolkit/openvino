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
    std::cout << "[OTD_PERF] gpu_hits=" << hits << ", gpu_misses=" << misses << ", gpu_hit_rate=" << (total > 0 ? 100.0 * hits / total : 0.0) << "%"
                   << ", tensor_loads=" << loads << ", avg_disk_io_us=" << (loads > 0 ? disk_io_ns.load(std::memory_order_relaxed) / 1000 / loads : 0)
                   << ", avg_transpose_us=" << (loads > 0 ? transpose_ns.load(std::memory_order_relaxed) / 1000 / loads : 0)
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

void maybe_transpose_scale_zp(const cldnn::MOECompressed::Config& config,
                              size_t offset_pos,
                              const cldnn::layout& layout,
                              std::vector<uint8_t>& payload,
                              size_t per_expert_size) {
    // Offset positions: 0-2 = weights (no transpose), 3-5 = scales, 6-8 = zero points.
    // Within each group of 3: index % 3 == 2 means "down" variant.
    const bool is_scale = (offset_pos >= 3 && offset_pos <= 5);
    const bool is_zp = (offset_pos >= 6 && offset_pos <= 8);
    if (!is_scale && !is_zp) {
        return;
    }

    const bool is_down = (offset_pos % 3 == 2);
    size_t oc = 0;
    size_t ic = 0;
    if (is_down) {
        oc = static_cast<size_t>(config.hidden_size);
        ic = static_cast<size_t>(config.inter_size);
    } else {
        oc = static_cast<size_t>(config.inter_size);
        ic = static_cast<size_t>(config.hidden_size);
    }

    const size_t group_size = static_cast<size_t>(config.group_size);
    size_t group_count = 1;
    if (group_size != 0 && group_size != std::numeric_limits<size_t>::max()) {
        OPENVINO_ASSERT(ic % group_size == 0, "Invalid group_size for OTD transpose: offset_pos=", offset_pos, ", ic=", ic, ", group_size=", group_size);
        group_count = ic / group_size;
    }

    OPENVINO_ASSERT(oc > 0 && group_count > 0, "Invalid dims for OTD transpose: offset_pos=", offset_pos, ", oc=", oc, ", group_count=", group_count);

    const size_t elem_count = oc * group_count;
    if (is_scale) {
        const size_t elem_size = static_cast<size_t>(cldnn::data_type_traits::size_of(layout.data_type));
        OPENVINO_ASSERT(elem_size > 0, "Invalid scale element size for offset_pos=", offset_pos);
        OPENVINO_ASSERT(elem_count * elem_size == per_expert_size,
                        "Unexpected scale payload size for offset_pos=",
                        offset_pos,
                        ", expected=",
                        elem_count * elem_size,
                        ", got=",
                        per_expert_size);

        std::vector<uint8_t> transposed(per_expert_size, 0);
        for (size_t o = 0; o < oc; o++) {
            for (size_t g = 0; g < group_count; g++) {
                const size_t src_elem_idx = o * group_count + g;
                const size_t dst_elem_idx = g * oc + o;
                std::memcpy(transposed.data() + dst_elem_idx * elem_size, payload.data() + src_elem_idx * elem_size, elem_size);
            }
        }
        payload.swap(transposed);
        return;
    }

    // ZP handling: u4 (packed 4-bit) requires unpack/transpose/repack;
    // u8 (INT8) is 1 byte per element, transposed like scales.
    const auto zp_et = ov::element::Type(layout.data_type);
    const size_t zp_bitwidth = zp_et.bitwidth();
    if (zp_bitwidth >= 8) {
        // u8 or larger: transpose like scale (1+ bytes per element)
        const size_t byte_per_elem = zp_bitwidth / 8;
        OPENVINO_ASSERT(elem_count * byte_per_elem == per_expert_size,
                        "Unexpected zp payload size for offset_pos=",
                        offset_pos,
                        ", expected=",
                        elem_count * byte_per_elem,
                        ", got=",
                        per_expert_size);

        std::vector<uint8_t> transposed(per_expert_size, 0);
        for (size_t o = 0; o < oc; o++) {
            for (size_t g = 0; g < group_count; g++) {
                const size_t src_idx = o * group_count + g;
                const size_t dst_idx = g * oc + o;
                std::memcpy(transposed.data() + dst_idx * byte_per_elem, payload.data() + src_idx * byte_per_elem, byte_per_elem);
            }
        }
        payload.swap(transposed);
        return;
    }

    // u4 packed: 2 elements per byte
    OPENVINO_ASSERT(elem_count % 2 == 0, "Unexpected odd element count for packed zp offset_pos=", offset_pos, ", elem_count=", elem_count);
    OPENVINO_ASSERT(elem_count / 2 == per_expert_size,
                    "Unexpected zp payload size for offset_pos=",
                    offset_pos,
                    ", expected=",
                    elem_count / 2,
                    ", got=",
                    per_expert_size);

    std::vector<uint8_t> unpacked(elem_count, 0);
    for (size_t i = 0; i < per_expert_size; i++) {
        const uint8_t byte = payload[i];
        unpacked[2 * i] = static_cast<uint8_t>(byte & 0x0F);
        unpacked[2 * i + 1] = static_cast<uint8_t>((byte >> 4) & 0x0F);
    }

    std::vector<uint8_t> transposed_unpacked(elem_count, 0);
    for (size_t o = 0; o < oc; o++) {
        for (size_t g = 0; g < group_count; g++) {
            const size_t src_idx = o * group_count + g;
            const size_t dst_idx = g * oc + o;
            transposed_unpacked[dst_idx] = unpacked[src_idx];
        }
    }

    std::vector<uint8_t> repacked(per_expert_size, 0);
    for (size_t i = 0; i < per_expert_size; i++) {
        repacked[i] = static_cast<uint8_t>((transposed_unpacked[2 * i] & 0x0F) | ((transposed_unpacked[2 * i + 1] & 0x0F) << 4));
    }
    payload.swap(repacked);
}

void fill_weights_memory(cldnn::stream& exec_stream,
                         const cldnn::MOECompressed::Config& config,
                         const std::vector<size_t>& weight_bin_offsets,
                         ParallelWeightReader& weight_reader,
                         cldnn::moe_weights& wei_mem,
                         const std::vector<uint32_t>& experts_list,
                         const std::vector<size_t>& lru_experts) {
    struct tensor_fill_plan {
        size_t per_expert_size = 0;
        size_t src_offset = 0;
        size_t dst_offset = 0;
    };

    const auto num_expert = static_cast<size_t>(config.num_expert);

    OPENVINO_ASSERT(weight_bin_offsets.size() == cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count, "Unexpected number of MOE weight offsets");

    static const std::array<const char*, cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count> tensor_names = {
        {"gate_w", "up_w", "down_w", "gate_s", "up_s", "down_s", "gate_z", "up_z", "down_z"}};
    const std::array<cldnn::memory_ptr, cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count> tensors_by_offset = {
        {wei_mem.gate_w, wei_mem.up_w, wei_mem.down_w, wei_mem.gate_s, wei_mem.up_s, wei_mem.down_s, wei_mem.gate_z, wei_mem.up_z, wei_mem.down_z}};

    auto* perf = get_perf_counters();

    const size_t weight_file_size = weight_reader.file_size();

    size_t index = 0;
    for (uint32_t expert : experts_list) {
        auto make_tensor_fill_plan = [&](size_t base_offset, cldnn::memory_ptr mem, size_t expert_no, size_t lru_expert_no, const char* tensor_name) {
            tensor_fill_plan plan;
            if (!mem) {
                return plan;
            }
            const auto total_bytes = mem->get_layout().bytes_count();
            OPENVINO_ASSERT(num_expert > 0, "Invalid expert count");
            plan.per_expert_size = total_bytes / num_expert;
            plan.src_offset = base_offset + expert_no * plan.per_expert_size;
            plan.dst_offset = lru_expert_no * plan.per_expert_size;
            OPENVINO_ASSERT(plan.src_offset <= weight_file_size, "Invalid src_offset out of file: ", plan.src_offset, ", file_size=", weight_file_size);
            OPENVINO_ASSERT(plan.per_expert_size <= weight_file_size - plan.src_offset,
                            "Read range out of file for tensor ",
                            tensor_name,
                            ": src_offset=",
                            plan.src_offset,
                            ", per_expert_size=",
                            plan.per_expert_size,
                            ", file_size=",
                            weight_file_size,
                            ", base_offset=",
                            base_offset,
                            ", expert=",
                            expert_no);
            return plan;
        };

        for (size_t offset_pos = 0; offset_pos < static_cast<size_t>(cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count); offset_pos++) {
            auto plan =
                make_tensor_fill_plan(weight_bin_offsets[offset_pos], tensors_by_offset[offset_pos], expert, lru_experts[index], tensor_names[offset_pos]);
            std::vector<uint8_t> payload;

            if (plan.per_expert_size != 0) {
                payload.resize(plan.per_expert_size);
                if (perf) {
                    auto t0 = std::chrono::steady_clock::now();
                    weight_reader.read(reinterpret_cast<char*>(payload.data()), plan.per_expert_size, plan.src_offset);
                    auto t1 = std::chrono::steady_clock::now();
                    perf->disk_io_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
                                               std::memory_order_relaxed);
                } else {
                    weight_reader.read(reinterpret_cast<char*>(payload.data()), plan.per_expert_size, plan.src_offset);
                }
            }

            // Transpose + GPU copy
            auto mem = tensors_by_offset[offset_pos];
            if (mem && plan.per_expert_size != 0) {
                if (perf) {
                    auto t0 = std::chrono::steady_clock::now();
                    maybe_transpose_scale_zp(config, offset_pos, mem->get_layout(), payload, plan.per_expert_size);
                    auto t1 = std::chrono::steady_clock::now();
                    mem->copy_from(exec_stream, payload.data(), 0, plan.dst_offset, plan.per_expert_size, true);
                    auto t2 = std::chrono::steady_clock::now();
                    perf->transpose_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
                                                 std::memory_order_relaxed);
                    perf->gpu_copy_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()),
                                                std::memory_order_relaxed);
                    perf->tensor_load_count.fetch_add(1, std::memory_order_relaxed);
                } else {
                    maybe_transpose_scale_zp(config, offset_pos, mem->get_layout(), payload, plan.per_expert_size);
                    mem->copy_from(exec_stream, payload.data(), 0, plan.dst_offset, plan.per_expert_size, true);
                }
            }
        }

        index++;
    }
}

std::vector<cldnn::event::ptr> fill_weights_memory_async(cldnn::stream& transfer_stream,
                                                      const cldnn::MOECompressed::Config& config,
                                                      const std::vector<size_t>& weight_bin_offsets,
                                                      ParallelWeightReader& weight_reader,
                                                      cldnn::moe_weights& weights,
                                                      const std::vector<uint32_t>& experts,
                                                      const std::vector<size_t>& slots,
                                                      uint64_t* uploaded_bytes,
                                                      LookaheadStagingPool* staging_pool) {
    constexpr size_t chunk_bytes = LookaheadStagingPool::chunk_bytes;
    const std::array<cldnn::memory::ptr, 9> tensors{weights.gate_w, weights.up_w, weights.down_w,
                                                weights.gate_s, weights.up_s, weights.down_s,
                                                weights.gate_z, weights.up_z, weights.down_z};
    struct ReadPlan {
        cldnn::memory::ptr memory;
        size_t component;
        size_t source;
        size_t destination;
        size_t bytes;
    };
    OPENVINO_ASSERT(weight_bin_offsets.size() == tensors.size() && experts.size() == slots.size(), "Invalid LES upload plan");
    const auto expert_count = static_cast<size_t>(config.num_expert);
    OPENVINO_ASSERT(expert_count > 0, "Invalid LES expert count");
    const auto file_size = weight_reader.file_size();
    std::vector<ReadPlan> plans;
    for (size_t index = 0; index < experts.size(); ++index) {
        OPENVINO_ASSERT(experts[index] < expert_count, "LES expert outside configured range");
        for (size_t component = 0; component < tensors.size(); ++component) {
            const auto& memory = tensors[component];
            if (!memory) {
                continue;
            }
            auto total = memory->get_layout().bytes_count();
            OPENVINO_ASSERT(total % expert_count == 0, "Invalid LES per-expert tensor size");
            auto bytes = total / expert_count;
            if (bytes == 0) {
                continue;
            }
            auto base = weight_bin_offsets[component];
            OPENVINO_ASSERT(base <= file_size && experts[index] <= (file_size - base) / bytes, "LES source offset overflow");
            auto source = base + experts[index] * bytes;
            OPENVINO_ASSERT(bytes <= file_size - source, "LES tensor read outside weight file");
            OPENVINO_ASSERT(slots[index] < memory->size() / bytes, "LES upload outside physical slot buffer");
            OPENVINO_ASSERT(component < 3 || bytes <= chunk_bytes, "LES transposed metadata exceeds 8 MiB per expert");
            plans.push_back({memory, component, source, slots[index] * bytes, bytes});
        }
    }
    LookaheadStagingPool local_pool;
    auto& pool = staging_pool ? *staging_pool : local_pool;
    auto& staging = pool.chunks;
    std::vector<cldnn::event::ptr> events;
    auto& cursor = pool.cursor;
    std::exception_ptr error;
    auto* perf = get_perf_counters();
    try {
        for (const auto& plan : plans) {
            for (size_t offset = 0; offset < plan.bytes;) {
                auto& buffer = staging[cursor++ % staging.size()];
                if (buffer.upload) {
                    buffer.upload->wait();
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
                if (plan.component >= 3) {
                    std::vector<uint8_t> metadata(payload.data(), payload.data() + bytes);
                    maybe_transpose_scale_zp(config, plan.component, plan.memory->get_layout(), metadata, bytes);
                    std::memcpy(payload.data(), metadata.data(), bytes);
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
            } catch (...) {
                if (!error) {
                    error = std::current_exception();
                }
            }
        }
    }
    if (error) {
        std::rethrow_exception(error);
    }
    return events;
}

}  // namespace ov::intel_gpu::ocl::moe_otd
