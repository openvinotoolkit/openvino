#pragma once

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#    ifdef min
#        undef min
#    endif
#    ifdef max
#        undef max
#    endif
#else
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

#include "LRUCache.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "moe_3gemm_fused_inst.h"

namespace ov::intel_gpu::ocl::moe_otd {

inline size_t get_layer_from_id(const std::string& id) {
    if (id == "moe:moe_router") {
        return 0;
    }

    size_t layer = 0;
    size_t pos = id.rfind('_');
    if (pos != std::string::npos && pos + 1 < id.size()) {
        std::string num_str = id.substr(pos + 1);
        layer = atoi(num_str.c_str());
    }
    return layer;
}

class parallel_weight_reader {
public:
    explicit parallel_weight_reader(const std::string& weights_path) : _weights_path(weights_path) {
        open_shared_handle();
    }

    ~parallel_weight_reader() {
        close_shared_handle();
    }

    const std::string& path() const {
        return _weights_path;
    }

    void read(char* dst, size_t size, size_t file_offset) {
        if (!single_read(get_shared_handle(), dst, size, file_offset)) {
            throw std::runtime_error("Failed to read enough bytes from OTD weight file");
        }
    }

private:
#ifdef _WIN32
    using native_handle_t = HANDLE;
    static constexpr native_handle_t invalid_handle() {
        return INVALID_HANDLE_VALUE;
    }

    static native_handle_t open_native_handle(const std::string& weights_path) {
        auto path = std::filesystem::path(weights_path);
        auto handle =
            CreateFileW(path.native().c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Failed to open weight file for OTD streaming read");
        }
        return handle;
    }

    static void close_native_handle(native_handle_t handle) {
        if (handle != INVALID_HANDLE_VALUE) {
            CloseHandle(handle);
        }
    }

    static bool single_read(native_handle_t handle, char* dst, size_t size, size_t file_offset) {
        char* current = dst;
        size_t remaining = size;
        size_t current_offset = file_offset;
        while (remaining > 0) {
            const DWORD to_read = static_cast<DWORD>(std::min(remaining, static_cast<size_t>(UINT_MAX - 1024u)));
            LARGE_INTEGER li = {};
            li.QuadPart = static_cast<LONGLONG>(current_offset);
            if (!SetFilePointerEx(handle, li, nullptr, FILE_BEGIN)) {
                return false;
            }
            DWORD bytes_read = 0;
            if (!ReadFile(handle, current, to_read, &bytes_read, nullptr) || bytes_read == 0) {
                return false;
            }
            current += bytes_read;
            current_offset += bytes_read;
            remaining -= bytes_read;
        }
        return true;
    }
#else
    using native_handle_t = int;
    static constexpr native_handle_t invalid_handle() {
        return -1;
    }

    static native_handle_t open_native_handle(const std::string& weights_path) {
        auto fd = ::open(weights_path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd == -1) {
            throw std::runtime_error("Failed to open weight file for OTD streaming read");
        }
        return fd;
    }

    static void close_native_handle(native_handle_t handle) {
        if (handle != -1) {
            ::close(handle);
        }
    }

    static bool single_read(native_handle_t handle, char* dst, size_t size, size_t file_offset) {
        char* current = dst;
        size_t remaining = size;
        off_t current_offset = static_cast<off_t>(file_offset);
        while (remaining > 0) {
            const ssize_t bytes_read = ::pread(handle, current, remaining, current_offset);
            if (bytes_read <= 0) {
                return false;
            }
            current += bytes_read;
            current_offset += bytes_read;
            remaining -= static_cast<size_t>(bytes_read);
        }
        return true;
    }
#endif

    void open_shared_handle() {
        _shared_handle = open_native_handle(_weights_path);
    }

    void close_shared_handle() {
        close_native_handle(_shared_handle);
        _shared_handle = invalid_handle();
    }

    native_handle_t get_shared_handle() const {
        return _shared_handle;
    }

    std::string _weights_path;
    native_handle_t _shared_handle = invalid_handle();
};

inline parallel_weight_reader& get_thread_local_weight_reader(const std::string& weights_path) {
    thread_local std::unique_ptr<parallel_weight_reader> reader;
    if (!reader || reader->path() != weights_path) {
        reader = std::make_unique<parallel_weight_reader>(weights_path);
    }
    return *reader;
}

inline void maybe_transpose_scale_zp(const cldnn::moe_3gemm_fused_compressed& desc,
                                     const char* tensor_name,
                                     const cldnn::layout& layout,
                                     std::vector<uint8_t>& payload,
                                     size_t per_expert_size) {
    const bool transpose_scale_zp = std::getenv("MOE_OTD_DISABLE_SCALE_ZP_TRANSPOSE") == nullptr;
    if (!transpose_scale_zp || tensor_name == nullptr) {
        return;
    }

    const std::string_view name(tensor_name);
    const bool is_scale = name.find("_s") != std::string_view::npos;
    const bool is_zp = name.find("_z") != std::string_view::npos;
    if (!is_scale && !is_zp) {
        return;
    }

    size_t oc = 0;
    size_t ic = 0;
    if (name.rfind("down_", 0) == 0) {
        oc = static_cast<size_t>(desc._config.hidden_size);
        ic = static_cast<size_t>(desc._config.inter_size);
    } else {
        oc = static_cast<size_t>(desc._config.inter_size);
        ic = static_cast<size_t>(desc._config.hidden_size);
    }

    const size_t group_size = static_cast<size_t>(desc._config.group_size);
    size_t group_count = 1;
    if (group_size != 0 && group_size != std::numeric_limits<size_t>::max()) {
        OPENVINO_ASSERT(ic % group_size == 0, "Invalid group_size for OTD transpose: tensor=", tensor_name, ", ic=", ic, ", group_size=", group_size);
        group_count = ic / group_size;
    }

    OPENVINO_ASSERT(oc > 0 && group_count > 0, "Invalid dims for OTD transpose: tensor=", tensor_name, ", oc=", oc, ", group_count=", group_count);

    const size_t elem_count = oc * group_count;
    if (is_scale) {
        const size_t elem_size = static_cast<size_t>(data_type_traits::size_of(layout.data_type));
        OPENVINO_ASSERT(elem_size > 0, "Invalid scale element size for tensor=", tensor_name);
        OPENVINO_ASSERT(elem_count * elem_size == per_expert_size,
                        "Unexpected scale payload size for tensor=",
                        tensor_name,
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

    OPENVINO_ASSERT(elem_count % 2 == 0, "Unexpected odd element count for packed zp tensor=", tensor_name, ", elem_count=", elem_count);
    OPENVINO_ASSERT(elem_count / 2 == per_expert_size,
                    "Unexpected zp payload size for tensor=",
                    tensor_name,
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

inline void fill_weights_memory(cldnn::stream& exec_stream,
                                const cldnn::moe_3gemm_fused_compressed& desc,
                                cldnn::moe_weights& wei_mem,
                                const std::vector<uint32_t>& experts_list,
                                const std::vector<uint32_t>& lru_experts) {
    struct tensor_fill_plan {
        size_t per_expert_size = 0;
        size_t src_offset = 0;
        size_t dst_offset = 0;
    };

    const auto num_expert = static_cast<size_t>(desc._config.num_expert);
    const auto& weight_bin_offsets = desc._weight_bin_offsets;
    const auto& weights_path = desc._weights_path;

    OPENVINO_ASSERT(!weights_path.empty(), "weights path is empty for OTD weight loading");
    OPENVINO_ASSERT(weight_bin_offsets.size() == cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count,
                    "Unexpected number of MOE weight offsets");

    std::ifstream size_file(weights_path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    OPENVINO_ASSERT(size_file.is_open(), "Failed to open weight file to query size: ", weights_path);
    auto end_pos = size_file.tellg();
    OPENVINO_ASSERT(end_pos >= 0, "Failed to query weight file size: ", weights_path);
    const size_t weight_file_size = static_cast<size_t>(end_pos);

    static const std::array<const char*, cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count> tensor_names = {
        {"gate_w", "up_w", "down_w", "gate_s", "up_s", "down_s", "gate_z", "up_z", "down_z"}};
    const std::array<cldnn::memory_ptr, cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count> tensors_by_offset = {
        {wei_mem.gate_w, wei_mem.up_w, wei_mem.down_w, wei_mem.gate_s, wei_mem.up_s, wei_mem.down_s, wei_mem.gate_z, wei_mem.up_z, wei_mem.down_z}};

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

    auto copy_tensor_to_memory = [&](cldnn::memory_ptr mem, const tensor_fill_plan& plan, std::vector<uint8_t>& payload, const char* tensor_name) {
        if (!mem || plan.per_expert_size == 0) {
            return;
        }

        maybe_transpose_scale_zp(desc, tensor_name, mem->get_layout(), payload, plan.per_expert_size);
        mem->copy_from(exec_stream, payload.data(), 0, plan.dst_offset, plan.per_expert_size, true);
    };

    size_t index = 0;
    for (uint32_t expert : experts_list) {
        auto& weight_reader = get_thread_local_weight_reader(weights_path);

        for (size_t offset_pos = 0; offset_pos < static_cast<size_t>(cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count); offset_pos++) {
            auto plan = make_tensor_fill_plan(weight_bin_offsets[offset_pos], tensors_by_offset[offset_pos], expert, lru_experts[index], tensor_names[offset_pos]);
            std::vector<uint8_t> payload;

            if (plan.per_expert_size != 0) {
                payload.resize(plan.per_expert_size);
                weight_reader.read(reinterpret_cast<char*>(payload.data()), plan.per_expert_size, plan.src_offset);
            }

            copy_tensor_to_memory(tensors_by_offset[offset_pos], plan, payload, tensor_names[offset_pos]);
        }

        index++;
    }
}

inline uint32_t get_lru_expert_no(typed_primitive_inst<cldnn::moe_3gemm_fused_compressed>& instance, uint32_t expert, LRUCache& cache) {
    auto cur_moe = instance.get_typed_desc<cldnn::moe_3gemm_fused_compressed>();
    auto& stream = instance.get_network().get_stream();
    size_t layer = get_layer_from_id(cur_moe->id);
    auto item = cache.get_lru_item(layer, expert);
    OPENVINO_ASSERT(item.first <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()), "LRU slot index overflow: ", item.first);
    const auto lru_slot = static_cast<uint32_t>(item.first);
    if (!item.second) {
        std::vector<uint32_t> experts_list_single;
        experts_list_single.push_back(expert);
        std::vector<uint32_t> lru_experts_list_single;
        lru_experts_list_single.push_back(lru_slot);
        fill_weights_memory(stream, *cur_moe, instance._weights, experts_list_single, lru_experts_list_single);
        cache.set_filled(lru_slot);
    }
    return lru_slot;
}

}  // namespace ov::intel_gpu::ocl::moe_otd