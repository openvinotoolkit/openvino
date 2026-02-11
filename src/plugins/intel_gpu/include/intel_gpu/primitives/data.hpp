// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <algorithm>
#include <atomic>
#include <climits>
#include <variant>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <mutex>
#include <future>
#include <cstring>
#include <cstdlib>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fstream>
#include <malloc.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#endif
#include "intel_gpu/runtime/itt.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"
#include "primitive.hpp"
#include "transformations/convert_precision.hpp"

using weights_memory_ptr = std::variant<std::shared_ptr<ov::MappedMemory>, std::shared_ptr<const ov::Model>>;
using offset_const_map_t = std::map<size_t, std::shared_ptr<ov::op::v0::Constant>>;
using shared_mapped_memory_ptr = std::shared_ptr<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>;
using constant_memory_ptr = std::variant<shared_mapped_memory_ptr, std::shared_ptr<ov::op::v0::Constant>>;

namespace {

bool is_alloc_host_accessible(const cldnn::allocation_type& alloc_type) {
    return alloc_type == cldnn::allocation_type::usm_host || alloc_type == cldnn::allocation_type::usm_shared;
}

void copy_to_dst_mem(cldnn::memory::ptr mem_ptr, const uint8_t* data_ptr) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "copy_to_dst_mem");
    if (is_alloc_host_accessible(mem_ptr->get_allocation_type())) {
        size_t data_size = mem_ptr->size();
        std::memcpy(reinterpret_cast<uint8_t*>(mem_ptr->buffer_ptr()),
                    data_ptr,
                    data_size);
    } else {
        auto& strm = mem_ptr->get_engine()->get_service_stream();
        mem_ptr->copy_from(strm, data_ptr);
    }
}

#ifdef __linux__
class fd_accessor : public std::filebuf {
public:
   int get_fd() { return _M_file.fd(); }
};
#endif

#ifdef _WIN32
size_t get_file_size(const std::string& path) {
    if (path.empty()) return 0;
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, NULL, 0);
    if (wlen <= 0) return 0;
    std::wstring wpath(wlen, 0);
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &wpath[0], wlen);
    if (wlen > 0) wpath.resize(wlen - 1);

    HANDLE hFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return 0;

    LARGE_INTEGER fileSize;
    size_t size = 0;
    if (GetFileSizeEx(hFile, &fileSize)) {
        size = static_cast<size_t>(fileSize.QuadPart);
    }
    CloseHandle(hFile);
    return size;
}
#else
size_t get_file_size(const std::string& path) {
     struct stat stat_buf;
     int rc = stat(path.c_str(), &stat_buf);
     return rc == 0 ? stat_buf.st_size : 0;
}
#endif

bool load_direct(std::istream& stream, void* buffer, size_t size) {
#ifdef __linux__
    auto* buf = stream.rdbuf();
    if (auto* fbuf = dynamic_cast<std::filebuf*>(buf)) {
         int fd = static_cast<fd_accessor*>(fbuf)->get_fd();
         if (fd >= 0) {
             size_t current_pos = stream.tellg();
             OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "load_weights::load_direct");

             std::string path = "/proc/self/fd/" + std::to_string(fd);
             int direct_fd = open(path.c_str(), O_RDONLY | O_DIRECT);
             if (direct_fd != -1) {
                 uintptr_t addr = reinterpret_cast<uintptr_t>(buffer);
                 size_t offset = current_pos;

                 bool addr_aligned = (addr % 4096) == 0;
                 bool offset_aligned = (offset % 4096) == 0;
                 bool size_aligned = (size % 4096) == 0;

                 if (addr_aligned && offset_aligned && size_aligned) {
                      ssize_t ret = pread(direct_fd, buffer, size, offset);
                      close(direct_fd);
                      if (ret == (ssize_t)size) {
                          stream.seekg(size, std::ios::cur);
                          return true;
                      }
                 } else {
                      size_t align_mask = 4095;
                      size_t file_start = offset & ~align_mask;
                      size_t file_end = (offset + size + align_mask) & ~align_mask;
                      size_t read_len = file_end - file_start;

                      void* tmp_buf = nullptr;
                      if (posix_memalign(&tmp_buf, 4096, read_len) == 0) {
                          ssize_t ret = pread(direct_fd, tmp_buf, read_len, file_start);
                          close(direct_fd);
                          if (ret == (ssize_t)read_len) {
                              size_t copy_off = offset - file_start;
                              memcpy(buffer, (char*)tmp_buf + copy_off, size);
                              free(tmp_buf);
                              stream.seekg(size, std::ios::cur);
                              return true;
                          }
                          free(tmp_buf);
                      } else {
                          close(direct_fd);
                      }
                 }
             }
         }
    }
#elif defined(_WIN32)
    auto* buf = stream.rdbuf();
    // On Windows, extracting file handle from std::filebuf is non-trivial and non-standard.
    // However, if we assume we are reading from the beginning or we know the path, we might re-open it.
    // But BinaryInputBuffer wraps a generic istream.
    return false;
#endif
    return false;
}

#ifdef _WIN32
bool load_parallel(const std::string& path, size_t offset, void* buffer, size_t size) {
    if (path.empty()) return false;

    // Convert UTF-8 path to Wide String for CreateFileW
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, NULL, 0);
    if (wlen <= 0) return false;
    std::wstring wpath(wlen, 0);
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &wpath[0], wlen);
    // wlen includes the null terminator, resizing to remove it for clean string object
    if (wlen > 0) wpath.resize(wlen - 1);

    HANDLE hFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return false;

    // Safety check: File size
    LARGE_INTEGER fileSize;
    if (GetFileSizeEx(hFile, &fileSize)) {
        if (static_cast<unsigned long long>(fileSize.QuadPart) < offset + size) {
            CloseHandle(hFile);
            return false;
        }
    }
    CloseHandle(hFile);

    const size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), size / (1024 * 1024));
    if (num_threads <= 1) {
        return false;
    }

    std::vector<std::future<void>> futures;
    size_t chunk_size = size / num_threads;
    chunk_size = (chunk_size + 4095) & ~4095;

    size_t current_offset = 0;
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "load_weights::load_parallel_win");

    std::atomic<bool> overall_status{true};

    for (size_t i = 0; i < num_threads; i++) {
        size_t read_size = (i == num_threads - 1) ? (size - current_offset) : chunk_size;
        if (read_size == 0) break;

        void* ptr = static_cast<char*>(buffer) + current_offset;
        size_t file_offset = offset + current_offset;

        futures.emplace_back(std::async(std::launch::async, [wpath, file_offset, ptr, read_size, &overall_status] {
            HANDLE t_hFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
            if (t_hFile == INVALID_HANDLE_VALUE) {
                overall_status = false;
                return;
            }
            OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "load_parallel_win_chunk");

            size_t remaining_size = read_size;
            char* current_ptr = static_cast<char*>(ptr);
            size_t current_file_offset = file_offset;

            // Loop to handle reads larger than DWORD max (4GB)
            while (remaining_size > 0 && overall_status) {
                DWORD to_read = static_cast<DWORD>(std::min(remaining_size, static_cast<size_t>(UINT_MAX - 1024)));
                OVERLAPPED ov = {0};
                ov.Offset = static_cast<DWORD>(current_file_offset & 0xFFFFFFFF);
                ov.OffsetHigh = static_cast<DWORD>((current_file_offset >> 32) & 0xFFFFFFFF);

                DWORD bytesRead = 0;
                if (!ReadFile(t_hFile, current_ptr, to_read, &bytesRead, &ov) || bytesRead != to_read) {
                    if (GetLastError() != ERROR_IO_PENDING) {
                        overall_status = false;
                        break;
                    }
                }

                remaining_size -= bytesRead;
                current_ptr += bytesRead;
                current_file_offset += bytesRead;
            }
            CloseHandle(t_hFile);
        }));

        current_offset += read_size;
    }

    for (auto& f : futures) {
        f.get();
    }
    return overall_status;
}
#else
bool load_parallel(const std::string& path, size_t offset, void* buffer, size_t size) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return false;

    const size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), size / (1024 * 1024));
    if (num_threads <= 1) return false;

    std::vector<std::future<void>> futures;
    size_t chunk_size = size / num_threads;
    // Align chunk size to 4KB for better performance
    chunk_size = (chunk_size + 4095) & ~4095;

    size_t current_offset = 0;
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "load_weights::load_parallel");

    // We open file in each thread to have independent file pointers
    for (size_t i = 0; i < num_threads; i++) {
        size_t read_size = (i == num_threads - 1) ? (size - current_offset) : chunk_size;
        if (read_size == 0) break;

        void* ptr = static_cast<char*>(buffer) + current_offset;
        size_t file_offset = offset + current_offset;

        futures.emplace_back(std::async(std::launch::async, [path, file_offset, ptr, read_size] {
            std::ifstream t_ifs(path, std::ios::binary);
            OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "load_parallel_chunk");
            if (t_ifs.is_open()) {
                t_ifs.seekg(file_offset, std::ios::beg);
                t_ifs.read(static_cast<char*>(ptr), read_size);
            }
        }));

        current_offset += read_size;
    }

    for (auto& f : futures) {
        f.get();
    }
    return true;
}
#endif

}  // namespace

namespace cldnn {

class WeightsMemory {
public:
    WeightsMemory(std::shared_ptr<const ov::Model> model,
                  std::shared_ptr<ov::intel_gpu::GpuWeightlessCacheMap> cache_attr_map = nullptr) : weights_memory(model) {
        fill_offset_to_constant_map(model, cache_attr_map);
    }

    WeightsMemory(std::shared_ptr<ov::MappedMemory> mapped_memory) : weights_memory(mapped_memory) {}

    constant_memory_ptr get_constant_buf(size_t bin_offset, size_t original_size) {
        if (std::holds_alternative<std::shared_ptr<ov::MappedMemory>>(weights_memory)) {
            auto mapped_memory = std::get<std::shared_ptr<ov::MappedMemory>>(weights_memory);
            return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
                mapped_memory->data() + bin_offset,
                original_size,
                mapped_memory);
        } else {
            auto model_ptr = std::get<std::shared_ptr<const ov::Model>>(weights_memory);
            auto const_it = offset_to_constant_map.find(bin_offset);
            if (const_it == offset_to_constant_map.end()) {
                OPENVINO_THROW("Constant with bin_offset ", bin_offset, " not found in the model");
            }
            auto const_ptr = const_it->second;
            return const_ptr;
        }
    }

private:
    void fill_offset_to_constant_map(std::shared_ptr<const ov::Model> model,
                                     std::shared_ptr<ov::intel_gpu::GpuWeightlessCacheMap> cache_attr_map = nullptr) {
        const auto& ops = model->get_ops();

        if (cache_attr_map != nullptr && cache_attr_map->size() > 0) {
            for (const auto& node : ops) {
                if (ov::op::util::is_constant(node)) {
                    auto it = cache_attr_map->find(node->get_instance_id());
                    if (it != cache_attr_map->end()) {
                        auto attr = it->second;
                        auto const_ptr = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
                        offset_to_constant_map.emplace(attr.bin_offset, const_ptr);
                    }
                }
            }
        } else {
            for (const auto& node : ops) {
                if (ov::op::util::is_constant(node)) {
                    auto rt_info = node->get_rt_info();
                    auto weightless_cache_attr = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());
                    if (weightless_cache_attr != rt_info.end()) {
                        auto& attr = weightless_cache_attr->second.as<ov::WeightlessCacheAttribute>();
                        auto const_ptr = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
                        offset_to_constant_map.emplace(attr.bin_offset, const_ptr);
                    }
                } else if (auto ti = ov::as_type<const ov::op::v0::TensorIterator>(node.get())) {
                    auto ti_body = ti->get_body();
                    fill_offset_to_constant_map(ti_body);
                }
            }
        }
    }

    weights_memory_ptr weights_memory;
    offset_const_map_t offset_to_constant_map{};
};

struct reorder_replication {
    std::shared_ptr<cldnn::layout> input_layout = nullptr;
    std::shared_ptr<cldnn::reorder> reorder = nullptr;
};

struct weightless_cache_manager {
    void set_constant_info(size_t bin_offset,
                           size_t original_size,
                           ov::element::Type original_dtype,
                           ov::element::Type curr_dtype,
                           ov::Shape shape) {
        this->bin_offset = bin_offset;
        this->original_size = original_size;
        this->original_dtype = original_dtype;
        this->curr_dtype = curr_dtype;
        this->shape = shape;
        do_weightless_caching = true;

        if (original_dtype != curr_dtype) {
            do_precision_conversion = true;
        }
    }

    void apply_reorder(std::shared_ptr<layout> input_layout, std::shared_ptr<reorder> reorder) {
        reorder_rep = {input_layout, reorder};
    }

    bool save(BinaryOutputBuffer& ob, size_t data_size) const {
        if (!do_weightless_caching) {
            ob << false;
            return false;
        }

        ob << true;
        ob << bin_offset;
        ob << do_precision_conversion;
        if (do_precision_conversion) {
            ob << original_size;
            ob << make_data(&original_dtype, sizeof(ov::element::Type));
            ob << make_data(&curr_dtype, sizeof(ov::element::Type));

            size_t num_dims = shape.size();
            ob << make_data(&num_dims, sizeof(size_t));
            ob << make_data(shape.data(), num_dims * sizeof(ov::Shape::value_type));
        }

        bool do_reorder = should_run_reorder();
        if (do_reorder) {
            ob << true;
            ob << *reorder_rep.input_layout;
            ob << *reorder_rep.reorder;
        } else {
            ob << false;
        }
        return true;
    }

    bool load(BinaryInputBuffer& ib, memory::ptr dst_mem, std::shared_ptr<WeightsMemory> weights_memory) {
        ib >> do_weightless_caching;
        if (!do_weightless_caching) {
            return false;
        }

        ib >> bin_offset;
        ib >> do_precision_conversion;
        if (do_precision_conversion) {
            ib >> original_size;
            ib >> make_data(&original_dtype, sizeof(ov::element::Type));
            ib >> make_data(&curr_dtype, sizeof(ov::element::Type));

            size_t num_dims = 0;
            ib >> make_data(&num_dims, sizeof(size_t));
            shape.resize(num_dims);
            ib >> make_data(shape.data(), num_dims * sizeof(ov::Shape::value_type));
        } else {
            original_size = dst_mem->size();
        }

        bool do_reorder = false;
        ib >> do_reorder;
        if (do_reorder) {
            reorder_rep.input_layout = std::make_shared<layout>();
            ib >> *reorder_rep.input_layout;
            reorder_rep.reorder = std::make_shared<reorder>();
            ib >> *reorder_rep.reorder;
        }

        OPENVINO_ASSERT(weights_memory != nullptr, "weights_memory is nullptr!!!");
        auto constant_ptr = weights_memory->get_constant_buf(bin_offset, original_size);

        if (should_run_transformations()) {
            run_transformations(ib.get_engine(), dst_mem, constant_ptr);
        } else {
            if (std::holds_alternative<std::shared_ptr<ov::op::v0::Constant>>(constant_ptr)) {
                auto cptr = std::get<std::shared_ptr<ov::op::v0::Constant>>(constant_ptr);
                copy_to_dst_mem(dst_mem, reinterpret_cast<const uint8_t*>(cptr->get_data_ptr()));
            } else {
                auto shared_buf = std::get<shared_mapped_memory_ptr>(constant_ptr);
                copy_to_dst_mem(dst_mem, shared_buf->get_ptr<uint8_t>());
            }
        }
        return true;
    }


private:
    bool do_weightless_caching = false;
    bool do_precision_conversion = false;
    reorder_replication reorder_rep{};

    size_t bin_offset = SIZE_MAX;
    size_t original_size = SIZE_MAX;
    ov::element::Type original_dtype = ov::element::Type_t::dynamic;
    ov::element::Type curr_dtype = ov::element::Type_t::dynamic;
    ov::Shape shape{};

    bool should_run_reorder() const {
        return reorder_rep.reorder != nullptr;
    }

    bool should_run_transformations() {
        return do_precision_conversion || should_run_reorder();
    }

    void run_transformations(engine& engine,
                             memory::ptr dst_mem,
                             constant_memory_ptr constant_ptr) {
        std::shared_ptr<ov::op::v0::Constant> transformed_constant = nullptr;
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "weightless_cache_manager::run_transformations");

        // Note: this works only until the data is copied to dst_mem.
        auto get_intermediate_data = [&]() -> const uint8_t* {
            if (transformed_constant) {
                return reinterpret_cast<const uint8_t*>(transformed_constant->get_data_ptr());
            }

            if (std::holds_alternative<std::shared_ptr<ov::op::v0::Constant>>(constant_ptr)) {
                auto cptr = std::get<std::shared_ptr<ov::op::v0::Constant>>(constant_ptr);
                return reinterpret_cast<const uint8_t*>(cptr->get_data_ptr());
            } else {
                auto shared_buf = std::get<shared_mapped_memory_ptr>(constant_ptr);
                return shared_buf->get_ptr<uint8_t>();
            }
        };

        // Note: this works only until the data is copied to dst_mem.
        auto get_current_data_size = [&]() -> size_t {
            if (transformed_constant) {
                return transformed_constant->get_byte_size();
            }
            return original_size;
        };

        if (do_precision_conversion) {
            std::shared_ptr<ov::op::v0::Constant> orig_constant = nullptr;
            if (std::holds_alternative<std::shared_ptr<ov::op::v0::Constant>>(constant_ptr)) {
                orig_constant = std::get<std::shared_ptr<ov::op::v0::Constant>>(constant_ptr);
            } else {
                auto shared_buf = std::get<shared_mapped_memory_ptr>(constant_ptr);
                orig_constant =
                    std::make_shared<ov::op::v0::Constant>(original_dtype, shape, get_intermediate_data(), shared_buf);
            }

            ov::ParameterVector inputParams;
            ov::ResultVector results;
            ov::pass::Manager manager("Plugin:GPU:weightless_cache_transformations");
            std::shared_ptr<ov::Model> model = nullptr;

            auto convert_op = std::make_shared<ov::op::v0::Convert>(orig_constant, curr_dtype);
            results.push_back(std::make_shared<ov::op::v0::Result>(convert_op->output(0)));
            model = std::make_shared<ov::Model>(results, inputParams, "aux");
            manager.register_pass<ov::pass::ConstantFolding>();

            manager.run_passes(model);
            const auto& ops = model->get_ops();
            auto it = std::find_if(ops.begin(), ops.end(), [](const std::shared_ptr<ov::Node>& node) {
                return ov::op::util::is_constant(node);
            });
            OPENVINO_ASSERT(it != ops.end());
            transformed_constant = ov::as_type_ptr<ov::op::v0::Constant>(*it);
            OPENVINO_ASSERT(transformed_constant->get_element_type() == curr_dtype);
        }

        if (should_run_reorder()) {
            const auto allocation_type = dst_mem->get_allocation_type();
            memory::ptr input_mem = engine.allocate_memory(*reorder_rep.input_layout, allocation_type, false);

            if (is_alloc_host_accessible(allocation_type)) {
                std::memcpy(reinterpret_cast<uint8_t*>(input_mem->buffer_ptr()),
                            get_intermediate_data(),
                            get_current_data_size());
            } else {
                auto& strm = engine.get_service_stream();
                input_mem->copy_from(strm, get_intermediate_data());
            }

            reorder_rep.reorder->input = {input_info("input")};
            topology topology(input_layout("input", *reorder_rep.input_layout),
                              *reorder_rep.reorder);

            ExecutionConfig config;
            config.set_property(ov::intel_gpu::optimize_data(false));
            cldnn::network network(engine, topology, config, true);
            network.set_input_data("input", input_mem);
            network.set_output_memory(reorder_rep.reorder->id, dst_mem);
            auto outputs = network.execute();
            for (const auto& output : outputs) {
                auto ev = output.second.get_event();
                if (ev) {
                    ev->wait();
                }
            }

            OPENVINO_ASSERT(outputs.size() == 1);
        } else {
            copy_to_dst_mem(dst_mem, get_intermediate_data());
        }
    }
};

/// @brief Provides input data to topology.
/// @details This primitive allows to pass data which is known at topology creation.
/// For example, weights and biases for scoring networks.
/// @note Passing data at topology may improve network performance if data optimization is enabled.
struct data : public primitive_base<data> {
    CLDNN_DECLARE_PRIMITIVE(data)

    data() : primitive_base("", {}) {
        cache_info = std::make_shared<weightless_cache_manager>();
    }

    /// @brief Constructs data primitive.
    /// @param id This primitive id.
    /// @param mem @ref memory object which contains data.
    /// @note If memory is attached by memory::attach(), the attached buffer should be valid till network build.
    data(const primitive_id& id, memory::ptr mem) : primitive_base(id, {}), mem(std::move(mem)) {
        cache_info = std::make_shared<weightless_cache_manager>();
    }

    data(const primitive_id& id, memory::ptr mem, std::shared_ptr<weightless_cache_manager> cache_info)
        : primitive_base(id, {}),
          mem(std::move(mem)),
          cache_info(cache_info) {
        if (!cache_info) {
            this->cache_info = std::make_shared<weightless_cache_manager>();
        }
    }

    /// @brief @ref memory object which contains data.
    /// @note If memory is attached by memory::attach(), the attached buffer should be valid till network build.
    memory::ptr mem;

    std::shared_ptr<weightless_cache_manager> cache_info;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, id);
        return seed;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<data>::save(ob);

        ob << mem->get_layout();

        const auto _allocation_type = mem->get_allocation_type();
        ob << make_data(&_allocation_type, sizeof(_allocation_type));

        size_t data_size = mem->size();
        ob << make_data(&data_size, sizeof(size_t));

        bool do_weightless_caching = cache_info->save(ob, data_size);
        if (!do_weightless_caching) {
            if (is_alloc_host_accessible(_allocation_type)) {
                ob << make_data(mem->buffer_ptr(), data_size);
            } else {
                std::vector<uint8_t> _buf;
                _buf.resize(data_size);
                stream* strm = reinterpret_cast<stream*>(ob.get_stream());
                mem->copy_to(*strm, _buf.data());
                ob << make_data(_buf.data(), data_size);
            }
        }
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<data>::load(ib);
    }

    void load_weights(BinaryInputBuffer& ib, std::shared_ptr<WeightsMemory> weights_memory, const std::string& weights_path = "") {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "load_weights");
        layout output_layout = layout();
        ib >> output_layout;

        allocation_type _allocation_type = allocation_type::unknown;
        ib >> make_data(&_allocation_type, sizeof(_allocation_type));

        size_t data_size = 0;
        ib >> make_data(&data_size, sizeof(size_t));

        mem = ib.get_engine().allocate_memory(output_layout, _allocation_type, false);

        bool is_weightless_caching = cache_info->load(ib, mem, weights_memory);

        if (!is_weightless_caching) {
            const size_t DATA_BLOCK_SIZE = 4 * 1024 * 1024;
            const size_t DIRECT_IO_THRESHOLD = 4 * 1024 * 1024; // Use DirectIO for >4MB

            if (is_alloc_host_accessible(_allocation_type)) {
                bool used_direct_io = false;
                if (data_size >= DIRECT_IO_THRESHOLD) {
                    used_direct_io = load_direct(ib.get_stream(), mem->buffer_ptr(), data_size);
                    if (!used_direct_io && !weights_path.empty()) {
                        auto cur_offset = ib.get_stream().tellg();

                        // Auto-detect header offset compensation for path-based loading
                        // This applies to both Windows and Linux Parallel loaders which open by path
                        size_t offset_compensation = 0;

                        // Save current position
                        auto restore_pos = ib.get_stream().tellg();
                        ib.get_stream().seekg(0, std::ios::end);
                        auto stream_end = (size_t)ib.get_stream().tellg();
                        ib.get_stream().seekg(restore_pos, std::ios::beg);

                        size_t physical_size = get_file_size(weights_path);
                        if (physical_size > stream_end) {
                            offset_compensation = physical_size - stream_end;
                        }

                        used_direct_io = load_parallel(weights_path, (size_t)cur_offset + offset_compensation, mem->buffer_ptr(), data_size);
                        if (used_direct_io) {
                            ib.get_stream().seekg(data_size, std::ios::cur);
                        }
                    }
                }

                if (!used_direct_io) {
                    if (data_size < DATA_BLOCK_SIZE) {
                        ib >> make_data(mem->buffer_ptr(), data_size);
                    } else {
                        struct AlignedBuffer {
                            uint8_t* ptr;
                            AlignedBuffer(size_t sz) {
#ifdef _WIN32
                                ptr = static_cast<uint8_t*>(_aligned_malloc(sz, 4096));
#else
                                if (posix_memalign((void**)&ptr, 4096, sz)) ptr = nullptr;
#endif
                            }
                            ~AlignedBuffer() {
#ifdef _WIN32
                                _aligned_free(ptr);
#else
                                free(ptr);
#endif
                            }
                            uint8_t* get() { return ptr; }
                        };

                        AlignedBuffer _buf1(DATA_BLOCK_SIZE);
                        AlignedBuffer _buf2(DATA_BLOCK_SIZE);

                        std::future<void> fut;
                        bool buf_flag = true;
                        size_t dst_offset = 0;
                        uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(mem->buffer_ptr());

                        while (dst_offset < data_size) {
                            size_t copy_size = std::min(DATA_BLOCK_SIZE, data_size - dst_offset);
                            if (buf_flag) {
                                ib >> make_data(_buf1.get(), copy_size);
                                if (fut.valid()) fut.get();
                                fut = std::async(std::launch::async, [dst_ptr, dst_offset, &_buf1, copy_size] {
                                    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "load_weights::cpy_gpu_0");
                                    std::memcpy(dst_ptr + dst_offset, _buf1.get(), copy_size);
                                });
                            } else {
                                ib >> make_data(_buf2.get(), copy_size);
                                if (fut.valid()) fut.get();
                                fut = std::async(std::launch::async, [dst_ptr, dst_offset, &_buf2, copy_size] {
                                    std::memcpy(dst_ptr + dst_offset, _buf2.get(), copy_size);
                                    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "load_weights::cpy_gpu_1");
                                });
                            }
                            dst_offset += copy_size;
                            buf_flag = !buf_flag;
                        }
                        if (fut.valid()) fut.get();
                    }
                }
            } else {
                auto& strm = ib.get_engine().get_service_stream();
                if (data_size < DATA_BLOCK_SIZE || output_layout.format.is_image_2d()) {
                    std::vector<uint8_t> _buf(data_size);
                    ib >> make_data(_buf.data(), data_size);
                    mem->copy_from(strm, _buf.data());
                } else {
                    std::vector<uint8_t> _buf1(DATA_BLOCK_SIZE);
                    std::vector<uint8_t> _buf2(DATA_BLOCK_SIZE);
                    bool buf_flag = true;
                    event::ptr ev1, ev2;
                    ev1 = ev2 = nullptr;
                    size_t dst_offset = 0;
                    while (dst_offset < data_size) {
                        const bool is_blocking = false;
                        const size_t src_offset = 0;
                        size_t copy_size =
                            (data_size > (dst_offset + DATA_BLOCK_SIZE)) ? DATA_BLOCK_SIZE : (data_size - dst_offset);
                        if (buf_flag) {
                            ib >> make_data(_buf1.data(), copy_size);
                            if (ev2 != nullptr) {
                                ev2->wait();
                                ev2 = nullptr;
                            }
                            ev1 = mem->copy_from(strm, _buf1.data(), src_offset, dst_offset, copy_size, is_blocking);
                        } else {
                            ib >> make_data(_buf2.data(), copy_size);
                            if (ev1 != nullptr) {
                                ev1->wait();
                                ev1 = nullptr;
                            }
                            ev2 = mem->copy_from(strm, _buf2.data(), src_offset, dst_offset, copy_size, is_blocking);
                        }
                        dst_offset += DATA_BLOCK_SIZE;
                        buf_flag = !buf_flag;
                    }
                    if (ev2 != nullptr) {
                        ev2->wait();
                    }
                    if (ev1 != nullptr) {
                        ev1->wait();
                    }
                }
            }
        }
    }
};
}  // namespace cldnn
