// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "sycl_memory.hpp"
#include "sycl_engine.hpp"
#include "sycl_stream.hpp"
#include "sycl_event.hpp"
#include <stdexcept>
#include <vector>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_sycl.hpp>
#endif

#define TRY_CATCH_SYCL_ERROR(...)               \
    try {                                     \
        __VA_ARGS__;                          \
    } catch (::sycl::exception const& err) {          \
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err)); \
    }

namespace cldnn {
namespace sycl {

static inline cldnn::event::ptr create_event(sycl_stream& stream, ::sycl::event& ev) {
    return stream.create_base_event(ev);
}

gpu_buffer::gpu_buffer(sycl_engine* engine,
                       const layout& layout)
    : lockable_gpu_mem(), memory(engine, layout, allocation_type::sycl_buffer, nullptr)
    , _byte_offset(0), _root_buffer(::sycl::range<1>(size())), _buffer(_root_buffer) {
    GPU_DEBUG_TRACE_DETAIL << "gpu_buffer: " << &_buffer << ", size: " << size() << std::endl;

    // TODO: remove this if possible
    // Make write accessor to ensure the buffer is created on the device
    try {
        auto q = ::sycl::queue(engine->get_sycl_context(), engine->get_sycl_device());

        q.submit([&](::sycl::handler& cgh) {
            _buffer.get_access<::sycl::access::mode::write>(cgh);
        });
        q.wait_and_throw();
    } catch (::sycl::exception const& err) {
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
    }

    if (engine->get_sycl_context().get_backend() == ::sycl::backend::opencl) {
        std::vector<cl_mem> cl_bufs = ::sycl::get_native<::sycl::backend::opencl>(_buffer);
        OPENVINO_ASSERT(cl_bufs.size() == 1, "Expected single OpenCL buffer handle from SYCL buffer");
    }

    m_mem_tracker = std::make_shared<MemoryTracker>(engine, static_cast<void*>(&_buffer),
                                                    layout.bytes_count(), allocation_type::sycl_buffer);
}

gpu_buffer::gpu_buffer(sycl_engine* engine,
                       const layout& new_layout,
                       const ::sycl::buffer<std::byte, 1>& root_buffer,
                       std::shared_ptr<MemoryTracker> mem_tracker)
    : lockable_gpu_mem(), memory(engine, new_layout, allocation_type::sycl_buffer, mem_tracker)
    , _byte_offset(0), _root_buffer(root_buffer), _buffer(root_buffer) {
        GPU_DEBUG_TRACE_DETAIL << "create gpu_buffer(buffer: " << &_buffer << ", size: " << size()
                               << ", offset: " << _byte_offset << ", root_buffer: " << &_root_buffer << std::endl;
}

gpu_buffer::gpu_buffer(sycl_engine* engine,
                       const layout& new_layout,
                       const size_t byte_offset,
                       const ::sycl::buffer<std::byte, 1>& root_buffer,
                       std::shared_ptr<MemoryTracker> mem_tracker)
    : lockable_gpu_mem(), memory(engine, new_layout, allocation_type::sycl_buffer, mem_tracker)
    , _byte_offset(byte_offset), _root_buffer(root_buffer)
    , _buffer(const_cast<::sycl::buffer<std::byte, 1>&>(root_buffer), ::sycl::id<1>(byte_offset), ::sycl::range<1>(new_layout.bytes_count())) {
        GPU_DEBUG_TRACE_DETAIL << "create gpu_buffer(buffer: " << &_buffer << ", size: " << size()
                               << ", offset: " << _byte_offset << ", root_buffer: " << &_root_buffer << std::endl;
}

void* gpu_buffer::lock(const stream& stream, mem_lock_type type) {
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        try {
            // TODO: take account of mem_lock_type
            _host_accessor = std::make_unique<::sycl::host_accessor<std::byte, 1, ::sycl::access::mode::read_write>>(_buffer, ::sycl::read_write);
            _mapped_ptr = _host_accessor->get_pointer();
        } catch (::sycl::exception const& err) {
            OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
        }
    }
    _lock_count++;
    return _mapped_ptr;
}

void gpu_buffer::unlock(const stream& stream) {
    std::lock_guard<std::mutex> locker(_mutex);
    _lock_count--;
    if (0 == _lock_count) {
        try {
            _host_accessor = nullptr;  // Release the host accessor
        } catch (::sycl::exception const& err) {
            OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
        }
        _mapped_ptr = nullptr;
    }
}

event::ptr gpu_buffer::fill(stream& stream, bool blocking) {
    return fill(stream, 0, blocking);
}

event::ptr gpu_buffer::fill(stream& stream, unsigned char pattern, bool blocking) {
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip EnqueueMemcpy for 0 size tensor" << std::endl;
        return nullptr;
    }
    auto& sycl_stream = downcast<sycl::sycl_stream>(stream);
    try {
        auto ev = sycl_stream.get_sycl_queue().fill(_buffer.get_access(::sycl::write_only), static_cast<std::byte>(pattern));

        if (blocking) {
            ev.wait_and_throw();
            return nullptr;
        } else {
            return create_event(sycl_stream, ev);
        }
    } catch (::sycl::exception const& err) {
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
    }
}

shared_mem_params gpu_buffer::get_internal_params() const {
    auto sycl_engine = downcast<const sycl::sycl_engine>(_engine);
    return {shared_mem_type::shared_mem_buffer, const_cast<shared_handle>(static_cast<const void*>(&(sycl_engine->get_sycl_context()))), nullptr,
            const_cast<shared_handle>(static_cast<const void*>(&_buffer)),
#ifdef _WIN32
        nullptr,
#else
        0,
#endif
        0};
}

event::ptr gpu_buffer::copy_from(stream& stream, const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) {
    if (size == 0)
        return nullptr;

    auto& sycl_stream = downcast<sycl::sycl_stream>(stream);
    auto src_ptr = reinterpret_cast<const char*>(data_ptr) + src_offset;

    try {
        auto event = sycl_stream.get_sycl_queue().submit([&](::sycl::handler& cgh) {
            ::sycl::accessor<std::byte, 1, ::sycl::access::mode::write> dst_acc(_buffer, cgh, ::sycl::range<1>(size), ::sycl::id<1>(dst_offset));
            cgh.copy(src_ptr, dst_acc);
        });
        if (blocking) {
            event.wait_and_throw();
            return nullptr;
        } else {
            return create_event(sycl_stream, event);
        }
    } catch (::sycl::exception const& err) {
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
    }
}

event::ptr gpu_buffer::copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) {
    if (size == 0)
        return nullptr;

    switch (src_mem.get_allocation_type()) {
        case allocation_type::usm_host:
        case allocation_type::usm_shared:
        case allocation_type::usm_device: {
            // If other is gpu_usm, down cast to gpu_buffer is not possible.
            // But it can read as host ptr if it's allocation type is either usm_host or usm_shared.
            OPENVINO_NOT_IMPLEMENTED;
            // TODO: implement
            //auto usm_mem = downcast<const gpu_usm>(&src_mem);
            //return copy_from(stream, usm_mem->buffer_ptr(), src_offset, dst_offset, size, blocking);
        }
        case allocation_type::sycl_buffer: {
            OPENVINO_ASSERT(!src_mem.get_layout().format.is_image_2d());

            auto& sycl_stream = downcast<sycl::sycl_stream>(stream);
            // const qualifier should be removed to construct ::sycl::accessor
            auto& src_buffer = const_cast<::sycl::buffer<std::byte, 1>&>(downcast<const gpu_buffer>(src_mem).get_buffer());

            try {
                auto event = sycl_stream.get_sycl_queue().submit([&](::sycl::handler& cgh) {
                    ::sycl::accessor<std::byte, 1, ::sycl::access::mode::read> src_acc(src_buffer, cgh, ::sycl::range<1>(size), ::sycl::id<1>(src_offset));
                    ::sycl::accessor<std::byte, 1, ::sycl::access::mode::write> dst_acc(_buffer, cgh, ::sycl::range<1>(size), ::sycl::id<1>(dst_offset));
                    cgh.copy(src_acc, dst_acc);
                });
                if (blocking) {
                    event.wait_and_throw();
                    return nullptr;
                } else {
                    return create_event(sycl_stream, event);
                }
            } catch (::sycl::exception const& err) {
                OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
            }
        }
        case allocation_type::cl_mem: {
            OPENVINO_THROW("[GPU] SYCL engine does not support allocation_type::cl_mem");
        }
        default:
            OPENVINO_THROW("[GPU] Unsupported buffer type for gpu_buffer::copy_from() function");
    }
}

event::ptr gpu_buffer::copy_to(stream& stream, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const {
    if (size == 0)
        return nullptr;

    auto& sycl_stream = downcast<sycl::sycl_stream>(stream);
    // const qualifier should be removed to construct ::sycl::accessor
    auto& src_buffer = const_cast<::sycl::buffer<std::byte, 1>&>(_buffer);
    auto dst_ptr = reinterpret_cast<char*>(data_ptr) + dst_offset;

    try {
        auto event = sycl_stream.get_sycl_queue().submit([&](::sycl::handler& cgh) {
            ::sycl::accessor<std::byte, 1, ::sycl::access::mode::read> src_acc(src_buffer, cgh, ::sycl::range<1>(size), ::sycl::id<1>(src_offset));
            cgh.copy(src_acc, dst_ptr);
        });

        if (blocking) {
            event.wait_and_throw();
            return nullptr;
        } else {
            return create_event(sycl_stream, event);
        }
    } catch (::sycl::exception const& err) {
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
    }
}

std::shared_ptr<gpu_buffer> gpu_buffer::create_subbuffer(const layout& new_layout, size_t byte_offset) const {
    auto new_layout_bytes_count = new_layout.bytes_count();
    OPENVINO_ASSERT(new_layout_bytes_count + byte_offset <= size(),
                    "Sub-buffer size (", new_layout_bytes_count, ") + offset (", byte_offset,
                    ") exceeds parent buffer size (", size(), ")");
    auto sycl_engine = downcast<sycl::sycl_engine>(_engine);
    auto mem_tracker = get_mem_tracker();

    // if new buffer is the same as root buffer, just copy root buffer
    if (new_layout_bytes_count == _root_buffer.byte_size() && _byte_offset + byte_offset == 0) {
        return std::make_shared<gpu_buffer>(sycl_engine, new_layout, _root_buffer, mem_tracker);
    }

    return std::make_shared<gpu_buffer>(sycl_engine, new_layout, _byte_offset + byte_offset, _root_buffer, mem_tracker);
}

std::shared_ptr<gpu_buffer> gpu_buffer::reinterpret(const layout& new_layout) const {
    // The reinterpreted GPU buffer can use a larger buffer size than the current one, as long as it does not exceed the root buffer size.
    // If the current buffer has offset, the reinterpreted buffer also keeps the offset.
    // |<-------------root buffer----------->|
    // |<-offset->|<--current buffer-->|
    // |<-offset->|<--reinterpreted buffer-->|
    auto new_layout_bytes_count = new_layout.bytes_count();
    OPENVINO_ASSERT(new_layout_bytes_count + _byte_offset <= _root_buffer.byte_size(),
                    "Reinterpret buffer size (", new_layout_bytes_count, ") + offset (", _byte_offset,
                    ") exceeds parent buffer size (", _root_buffer.byte_size(), ")");
    auto sycl_engine = downcast<sycl::sycl_engine>(_engine);
    auto mem_tracker = get_mem_tracker();

    // if new buffer is the same as root buffer, just copy root buffer
    if (new_layout_bytes_count == _root_buffer.byte_size() && _byte_offset == 0) {
        return std::make_shared<gpu_buffer>(sycl_engine, new_layout, _root_buffer, mem_tracker);
    }

    return std::make_shared<gpu_buffer>(sycl_engine, new_layout, _byte_offset, _root_buffer, mem_tracker);
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::memory gpu_buffer::get_onednn_memory(dnnl::memory::desc desc, int64_t offset) const {
    OPENVINO_ASSERT(offset + _byte_offset == 0, "get_onednn_memory with offset is not supported for sycl_buffer");

    auto onednn_engine = _engine->get_onednn_engine();
    dnnl::memory dnnl_mem(desc, onednn_engine, DNNL_MEMORY_NONE);
    dnnl::sycl_interop::set_buffer(dnnl_mem, const_cast<::sycl::buffer<std::byte, 1>&>(_root_buffer));
    return dnnl_mem;
}
#endif

// gpu_image2d::gpu_image2d(sycl_engine* engine, const layout& layout)
//     : lockable_gpu_mem()
//     , memory(engine, layout, allocation_type::sycl_buffer, nullptr)
//     , _width(0)
//     , _height(0)
//     , _row_pitch(0)
//     , _slice_pitch(0) {
//     ::sycl::image_channel_type type = layout.data_type == data_types::f16 ? ::sycl::image_channel_type::fp16 : ::sycl::image_channel_type::fp32;
//     ::sycl::image_channel_order order = ::sycl::image_channel_order::r;
//     switch (layout.format) {
//         case format::image_2d_weights_c1_b_fyx:
//             _width = layout.batch();
//             _height = layout.spatial(0) * layout.feature() * layout.spatial(1);
//             break;
//         case format::image_2d_weights_winograd_6x3_s1_fbxyb:
//             _height = layout.feature();
//             _width = layout.spatial(0) * layout.batch() * layout.spatial(1) * 8 / 3;
//             break;
//         case format::image_2d_weights_winograd_6x3_s1_xfbyb:
//             _height = layout.feature() * layout.spatial(0) * 8 / 3;
//             _width = layout.batch() * layout.spatial(1);
//             break;
//         case format::image_2d_weights_c4_fyx_b:
//             _width = layout.batch();
//             _height = layout.spatial(0) * layout.feature() * layout.spatial(1);
//             order = ::sycl::image_channel_order::rgba;
//             break;
//         case format::image_2d_rgba:
//             _width = layout.spatial(0);
//             _height = layout.spatial(1);
//             order = ::sycl::image_channel_order::rgba;
//             if (layout.feature() != 3 && layout.feature() != 4) {
//                 CLDNN_ERROR_MESSAGE("2D image allocation", "invalid number of channels in image_2d_rgba input image (should be 3 or 4)!");
//             }
//             type = ::sycl::image_channel_type::unorm_int8;
//             break;
//         case format::nv12:
//         {
//             // [NHWC] dimensions order
//             auto shape = layout.get_shape();
//             _width = shape[2];
//             _height = shape[1];
//             if (shape[3] == 2) {
//                 order = ::sycl::image_channel_order::rg;
//             } else if (shape[3] > 2) {
//                 CLDNN_ERROR_MESSAGE("2D image allocation", "invalid number of channels in NV12 input image!");
//             }
//             type = ::sycl::image_channel_type::unorm_int8;
//             break;
//         }
//         default:
//             CLDNN_ERROR_MESSAGE("2D image allocation", "unsupported image type!");
//     }
//
//     cl::ImageFormat imageFormat(order, type);
//     _buffer = ::sycl::image<2>(order, type, ::sycl::range<2>(_width, _height));
//     _bytes_count = _buffer.byte_size();
//     m_mem_tracker = std::make_shared<MemoryTracker>(engine, _buffer.get(), layout.bytes_count(), allocation_type::sycl_buffer);
// }
//
// gpu_image2d::gpu_image2d(sycl_engine* engine,
//                          const layout& new_layout,
//                          const ::sycl::image<2>& buffer,
//                          std::shared_ptr<MemoryTracker> mem_tracker)
//     : lockable_gpu_mem(), memory(engine, new_layout, allocation_type::sycl_buffer, mem_tracker),
//       _buffer(buffer) {
//     auto& range = _buffer.get_range();
//     auto& pitch = _buffer.get_pitch();
//     _width = range[0];
//     _height = range[1];
//     _row_pitch = pitch[0];
//     _slice_pitch = pitch[1];
// }
//
// event::ptr gpu_image2d::fill(stream& stream, bool blocking) {
//     return fill(stream, 0, blocking);
// }
//
// event::ptr gpu_image2d::fill(stream& stream, unsigned char pattern, bool blocking) {
//     if (_bytes_count == 0) {
//         GPU_DEBUG_TRACE_DETAIL << "Skip EnqueueMemcpy for 0 size tensor" << std::endl;
//         return nullptr;
//     }
//     auto& sycl_stream = downcast<sycl_stream>(stream);
//     auto ev = stream.create_base_event();
//     cl::Event& ev_ocl = downcast<ocl_event>(ev.get())->get();
//     cl_uint4 pattern_uint4 = {{pattern, pattern, pattern, pattern}};
//     try {
//         cl_stream.get_cl_queue().enqueueFillImage(_buffer, pattern_uint4, {0, 0, 0}, {_width, _height, 1}, 0, &ev_ocl);
//         if (blocking) {
//             ev_ocl.wait();
//         }
//     } catch (cl::Error const& err) {
//         OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
//     }
//     // TODO: do we need sync here?
//     cl_stream.finish();
//
//     return ev;
// }
//
// void* gpu_image2d::lock(const stream& stream, mem_lock_type type) {
//     auto& cl_stream = downcast<const ocl_stream>(stream);
//     std::lock_guard<std::mutex> locker(_mutex);
//     if (0 == _lock_count) {
//         try {
//             _mapped_ptr = cl_stream.get_cl_queue()
//                     .enqueueMapImage(_buffer,
//                                     CL_TRUE,
//                                     get_cl_map_type(type),
//                                     {0, 0, 0},
//                                     {_width, _height, 1},
//                                     &_row_pitch,
//                                     &_slice_pitch);
//         } catch (cl::Error const& err) {
//             OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
//         }
//     }
//     _lock_count++;
//     return _mapped_ptr;
// }
//
// void gpu_image2d::unlock(const stream& stream) {
//     auto& cl_stream = downcast<const ocl_stream>(stream);
//     std::lock_guard<std::mutex> locker(_mutex);
//     _lock_count--;
//     if (0 == _lock_count) {
//         try {
//             cl_stream.get_cl_queue().enqueueUnmapMemObject(_buffer, _mapped_ptr);
//         } catch (cl::Error const& err) {
//             OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
//         }
//         _mapped_ptr = nullptr;
//     }
// }
//
//
// shared_mem_params gpu_image2d::get_internal_params() const {
//     auto cl_engine = downcast<const ocl_engine>(_engine);
//     return {shared_mem_type::shared_mem_image, static_cast<shared_handle>(cl_engine->get_cl_context().get()), nullptr,
//             static_cast<shared_handle>(_buffer.get()),
// #ifdef _WIN32
//         nullptr,
// #else
//         0,
// #endif
//         0};
// }
//
// event::ptr gpu_image2d::copy_from(stream& stream, const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) {
//     auto result_event = create_event(stream, size, blocking);
//     if (size == 0)
//         return result_event;
//
//     OPENVINO_ASSERT(dst_offset == 0, "[GPU] Unsupported dst_offset value for gpu_image2d::copy_from() function");
//     OPENVINO_ASSERT(size == _bytes_count, "[GPU] Unsupported data_size value for gpu_image2d::copy_from() function");
//
//     auto cl_stream = downcast<ocl_stream>(&stream);
//     auto cl_event = blocking ? nullptr : &downcast<ocl_event>(result_event.get())->get();
//     auto src_ptr = reinterpret_cast<const char*>(data_ptr) + src_offset;
//
//     TRY_CATCH_CL_ERROR(
//         cl_stream->get_cl_queue().enqueueWriteImage(_buffer, blocking, {0, 0, 0}, {_width, _height, 1}, _row_pitch, _slice_pitch,
//                                                     src_ptr, nullptr, cl_event));
//
//     return result_event;
// }
//
// event::ptr gpu_image2d::copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) {
//     auto result_event = create_event(stream, size, false);
//     if (size == 0)
//         return result_event;
//
//     OPENVINO_ASSERT(src_mem.get_layout().format.is_image_2d(), "Unsupported buffer type for gpu_image2d::copy_from() function");
//     OPENVINO_ASSERT(src_offset == 0, "[GPU] Unsupported dst_offset value for gpu_image2d::copy_from() function");
//     OPENVINO_ASSERT(dst_offset == 0, "[GPU] Unsupported dst_offset value for gpu_image2d::copy_from() function");
//     OPENVINO_ASSERT(size == _bytes_count, "[GPU] Unsupported data_size value for gpu_image2d::copy_from() function");
//
//     auto cl_stream = downcast<ocl_stream>(&stream);
//     auto cl_event = &downcast<ocl_event>(result_event.get())->get();
//     auto cl_image_mem = downcast<const gpu_image2d>(&src_mem);
//
//     TRY_CATCH_CL_ERROR(
//         cl_stream->get_cl_queue().enqueueCopyImage(cl_image_mem->get_buffer(), get_buffer(), {0, 0, 0}, {0, 0, 0}, {_width, _height, 1}, nullptr, cl_event));
//
//     if (blocking)
//         cl_event->wait();
//
//     return result_event;
// }
//
// event::ptr gpu_image2d::copy_to(stream& stream, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const {
//     auto result_event = create_event(stream, size, blocking);
//     if (size == 0)
//         return result_event;
//
//     OPENVINO_ASSERT(src_offset == 0, "[GPU] Unsupported src_offset value for gpu_image2d::copy_from() function");
//     OPENVINO_ASSERT(size == _bytes_count, "[GPU] Unsupported data_size value for gpu_image2d::copy_from() function");
//
//     auto cl_stream = downcast<ocl_stream>(&stream);
//     auto cl_event = blocking ? nullptr : &downcast<ocl_event>(result_event.get())->get();
//     auto dst_ptr = reinterpret_cast<char*>(data_ptr) + dst_offset;
//
//     TRY_CATCH_CL_ERROR(
//         cl_stream->get_cl_queue().enqueueReadImage(_buffer, blocking, {0, 0, 0}, {_width, _height, 1}, _row_pitch, _slice_pitch,
//                                                     dst_ptr, nullptr, cl_event));
//
//     return result_event;
// }
//
// gpu_media_buffer::gpu_media_buffer(ocl_engine* engine,
//                                    const layout& new_layout,
//                                    shared_mem_params params)
//     : gpu_image2d(engine, new_layout, cl::ImageVA(engine->get_cl_context(), CL_MEM_READ_WRITE, params.surface, params.plane), nullptr),
//     device(params.user_device),
//     surface(params.surface),
//     plane(params.plane) { }
//
// shared_mem_params gpu_media_buffer::get_internal_params() const {
//     auto cl_engine = downcast<const ocl_engine>(_engine);
//     return {shared_mem_type::shared_mem_vasurface, static_cast<shared_handle>(cl_engine->get_cl_context().get()), device,
//             static_cast<shared_handle>(_buffer.get()), surface, plane };
// }
//
// #ifdef _WIN32
// gpu_dx_buffer::gpu_dx_buffer(ocl_engine* engine,
//                              const layout& new_layout,
//                              shared_mem_params params)
//     : gpu_buffer(engine, new_layout,
//                 cl::BufferDX(engine->get_cl_context(), CL_MEM_READ_WRITE, params.mem), nullptr),
//     device(params.user_device),
//     resource(params.mem) { }
//
// shared_mem_params gpu_dx_buffer::get_internal_params() const {
//     auto cl_engine = downcast<const ocl_engine>(_engine);
//     return {shared_mem_type::shared_mem_dxbuffer, static_cast<shared_handle>(cl_engine->get_cl_context().get()), device,
//             static_cast<shared_handle>(_buffer.get()), resource, 0 };
// }
// #endif
//
// gpu_usm::gpu_usm(ocl_engine* engine, const layout& new_layout, const cl::UsmMemory& buffer, allocation_type type, std::shared_ptr<MemoryTracker> mem_tracker)
//     : lockable_gpu_mem()
//     , memory(engine, new_layout, type, mem_tracker)
//     , _buffer(buffer)
//     , _host_buffer(engine->get_usm_helper()) {
// }
//
// gpu_usm::gpu_usm(ocl_engine* engine, const layout& new_layout, const cl::UsmMemory& buffer, std::shared_ptr<MemoryTracker> mem_tracker)
//     : lockable_gpu_mem()
//     , memory(engine, new_layout, detect_allocation_type(engine, buffer), mem_tracker)
//     , _buffer(buffer)
//     , _host_buffer(engine->get_usm_helper()) {
// }
//
// gpu_usm::gpu_usm(ocl_engine* engine, const layout& layout, allocation_type type)
//     : lockable_gpu_mem()
//     , memory(engine, layout, type, nullptr)
//     , _buffer(engine->get_usm_helper())
//     , _host_buffer(engine->get_usm_helper()) {
//     auto actual_bytes_count = _bytes_count;
//     if (actual_bytes_count == 0)
//         actual_bytes_count = 1;
//     switch (get_allocation_type()) {
//     case allocation_type::usm_host:
//         _buffer.allocateHost(actual_bytes_count);
//         break;
//     case allocation_type::usm_shared:
//         _buffer.allocateShared(actual_bytes_count);
//         break;
//     case allocation_type::usm_device:
//         _buffer.allocateDevice(actual_bytes_count);
//         break;
//     default:
//         CLDNN_ERROR_MESSAGE("gpu_usm allocation type",
//             "Unknown unified shared memory type!");
//     }
//
//     m_mem_tracker = std::make_shared<MemoryTracker>(engine, _buffer.get(), actual_bytes_count, type);
// }
//
// void* gpu_usm::lock(const stream& stream, mem_lock_type type) {
//     std::lock_guard<std::mutex> locker(_mutex);
//     if (0 == _lock_count) {
//         auto& cl_stream = downcast<const ocl_stream>(stream);
//         if (get_allocation_type() == allocation_type::usm_device) {
//             if (type != mem_lock_type::read) {
//                 throw std::runtime_error("Unable to lock allocation_type::usm_device with write lock_type.");
//             }
//             GPU_DEBUG_LOG << "Copy usm_device buffer to host buffer." << std::endl;
//             _host_buffer.allocateHost(_bytes_count);
//             try {
//                 cl_stream.get_usm_helper().enqueue_memcpy(cl_stream.get_cl_queue(), _host_buffer.get(), _buffer.get(), _bytes_count, CL_TRUE);
//             } catch (cl::Error const& err) {
//                 OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
//             }
//             _mapped_ptr = _host_buffer.get();
//         } else {
//             _mapped_ptr = _buffer.get();
//         }
//     }
//     _lock_count++;
//     return _mapped_ptr;
// }
//
// void gpu_usm::unlock(const stream& /* stream */) {
//     std::lock_guard<std::mutex> locker(_mutex);
//     _lock_count--;
//     if (0 == _lock_count) {
//         if (get_allocation_type() == allocation_type::usm_device) {
//             _host_buffer.freeMem();
//         }
//         _mapped_ptr = nullptr;
//     }
// }
//
// event::ptr gpu_usm::fill(stream& stream, unsigned char pattern, bool blocking) {
//     if (_bytes_count == 0) {
//         GPU_DEBUG_TRACE_DETAIL << "Skip gpu_usm::fill for 0 size tensor" << std::endl;
//         return nullptr;
//     }
//     auto& cl_stream = downcast<ocl_stream>(stream);
//     auto ev = stream.create_base_event();
//     cl::Event& ev_ocl = downcast<ocl_event>(ev.get())->get();
//     try {
//         cl_stream.get_usm_helper().enqueue_fill_mem(
//                 cl_stream.get_cl_queue(), _buffer.get(), static_cast<const void*>(&pattern), sizeof(unsigned char), _bytes_count, nullptr, &ev_ocl);
//         if (blocking) {
//             ev_ocl.wait();
//         }
//     } catch (cl::Error const& err) {
//         OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
//     }
//
//     return ev;
// }
//
// event::ptr gpu_usm::fill(stream& stream, bool blocking) {
//     return fill(stream, 0, blocking);
// }
//
// event::ptr gpu_usm::copy_from(stream& stream, const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) {
//     auto result_event = create_event(stream, size, blocking);
//     if (size == 0)
//         return result_event;
//
//     auto cl_stream = downcast<ocl_stream>(&stream);
//     auto cl_event = blocking ? nullptr : &downcast<ocl_event>(result_event.get())->get();
//     auto src_ptr = reinterpret_cast<const char*>(data_ptr) + src_offset;
//     auto dst_ptr = reinterpret_cast<char*>(buffer_ptr()) + dst_offset;
//
//     TRY_CATCH_CL_ERROR(cl_stream->get_usm_helper().enqueue_memcpy(cl_stream->get_cl_queue(), dst_ptr, src_ptr, size, blocking, nullptr, cl_event));
//
//     return result_event;
// }
//
// event::ptr gpu_usm::copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) {
//     auto result_event = create_event(stream, size, blocking);
//     if (size == 0)
//         return result_event;
//
//     auto cl_stream = downcast<ocl_stream>(&stream);
//     auto cl_event = blocking ? nullptr : &downcast<ocl_event>(result_event.get())->get();
//
//     if (src_mem.get_allocation_type() == allocation_type::cl_mem) {
//         auto cl_mem_buffer = downcast<const gpu_buffer>(&src_mem);
//         auto dst_ptr = reinterpret_cast<char*>(buffer_ptr());
//
//         return cl_mem_buffer->copy_to(stream, dst_ptr, src_offset, dst_offset, size, blocking);
//     } else if (memory_capabilities::is_usm_type(src_mem.get_allocation_type())) {
//         auto usm_mem = downcast<const gpu_usm>(&src_mem);
//         auto src_ptr = reinterpret_cast<const char*>(usm_mem->buffer_ptr()) + src_offset;
//         auto dst_ptr = reinterpret_cast<char*>(buffer_ptr()) + dst_offset;
//
//         TRY_CATCH_CL_ERROR(cl_stream->get_usm_helper().enqueue_memcpy(cl_stream->get_cl_queue(), dst_ptr, src_ptr, size, blocking, nullptr, cl_event));
//     } else {
//         std::vector<char> tmp_buf;
//         tmp_buf.resize(size);
//         src_mem.copy_to(stream, tmp_buf.data(), src_offset, 0, size, true);
//
//         GPU_DEBUG_TRACE_DETAIL << "Suboptimal copy call from " << src_mem.get_allocation_type() << " to " << get_allocation_type() << "\n";
//         return copy_from(stream, tmp_buf.data(), 0, 0, size, blocking);
//     }
//
//     return result_event;
// }
//
// event::ptr gpu_usm::copy_to(stream& stream, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const {
//     auto result_event = create_event(stream, size, blocking);
//     if (size == 0)
//         return result_event;
//
//     auto cl_stream = downcast<ocl_stream>(&stream);
//     auto cl_event = blocking ? nullptr : &downcast<ocl_event>(result_event.get())->get();
//     auto src_ptr = reinterpret_cast<const char*>(buffer_ptr()) + src_offset;
//     auto dst_ptr = reinterpret_cast<char*>(data_ptr) + dst_offset;
//
//     TRY_CATCH_CL_ERROR(cl_stream->get_usm_helper().enqueue_memcpy(cl_stream->get_cl_queue(), dst_ptr, src_ptr, size, blocking, nullptr, cl_event));
//
//     return result_event;
// }
//
// #ifdef ENABLE_ONEDNN_FOR_GPU
// dnnl::memory gpu_usm::get_onednn_memory(dnnl::memory::desc desc, int64_t offset) const {
//     auto onednn_engine = _engine->get_onednn_engine();
//     dnnl::memory dnnl_mem = dnnl::ocl_interop::make_memory(desc, onednn_engine, dnnl::ocl_interop::memory_kind::usm,
//         reinterpret_cast<uint8_t*>(_buffer.get()) + offset);
//     return dnnl_mem;
// }
// #endif
//
// shared_mem_params gpu_usm::get_internal_params() const {
//     auto cl_engine = downcast<const ocl_engine>(_engine);
//     return {
//         shared_mem_type::shared_mem_usm,  // shared_mem_type
//         static_cast<shared_handle>(cl_engine->get_cl_context().get()),  // context handle
//         nullptr,        // user_device handle
//         _buffer.get(),  // mem handle
// #ifdef _WIN32
//         nullptr,  // surface handle
// #else
//         0,  // surface handle
// #endif
//         0  // plane
//     };
// }

// allocation_type gpu_usm::detect_allocation_type(const ocl_engine* engine, const void* mem_ptr) {
//     auto cl_alloc_type = engine->get_usm_helper().get_usm_allocation_type(mem_ptr);
//
//     allocation_type res;
//     switch (cl_alloc_type) {
//         case CL_MEM_TYPE_DEVICE_INTEL: res = allocation_type::usm_device; break;
//         case CL_MEM_TYPE_HOST_INTEL: res = allocation_type::usm_host; break;
//         case CL_MEM_TYPE_SHARED_INTEL: res = allocation_type::usm_shared; break;
//         default: res = allocation_type::unknown;
//     }
//
//     return res;
// }
//
// allocation_type gpu_usm::detect_allocation_type(const ocl_engine* engine, const cl::UsmMemory& buffer) {
//     auto alloc_type = detect_allocation_type(engine, buffer.get());
//     OPENVINO_ASSERT(alloc_type == allocation_type::usm_device ||
//                     alloc_type == allocation_type::usm_host ||
//                     alloc_type == allocation_type::usm_shared, "[GPU] Unsupported USM alloc type: " + to_string(alloc_type));
//     return alloc_type;
// }
//
std::vector<::sycl::buffer<std::byte, 1>> sycl_surfaces_lock::get_handles(std::vector<memory::ptr> mem) const {
    std::vector<::sycl::buffer<std::byte, 1>> res;

    // Do nothing because we don't support sycl surfaces lock
    return res;
}

sycl_surfaces_lock::sycl_surfaces_lock(std::vector<memory::ptr> mem, const stream& stream)
    : surfaces_lock() {
    // , _handles(get_handles(mem))
    // , _lock(nullptr) {
    OPENVINO_ASSERT(mem.empty(), "[GPU] SYCL surfaces lock is not supported");
}
}  // namespace sycl
}  // namespace cldnn
