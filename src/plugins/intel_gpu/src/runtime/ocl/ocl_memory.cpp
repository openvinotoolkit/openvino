// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "ocl_memory.hpp"
#include "ocl_engine.hpp"
#include "ocl_stream.hpp"
#include "ocl_event.hpp"
#include <stdexcept>
#include <vector>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_ocl.hpp>
#endif

namespace cldnn {
namespace ocl {

static int get_cl_map_type(mem_lock_type type) {
    switch (type) {
        case mem_lock_type::read:
            return CL_MAP_READ;
        case mem_lock_type::write:
            return CL_MAP_WRITE;
        case mem_lock_type::read_write:
            return CL_MAP_READ | CL_MAP_WRITE;
        default:
            throw std::runtime_error("Unsupported lock type for cl_memory buffer\n");
    }
}

gpu_buffer::gpu_buffer(ocl_engine* engine,
                       const layout& layout)
    : lockable_gpu_mem(), memory(engine, layout, allocation_type::cl_mem, nullptr)
    , _buffer(engine->get_cl_context(), CL_MEM_READ_WRITE, size()) {
    m_mem_tracker = std::make_shared<MemoryTracker>(engine, _buffer.get(), layout.bytes_count(), allocation_type::cl_mem);
}

gpu_buffer::gpu_buffer(ocl_engine* engine,
                       const layout& new_layout,
                       const cl::Buffer& buffer,
                       std::shared_ptr<MemoryTracker> mem_tracker)
    : lockable_gpu_mem(), memory(engine, new_layout, allocation_type::cl_mem, mem_tracker)
    , _buffer(buffer) {}

void* gpu_buffer::lock(const stream& stream, mem_lock_type type) {
    auto& cl_stream = downcast<const ocl_stream>(stream);
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        try {
             _mapped_ptr = cl_stream.get_cl_queue().enqueueMapBuffer(_buffer, CL_TRUE, get_cl_map_type(type), 0, size());
        } catch (cl::Error const& err) {
            OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
        }
    }
    _lock_count++;
    return _mapped_ptr;
}

void gpu_buffer::unlock(const stream& stream) {
    auto& cl_stream = downcast<const ocl_stream>(stream);
    std::lock_guard<std::mutex> locker(_mutex);
    _lock_count--;
    if (0 == _lock_count) {
        try {
            cl_stream.get_cl_queue().enqueueUnmapMemObject(_buffer, _mapped_ptr);
        } catch (cl::Error const& err) {
            OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
        }
        _mapped_ptr = nullptr;
    }
}

event::ptr gpu_buffer::fill(stream& stream) {
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip EnqueueMemcpy for 0 size tensor" << std::endl;
        return stream.create_user_event(true);
    }
    return fill(stream, 0);
}

event::ptr gpu_buffer::fill(stream& stream, unsigned char pattern) {
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip EnqueueMemcpy for 0 size tensor" << std::endl;
        return stream.create_user_event(true);
    }
    auto& cl_stream = downcast<ocl_stream>(stream);
    auto ev = stream.create_base_event();
    cl::Event& ev_ocl = downcast<ocl_event>(ev.get())->get();
    try {
        cl_stream.get_cl_queue().enqueueFillBuffer<unsigned char>(_buffer, pattern, 0, size(), nullptr, &ev_ocl);
    } catch (cl::Error const& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }

    return ev;
}

shared_mem_params gpu_buffer::get_internal_params() const {
    auto cl_engine = downcast<const ocl_engine>(_engine);
    return {shared_mem_type::shared_mem_buffer, static_cast<shared_handle>(cl_engine->get_cl_context().get()), nullptr,
            static_cast<shared_handle>(_buffer.get()),
#ifdef _WIN32
        nullptr,
#else
        0,
#endif
        0};
}

event::ptr gpu_buffer::copy_from(stream& stream, const memory& other, bool blocking) {
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip EnqueueMemcpy for 0 size tensor" << std::endl;
        return stream.create_user_event(true);
    }
    switch (other.get_allocation_type()) {
    case allocation_type::usm_host:
    case allocation_type::usm_shared:
        {
            // If other is gpu_usm, down cast to gpu_buffer is not possible.
            // But it can read as host ptr if it's allocation type is either usm_host or usm_shared.
            auto& mem_inst = downcast<const gpu_usm>(other);
            return copy_from(stream, mem_inst.buffer_ptr(), blocking, 0, 0);
        }
    case allocation_type::cl_mem:
        {
            auto& cl_stream = downcast<ocl_stream>(stream);
            auto& mem_inst = downcast<const gpu_buffer>(other);
            auto ev = stream.create_base_event();
            cl::Event& ev_ocl = downcast<ocl_event>(ev.get())->get();
            try {
                cl_stream.get_cl_queue().enqueueCopyBuffer(mem_inst.get_buffer(), get_buffer(), 0, 0, other.size(), nullptr, &ev_ocl);
            } catch (cl::Error const& err) {
                OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
            }
            if (blocking)
                ev->wait();

            return ev;
        }
    case allocation_type::usm_device:
    default:
        throw std::runtime_error("Unsupported allocation_type.");
    }
}

event::ptr gpu_buffer::copy_from(stream& stream, const void* host_ptr, bool blocking, size_t dst_offset, size_t data_size) {
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip EnqueueMemcpy for 0 size tensor" << std::endl;
        return stream.create_user_event(true);
    }
    auto& cl_stream = downcast<ocl_stream>(stream);
    auto ev = blocking ? stream.create_user_event(true) : stream.create_base_event();
    cl::Event* ev_ocl = blocking ? nullptr : &downcast<ocl_event>(ev.get())->get();
    data_size = (data_size == 0) ? size() : data_size;
    try {
        cl_stream.get_cl_queue().enqueueWriteBuffer(_buffer, blocking, dst_offset, data_size, host_ptr, nullptr, ev_ocl);
    } catch (cl::Error const& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }

    return ev;
}

event::ptr gpu_buffer::copy_to(stream& stream, void* host_ptr, bool blocking) {
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip EnqueueMemcpy for 0 size tensor" << std::endl;
        return stream.create_user_event(true);
    }
    auto& cl_stream = downcast<ocl_stream>(stream);
    auto ev = blocking ? stream.create_user_event(true) : stream.create_base_event();
    cl::Event* ev_ocl = blocking ? nullptr : &downcast<ocl_event>(ev.get())->get();
    try {
        cl_stream.get_cl_queue().enqueueReadBuffer(_buffer, blocking, 0, size(), host_ptr, nullptr, ev_ocl);
    } catch (cl::Error const& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }

    return ev;
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::memory gpu_buffer::get_onednn_memory(dnnl::memory::desc desc, int64_t offset) const {
    auto onednn_engine = _engine->get_onednn_engine();
    dnnl::memory dnnl_mem(desc, onednn_engine, DNNL_MEMORY_NONE);
    dnnl::ocl_interop::set_mem_object(dnnl_mem, _buffer.get());
    return dnnl_mem;
}
#endif

gpu_image2d::gpu_image2d(ocl_engine* engine, const layout& layout)
    : lockable_gpu_mem()
    , memory(engine, layout, allocation_type::cl_mem, nullptr)
    , _width(0)
    , _height(0)
    , _row_pitch(0)
    , _slice_pitch(0) {
    cl_channel_type type = layout.data_type == data_types::f16 ? CL_HALF_FLOAT : CL_FLOAT;
    cl_channel_order order = CL_R;
    switch (layout.format) {
        case format::image_2d_weights_c1_b_fyx:
            _width = layout.batch();
            _height = layout.spatial(0) * layout.feature() * layout.spatial(1);
            break;
        case format::image_2d_weights_winograd_6x3_s1_fbxyb:
            _height = layout.feature();
            _width = layout.spatial(0) * layout.batch() * layout.spatial(1) * 8 / 3;
            break;
        case format::image_2d_weights_winograd_6x3_s1_xfbyb:
            _height = layout.feature() * layout.spatial(0) * 8 / 3;
            _width = layout.batch() * layout.spatial(1);
            break;
        case format::image_2d_weights_c4_fyx_b:
            _width = layout.batch();
            _height = layout.spatial(0) * layout.feature() * layout.spatial(1);
            order = CL_RGBA;
            break;
        case format::image_2d_rgba:
            _width = layout.spatial(0);
            _height = layout.spatial(1);
            order = CL_RGBA;
            if (layout.feature() != 3 && layout.feature() != 4) {
                CLDNN_ERROR_MESSAGE("2D image allocation", "invalid number of channels in image_2d_rgba input image (should be 3 or 4)!");
            }
            type = CL_UNORM_INT8;
            break;
        case format::nv12:
        {
            // [NHWC] dimensions order
            auto shape = layout.get_shape();
            _width = shape[2];
            _height = shape[1];
            if (shape[3] == 2) {
                order = CL_RG;
            } else if (shape[3] > 2) {
                CLDNN_ERROR_MESSAGE("2D image allocation", "invalid number of channels in NV12 input image!");
            }
            type = CL_UNORM_INT8;
            break;
        }
        default:
            CLDNN_ERROR_MESSAGE("2D image allocation", "unsupported image type!");
    }

    cl::ImageFormat imageFormat(order, type);
    _buffer = cl::Image2D(engine->get_cl_context(), CL_MEM_READ_WRITE, imageFormat, _width, _height, 0);
    size_t elem_size = _buffer.getImageInfo<CL_IMAGE_ELEMENT_SIZE>();
    _bytes_count = elem_size * _width * _height;
    m_mem_tracker = std::make_shared<MemoryTracker>(engine, _buffer.get(), layout.bytes_count(), allocation_type::cl_mem);
}

gpu_image2d::gpu_image2d(ocl_engine* engine,
                         const layout& new_layout,
                         const cl::Image2D& buffer,
                         std::shared_ptr<MemoryTracker> mem_tracker)
    : lockable_gpu_mem(), memory(engine, new_layout, allocation_type::cl_mem, mem_tracker),
      _buffer(buffer) {
    _width = _buffer.getImageInfo<CL_IMAGE_WIDTH>();
    _height = _buffer.getImageInfo<CL_IMAGE_HEIGHT>();
    _row_pitch = _buffer.getImageInfo<CL_IMAGE_ROW_PITCH>();
    _slice_pitch = _buffer.getImageInfo<CL_IMAGE_SLICE_PITCH>();
}

event::ptr gpu_image2d::fill(stream& stream) {
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip EnqueueMemcpy for 0 size tensor" << std::endl;
        return stream.create_user_event(true);
    }
    return fill(stream, 0);
}

event::ptr gpu_image2d::fill(stream& stream, unsigned char pattern) {
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip EnqueueMemcpy for 0 size tensor" << std::endl;
        return stream.create_user_event(true);
    }
    auto& cl_stream = downcast<ocl_stream>(stream);
    auto ev = stream.create_base_event();
    cl::Event& ev_ocl = downcast<ocl_event>(ev.get())->get();
    cl_uint4 pattern_uint4 = {{pattern, pattern, pattern, pattern}};
    try {
        cl_stream.get_cl_queue().enqueueFillImage(_buffer, pattern_uint4, {0, 0, 0}, {_width, _height, 1}, 0, &ev_ocl);
    } catch (cl::Error const& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }
    // TODO: do we need sync here?
    cl_stream.finish();

    return ev;
}

void* gpu_image2d::lock(const stream& stream, mem_lock_type type) {
    auto& cl_stream = downcast<const ocl_stream>(stream);
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        try {
            _mapped_ptr = cl_stream.get_cl_queue()
                    .enqueueMapImage(_buffer,
                                    CL_TRUE,
                                    get_cl_map_type(type),
                                    {0, 0, 0},
                                    {_width, _height, 1},
                                    &_row_pitch,
                                    &_slice_pitch);
        } catch (cl::Error const& err) {
            OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
        }
    }
    _lock_count++;
    return _mapped_ptr;
}

void gpu_image2d::unlock(const stream& stream) {
    auto& cl_stream = downcast<const ocl_stream>(stream);
    std::lock_guard<std::mutex> locker(_mutex);
    _lock_count--;
    if (0 == _lock_count) {
        try {
            cl_stream.get_cl_queue().enqueueUnmapMemObject(_buffer, _mapped_ptr);
        } catch (cl::Error const& err) {
            OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
        }
        _mapped_ptr = nullptr;
    }
}


shared_mem_params gpu_image2d::get_internal_params() const {
    auto cl_engine = downcast<const ocl_engine>(_engine);
    return {shared_mem_type::shared_mem_image, static_cast<shared_handle>(cl_engine->get_cl_context().get()), nullptr,
            static_cast<shared_handle>(_buffer.get()),
#ifdef _WIN32
        nullptr,
#else
        0,
#endif
        0};
}

event::ptr gpu_image2d::copy_from(stream& stream, const memory& other, bool blocking) {
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip EnqueueMemcpy for 0 size tensor" << std::endl;
        return stream.create_user_event(true);
    }
    auto& cl_stream = downcast<const ocl_stream>(stream);
    auto& casted = downcast<const gpu_image2d>(other);
    auto ev = stream.create_base_event();
    cl::Event* ev_ocl = &downcast<ocl_event>(ev.get())->get();
    try {
        cl_stream.get_cl_queue().enqueueCopyImage(casted.get_buffer(), get_buffer(),
                                                {0, 0, 0}, {0, 0, 0}, {_width, _height, 1},
                                                nullptr, ev_ocl);
    } catch (cl::Error const& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }

    if (blocking)
        ev->wait();

    return ev;
}

event::ptr gpu_image2d::copy_from(stream& stream, const void* host_ptr, bool blocking, size_t dst_offset, size_t data_size) {
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip EnqueueMemcpy for 0 size tensor" << std::endl;
        return stream.create_user_event(true);
    }
    OPENVINO_ASSERT(dst_offset == 0, "[GPU] dst_offset should be zero for gpu_image2d::copy_from.");
    OPENVINO_ASSERT(data_size == 0, "[GPU] data_size should be zero for gpu_image2d::copy_from.");
    auto& cl_stream = downcast<ocl_stream>(stream);
    auto ev = blocking ? stream.create_user_event(true) : stream.create_base_event();
    cl::Event* ev_ocl = blocking ? nullptr : &downcast<ocl_event>(ev.get())->get();
    try {
        cl_stream.get_cl_queue().enqueueWriteImage(_buffer, blocking, {0, 0, 0}, {_width, _height, 1},
                                                    _row_pitch, _slice_pitch, host_ptr, nullptr, ev_ocl);
    } catch (cl::Error const& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }

    return ev;
}

event::ptr gpu_image2d::copy_to(stream& stream, memory& other, bool blocking) {
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip EnqueueMemcpy for 0 size tensor" << std::endl;
        return stream.create_user_event(true);
    }
    auto& cl_stream = downcast<const ocl_stream>(stream);
    auto& casted = downcast<const gpu_image2d>(other);
    auto ev = stream.create_base_event();
    cl::Event* ev_ocl = &downcast<ocl_event>(ev.get())->get();
    try {
        cl_stream.get_cl_queue().enqueueCopyImage(get_buffer(), casted.get_buffer(),
                                                {0, 0, 0}, {0, 0, 0}, {_width, _height, 1},
                                                nullptr, ev_ocl);
    } catch (cl::Error const& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }


    if (blocking)
        ev->wait();

    return ev;
}

event::ptr gpu_image2d::copy_to(stream& stream, void* host_ptr, bool blocking) {
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip EnqueueMemcpy for 0 size tensor" << std::endl;
        return stream.create_user_event(true);
    }
    auto& cl_stream = downcast<ocl_stream>(stream);
    auto ev = blocking ? stream.create_user_event(true) : stream.create_base_event();
    cl::Event* ev_ocl = blocking ? nullptr : &downcast<ocl_event>(ev.get())->get();
    try {
        cl_stream.get_cl_queue().enqueueReadImage(_buffer, blocking, {0, 0, 0}, {_width, _height, 1},
                                                _row_pitch, _slice_pitch, host_ptr, nullptr, ev_ocl);
    } catch (cl::Error const& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }

    return ev;
}

gpu_media_buffer::gpu_media_buffer(ocl_engine* engine,
                                   const layout& new_layout,
                                   shared_mem_params params)
    : gpu_image2d(engine, new_layout, cl::ImageVA(engine->get_cl_context(), CL_MEM_READ_WRITE, params.surface, params.plane), nullptr),
    device(params.user_device),
    surface(params.surface),
    plane(params.plane) { }

shared_mem_params gpu_media_buffer::get_internal_params() const {
    auto cl_engine = downcast<const ocl_engine>(_engine);
    return {shared_mem_type::shared_mem_vasurface, static_cast<shared_handle>(cl_engine->get_cl_context().get()), device,
            static_cast<shared_handle>(_buffer.get()), surface, plane };
}

#ifdef _WIN32
gpu_dx_buffer::gpu_dx_buffer(ocl_engine* engine,
                             const layout& new_layout,
                             shared_mem_params params)
    : gpu_buffer(engine, new_layout,
                cl::BufferDX(engine->get_cl_context(), CL_MEM_READ_WRITE, params.mem), nullptr),
    device(params.user_device),
    resource(params.mem) { }

shared_mem_params gpu_dx_buffer::get_internal_params() const {
    auto cl_engine = downcast<const ocl_engine>(_engine);
    return {shared_mem_type::shared_mem_dxbuffer, static_cast<shared_handle>(cl_engine->get_cl_context().get()), device,
            static_cast<shared_handle>(_buffer.get()), resource, 0 };
}
#endif

gpu_usm::gpu_usm(ocl_engine* engine, const layout& new_layout, const cl::UsmMemory& buffer, allocation_type type, std::shared_ptr<MemoryTracker> mem_tracker)
    : lockable_gpu_mem()
    , memory(engine, new_layout, type, mem_tracker)
    , _buffer(buffer)
    , _host_buffer(engine->get_usm_helper()) {
}

gpu_usm::gpu_usm(ocl_engine* engine, const layout& new_layout, const cl::UsmMemory& buffer, std::shared_ptr<MemoryTracker> mem_tracker)
    : lockable_gpu_mem()
    , memory(engine, new_layout, detect_allocation_type(engine, buffer), mem_tracker)
    , _buffer(buffer)
    , _host_buffer(engine->get_usm_helper()) {
}

gpu_usm::gpu_usm(ocl_engine* engine, const layout& layout, allocation_type type)
    : lockable_gpu_mem()
    , memory(engine, layout, type, nullptr)
    , _buffer(engine->get_usm_helper())
    , _host_buffer(engine->get_usm_helper()) {
    _bytes_count = _bytes_count == 0 ? 1 : _bytes_count;
    switch (get_allocation_type()) {
    case allocation_type::usm_host:
        _buffer.allocateHost(_bytes_count);
        break;
    case allocation_type::usm_shared:
        _buffer.allocateShared(_bytes_count);
        break;
    case allocation_type::usm_device:
        _buffer.allocateDevice(_bytes_count);
        break;
    default:
        CLDNN_ERROR_MESSAGE("gpu_usm allocation type",
            "Unknown unified shared memory type!");
    }

    m_mem_tracker = std::make_shared<MemoryTracker>(engine, _buffer.get(), layout.bytes_count(), type);
}

void* gpu_usm::lock(const stream& stream, mem_lock_type type) {
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        auto& cl_stream = downcast<const ocl_stream>(stream);
        if (get_allocation_type() == allocation_type::usm_device) {
            if (type != mem_lock_type::read) {
                throw std::runtime_error("Unable to lock allocation_type::usm_device with write lock_type.");
            }
            GPU_DEBUG_LOG << "Copy usm_device buffer to host buffer." << std::endl;
            _host_buffer.allocateHost(_bytes_count);
            try {
                cl_stream.get_usm_helper().enqueue_memcpy(cl_stream.get_cl_queue(), _host_buffer.get(), _buffer.get(), _bytes_count, CL_TRUE);
            } catch (cl::Error const& err) {
                OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
            }
            _mapped_ptr = _host_buffer.get();
        } else {
            _mapped_ptr = _buffer.get();
        }
    }
    _lock_count++;
    return _mapped_ptr;
}

void gpu_usm::unlock(const stream& /* stream */) {
    std::lock_guard<std::mutex> locker(_mutex);
    _lock_count--;
    if (0 == _lock_count) {
        if (get_allocation_type() == allocation_type::usm_device) {
            _host_buffer.freeMem();
        }
        _mapped_ptr = nullptr;
    }
}

event::ptr gpu_usm::fill(stream& stream, unsigned char pattern) {
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip gpu_usm::fill for 0 size tensor" << std::endl;
        return stream.create_user_event(true);
    }
    auto& cl_stream = downcast<ocl_stream>(stream);
    auto ev = stream.create_base_event();
    cl::Event& ev_ocl = downcast<ocl_event>(ev.get())->get();
    // enqueueFillUsm call will never finish. Driver bug? Uncomment when fixed. Some older drivers doesn't support enqueueFillUsm call at all.
    // cl_stream.get_usm_helper().enqueue_fill_mem<unsigned char>(cl_stream.get_cl_queue(), _buffer.get(), pattern, _bytes_count, nullptr, &ev_ocl)
    // Workarounded with enqeue_memcopy. ToDo: Remove below code. Uncomment above.
    std::vector<unsigned char> temp_buffer(_bytes_count, pattern);
    // TODO: Do we really need blocking call here? Non-blocking one causes accuracy issues right now, but hopefully it can be fixed in more performant way.
    const bool blocking = true;
    try {
        cl_stream.get_usm_helper().enqueue_memcpy(cl_stream.get_cl_queue(), _buffer.get(), temp_buffer.data(), _bytes_count, blocking, nullptr, &ev_ocl);
    } catch (cl::Error const& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }

    return ev;
}

event::ptr gpu_usm::fill(stream& stream) {
    // event::ptr ev{ new base_event(_context), false };
    // cl::Event ev_ocl = downcast<ocl_event>(ev.get())->get();
    // cl::usm::enqueue_set_mem(cl_stream.get_cl_queue(), _buffer.get(), 0, _bytes_count, nullptr, &ev_ocl);
    // ev->wait();

    // [WA]
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip EnqueueMemcpy for 0 size tensor" << std::endl;
        return stream.create_user_event(true);
    }
    return fill(stream, 0);
}

event::ptr gpu_usm::copy_from(stream& stream, const memory& other, bool blocking) {
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip EnqueueMemcpy for 0 size tensor" << std::endl;
        return stream.create_user_event(true);
    }
    auto& cl_stream = downcast<const ocl_stream>(stream);
    auto ev = blocking ? stream.create_user_event(true) : stream.create_base_event();
    cl::Event* ev_ocl = blocking ? nullptr : &downcast<ocl_event>(ev.get())->get();
    if (other.get_allocation_type() == allocation_type::cl_mem) {
        // Copy cl_mem to usm_memory by cl::CommandQueue::enqueueReadBuffer()
        auto& mem_inst = downcast<const gpu_buffer>(other);
        try {
            cl_stream.get_cl_queue().enqueueReadBuffer(mem_inst.get_buffer(), blocking, 0, size(), this->buffer_ptr(), nullptr, ev_ocl);
        } catch (cl::Error const& err) {
            OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
        }
    } else {
        auto& casted = downcast<const gpu_usm>(other);
        auto dst_ptr = get_buffer().get();
        auto src_ptr = casted.get_buffer().get();
        try {
            cl_stream.get_usm_helper().enqueue_memcpy(cl_stream.get_cl_queue(),
                                                    dst_ptr,
                                                    src_ptr,
                                                    _bytes_count,
                                                    blocking,
                                                    nullptr,
                                                    ev_ocl);
        } catch (cl::Error const& err) {
            OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
        }
    }
    return ev;
}

event::ptr gpu_usm::copy_from(stream& stream, const void* host_ptr, bool blocking, size_t dst_offset, size_t data_size) {
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip EnqueueMemcpy for 0 size tensor" << std::endl;
        return stream.create_user_event(true);
    }
    auto& cl_stream = downcast<ocl_stream>(stream);
    auto dst_ptr = get_buffer().get();
    if (dst_offset > 0) {
        auto tmp_dst_ptr = reinterpret_cast<char*>(dst_ptr);
        tmp_dst_ptr += dst_offset;
        dst_ptr = reinterpret_cast<void*>(tmp_dst_ptr);
    }
    data_size = (data_size == 0) ? _bytes_count : data_size;
    auto ev = blocking ? stream.create_user_event(true) : stream.create_base_event();
    cl::Event* ev_ocl = blocking ? nullptr : &downcast<ocl_event>(ev.get())->get();
    try {
        cl_stream.get_usm_helper().enqueue_memcpy(cl_stream.get_cl_queue(),
                                                dst_ptr,
                                                host_ptr,
                                                data_size,
                                                blocking,
                                                nullptr,
                                                ev_ocl);
    } catch (cl::Error const& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }
    return ev;
}

event::ptr gpu_usm::copy_to(stream& stream, void* host_ptr, bool blocking) {
    if (_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "Skip EnqueueMemcpy for 0 size tensor" << std::endl;
        return stream.create_user_event(true);
    }
    auto& cl_stream = downcast<ocl_stream>(stream);
    auto ev = blocking ? stream.create_user_event(true) : stream.create_base_event();
    cl::Event* ev_ocl = blocking ? nullptr : &downcast<ocl_event>(ev.get())->get();
    auto src_ptr = get_buffer().get();
    try {
        cl_stream.get_usm_helper().enqueue_memcpy(cl_stream.get_cl_queue(),
                                                host_ptr,
                                                src_ptr,
                                                _bytes_count,
                                                blocking,
                                                nullptr,
                                                ev_ocl);
    } catch (cl::Error const& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }
    return ev;
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::memory gpu_usm::get_onednn_memory(dnnl::memory::desc desc, int64_t offset) const {
    auto onednn_engine = _engine->get_onednn_engine();
    dnnl::memory dnnl_mem = dnnl::ocl_interop::make_memory(desc, onednn_engine, dnnl::ocl_interop::memory_kind::usm,
        reinterpret_cast<uint8_t*>(_buffer.get()) + offset);
    return dnnl_mem;
}
#endif

shared_mem_params gpu_usm::get_internal_params() const {
    auto cl_engine = downcast<const ocl_engine>(_engine);
    return {
        shared_mem_type::shared_mem_usm,  // shared_mem_type
        static_cast<shared_handle>(cl_engine->get_cl_context().get()),  // context handle
        nullptr,        // user_device handle
        _buffer.get(),  // mem handle
#ifdef _WIN32
        nullptr,  // surface handle
#else
        0,  // surface handle
#endif
        0  // plane
    };
}

allocation_type gpu_usm::detect_allocation_type(const ocl_engine* engine, const void* mem_ptr) {
    auto cl_alloc_type = engine->get_usm_helper().get_usm_allocation_type(mem_ptr);

    allocation_type res;
    switch (cl_alloc_type) {
        case CL_MEM_TYPE_DEVICE_INTEL: res = allocation_type::usm_device; break;
        case CL_MEM_TYPE_HOST_INTEL: res = allocation_type::usm_host; break;
        case CL_MEM_TYPE_SHARED_INTEL: res = allocation_type::usm_shared; break;
        default: res = allocation_type::unknown;
    }

    return res;
}

allocation_type gpu_usm::detect_allocation_type(const ocl_engine* engine, const cl::UsmMemory& buffer) {
    auto alloc_type = detect_allocation_type(engine, buffer.get());
    OPENVINO_ASSERT(alloc_type == allocation_type::usm_device ||
                    alloc_type == allocation_type::usm_host ||
                    alloc_type == allocation_type::usm_shared, "[GPU] Unsupported USM alloc type: " + to_string(alloc_type));
    return alloc_type;
}

std::vector<cl_mem> ocl_surfaces_lock::get_handles(std::vector<memory::ptr> mem) const {
    std::vector<cl_mem> res;
    for (auto& m : mem) {
        auto mem_type = m->get_internal_params().mem_type;
        if (mem_type == shared_mem_type::shared_mem_vasurface || mem_type == shared_mem_type::shared_mem_dxbuffer) {
            res.push_back(static_cast<cl_mem>(m->get_internal_params().mem));
        }
    }

    return res;
}

ocl_surfaces_lock::ocl_surfaces_lock(std::vector<memory::ptr> mem, const stream& stream)
    : surfaces_lock()
    , _handles(get_handles(mem))
    , _lock(nullptr) {
    cl_int err = CL_SUCCESS;

    auto& cl_stream = downcast<const ocl_stream>(stream);
    auto queue = cl_stream.get_cl_queue();
    _lock.reset(new cl::SharedSurfLock(queue.get(), _handles, &err));
    // TODO: err code for some reason is 32766
    if (/* err != CL_SUCCESS || */ !_lock) {
        throw std::runtime_error("Unable to lock shared surface (" + std::to_string(err) + ")");
    }
}

}  // namespace ocl
}  // namespace cldnn
