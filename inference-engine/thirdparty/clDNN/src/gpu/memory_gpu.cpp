/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "error_handler.h"
#include "memory_gpu.h"
#include "engine_impl.h"
#include "ocl_base_event.h"
#include <stdexcept>
#include <vector>

namespace cldnn {
namespace gpu {

gpu_buffer::gpu_buffer(const refcounted_obj_ptr<engine_impl>& engine,
                       const layout& layout,
                       uint32_t net_id,
                       bool reset)
    : lockable_gpu_mem(engine), memory_impl(engine, layout, net_id, allocation_type::cl_mem, false),
      _buffer(_context->context(), CL_MEM_READ_WRITE, size()) {
    if (reset) zero_buffer();
}

gpu_buffer::gpu_buffer(const refcounted_obj_ptr<engine_impl>& engine,
                       const layout& new_layout,
                       const cl::Buffer& buffer,
                       uint32_t net_id)
    : lockable_gpu_mem(engine), memory_impl(engine, new_layout, net_id, allocation_type::cl_mem, true),
      _buffer(buffer) {}

void* gpu_buffer::lock() {
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        _mapped_ptr = _context->queue(_net_id).enqueueMapBuffer(_buffer, CL_TRUE, CL_MAP_WRITE, 0, size());
    }
    _lock_count++;
    return _mapped_ptr;
}

void gpu_buffer::unlock() {
    std::lock_guard<std::mutex> locker(_mutex);
    _lock_count--;
    if (0 == _lock_count) {
        _context->queue(_net_id).enqueueUnmapMemObject(_buffer, _mapped_ptr);
        _mapped_ptr = nullptr;
    }
}

void gpu_buffer::zero_buffer() {
    _context->queue(_net_id).enqueueFillBuffer<unsigned char>(_buffer, 0, 0, size());
    _context->queue(_net_id).flush();
}

void gpu_buffer::fill(unsigned char pattern, event_impl::ptr ev) {
    cl::Event ev_ocl = dynamic_cast<base_event*>(ev.get())->get();
    _context->queue(_net_id).enqueueFillBuffer<unsigned char>(_buffer, pattern, 0, size(), 0, &ev_ocl);
}

shared_mem_params gpu_buffer::get_internal_params() const {
    return {shared_mem_type::shared_mem_buffer, static_cast<shared_handle>(_context->context().get()), nullptr,
            static_cast<shared_handle>(_buffer.get()),
#ifdef WIN32
        nullptr,
#else
        0,
#endif
        0};
}

gpu_image2d::gpu_image2d(const refcounted_obj_ptr<engine_impl>& engine, const layout& layout, uint32_t net_id,
                         bool reset)
    : lockable_gpu_mem(engine), memory_impl(engine, layout, net_id, allocation_type::cl_mem, false),
    _row_pitch(0), _slice_pitch(0) {
    cl_channel_type type = layout.data_type == data_types::f16 ? CL_HALF_FLOAT : CL_FLOAT;
    cl_channel_order order = CL_R;
    switch (layout.format) {
        case format::image_2d_weights_c1_b_fyx:
            _width = layout.size.batch[0];
            _height = layout.size.spatial[0] * layout.size.feature[0] * layout.size.spatial[1];
            break;
        case format::image_2d_weights_winograd_6x3_s1_fbxyb:
            _height = layout.size.feature[0];
            _width = layout.size.spatial[0] * layout.size.batch[0] * layout.size.spatial[1] * 8 / 3;
            break;
        case format::image_2d_weights_winograd_6x3_s1_xfbyb:
            _height = layout.size.feature[0] * layout.size.spatial[0] * 8 / 3;
            _width = layout.size.batch[0] * layout.size.spatial[1];
            break;
        case format::image_2d_weights_c4_fyx_b:
            _width = layout.size.batch[0];
            _height = layout.size.spatial[0] * layout.size.feature[0] * layout.size.spatial[1];
            order = CL_RGBA;
            break;
        case format::nv12:
            _width = layout.size.spatial[1];
            _height = layout.size.spatial[0];
            if (layout.size.feature[0] == 2) {
                order = CL_RG;
            } else if (layout.size.feature[0] > 2) {
                CLDNN_ERROR_MESSAGE("2D image allocation", "invalid number of channels in NV12 input image!");
            }
            type = CL_UNORM_INT8;
            break;
        default:
            CLDNN_ERROR_MESSAGE("2D image allocation", "unsupported image type!");
    }

    cl::ImageFormat imageFormat(order, type);
    _buffer = cl::Image2D(_context->context(), CL_MEM_READ_WRITE, imageFormat, _width, _height, 0);

    if (reset) zero_image();
}

gpu_image2d::gpu_image2d(const refcounted_obj_ptr<engine_impl>& engine,
                         const layout& new_layout,
                         const cl::Image2D& buffer,
                         uint32_t net_id)
    : lockable_gpu_mem(engine), memory_impl(engine, new_layout, net_id, allocation_type::cl_mem, true),
      _buffer(buffer) {
    _width = _buffer.getImageInfo<CL_IMAGE_WIDTH>();
    _height = _buffer.getImageInfo<CL_IMAGE_HEIGHT>();
    _row_pitch = _buffer.getImageInfo<CL_IMAGE_ROW_PITCH>();
    _slice_pitch = _buffer.getImageInfo<CL_IMAGE_SLICE_PITCH>();
}

void gpu_image2d::zero_image() {
    cl_uint4 pattern_uint4 = { 0, 0, 0, 0 };
    _context->queue(_net_id).enqueueFillImage(_buffer, pattern_uint4, { 0, 0, 0 }, { _width, _height, 1 });
    _context->queue(_net_id).flush();
}

void* gpu_image2d::lock() {
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        _mapped_ptr = _context->queue(_net_id)
                          .enqueueMapImage(_buffer,
                                           CL_TRUE,
                                           CL_MAP_WRITE,
                                           {0, 0, 0},
                                           {_width, _height, 1},
                                           &_row_pitch,
                                           &_slice_pitch);
    }
    _lock_count++;
    return _mapped_ptr;
}

void gpu_image2d::unlock() {
    std::lock_guard<std::mutex> locker(_mutex);
    _lock_count--;
    if (0 == _lock_count) {
        _context->queue(_net_id).enqueueUnmapMemObject(_buffer, _mapped_ptr);
        _mapped_ptr = nullptr;
    }
}

void gpu_image2d::fill(unsigned char pattern, event_impl::ptr ev) {
    cl::Event ev_ocl = dynamic_cast<base_event*>(ev.get())->get();
    cl_uint4 pattern_uint4 = {pattern, pattern, pattern, pattern};
    _context->queue(_net_id).enqueueFillImage(_buffer, pattern_uint4, {0, 0, 0}, {_width, _height, 1}, 0, &ev_ocl);
}

shared_mem_params gpu_image2d::get_internal_params() const {
    return {shared_mem_type::shared_mem_image, static_cast<shared_handle>(_context->context().get()), nullptr,
            static_cast<shared_handle>(_buffer.get()),
#ifdef WIN32
        nullptr,
#else
        0,
#endif
        0};
}

gpu_media_buffer::gpu_media_buffer(const refcounted_obj_ptr<engine_impl>& engine,
    const layout& new_layout,
    const shared_mem_params* params,
    uint32_t net_id)
    : gpu_image2d(engine, new_layout,
        cl::ImageVA(engine->get_context()->context(), CL_MEM_READ_ONLY,
                    params->surface, params->plane),
        net_id),
    device(params->user_device),
    surface(params->surface),
    plane(params->plane) {
}

shared_mem_params gpu_media_buffer::get_internal_params() const {
    return {shared_mem_type::shared_mem_vasurface, static_cast<shared_handle>(_context->context().get()), device,
            static_cast<shared_handle>(_buffer.get()), surface, plane };
}

#ifdef WIN32
gpu_dx_buffer::gpu_dx_buffer(const refcounted_obj_ptr<engine_impl>& engine,
    const layout& new_layout,
    const shared_mem_params* params,
    uint32_t net_id)
    : gpu_buffer(engine, new_layout,
                cl::BufferDX(engine->get_context()->context(), CL_MEM_READ_WRITE, params->mem),
                net_id),
    device(params->user_device),
    resource(params->mem) { }

shared_mem_params gpu_dx_buffer::get_internal_params() const {
    return {shared_mem_type::shared_mem_dxbuffer, static_cast<shared_handle>(_context->context().get()), device,
            static_cast<shared_handle>(_buffer.get()), resource, 0 };
}
#endif

gpu_usm::gpu_usm(const refcounted_obj_ptr<engine_impl>& engine,
    const layout& new_layout, const cl::UsmMemory& buffer,
    allocation_type type, uint32_t net_id)
    : lockable_gpu_mem(engine)
    , memory_impl(engine, new_layout, net_id, type, true)
    , _buffer(buffer) {
}

gpu_usm::gpu_usm(const refcounted_obj_ptr<engine_impl>& engine, const layout& layout, uint32_t net_id, allocation_type type, bool reset)
    : lockable_gpu_mem(engine)
    , memory_impl(engine, layout, net_id, type, false)
    , _buffer(_engine->get_context()->context()) {
    auto device = _engine->get_context()->device();
    switch (get_allocation_type()) {
    case allocation_type::usm_host:
        _buffer.allocateHost(_bytes_count);
        break;
    case allocation_type::usm_shared:
        _buffer.allocateShared(device, _bytes_count);
        break;
    case allocation_type::usm_device:
        _buffer.allocateDevice(device, _bytes_count);
        break;
    default:
        CLDNN_ERROR_MESSAGE("gpu_usm allocation type",
            "Unknown unified shared memory type!");
    }

    if (reset) zero_buffer();
}

void* gpu_usm::lock() {
    assert(get_allocation_type() != allocation_type::usm_device && "Can't lock usm device memory!");
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        _engine->get_context()->queue(_net_id).finish();  // Synchronization needed for OOOQ.
        _mapped_ptr = _buffer.get();
    }
    _lock_count++;
    return _mapped_ptr;
}

void gpu_usm::unlock() {
    std::lock_guard<std::mutex> locker(_mutex);
    _lock_count--;
    if (0 == _lock_count) {
        _mapped_ptr = nullptr;
    }
}

void gpu_usm::fill(unsigned char pattern, event_impl::ptr ev) {
    cl::Event ev_ocl = dynamic_cast<base_event*>(ev.get())->get();
    // enqueueFillUsm call will never finish. Driver bug? Uncomment when fixed. Some older drivers doesn't support enqueueFillUsm call at all.
    // _engine->get_context()->queue(_net_id).enqueueFillUsm<unsigned char>(_buffer, pattern, _bytes_count, nullptr, &ev_ocl)
    // Workarounded with enqeue_memcopy. ToDo: Remove below code. Uncomment above.
    std::vector<unsigned char> temp_buffer(_bytes_count, pattern);
    cl::usm::enqueue_memcpy(_engine->get_context()->queue(_net_id), _buffer.get(), temp_buffer.data(), _bytes_count, true, nullptr, &ev_ocl);
}

void gpu_usm::zero_buffer() {
    // event_impl::ptr ev{ new base_event(_engine->get_context()), false };
    // cl::Event ev_ocl = dynamic_cast<base_event*>(ev.get())->get();
    // cl::usm::enqueue_set_mem(_engine->get_context()->queue(_net_id), _buffer.get(), 0, _bytes_count, nullptr, &ev_ocl);
    // ev->wait();

    // [WA]
    event_impl::ptr ev{ new base_event(_engine->get_context()), false };
    fill(0, ev);
    ev->wait();
}

void gpu_usm::copy_from_other(const gpu_usm& other) {
    _engine->get_context()->queue(_net_id).enqueueCopyUsm(other.get_buffer(), get_buffer(), _bytes_count, true);
}

shared_mem_params gpu_usm::get_internal_params() const {
    return {
        shared_mem_type::shared_mem_empty,  // shared_mem_type
        static_cast<shared_handle>(_engine->get_context()->context().get()),  // context handle
        nullptr,  // user_device handle
        nullptr,  // mem handle
#ifdef WIN32
        nullptr,  // surface handle
#else
        0,  // surface handle
#endif
        0  // plane
    };
}

}  // namespace gpu
}  // namespace cldnn
