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

namespace cldnn {
namespace gpu {

gpu_buffer::gpu_buffer(const refcounted_obj_ptr<engine_impl>& engine, const layout& layout, uint32_t net_id)
    : lockable_gpu_mem(engine), memory_impl(engine, layout, net_id, false),
      _buffer(_context->context(), CL_MEM_READ_WRITE, size()) {
    void* ptr = gpu_buffer::lock();
    memset(ptr, 0, size());
    gpu_buffer::unlock();
}

gpu_buffer::gpu_buffer(const refcounted_obj_ptr<engine_impl>& engine,
                       const layout& new_layout,
                       const cl::Buffer& buffer,
                       uint32_t net_id)
    : lockable_gpu_mem(engine), memory_impl(engine, new_layout, net_id, true),
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

gpu_image2d::gpu_image2d(const refcounted_obj_ptr<engine_impl>& engine, const layout& layout, uint32_t net_id)
    : lockable_gpu_mem(engine), memory_impl(engine, layout, net_id, false) {
    cl_channel_type type = layout.data_type == data_types::f16 ? CL_HALF_FLOAT : CL_FLOAT;
    cl_channel_order order = CL_R;
    switch (layout.format) {
        case format::image_2d_weights_c1_b_fyx:
            _width = layout.size.batch[0];
            _height = layout.size.spatial[0] * layout.size.feature[0] * layout.size.spatial[1];
            order = CL_R;
            break;
        case format::image_2d_weights_winograd_6x3_s1_fbxyb:
            _height = layout.size.feature[0];
            _width = layout.size.spatial[0] * layout.size.batch[0] * layout.size.spatial[1] * 8 / 3;
            order = CL_R;
            break;
        case format::image_2d_weights_winograd_6x3_s1_xfbyb:
            _height = layout.size.feature[0] * layout.size.spatial[0] * 8 / 3;
            _width = layout.size.batch[0] * layout.size.spatial[1];
            order = CL_R;
            break;
        case format::image_2d_weights_c4_fyx_b:
            _width = layout.size.batch[0];
            _height = layout.size.spatial[0] * layout.size.feature[0] * layout.size.spatial[1];
            order = CL_RGBA;
            break;
        case format::nv12:
            _width = layout.size.spatial[1];
            _height = layout.size.spatial[0];
            if (layout.size.feature[0] == 1) {
                order = CL_R;
            } else if (layout.size.feature[0] == 2) {
                order = CL_RG;
            } else {
                CLDNN_ERROR_MESSAGE("2D image allocation", "invalid number of channels in NV12 input image!");
            }
            type = CL_UNORM_INT8;
            break;
        default:
            CLDNN_ERROR_MESSAGE("2D image allocation", "unsupported image type!");
    }

    cl::ImageFormat imageFormat(order, type);
    _buffer = cl::Image2D(_context->context(), CL_MEM_READ_WRITE, imageFormat, _width, _height, 0);

    void* ptr = gpu_image2d::lock();
    for (uint64_t y = 0; y < static_cast<uint64_t>(_height); y++) memset(ptr, 0, static_cast<size_t>(y * _row_pitch));
    gpu_image2d::unlock();
}

gpu_image2d::gpu_image2d(const refcounted_obj_ptr<engine_impl>& engine,
                         const layout& new_layout,
                         const cl::Image2D& buffer,
                         uint32_t net_id)
    : lockable_gpu_mem(engine), memory_impl(engine, new_layout, net_id, true),
      _buffer(buffer) {
    _width = _buffer.getImageInfo<CL_IMAGE_WIDTH>();
    _height = _buffer.getImageInfo<CL_IMAGE_HEIGHT>();
    _row_pitch = _buffer.getImageInfo<CL_IMAGE_ROW_PITCH>();
    _slice_pitch = _buffer.getImageInfo<CL_IMAGE_SLICE_PITCH>();
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
    resource(params->mem) {
}

shared_mem_params gpu_dx_buffer::get_internal_params() const {
    return {shared_mem_type::shared_mem_dxbuffer, static_cast<shared_handle>(_context->context().get()), device,
            static_cast<shared_handle>(_buffer.get()), resource, 0 };
}
#endif

}  // namespace gpu
}  // namespace cldnn
