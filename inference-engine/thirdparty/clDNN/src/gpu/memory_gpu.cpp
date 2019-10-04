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
#include "memory_gpu.h"
#include "engine_impl.h"
#include "ocl_base_event.h"
#include <stdexcept>

namespace cldnn {
namespace gpu {

gpu_buffer::gpu_buffer(const refcounted_obj_ptr<engine_impl>& engine, const layout& layout, uint16_t stream_id)
    : memory_impl(engine, layout, stream_id, false),
      _context(engine->get_context()),
      _lock_count(0),
      _buffer(_context->context(), CL_MEM_READ_WRITE, size()),
      _mapped_ptr(nullptr) {
    void* ptr = gpu_buffer::lock();
    memset(ptr, 0, size());
    gpu_buffer::unlock();
}

gpu_buffer::gpu_buffer(const refcounted_obj_ptr<engine_impl>& engine,
                       const layout& new_layout,
                       const cl::Buffer& buffer,
                       uint16_t stream_id)
    : memory_impl(engine, new_layout, stream_id, true),
      _context(engine->get_context()),
      _lock_count(0),
      _buffer(buffer),
      _mapped_ptr(nullptr) {}

void* gpu_buffer::lock() {
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        _mapped_ptr = _context->queue(_stream_id).enqueueMapBuffer(_buffer, CL_TRUE, CL_MAP_WRITE, 0, size());
    }
    _lock_count++;
    return _mapped_ptr;
}

void gpu_buffer::unlock() {
    std::lock_guard<std::mutex> locker(_mutex);
    _lock_count--;
    if (0 == _lock_count) {
        _context->queue(_stream_id).enqueueUnmapMemObject(_buffer, _mapped_ptr);
        _mapped_ptr = nullptr;
    }
}

void gpu_buffer::fill(unsigned char pattern, event_impl::ptr ev) {
    cl::Event ev_ocl = dynamic_cast<base_event*>(ev.get())->get();
    _context->queue(_stream_id).enqueueFillBuffer<unsigned char>(_buffer, pattern, 0, size(), 0, &ev_ocl);
}

gpu_image2d::gpu_image2d(const refcounted_obj_ptr<engine_impl>& engine, const layout& layout, uint16_t stream_id)
    : memory_impl(engine, layout, stream_id, false),
      _context(engine->get_context()),
      _lock_count(0),
      _mapped_ptr(nullptr) {
    cl_channel_order order;
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
        default:
            throw std::invalid_argument("unsupported image type!");
    }

    cl_channel_type type = layout.data_type == data_types::f16 ? CL_HALF_FLOAT : CL_FLOAT;
    cl::ImageFormat imageFormat(order, type);
    _buffer = cl::Image2D(_context->context(), CL_MEM_READ_WRITE, imageFormat, _width, _height, 0);

    void* ptr = gpu_image2d::lock();
    for (uint64_t y = 0; y < static_cast<uint64_t>(_height); y++) memset(ptr, 0, static_cast<size_t>(y * _row_pitch));
    gpu_image2d::unlock();
}

gpu_image2d::gpu_image2d(const refcounted_obj_ptr<engine_impl>& engine,
                         const layout& new_layout,
                         const cl::Image2D& buffer,
                         uint16_t stream_id)
    : memory_impl(engine, new_layout, stream_id, true),
      _context(engine->get_context()),
      _lock_count(0),
      _buffer(buffer),
      _width(0), _height(0), _row_pitch(0), _slice_pitch(0),
      _mapped_ptr(nullptr) {}

void* gpu_image2d::lock() {
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        _mapped_ptr = _context->queue(_stream_id)
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
        _context->queue(_stream_id).enqueueUnmapMemObject(_buffer, _mapped_ptr);
        _mapped_ptr = nullptr;
    }
}

void gpu_image2d::fill(unsigned char pattern, event_impl::ptr ev) {
    cl::Event ev_ocl = dynamic_cast<base_event*>(ev.get())->get();
    cl_uint4 pattern_uint4 = {pattern, pattern, pattern, pattern};
    _context->queue(_stream_id).enqueueFillImage(_buffer, pattern_uint4, {0, 0, 0}, {_width, _height, 1}, 0, &ev_ocl);
}

}  // namespace gpu
}  // namespace cldnn
