// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn/runtime/error_handler.hpp"
#include "sycl_memory.hpp"
#include "sycl_engine.hpp"
#include "sycl_stream.hpp"
#include "sycl_event.hpp"
#include <stdexcept>
#include <vector>

namespace cldnn {
namespace sycl {

gpu_buffer::gpu_buffer(sycl_engine* engine, const layout& layout)
    : lockable_gpu_mem(), memory(engine, layout, allocation_type::cl_mem, false)
    , _buffer(cl::sycl::range<1>(size())) { }

gpu_buffer::gpu_buffer(sycl_engine* engine, const layout& new_layout, const buffer_type& buffer)
    : memory(engine, new_layout, allocation_type::cl_mem, true)
    , _buffer(buffer) {}

void* gpu_buffer::lock(const stream& /* stream */) {
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        access_type acc = _buffer.get_access<cl::sycl::access::mode::read_write>();
        _access.reset(new access_type(acc));
        _mapped_ptr = static_cast<void *>(_access->get_pointer());
    }
    _lock_count++;
    return _mapped_ptr;
}

void gpu_buffer::unlock(const stream& /* stream */) {
    std::lock_guard<std::mutex> locker(_mutex);
    _lock_count--;
    if (0 == _lock_count) {
        _mapped_ptr = nullptr;
        _access.reset();
    }
}

event::ptr gpu_buffer::fill(stream& stream) {
    return fill(stream, 0);
}

event::ptr gpu_buffer::fill(stream& stream, unsigned char pattern) {
    auto& casted_stream = dynamic_cast<sycl_stream&>(stream);
    using access = cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer>;
    auto out_event = casted_stream.queue().submit([&](cl::sycl::handler &cgh) {
        access acc_dst(_buffer, cgh, cl::sycl::range<1>(size()), cl::sycl::id<1>(0));
        cgh.fill(acc_dst, pattern);
    });
    return std::make_shared<sycl_event>(casted_stream.get_sycl_engine().get_sycl_context(), out_event);
}

shared_mem_params gpu_buffer::get_internal_params() const {
    // auto cl_engine = dynamic_cast<const sycl_engine*>(_engine);
    return {};
//     return {shared_mem_type::shared_mem_buffer, static_cast<shared_handle>(cl_engine->get_cl_context().get()), nullptr,
//             static_cast<shared_handle>(_buffer.get()),
// #ifdef _WIN32
//         nullptr,
// #else
//         0,
// #endif
//         0};
}

// gpu_image2d::gpu_image2d(sycl_engine* engine, const layout& layout)
//     : lockable_gpu_mem(), memory(engine, layout, allocation_type::cl_mem, false), _row_pitch(0), _slice_pitch(0) {
//     cl_channel_type type = layout.data_type == data_types::f16 ? CL_HALF_FLOAT : CL_FLOAT;
//     cl_channel_order order = CL_R;
//     switch (layout.format) {
//         case format::image_2d_weights_c1_b_fyx:
//             _width = layout.size.batch[0];
//             _height = layout.size.spatial[0] * layout.size.feature[0] * layout.size.spatial[1];
//             break;
//         case format::image_2d_weights_winograd_6x3_s1_fbxyb:
//             _height = layout.size.feature[0];
//             _width = layout.size.spatial[0] * layout.size.batch[0] * layout.size.spatial[1] * 8 / 3;
//             break;
//         case format::image_2d_weights_winograd_6x3_s1_xfbyb:
//             _height = layout.size.feature[0] * layout.size.spatial[0] * 8 / 3;
//             _width = layout.size.batch[0] * layout.size.spatial[1];
//             break;
//         case format::image_2d_weights_c4_fyx_b:
//             _width = layout.size.batch[0];
//             _height = layout.size.spatial[0] * layout.size.feature[0] * layout.size.spatial[1];
//             order = CL_RGBA;
//             break;
//         case format::image_2d_rgba:
//             _width = layout.size.spatial[0];
//             _height = layout.size.spatial[1];
//             order = CL_RGBA;
//             if (layout.size.feature[0] != 3 && layout.size.feature[0] != 4) {
//                 CLDNN_ERROR_MESSAGE("2D image allocation", "invalid number of channels in image_2d_rgba input image (should be 3 or 4)!");
//             }
//             type = CL_UNORM_INT8;
//             break;
//         case format::nv12:
//             _width = layout.size.spatial[1];
//             _height = layout.size.spatial[0];
//             if (layout.size.feature[0] == 2) {
//                 order = CL_RG;
//             } else if (layout.size.feature[0] > 2) {
//                 CLDNN_ERROR_MESSAGE("2D image allocation", "invalid number of channels in NV12 input image!");
//             }
//             type = CL_UNORM_INT8;
//             break;
//         default:
//             CLDNN_ERROR_MESSAGE("2D image allocation", "unsupported image type!");
//     }

//     cl::ImageFormat imageFormat(order, type);
//     _buffer = cl::Image2D(engine->get_cl_context(), CL_MEM_READ_WRITE, imageFormat, _width, _height, 0);
// }

// gpu_image2d::gpu_image2d(sycl_engine* engine,
//                          const layout& new_layout,
//                          const cl::Image2D& buffer)
//     : lockable_gpu_mem(), memory(engine, new_layout, allocation_type::cl_mem, true),
//       _buffer(buffer) {
//     _width = _buffer.getImageInfo<CL_IMAGE_WIDTH>();
//     _height = _buffer.getImageInfo<CL_IMAGE_HEIGHT>();
//     _row_pitch = _buffer.getImageInfo<CL_IMAGE_ROW_PITCH>();
//     _slice_pitch = _buffer.getImageInfo<CL_IMAGE_SLICE_PITCH>();
// }

// event::ptr gpu_image2d::fill(stream& stream) {
//     return fill(stream, 0);
// }

// event::ptr gpu_image2d::fill(stream& stream, unsigned char pattern) {
//     auto& cl_stream = dynamic_cast<sycl_stream&>(stream);
//     auto ev = stream.create_base_event();
//     cl::Event ev_sycl = dynamic_cast<base_event*>(ev.get())->get();
//     cl_uint4 pattern_uint4 = {pattern, pattern, pattern, pattern};
//     cl_stream.queue().enqueueFillImage(_buffer, pattern_uint4, {0, 0, 0}, {_width, _height, 1}, 0, &ev_sycl);

//     return ev;
// }

// void* gpu_image2d::lock(const stream& stream) {
//     auto& cl_stream = dynamic_cast<const sycl_stream&>(stream);
//     std::lock_guard<std::mutex> locker(_mutex);
//     if (0 == _lock_count) {
//         _mapped_ptr = cl_stream.queue()
//                           .enqueueMapImage(_buffer,
//                                            CL_TRUE,
//                                            CL_MAP_WRITE,
//                                            {0, 0, 0},
//                                            {_width, _height, 1},
//                                            &_row_pitch,
//                                            &_slice_pitch);
//     }
//     _lock_count++;
//     return _mapped_ptr;
// }

// void gpu_image2d::unlock(const stream& stream) {
//     auto& cl_stream = dynamic_cast<const sycl_stream&>(stream);
//     std::lock_guard<std::mutex> locker(_mutex);
//     _lock_count--;
//     if (0 == _lock_count) {
//         cl_stream.queue().enqueueUnmapMemObject(_buffer, _mapped_ptr);
//         _mapped_ptr = nullptr;
//     }
// }


// shared_mem_params gpu_image2d::get_internal_params() const {
//     auto cl_engine = dynamic_cast<const sycl_engine*>(_engine);
//     return {shared_mem_type::shared_mem_image, static_cast<shared_handle>(cl_engine->get_cl_context().get()), nullptr,
//             static_cast<shared_handle>(_buffer.get()),
// #ifdef _WIN32
//         nullptr,
// #else
//         0,
// #endif
//         0};
// }

// gpu_media_buffer::gpu_media_buffer(sycl_engine* engine,
//                                    const layout& new_layout,
//                                    shared_mem_params params)
//     : gpu_image2d(engine, new_layout, cl::ImageVA(engine->get_cl_context(), CL_MEM_READ_WRITE, params.surface, params.plane)),
//     device(params.user_device),
//     surface(params.surface),
//     plane(params.plane) { }

// shared_mem_params gpu_media_buffer::get_internal_params() const {
//     auto cl_engine = dynamic_cast<const sycl_engine*>(_engine);
//     return {shared_mem_type::shared_mem_vasurface, static_cast<shared_handle>(cl_engine->get_cl_context().get()), device,
//             static_cast<shared_handle>(_buffer.get()), surface, plane };
// }

// #ifdef _WIN32
// gpu_dx_buffer::gpu_dx_buffer(sycl_engine* engine,
//     const layout& new_layout,
//     shared_mem_params params,
//     uint32_t net_id)
//     : gpu_buffer(engine, new_layout,
//                 cl::BufferDX(engine->get_context()->context(), CL_MEM_READ_WRITE, params.mem),
//                 net_id),
//     device(params.user_device),
//     resource(params.mem) { }

// shared_mem_params gpu_dx_buffer::get_internal_params() const {
//     auto cl_engine = dynamic_cast<const sycl_engine*>(_engine);
//     return {shared_mem_type::shared_mem_dxbuffer, static_cast<shared_handle>(cl_engine->get_cl_context().get()), device,
//             static_cast<shared_handle>(_buffer.get()), resource, 0 };
// }
// #endif

// gpu_usm::gpu_usm(sycl_engine* engine, const layout& new_layout, void* buffer, allocation_type type)
//     : lockable_gpu_mem()
//     , memory(engine, new_layout, type, true)
//     , _buffer(buffer) {
// }

// gpu_usm::gpu_usm(sycl_engine* engine, const layout& layout, allocation_type type)
//     : lockable_gpu_mem()
//     , memory(engine, layout, type, false)
//     , _buffer(nullptr) {
//     auto device = engine->get_cl_device();
//     switch (get_allocation_type()) {
//     case allocation_type::usm_host:
//         _buffer.allocateHost(_bytes_count);
//         break;
//     case allocation_type::usm_shared:
//         _buffer.allocateShared(device, _bytes_count);
//         break;
//     case allocation_type::usm_device:
//         _buffer.allocateDevice(device, _bytes_count);
//         break;
//     default:
//         CLDNN_ERROR_MESSAGE("gpu_usm allocation type",
//             "Unknown unified shared memory type!");
//     }
// }

// void* gpu_usm::lock(const stream& stream) {
//     assert(get_allocation_type() != allocation_type::usm_device && "Can't lock usm device memory!");
//     std::lock_guard<std::mutex> locker(_mutex);
//     if (0 == _lock_count) {
//         // stream.finish();  // Synchronization needed for OOOQ.
//         _mapped_ptr = _buffer.get();
//     }
//     _lock_count++;
//     return _mapped_ptr;
// }

// void gpu_usm::unlock(const stream& /* stream */) {
//     std::lock_guard<std::mutex> locker(_mutex);
//     _lock_count--;
//     if (0 == _lock_count) {
//         _mapped_ptr = nullptr;
//     }
// }

// event::ptr gpu_usm::fill(stream& stream, unsigned char pattern) {
//     auto& cl_stream = dynamic_cast<sycl_stream&>(stream);
//     auto ev = stream.create_base_event();
//     cl::Event ev_sycl = dynamic_cast<base_event*>(ev.get())->get();
//     // enqueueFillUsm call will never finish. Driver bug? Uncomment when fixed. Some older drivers doesn't support enqueueFillUsm call at all.
//     // cl_stream.queue().enqueueFillUsm<unsigned char>(_buffer, pattern, _bytes_count, nullptr, &ev_sycl)
//     // Workarounded with enqeue_memcopy. ToDo: Remove below code. Uncomment above.
//     std::vector<unsigned char> temp_buffer(_bytes_count, pattern);
//     cl::usm::enqueue_memcpy(cl_stream.queue(), _buffer.get(), temp_buffer.data(), _bytes_count, false, nullptr, &ev_sycl);

//     //  dynamic_cast<base_event*>(ev.get())->set();
//     return ev;
// }

// event::ptr gpu_usm::fill(stream& stream) {
//     // event::ptr ev{ new base_event(_context), false };
//     // cl::Event ev_sycl = dynamic_cast<base_event*>(ev.get())->get();
//     // cl::usm::enqueue_set_mem(cl_stream.queue(), _buffer.get(), 0, _bytes_count, nullptr, &ev_sycl);
//     // ev->wait();

//     // [WA]
//     return fill(stream, 0);
// }

// void gpu_usm::copy_from_other(const stream& stream, const memory& other) {
//     auto& cl_stream = dynamic_cast<const sycl_stream&>(stream);
//     auto& casted = dynamic_cast<const gpu_usm&>(other);
//     cl_stream.queue().enqueueCopyUsm(casted.get_buffer(), get_buffer(), _bytes_count, true);
// }

// shared_mem_params gpu_usm::get_internal_params() const {
//     auto cl_engine = dynamic_cast<const sycl_engine*>(_engine);
//     return {
//         shared_mem_type::shared_mem_empty,  // shared_mem_type
//         static_cast<shared_handle>(cl_engine->get_cl_context().get()),  // context handle
//         nullptr,  // user_device handle
//         nullptr,  // mem handle
// #ifdef _WIN32
//         nullptr,  // surface handle
// #else
//         0,  // surface handle
// #endif
//         0  // plane
//     };
// }

}  // namespace sycl
}  // namespace cldnn
