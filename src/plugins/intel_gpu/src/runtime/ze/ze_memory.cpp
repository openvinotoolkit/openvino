// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn/runtime/error_handler.hpp"
#include "cldnn/runtime/utils.hpp"
#include "ze_memory.hpp"
#include "ze_engine.hpp"
#include "ze_stream.hpp"
#include "ze_event.hpp"
#include <stdexcept>
#include <vector>

namespace cldnn {
namespace ze {

// gpu_buffer::gpu_buffer(ze_engine* engine,
//                        const layout& layout)
//     : lockable_gpu_mem(), memory(engine, layout, allocation_type::cl_mem, false)
//     , _buffer(engine->get_cl_context(), CL_MEM_READ_WRITE, size()) { }

// gpu_buffer::gpu_buffer(ze_engine* engine,
//                        const layout& new_layout,
//                        const ze::Buffer& buffer)
//     : lockable_gpu_mem(), memory(engine, new_layout, allocation_type::cl_mem, true)
//     , _buffer(buffer) {}

// void* gpu_buffer::lock(const stream& stream) {
//     auto& cl_stream = downcast<const ze_stream>(stream);
//     std::lock_guard<std::mutex> locker(_mutex);
//     if (0 == _lock_count) {
//         _mapped_ptr = cl_stream.get_cl_queue().enqueueMapBuffer(_buffer, CL_TRUE, CL_MAP_WRITE, 0, size());
//     }
//     _lock_count++;
//     return _mapped_ptr;
// }

// void gpu_buffer::unlock(const stream& stream) {
//     auto& cl_stream = downcast<const ze_stream>(stream);
//     std::lock_guard<std::mutex> locker(_mutex);
//     _lock_count--;
//     if (0 == _lock_count) {
//         cl_stream.get_cl_queue().enqueueUnmapMemObject(_buffer, _mapped_ptr);
//         _mapped_ptr = nullptr;
//     }
// }

// event::ptr gpu_buffer::fill(stream& stream) {
//     return fill(stream, 0);
// }

// event::ptr gpu_buffer::fill(stream& stream, unsigned char pattern) {
//     auto& cl_stream = downcast<ze_stream>(stream);
//     auto ev = stream.create_base_event();
//     ze::Event ev_ze = std::dynamic_pointer_cast<ze_event>(ev)->get();
//     cl_stream.get_cl_queue().enqueueFillBuffer<unsigned char>(_buffer, pattern, 0, size(), nullptr, &ev_ze);

//     // TODO: do we need sync here?
//     cl_stream.finish();

//     return ev;
// }

// shared_mem_params gpu_buffer::get_internal_params() const {
//     auto cl_engine = downcast<const ze_engine>(_engine);
//     return {shared_mem_type::shared_mem_buffer, static_cast<shared_handle>(cl_engine->get_cl_context().get()), nullptr,
//             static_cast<shared_handle>(_buffer.get()),
// #ifdef _WIN32
//         nullptr,
// #else
//         0,
// #endif
//         0};
// }

// event::ptr gpu_buffer::copy_from(stream& /* stream */, const memory& /* other */) {
//     throw std::runtime_error("[clDNN] copy_from is not implemented for gpu_buffer");
// }

// event::ptr gpu_buffer::copy_from(stream& stream, const void* host_ptr) {
//     auto& cl_stream = downcast<ze_stream>(stream);
//     auto ev = stream.create_base_event();
//     ze::Event ev_ze = std::dynamic_pointer_cast<ze_event>(ev)->get();
//     cl_stream.get_cl_queue().enqueueWriteBuffer(_buffer, false, 0, size(), host_ptr, nullptr, &ev_ze);

//     return ev;
// }

// gpu_image2d::gpu_image2d(ze_engine* engine, const layout& layout)
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

//     ze::ImageFormat imageFormat(order, type);
//     _buffer = ze::Image2D(engine->get_cl_context(), CL_MEM_READ_WRITE, imageFormat, _width, _height, 0);
// }

// gpu_image2d::gpu_image2d(ze_engine* engine,
//                          const layout& new_layout,
//                          const ze::Image2D& buffer)
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
//     auto& cl_stream = downcast<ze_stream>(stream);
//     auto ev = stream.create_base_event();
//     ze::Event ev_ze = downcast<ze_event>(ev.get())->get();
//     cl_uint4 pattern_uint4 = {pattern, pattern, pattern, pattern};
//     cl_stream.get_cl_queue().enqueueFillImage(_buffer, pattern_uint4, {0, 0, 0}, {_width, _height, 1}, 0, &ev_ze);

//     // TODO: do we need sync here?
//     cl_stream.finish();

//     return ev;
// }

// void* gpu_image2d::lock(const stream& stream) {
//     auto& cl_stream = downcast<const ze_stream>(stream);
//     std::lock_guard<std::mutex> locker(_mutex);
//     if (0 == _lock_count) {
//         _mapped_ptr = cl_stream.get_cl_queue()
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
//     auto& cl_stream = downcast<const ze_stream>(stream);
//     std::lock_guard<std::mutex> locker(_mutex);
//     _lock_count--;
//     if (0 == _lock_count) {
//         cl_stream.get_cl_queue().enqueueUnmapMemObject(_buffer, _mapped_ptr);
//         _mapped_ptr = nullptr;
//     }
// }


// shared_mem_params gpu_image2d::get_internal_params() const {
//     auto cl_engine = downcast<const ze_engine>(_engine);
//     return {shared_mem_type::shared_mem_image, static_cast<shared_handle>(cl_engine->get_cl_context().get()), nullptr,
//             static_cast<shared_handle>(_buffer.get()),
// #ifdef _WIN32
//         nullptr,
// #else
//         0,
// #endif
//         0};
// }

// event::ptr gpu_image2d::copy_from(stream& /* stream */, const memory& /* other */) {
//     throw std::runtime_error("[clDNN] copy_from is not implemented for gpu_image2d");
// }

// event::ptr gpu_image2d::copy_from(stream& /* stream */, const void* /* host_ptr */) {
//     throw std::runtime_error("[clDNN] copy_from is not implemented for gpu_image2d");
// }

// gpu_media_buffer::gpu_media_buffer(ze_engine* engine,
//                                    const layout& new_layout,
//                                    shared_mem_params params)
//     : gpu_image2d(engine, new_layout, ze::ImageVA(engine->get_cl_context(), CL_MEM_READ_WRITE, params.surface, params.plane)),
//     device(params.user_device),
//     surface(params.surface),
//     plane(params.plane) { }

// shared_mem_params gpu_media_buffer::get_internal_params() const {
//     auto cl_engine = downcast<const ze_engine>(_engine);
//     return {shared_mem_type::shared_mem_vasurface, static_cast<shared_handle>(cl_engine->get_cl_context().get()), device,
//             static_cast<shared_handle>(_buffer.get()), surface, plane };
// }

// #ifdef _WIN32
// gpu_dx_buffer::gpu_dx_buffer(ze_engine* engine,
//                              const layout& new_layout,
//                              shared_mem_params params)
//     : gpu_buffer(engine, new_layout,
//                 ze::BufferDX(engine->get_cl_context(), CL_MEM_READ_WRITE, params.mem)),
//     device(params.user_device),
//     resource(params.mem) { }

// shared_mem_params gpu_dx_buffer::get_internal_params() const {
//     auto cl_engine = downcast<const ze_engine>(_engine);
//     return {shared_mem_type::shared_mem_dxbuffer, static_cast<shared_handle>(cl_engine->get_cl_context().get()), device,
//             static_cast<shared_handle>(_buffer.get()), resource, 0 };
// }
// #endif

gpu_usm::gpu_usm(ze_engine* engine, const layout& new_layout, const ze::UsmMemory& buffer, allocation_type type)
    : lockable_gpu_mem()
    , memory(engine, new_layout, type, true)
    , _buffer(buffer) {}

gpu_usm::gpu_usm(ze_engine* engine, const layout& layout, allocation_type type)
    : lockable_gpu_mem()
    , memory(engine, layout, type, false)
    , _buffer(engine->get_context(), engine->get_device()) {
    switch (get_allocation_type()) {
    case allocation_type::usm_host:
        _buffer.allocateHost(_bytes_count);
        fill(engine->get_program_stream());
        break;
    case allocation_type::usm_shared:
        _buffer.allocateShared(_bytes_count);
        fill(engine->get_program_stream());
        break;
    case allocation_type::usm_device:
        _buffer.allocateDevice(_bytes_count);
        fill(engine->get_program_stream());
        break;
    default:
        CLDNN_ERROR_MESSAGE("gpu_usm allocation type",
            "Unknown unified shared memory type!");
    }
}

void* gpu_usm::lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) {
    assert(get_allocation_type() != allocation_type::usm_device && "Can't lock usm device memory!");
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        stream.finish();  // Synchronization needed for OOOQ.
        _mapped_ptr = _buffer.get();
    }
    _lock_count++;
    return _mapped_ptr;
}

void gpu_usm::unlock(const stream& /* stream */) {
    std::lock_guard<std::mutex> locker(_mutex);
    _lock_count--;
    if (0 == _lock_count) {
        _mapped_ptr = nullptr;
    }
}

event::ptr gpu_usm::fill(stream& stream, unsigned char pattern) {
    // auto& cl_stream = downcast<ze_stream>(stream);
    //throw std::runtime_error("[clDNN] gpu_usm::fill is not implemented for gpu_usm");

    auto& _ze_stream = downcast<ze_stream>(stream);
    auto ev = _ze_stream.create_base_event();
    auto ev_ze = downcast<ze::ze_base_event>(ev.get())->get();
    std::vector<unsigned char> temp_buffer(_bytes_count, pattern);
    ZE_CHECK(zeCommandListAppendMemoryFill(_ze_stream.get_queue(), _buffer.get(), temp_buffer.data(), 1, _bytes_count, ev_ze, 0, nullptr));
    // ze::Event ev_ze = downcast<ze_event>(ev.get())->get();
    // // enqueueFillUsm call will never finish. Driver bug? Uncomment when fixed. Some older drivers doesn't support enqueueFillUsm call at all.
    // // cl_stream.get_usm_helper().enqueue_fill_mem<unsigned char>(cl_stream.get_cl_queue(), _buffer.get(), pattern, _bytes_count, nullptr, &ev_ze)
    // // Workarounded with enqeue_memcopy. ToDo: Remove below code. Uncomment above.
    // std::vector<unsigned char> temp_buffer(_bytes_count, pattern);
    // // TODO: Do we really need blocking call here? Non-blocking one causes accuracy issues right now, but hopefully it can be fixed in more performant way.
    // const bool blocking = true;
    // cl_stream.get_usm_helper().enqueue_memcpy(cl_stream.get_cl_queue(), _buffer.get(), temp_buffer.data(), _bytes_count, blocking, nullptr, &ev_ze);
    return ev;
}

event::ptr gpu_usm::fill(stream& stream) {
    //throw std::runtime_error("[clDNN] gpu_usm::fill is not implemented for gpu_usm");
    //event::ptr ev{ new ze_base_event(), false };
    //ze::Event ev_ze = downcast<ze_event>(ev.get())->get();
    // ze::usm::enqueue_set_mem(cl_stream.get_cl_queue(), _buffer.get(), 0, _bytes_count, nullptr, &ev_ze);
    // ev->wait();
    //auto& _ze_stream = downcast<const ze_stream>(stream);
    //auto ev = stream.create_user_event(true);
    //cldnn::ze::ze_event ev_ze = downcast<ze_event>(ev.get())->get();

    //ZE_CHECK(zeCommandListAppendMemoryFill(_ze_stream.get_queue(), _buffer.get(), 0, 0, _bytes_count, ev_ze, 0, nullptr));
    // [WA]
    return fill(stream, 0);
}

event::ptr gpu_usm::copy_from(stream& stream, const memory& other) {
    auto& _ze_stream = downcast<ze_stream>(stream);
    //auto& casted = downcast<const gpu_usm>(other);
    auto dst_ptr = get_buffer().get();
    auto src_ptr = downcast<const gpu_usm>(other).get_buffer().get();
    //_ze_stream.get_usm_helper().enqueue_memcpy(_ze_stream.get_queue(),
    auto ev = _ze_stream.create_base_event();
    auto ev_ze = downcast<ze::ze_base_event>(ev.get())->get();
    ZE_CHECK(zeCommandListAppendMemoryCopy(_ze_stream.get_queue(),
                                               dst_ptr,
                                               src_ptr,
                                               _bytes_count,
                                               ev_ze,
                                               0,
                                               nullptr));
    ZE_CHECK(zeCommandListAppendWaitOnEvents(_ze_stream.get_queue(), 1, &ev_ze));
    _ze_stream.finish();
    return ev;
}

event::ptr gpu_usm::copy_from(stream& stream, const void* host_ptr) {
    //throw std::runtime_error("[clDNN] copy_from is not implemented for gpu_usm");
    auto& _ze_stream = downcast<ze_stream>(stream);
    //auto& casted = downcast<const gpu_usm>(other);
    auto dst_ptr = get_buffer().get();
    auto src_ptr = host_ptr;//downcast<const gpu_usm>(other).get_buffer().get();
    //_ze_stream.get_usm_helper().enqueue_memcpy(_ze_stream.get_queue(),
    auto ev = _ze_stream.create_base_event();
    auto ev_ze = downcast<ze::ze_base_event>(ev.get())->get();
    ZE_CHECK(zeCommandListAppendMemoryCopy(_ze_stream.get_queue(),
                                               dst_ptr,
                                               src_ptr,
                                               _bytes_count,
                                               ev_ze,
                                               0,
                                               nullptr));
    ZE_CHECK(zeCommandListAppendWaitOnEvents(_ze_stream.get_queue(), 1, &ev_ze));
    _ze_stream.finish();
    return ev;
}

shared_mem_params gpu_usm::get_internal_params() const {
    auto casted = downcast<ze_engine>(_engine);
    return {
        shared_mem_type::shared_mem_usm,  // shared_mem_type
        static_cast<shared_handle>(casted->get_context()),  // context handle
        static_cast<shared_handle>(casted->get_device()),  // user_device handle
        static_cast<shared_handle>(_buffer.get()),  // mem handle
#ifdef _WIN32
        nullptr,  // surface handle
#else
        0,  // surface handle
#endif
        0  // plane
    };
}

// std::vector<cl_mem> ze_surfaces_lock::get_handles(std::vector<memory::ptr> mem) const {
//     std::vector<cl_mem> res;
//     for (auto& m : mem) {
//         auto mem_type = m->get_internal_params().mem_type;
//         if (mem_type == shared_mem_type::shared_mem_vasurface || mem_type == shared_mem_type::shared_mem_dxbuffer) {
//             res.push_back(static_cast<cl_mem>(m->get_internal_params().mem));
//         }
//     }

//     return res;
// }

// ze_surfaces_lock::ze_surfaces_lock(std::vector<memory::ptr> mem, const stream& stream)
//     : surfaces_lock()
//     , _stream(stream)
//     , _handles(get_handles(mem))
//     , _lock(nullptr) {
//     cl_int err = CL_SUCCESS;

//     auto& cl_stream = downcast<const ze_stream>(stream);
//     auto queue = cl_stream.get_cl_queue();
//     _lock.reset(new ze::SharedSurfLock(queue.get(), _handles, &err));
//     // TODO: err code for some reason is 32766
//     if (/* err != CL_SUCCESS || */ !_lock) {
//         throw std::runtime_error("Unable to lock shared surface (" + std::to_string(err) + ")");
//     }
// }

}  // namespace ze
}  // namespace cldnn
