// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_test_context.hpp"

#include "intel_gpu/runtime/memory.hpp"
#include "ze/ze_engine.hpp"
#include "ze/ze_memory.hpp"
#include "ze/ze_stream.hpp"
#include "ze/ze_ocl_interop.hpp"

#include <vector>

using namespace cldnn;
using namespace ze_tests;

namespace {

struct cl_event_guard {
    cl_event event = nullptr;
    ~cl_event_guard() {
        if (event != nullptr) {
            clReleaseEvent(event);
        }
    }
};

}  // namespace

class ze_ocl_interop_tests : public testing::Test {
protected:
    void SetUp() override {
        ctx = create_ze_test_context();
        if (!ctx.ze_test_engine->get_device_info().supports_leo) {
            GTEST_SKIP() << "Level Zero - OpenCL interoperability not supported";
        }
        engine = std::dynamic_pointer_cast<ze::ze_engine>(ctx.ze_test_engine);
        ASSERT_NE(engine, nullptr);
        interop = &ze::ze_ocl_interop::get_instance();
        ze_device = engine->get_device().handle();
        ocl_device = interop->find_ocl_device(ze_device);
    }

    ze_test_context ctx;
    std::shared_ptr<ze::ze_engine> engine;
    ze::ze_ocl_interop* interop = nullptr;
    ze_device_handle_t ze_device = nullptr;
    cl_device_id ocl_device = nullptr;
};

TEST_F(ze_ocl_interop_tests, finds_matching_device_and_driver) {
    EXPECT_TRUE(interop->check_support(ze_device));
    EXPECT_EQ(interop->get_ze_device(ocl_device), ze_device);
    EXPECT_EQ(interop->find_ze_driver(ocl_device), engine->get_driver().handle());
    EXPECT_EQ(interop->find_ocl_device(ze_device), ocl_device);
}

TEST_F(ze_ocl_interop_tests, creates_cl_context_and_queries_ze_context) {
    ze::ocl_context_args args{};
    args.device = ocl_device;

    cl_context context = interop->create_cl_context(engine->get_context().handle(), args);
    ASSERT_NE(context, nullptr);
    ze::ocl_owner<ze::ocl_resource_type::context> context_owner(context);
    EXPECT_EQ(interop->get_ze_context(context), engine->get_context().handle());
}

TEST_F(ze_ocl_interop_tests, creates_cl_queue_and_queries_command_list) {
    auto stream = std::dynamic_pointer_cast<ze::ze_stream>(ctx.ze_test_stream);
    ASSERT_NE(stream, nullptr);

    ze::ocl_queue_args args{};
    args.context = static_cast<cl_context>(engine->get_user_context(runtime_types::ocl));
    args.device = ocl_device;
    auto command_list = stream->get_immediate_command_list().handle();

    cl_command_queue queue = interop->create_cl_queue(command_list, args);
    ASSERT_NE(queue, nullptr);
    ze::ocl_owner<ze::ocl_resource_type::command_queue> queue_owner(queue);
    EXPECT_EQ(interop->get_ze_cmd_list(queue), command_list);
}

TEST_F(ze_ocl_interop_tests, queries_ze_event_from_cl_event) {
    auto stream = std::dynamic_pointer_cast<ze::ze_stream>(ctx.ze_test_stream);
    ASSERT_NE(stream, nullptr);

    ze::ocl_queue_args args{};
    args.context = static_cast<cl_context>(engine->get_user_context(runtime_types::ocl));
    args.device = ocl_device;
    auto command_list = stream->get_immediate_command_list().handle();
    cl_command_queue queue = interop->create_cl_queue(command_list, args);
    ASSERT_NE(queue, nullptr);
    ze::ocl_owner<ze::ocl_resource_type::command_queue> queue_owner(queue);

    cl_event_guard event;
    ASSERT_EQ(clEnqueueMarkerWithWaitList(queue, 0, nullptr, &event.event), CL_SUCCESS);
    ASSERT_NE(event.event, nullptr);
    ASSERT_EQ(clFinish(queue), CL_SUCCESS);
    EXPECT_NE(interop->get_ze_event(event.event), nullptr);
}

TEST_F(ze_ocl_interop_tests, creates_cl_buffer_and_queries_ze_usm_pointer) {
    const layout buffer_layout{{1, 1, 64, 1}, data_types::u8, format::bfyx};
    auto memory = ctx.ze_test_engine->allocate_memory(buffer_layout, allocation_type::usm_device);
    ASSERT_NE(memory, nullptr);
    auto ze_ptr = memory->get_internal_params(runtime_types::ze).mem;
    ASSERT_NE(ze_ptr, nullptr);

    ze::ocl_buffer_args args{};
    args.context = static_cast<cl_context>(engine->get_user_context(runtime_types::ocl));
    args.flags = CL_MEM_READ_WRITE;
    args.size = memory->size();

    cl_mem buffer = interop->create_cl_buffer(ze_ptr, args);
    ASSERT_NE(buffer, nullptr);
    ze::ocl_owner<ze::ocl_resource_type::mem_object> buffer_owner(buffer);
    EXPECT_EQ(interop->get_ze_usm(buffer), ze_ptr);
}

class ze_image_interop_tests : public testing::Test {
protected:
    void SetUp() override {
        ctx = create_ze_test_context();
        if (!ctx.ze_test_engine->supports_allocation(allocation_type::ze_image) ||
            !ctx.ze_test_engine->get_device_info().supports_leo) {
            GTEST_SKIP() << "Level Zero - OpenCL image interoperability not supported";
        }
    }

    ze_test_context ctx;
    const layout image_layout{ov::PartialShape{1, 4, 8, 1}, data_types::u8, format::image_2d_rgba};
};

TEST_F(ze_image_interop_tests, can_round_trip_ocl_image_handle) {
    auto engine = std::dynamic_pointer_cast<ze::ze_engine>(ctx.ze_test_engine);
    ASSERT_NE(engine, nullptr);
    auto& interop = ze::ze_ocl_interop::get_instance();
    auto device = interop.find_ocl_device(engine->get_device().handle());
    auto mem = ctx.ze_test_engine->allocate_memory(image_layout, allocation_type::ze_image);
    auto image = std::dynamic_pointer_cast<ze::gpu_image2d>(mem);
    ASSERT_NE(image, nullptr);
    auto ze_handle = image->get_resource().handle();
    auto ocl_handle = static_cast<cl_mem>(image->get_internal_params(runtime_types::ocl).mem);
    ASSERT_NE(ocl_handle, nullptr);
    EXPECT_EQ(interop.get_ze_image(ocl_handle), ze_handle);
    EXPECT_EQ(interop.get_ze_images(ocl_handle), std::vector<ze_image_handle_t>{ze_handle});
    EXPECT_EQ(interop.find_ze_driver(device), engine->get_driver().handle());

    size_t properties_size = 0;
    ASSERT_EQ(clGetMemObjectInfo(ocl_handle, CL_MEM_PROPERTIES, 0, nullptr, &properties_size), CL_SUCCESS);
    ASSERT_EQ(properties_size, 4 * sizeof(cl_mem_properties));
    std::vector<cl_mem_properties> properties(properties_size / sizeof(cl_mem_properties));
    ASSERT_EQ(clGetMemObjectInfo(ocl_handle, CL_MEM_PROPERTIES, properties_size, properties.data(), nullptr), CL_SUCCESS);
    EXPECT_EQ(properties[0], 0x42B6);
    EXPECT_EQ(properties[1], reinterpret_cast<cl_mem_properties>(ze_handle));
    EXPECT_EQ(properties[2], 0);
    EXPECT_EQ(properties[3], 0);
}

TEST_F(ze_image_interop_tests, create_cl_image_preserves_handle_and_properties) {
    auto mem = ctx.ze_test_engine->allocate_memory(image_layout, allocation_type::ze_image);
    auto image = std::dynamic_pointer_cast<ze::gpu_image2d>(mem);
    ASSERT_NE(image, nullptr);
    auto ze_handle = image->get_resource().handle();
    ze::ocl_image_args args{};
    args.context = static_cast<cl_context>(ctx.ze_test_engine->get_user_context(runtime_types::ocl));
    args.flags = CL_MEM_READ_WRITE;
    args.format = {CL_RGBA, CL_UNSIGNED_INT8};
    args.desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    args.desc.image_width = 8;
    args.desc.image_height = 4;

    auto& interop = ze::ze_ocl_interop::get_instance();
    for (bool use_list : {false, true}) {
        SCOPED_TRACE(use_list ? "image list" : "single image");
        cl_mem handle = use_list ? interop.create_cl_image(std::vector<ze_image_handle_t>{ze_handle}, args)
                                 : interop.create_cl_image(ze_handle, args);
        ze::ocl_owner<ze::ocl_resource_type::mem_object> owner(handle);
        ASSERT_NE(handle, nullptr);
        EXPECT_EQ(interop.get_ze_image(handle), ze_handle);

        size_t properties_size = 0;
        ASSERT_EQ(clGetMemObjectInfo(handle, CL_MEM_PROPERTIES, 0, nullptr, &properties_size), CL_SUCCESS);
        ASSERT_EQ(properties_size, 4 * sizeof(cl_mem_properties));
        std::vector<cl_mem_properties> properties(properties_size / sizeof(cl_mem_properties));
        ASSERT_EQ(clGetMemObjectInfo(handle, CL_MEM_PROPERTIES, properties_size, properties.data(), nullptr), CL_SUCCESS);
        EXPECT_EQ(properties[0], 0x42B6);
        EXPECT_EQ(properties[1], reinterpret_cast<cl_mem_properties>(ze_handle));
        EXPECT_EQ(properties[2], 0);
        EXPECT_EQ(properties[3], 0);
    }
}

TEST_F(ze_image_interop_tests, create_cl_image_rejects_invalid_arguments) {
    auto mem = ctx.ze_test_engine->allocate_memory(image_layout, allocation_type::ze_image);
    auto image = std::dynamic_pointer_cast<ze::gpu_image2d>(mem);
    ASSERT_NE(image, nullptr);
    auto ze_handle = image->get_resource().handle();
    ze::ocl_image_args args{};
    args.context = static_cast<cl_context>(ctx.ze_test_engine->get_user_context(runtime_types::ocl));
    args.flags = CL_MEM_READ_WRITE;
    args.format = {CL_RGBA, CL_UNSIGNED_INT8};
    args.desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    args.desc.image_width = 8;
    args.desc.image_height = 4;

    auto& interop = ze::ze_ocl_interop::get_instance();
    EXPECT_ANY_THROW(interop.create_cl_image(std::vector<ze_image_handle_t>{}, args));
    EXPECT_ANY_THROW(interop.create_cl_image(static_cast<ze_image_handle_t>(nullptr), args));
    EXPECT_ANY_THROW(interop.create_cl_image(std::vector<ze_image_handle_t>{ze_handle, nullptr}, args));
    args.context = nullptr;
    EXPECT_ANY_THROW(interop.create_cl_image(ze_handle, args));
}