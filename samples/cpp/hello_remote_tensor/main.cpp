// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// clang-format off
#ifndef CL_TARGET_OPENCL_VERSION
#    define CL_TARGET_OPENCL_VERSION 300
#endif
#include <CL/cl.h>
#include <level_zero/ze_api.h>

#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "samples/slog.hpp"
#include "format_reader_ptr.h"
// clang-format on

namespace {

void check_ze(ze_result_t result, const char* operation) {
    OPENVINO_ASSERT(result == ZE_RESULT_SUCCESS, operation, " failed with Level Zero error ", result);
}

void initialize_level_zero() {
    ze_init_driver_type_desc_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC;
    desc.flags = ZE_INIT_DRIVER_TYPE_FLAG_GPU;
    uint32_t driver_count = 0;
    check_ze(zeInitDrivers(&driver_count, nullptr, &desc), "zeInitDrivers");
    OPENVINO_ASSERT(driver_count > 0, "No Level Zero GPU driver found");
}

struct CommandListDeleter {
    void operator()(ze_command_list_handle_t command_list) const {
        const auto result = zeCommandListDestroy(command_list);
        if (result != ZE_RESULT_SUCCESS) {
            std::cerr << "zeCommandListDestroy failed with Level Zero error " << result << std::endl;
        }
    }
};

using CommandList = std::unique_ptr<std::remove_pointer_t<ze_command_list_handle_t>, CommandListDeleter>;

CommandList create_command_list(ze_context_handle_t context, ze_device_handle_t device) {
    uint32_t count = 0;
    check_ze(zeDeviceGetCommandQueueGroupProperties(device, &count, nullptr), "zeDeviceGetCommandQueueGroupProperties");
    std::vector<ze_command_queue_group_properties_t> groups(count);
    for (auto& group : groups) {
        group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    }
    check_ze(zeDeviceGetCommandQueueGroupProperties(device, &count, groups.data()),
             "zeDeviceGetCommandQueueGroupProperties");

    for (uint32_t ordinal = 0; ordinal < count; ++ordinal) {
        if (groups[ordinal].numQueues > 0 && (groups[ordinal].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY)) {
            ze_command_queue_desc_t desc = {};
            desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
            desc.ordinal = ordinal;
            desc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
            ze_command_list_handle_t handle = nullptr;
            check_ze(zeCommandListCreateImmediate(context, device, &desc, &handle), "zeCommandListCreateImmediate");
            return CommandList(handle);
        }
    }
    OPENVINO_THROW("The GPU has no command queue group supporting memory copies");
}

struct HostMemoryDeleter {
    ze_context_handle_t context;
    void operator()(void* pointer) const {
        const auto result = zeMemFree(context, pointer);
        if (result != ZE_RESULT_SUCCESS) {
            std::cerr << "zeMemFree failed with Level Zero error " << result << std::endl;
        }
    }
};

using HostMemory = std::unique_ptr<void, HostMemoryDeleter>;

// Level Zero copies require memory accessible by the device; pageable host memory is not guaranteed to be.
HostMemory allocate_host_memory(ze_context_handle_t context, size_t bytes) {
    ze_host_mem_alloc_desc_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    void* pointer = nullptr;
    check_ze(zeMemAllocHost(context, &desc, bytes, 0, &pointer), "zeMemAllocHost");
    return HostMemory(pointer, HostMemoryDeleter{context});
}

void copy_memory(ze_command_list_handle_t command_list, void* destination, const void* source, size_t bytes) {
    check_ze(zeCommandListAppendMemoryCopy(command_list, destination, source, bytes, nullptr, 0, nullptr),
             "zeCommandListAppendMemoryCopy");
    // Wait explicitly so the data is complete before OpenVINO or the host uses it.
    check_ze(zeCommandListHostSynchronize(command_list, std::numeric_limits<uint64_t>::max()),
             "zeCommandListHostSynchronize");
}

void* get_usm_pointer(const ov::RemoteTensor& tensor) {
    void* pointer = tensor.get_params().at(ov::intel_gpu::mem_handle.name()).as<void*>();
    OPENVINO_ASSERT(pointer != nullptr, "The GPU plugin returned a null USM pointer");
    return pointer;
}

// cl_context query of the Level Zero - OpenCL interoperability (LEO) driver feature.
constexpr cl_context_info cl_l0_context_handle = 0x42B0;

ze_context_handle_t get_level_zero_context(const ov::AnyMap& context_params) {
    const auto type = context_params.at(ov::intel_gpu::context_type.name()).as<ov::intel_gpu::ContextType>();
    // The current GPU API uses the OCL_CONTEXT key also for the native handle of a ZE context.
    void* handle = context_params.at(ov::intel_gpu::ocl_context.name()).as<void*>();
    OPENVINO_ASSERT(handle != nullptr, "The GPU plugin returned a null context handle");

    if (type == ov::intel_gpu::ContextType::ZE) {
        return static_cast<ze_context_handle_t>(handle);
    }
    OPENVINO_ASSERT(type == ov::intel_gpu::ContextType::OCL, "Unsupported GPU context type: ", type);

    // With LEO the GPU plugin runs on Level Zero but its default context is an OpenCL context created
    // on top of the plugin's Level Zero context. This is the only OpenCL call in the sample.
    ze_context_handle_t ze_context = nullptr;
    const cl_int result = clGetContextInfo(static_cast<cl_context>(handle),
                                           cl_l0_context_handle,
                                           sizeof(ze_context),
                                           &ze_context,
                                           nullptr);
    OPENVINO_ASSERT(result == CL_SUCCESS && ze_context != nullptr,
                    "Cannot get a Level Zero context from the GPU plugin's OpenCL context (OpenCL error ",
                    result,
                    "). The driver does not support Level Zero - OpenCL interoperability (LEO).");
    return ze_context;
}

}  // namespace

/**
 * @brief Main with support Unicode paths, wide strings
 */
int tmain(int argc, tchar* argv[]) {
    try {
        // -------- Set OpenVINO log message capturing callback --------
        const std::function<void(std::string_view)> log_callback{[](std::string_view msg) {
            slog::info << msg;
        }};
        ov::util::set_log_callback(log_callback);

        // -------- Get OpenVINO runtime version --------
        slog::info << ov::get_openvino_version() << slog::endl;

        // -------- Parsing and validation of input arguments --------
        if (argc != 4) {
            slog::info << "Usage : " << TSTRING2STRING(argv[0]) << " <path_to_model> <path_to_image> <device_name>"
                       << slog::endl;
            return EXIT_FAILURE;
        }

        const std::string model_path = TSTRING2STRING(argv[1]);
        const std::string image_path = TSTRING2STRING(argv[2]);
        const std::string device_name = TSTRING2STRING(argv[3]);
        OPENVINO_ASSERT(device_name == "GPU" || device_name.rfind("GPU.", 0) == 0,
                        "Select a GPU device directly (e.g. GPU or GPU.0), not CPU, AUTO, MULTI or HETERO");

        // -------- Step 1. Initialize Level Zero and OpenVINO Runtime Core --------
        initialize_level_zero();
        ov::Core core;

        // -------- Step 2. Get the Level Zero context of the GPU plugin --------
        // The default context is ZE, or OCL on drivers with Level Zero - OpenCL interoperability (LEO).
        auto context = core.get_default_context(device_name);
        const auto ze_context = get_level_zero_context(context.get_params());

        // -------- Step 3. Read a model --------
        slog::info << "Loading model files: " << model_path << slog::endl;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        printInputAndOutputsInfo(*model);

        OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
        OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

        // -------- Step 4. Read input image --------
        FormatReader::ReaderPtr reader(image_path.c_str());
        if (reader.get() == nullptr) {
            std::stringstream ss;
            ss << "Image " + image_path + " cannot be read!";
            throw std::logic_error(ss.str());
        }

        ov::element::Type input_type = ov::element::u8;
        ov::Shape input_shape = {1, reader->height(), reader->width(), 3};
        std::shared_ptr<unsigned char> input_data = reader->getData();

        const ov::Layout tensor_layout{"NHWC"};

        // -------- Step 5. Configure preprocessing --------
        ov::preprocess::PrePostProcessor ppp(model);

        // 1) Set input tensor information: shape and precision of the image, layout 'NHWC'
        ppp.input().tensor().set_shape(input_shape).set_element_type(input_type).set_layout(tensor_layout);
        // 2) Apply linear resize from tensor spatial dims to model spatial dims
        ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
        // 3) Suppose model has 'NCHW' layout for input
        ppp.input().model().set_layout("NCHW");
        // 4) Set output tensor precision to 'f32'
        ppp.output().tensor().set_element_type(ov::element::f32);

        // 5) Apply preprocessing modifying the original 'model'
        model = ppp.build();

        // -------- Step 6. Loading a model to the device using the Level Zero context --------
        ov::CompiledModel compiled_model = core.compile_model(model, context);
        OPENVINO_ASSERT(compiled_model.output().get_partial_shape().is_static(),
                        "Sample supports models with static output shape only");

        // -------- Step 7. Create an infer request --------
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        // -------- Step 8. Prepare Level Zero remote tensors --------
        // OpenVINO owns the device allocations; the tensors must outlive all Level Zero accesses to them.
        const ov::AnyMap device_memory{ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER)};
        auto remote_input = context.create_tensor(input_type, input_shape, device_memory);
        auto remote_output =
            context.create_tensor(ov::element::f32, compiled_model.output().get_shape(), device_memory);
        void* input_pointer = get_usm_pointer(remote_input);
        void* output_pointer = get_usm_pointer(remote_output);

        // Query the allocation's device rather than assuming that GPU.0 is the first Level Zero device.
        ze_memory_allocation_properties_t allocation = {};
        allocation.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
        ze_device_handle_t ze_device = nullptr;
        check_ze(zeMemGetAllocProperties(ze_context, input_pointer, &allocation, &ze_device),
                 "zeMemGetAllocProperties");
        OPENVINO_ASSERT(allocation.type == ZE_MEMORY_TYPE_DEVICE && ze_device != nullptr,
                        "Expected a Level Zero device allocation");

        // Upload the image to device memory through a Level Zero command list
        auto input_staging = allocate_host_memory(ze_context, remote_input.get_byte_size());
        auto output_staging = allocate_host_memory(ze_context, remote_output.get_byte_size());
        auto command_list = create_command_list(ze_context, ze_device);
        std::memcpy(input_staging.get(), input_data.get(), remote_input.get_byte_size());
        copy_memory(command_list.get(), input_pointer, input_staging.get(), remote_input.get_byte_size());

        infer_request.set_input_tensor(remote_input);
        infer_request.set_output_tensor(remote_output);

        // -------- Step 9. Do inference synchronously --------
        infer_request.infer();

        // -------- Step 10. Download and process output --------
        copy_memory(command_list.get(), output_staging.get(), output_pointer, remote_output.get_byte_size());
        const ov::Tensor output_tensor(ov::element::f32, remote_output.get_shape(), output_staging.get());

        // Print classification results
        ClassificationResult classification_result(output_tensor, {image_path});
        classification_result.show();
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
