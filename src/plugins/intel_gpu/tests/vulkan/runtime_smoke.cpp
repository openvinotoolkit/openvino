// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

#include "openvino/core/except.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

namespace {

bool is_gpu_device(const std::string& device) {
    return device == "GPU" || device.rfind("GPU.", 0) == 0;
}

void check_no_driver(ov::Core& core) {
    OPENVINO_ASSERT(core.get_property("GPU", ov::available_devices).empty(), "Vulkan reported a device without a driver");
    const auto devices = core.get_available_devices();
    OPENVINO_ASSERT(std::none_of(devices.begin(), devices.end(), is_gpu_device), "The Core device list contains a GPU without a driver");

    try {
        core.get_property("GPU", ov::device::full_name);
    } catch (const ov::Exception& error) {
        OPENVINO_ASSERT(std::string_view(error.what()).find("no supported devices found") != std::string_view::npos,
                        "Expected a diagnostic about unavailable GPU devices: ",
                        error.what());
        return;
    }
    OPENVINO_THROW("A device property query succeeded without a Vulkan device");
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        ov::Core core;
        if (argc == 2 && std::string_view(argv[1]) == "--expect-no-driver") {
            check_no_driver(core);
            return 0;
        }
        const auto devices = core.get_available_devices();
        const auto gpu = std::find_if(devices.begin(), devices.end(), is_gpu_device);
        OPENVINO_ASSERT(gpu != devices.end(), "No Vulkan GPU detected");
        const auto runtime = core.get_property(*gpu, "GPU_RUNTIME_TYPE").as<std::string>();
        OPENVINO_ASSERT(runtime == "VULKAN", "Expected the Vulkan runtime");
        OPENVINO_ASSERT(core.get_property(*gpu, ov::device::capabilities).empty(), "Vulkan must not advertise model execution or caching");
        const auto properties = core.get_property(*gpu, ov::supported_properties);
        for (const auto* name : {ov::optimal_batch_size.name(), ov::max_batch_size.name(), "CONFIG_FILE"}) {
            OPENVINO_ASSERT(std::find(properties.begin(), properties.end(), name) == properties.end(),
                            "Vulkan advertised an unsupported model property: ",
                            name);
        }
        OPENVINO_ASSERT(core.get_property(*gpu, ov::compatibility_check) == ov::CompatibilityCheck::UNSUPPORTED,
                        "Vulkan must report model compatibility as unsupported");

        const auto tracked_buffer_bytes = [&] {
            const auto statistics = core.get_property(*gpu, ov::intel_gpu::memory_statistics);
            const auto device_buffer = statistics.find("device_buffer");
            OPENVINO_ASSERT(device_buffer != statistics.end(), "Vulkan memory statistics omit device_buffer");
            return device_buffer->second;
        };
        OPENVINO_ASSERT(tracked_buffer_bytes() == 0, "Vulkan reported device-buffer bytes before context creation");

        auto context = core.get_default_context(*gpu);
        OPENVINO_ASSERT(tracked_buffer_bytes() == 0, "Vulkan reported device-buffer bytes before tensor creation");
        for (const auto& shape : {ov::Shape{}, ov::Shape{256}, ov::Shape{257}}) {
            const auto count = ov::shape_size(shape);
            ov::Tensor input(ov::element::u8, shape);
            ov::Tensor output(ov::element::u8, shape);
            auto* bytes = input.data<uint8_t>();
            for (size_t i = 0; i < count; ++i) {
                bytes[i] = static_cast<uint8_t>(i ^ 0xA5);
            }
            {
                auto source = context.create_tensor(ov::element::u8, shape);
                auto destination = context.create_tensor(ov::element::u8, shape);
                OPENVINO_ASSERT(tracked_buffer_bytes() == 2 * count, "Vulkan reported incorrect live device-buffer bytes");
                source.copy_from(input);
                source.copy_to(destination);
                destination.copy_to(output);
            }
            OPENVINO_ASSERT(tracked_buffer_bytes() == 0, "Vulkan reported device-buffer bytes after tensor release");
            OPENVINO_ASSERT(std::equal(bytes, bytes + count, output.data<const uint8_t>()), "Vulkan buffer round trip changed the data");
        }

        bool string_rejected = false;
        try {
            context.create_tensor(ov::element::string, {1});
        } catch (const ov::Exception& error) {
            string_rejected = std::string_view(error.what()).find("String tensors are unsupported") != std::string_view::npos;
        }
        OPENVINO_ASSERT(string_rejected, "Vulkan must reject string tensors");

        {
            const ov::Shape original_shape{3};
            auto remote = context.create_tensor(ov::element::u8, original_shape);
            ov::Tensor input(ov::element::u8, original_shape);
            std::fill_n(input.data<uint8_t>(), input.get_size(), 42);
            remote.copy_from(input);
            const auto limit = core.get_property(*gpu, ov::intel_gpu::device_max_alloc_mem_size);
            bool resize_rejected = false;
            try {
                remote.set_shape({static_cast<size_t>(limit + 1)});
            } catch (const ov::Exception&) {
                resize_rejected = true;
            }
            OPENVINO_ASSERT(resize_rejected, "Vulkan must reject a resize beyond the allocation limit");
            OPENVINO_ASSERT(remote.get_shape() == original_shape, "Rejected resize changed the tensor shape");
            remote.set_shape(original_shape);
            ov::Tensor output(ov::element::u8, original_shape);
            remote.copy_to(output);
            OPENVINO_ASSERT(std::equal(input.data<uint8_t>(), input.data<uint8_t>() + input.get_size(), output.data<uint8_t>()),
                            "Rejected resize changed the tensor contents");
            remote.set_shape({5});
            OPENVINO_ASSERT(remote.get_shape() == ov::Shape{5}, "Vulkan tensor did not grow after a rejected resize");
        }

        const ov::Shape shape{257};
        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto add = std::make_shared<ov::op::v1::Add>(parameter, parameter);
        auto model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{parameter});
        OPENVINO_ASSERT(core.query_model(model, *gpu).empty(), "The Vulkan runtime reported inference support");
        bool rejected = false;
        try {
            core.compile_model(model, *gpu);
        } catch (const ov::Exception& error) {
            rejected = std::string_view(error.what()).find("Model compilation is unavailable") != std::string_view::npos;
        }
        OPENVINO_ASSERT(rejected, "The Vulkan runtime must explicitly reject model compilation");

        std::cout << "Vulkan discovery and buffer round trip passed on " << core.get_property(*gpu, ov::device::full_name) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
