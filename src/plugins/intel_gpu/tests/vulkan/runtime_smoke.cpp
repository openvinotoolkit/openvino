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

int main() {
    try {
        ov::Core core;
        const auto devices = core.get_available_devices();
        const auto gpu = std::find_if(devices.begin(), devices.end(), [](const auto& device) {
            return device == "GPU" || device.rfind("GPU.", 0) == 0;
        });
        OPENVINO_ASSERT(gpu != devices.end(), "No Vulkan GPU detected");
        const auto runtime = core.get_property(*gpu, "GPU_RUNTIME_TYPE").as<std::string>();
        OPENVINO_ASSERT(runtime == "VULKAN", "Expected the Vulkan runtime");

        const auto tracked_buffer_bytes = [&] {
            const auto statistics = core.get_property(*gpu, ov::intel_gpu::memory_statistics);
            const auto device_buffer = statistics.find("device_buffer");
            OPENVINO_ASSERT(device_buffer != statistics.end(), "Vulkan memory statistics omit device_buffer");
            return device_buffer->second;
        };
        OPENVINO_ASSERT(tracked_buffer_bytes() == 0, "Vulkan reported device-buffer bytes before context creation");

        auto context = core.get_default_context(*gpu);
        OPENVINO_ASSERT(tracked_buffer_bytes() == 0, "Vulkan reported device-buffer bytes before tensor creation");
        for (const size_t count : {256, 257}) {
            const ov::Shape shape{count};
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
