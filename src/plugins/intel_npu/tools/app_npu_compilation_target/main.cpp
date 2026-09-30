// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "intel_npu/npu_private_properties.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"

enum class ModelType { SIMPLE, WEIGHT_SEPARATION, DYNAMIC };

ModelType readModelType() {
    std::string choice;
    std::cout << "Choose model type:\n1 - simple\n2 - weight separation\n3 - dynamic\n";
    if (!(std::cin >> choice)) {
        throw std::runtime_error("Failed to read model type");
    }

    if (choice == "1") {
        return ModelType::SIMPLE;
    }
    if (choice == "2") {
        return ModelType::WEIGHT_SEPARATION;
    }
    if (choice == "3") {
        return ModelType::DYNAMIC;
    }
    throw std::runtime_error("Invalid model type; enter 1, 2, or 3");
}

std::string_view getModelTypeName(ModelType type) {
    switch (type) {
    case ModelType::SIMPLE:
        return "simple";
    case ModelType::WEIGHT_SEPARATION:
        return "weight separation";
    case ModelType::DYNAMIC:
        return "dynamic";
    }
    OPENVINO_THROW("Unknown model type");
}

std::shared_ptr<ov::Model> createModel(ModelType type) {
    if (type == ModelType::WEIGHT_SEPARATION) {
        ov::ParameterVector parameters;
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
        parameters.push_back(input);
        auto add = std::make_shared<ov::op::v1::Add>(
            input,
            ov::op::v0::Constant::create(ov::element::f16, ov::Shape{50}, {0.5f}));

        for (std::size_t i = 0; i < 10; ++i) {
            auto intermediate_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
            parameters.push_back(intermediate_input);
            auto weights = ov::op::v0::Constant::create(
                ov::element::f16, ov::Shape{10, 10, 20, 50}, {static_cast<float>(i)});
            // The one-shot exporter uses this metadata to extract constants into weight-init schedules.
            weights->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
                ov::WeightlessCacheAttribute(weights->get_byte_size(), i, weights->get_element_type());
            add = std::make_shared<ov::op::v1::Add>(add, intermediate_input);
            add = std::make_shared<ov::op::v1::Add>(add, weights);
        }
        return std::make_shared<ov::Model>(ov::OutputVector{add}, parameters);
    }

    if (type == ModelType::DYNAMIC) {
        auto input = std::make_shared<ov::op::v0::Parameter>(
            ov::element::f16,
            ov::PartialShape{1, 16, ov::Dimension(10, 1280), ov::Dimension(10, 1280)});
        input->set_friendly_name("input1");
        input->get_output_tensor(0).set_names({"input1"});
        input->set_layout("NCHW");
        auto max_pool = std::make_shared<ov::op::v1::MaxPool>(input,
                                                              ov::Strides{1, 1},
                                                              ov::Shape{0, 0},
                                                              ov::Shape{0, 0},
                                                              ov::Shape{1, 1},
                                                              ov::op::RoundingType::FLOOR,
                                                              ov::op::PadType::EXPLICIT);
        max_pool->set_friendly_name("MaxPool_2");
        auto result = std::make_shared<ov::op::v0::Result>(max_pool);
        result->set_friendly_name("output");
        result->get_output_tensor(0).set_names({"output"});
        result->set_layout("NCHW");
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "MaxPool");
    }

    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto relu = std::make_shared<ov::op::v0::Relu>(input);
    return std::make_shared<ov::Model>(ov::OutputVector{relu}, ov::ParameterVector{input});
}

ov::Shape makeInferenceInputShape(const ov::PartialShape& partialShape, ModelType type, std::size_t inferenceIndex) {
    if (type == ModelType::DYNAMIC) {
        // These non-minimum shapes exercise two VM runs without triggering its small-shape backtracking failure.
        return inferenceIndex == 0 ? ov::Shape{1, 16, 720, 1280} : ov::Shape{1, 16, 720, 720};
    }

    ov::Shape shape;
    shape.reserve(partialShape.size());
    for (const auto& dimension : partialShape) {
        auto length = std::max<std::int64_t>(dimension.get_min_length(), 1);
        if (!dimension.is_static() && inferenceIndex > 0 && length < dimension.get_max_length()) {
            ++length;
        }
        shape.push_back(static_cast<std::size_t>(length));
    }
    return shape;
}

void interrogateOfflineCompilationTargets(ov::Core& core,
                                         const std::string& device,
                                         const std::vector<ov::CompilationTarget>& targets) {
    std::cout << "Offline compilation targets for " << device << " (" << targets.size() << "):" << std::endl;
    for (const auto& target : targets) {
        std::cout << "  " << target << std::endl;
    }

    // This platform argument demonstrates the NPU plugin's filtered-enumeration extension.
    const auto filtered_targets =
        core.get_property(device,
                          ov::offline_compilation_targets,
                          {{ov::intel_npu::platform.name(), std::string(ov::intel_npu::Platform::NPU6010)}});
    std::cout << "Offline compilation targets filtered by platform " << ov::intel_npu::Platform::NPU6010 << " ("
              << filtered_targets.size() << "):" << std::endl;
    for (const auto& target : filtered_targets) {
        std::cout << "  " << target << std::endl;
    }
}

int main(int argc, char* argv[]) {
    const std::string device = argc > 1 ? argv[1] : "NPU";
    try {
        ov::Core core;
        const auto offline_targets = core.get_property(device, ov::offline_compilation_targets);

        std::string action;
        std::cout << "Choose action:\n0 - interrogate offline compilation targets\n1 - compile and run a model\n";
        if (!(std::cin >> action)) {
            throw std::runtime_error("Failed to read action");
        }
        if (action == "0") {
            interrogateOfflineCompilationTargets(core, device, offline_targets);
            return 0;
        }
        if (action != "1") {
            throw std::runtime_error("Invalid action; enter 0 or 1");
        }

        std::string platform;
        std::cout << "Enter the target platform (e.g. 3720, 4000, 5010, 6010): ";
        if (!(std::cin >> platform)) {
            throw std::runtime_error("Failed to read NPU platform number");
        }

        // Check if target is available in the offline compilation targets
        const auto compile_target =
            std::find_if(offline_targets.begin(), offline_targets.end(), [&platform](const auto& target) {
                return target.platform == platform;
            });
        if (compile_target == offline_targets.end()) {
            throw std::runtime_error("Platform " + platform + " is not present in the offline compilation targets");
        }

        const auto model_type = readModelType();
        const auto model_name = getModelTypeName(model_type);
        const auto model = createModel(model_type);
        ov::AnyMap compile_properties;
        compile_properties.emplace(ov::compilation_target.name(), *compile_target);
        std::size_t standard_blob_size = 0;
        if (model_type == ModelType::WEIGHT_SEPARATION) {
            auto standard_compiled_model = core.compile_model(model, device, compile_properties);
            std::stringstream standard_blob;
            standard_compiled_model.export_model(standard_blob);
            standard_blob_size = standard_blob.str().size();
            std::cout << "Standard model blob size: " << standard_blob_size << " bytes" << std::endl;
            compile_properties.emplace(ov::enable_weightless.name(), true);
        }

        // Compile the model for the selected target
        std::cout << "Compiling " << model_name << " model for " << compile_target->platform << std::endl;
        auto compiled_model = core.compile_model(model, device, compile_properties);
        std::cout << model_name << " model compiled successfully for " << compile_target->platform << std::endl;

        const auto compiled_properties = compiled_model.get_property(ov::supported_properties);
        if (std::find(compiled_properties.begin(), compiled_properties.end(), ov::runtime_requirements) !=
            compiled_properties.end()) {
            std::cout << "#####################################################" << std::endl;
            const auto requirements = compiled_model.get_property(ov::runtime_requirements);
            std::cout << "Runtime requirements: " << requirements << std::endl;
            std::cout << "Compatibility check against " << device << ": "
                      << core.get_property(device,
                                           ov::compatibility_check,
                                           {{ov::runtime_requirements.name(), requirements}})
                      << std::endl;
            std::cout << "#####################################################" << std::endl;
        } else {
            std::cout << "Runtime requirements and compatibility check are unavailable for this compiled model."
                      << std::endl;
        }

        // Exercise the export/import/infer
        std::stringstream blob;
        compiled_model.export_model(blob);
        std::cout << "Model exported successfully for " << compile_target->platform << std::endl;
        if (model_type == ModelType::WEIGHT_SEPARATION) {
            const auto weight_separated_blob_size = blob.str().size();
            std::cout << "Weight-separated blob size: " << weight_separated_blob_size << " bytes" << std::endl;
            if (weight_separated_blob_size >= standard_blob_size) {
                throw std::runtime_error("Weight-separated blob is not smaller than the standard model blob");
            }
            std::cout << "Verified: weight separation reduced the exported blob size." << std::endl;
        }

        try {
            auto imported_model = model_type == ModelType::WEIGHT_SEPARATION
                                      ? core.import_model(blob, device, {ov::hint::model(model)})
                                      : core.import_model(blob, device);
            std::cout << "Model imported successfully for " << compile_target->platform << std::endl;

            // Exercise the inference
            try {
                if (model_type == ModelType::DYNAMIC) {
                    const auto device_architecture = core.get_property(device, ov::device::architecture);
                    if (device_architecture != compile_target->platform) {
                        throw std::runtime_error("Dynamic inference requires an attached NPU with platform " +
                                                 compile_target->platform + "; detected " + device_architecture);
                    }
                }
                auto infer_request = imported_model.create_infer_request();
                for (std::size_t inferenceIndex = 0; inferenceIndex < 2; ++inferenceIndex) {
                    for (std::size_t inputIndex = 0; inputIndex < model->inputs().size(); ++inputIndex) {
                        const auto input = model->input(inputIndex);
                        const auto input_shape = makeInferenceInputShape(input.get_partial_shape(), model_type, inferenceIndex);
                        ov::Tensor input_tensor(input.get_element_type(), input_shape);
                        std::memset(input_tensor.data(), 0, input_tensor.get_byte_size());
                        infer_request.set_input_tensor(inputIndex, input_tensor);
                    }
                    infer_request.infer();
                    std::cout << "Inference " << inferenceIndex + 1 << " succeeded for " << compile_target->platform
                              << std::endl;
                }
            } catch (const std::exception& ex) {
                std::cout << "NPU inference failed: " << ex.what() << std::endl;
            }
        } catch (const std::exception& ex) {
            std::cout << "Model import failed: " << ex.what() << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
