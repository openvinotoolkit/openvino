// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remote_tensors_filling.hpp"

#include <memory>
#include <random>
#include <samples/slog.hpp>
#include <string>
#include <utility>
#include <vector>

#ifdef HAVE_GPU_DEVICE_MEM_SUPPORT
#    include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#    include <openvino/runtime/intel_gpu/ocl/ocl_wrapper.hpp>
#endif

#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#include <openvino/runtime/intel_npu/remote_properties.hpp>

namespace {

template <typename T>
using uniformDistribution = typename std::conditional<
    std::is_floating_point<T>::value,
    std::uniform_real_distribution<T>,
    typename std::conditional<std::is_integral<T>::value, std::uniform_int_distribution<T>, void>::type>::type;

template <typename T, typename T2>
void fill_buffer_random(void* inputBuffer,
                        size_t elementsNum,
                        T rand_min = std::numeric_limits<uint8_t>::min(),
                        T rand_max = std::numeric_limits<uint8_t>::max()) {
    std::mt19937 gen(0);
    uniformDistribution<T2> distribution(rand_min, rand_max);
    auto inputBufferData = static_cast<T*>(inputBuffer);
    for (size_t i = 0; i < elementsNum; i++) {
        inputBufferData[i] = static_cast<T>(distribution(gen));
    }
}

void fill_buffer(void* inputBuffer, size_t elementsNum, const ov::element::Type& type) {
    if (type == ov::element::f32) {
        fill_buffer_random<float, float>(inputBuffer, elementsNum);
    } else if (type == ov::element::f64) {
        fill_buffer_random<double, double>(inputBuffer, elementsNum);
    } else if (type == ov::element::f16) {
        fill_buffer_random<ov::float16, float>(inputBuffer, elementsNum);
    } else if (type == ov::element::i32) {
        fill_buffer_random<int32_t, int32_t>(inputBuffer, elementsNum);
    } else if (type == ov::element::i64) {
        fill_buffer_random<int64_t, int64_t>(inputBuffer, elementsNum);
    } else if (type == ov::element::u8) {
        // uniform_int_distribution<uint8_t> is not allowed in the C++17
        // standard and vs2017/19
        fill_buffer_random<uint8_t, uint32_t>(inputBuffer, elementsNum);
    } else if (type == ov::element::i8) {
        // uniform_int_distribution<int8_t> is not allowed in the C++17 standard
        // and vs2017/19
        fill_buffer_random<int8_t, int32_t>(inputBuffer, elementsNum);
    } else if (type == ov::element::u16) {
        fill_buffer_random<uint16_t, uint16_t>(inputBuffer, elementsNum);
    } else if (type == ov::element::i16) {
        fill_buffer_random<int16_t, int16_t>(inputBuffer, elementsNum);
    } else if (type == ov::element::boolean) {
        fill_buffer_random<uint8_t, uint32_t>(inputBuffer, elementsNum, 0, 1);
    } else {
        OPENVINO_THROW("Requested type is not supported");
    }
}
}  // namespace

namespace gpu {

std::map<std::string, ov::TensorVector> get_remote_input_tensors(
    const std::map<std::string, std::vector<std::string>>& inputFiles,
    const std::vector<benchmark_app::InputsInfo>& app_inputs_info,
    const ov::CompiledModel& compiledModel,
    std::vector<BufferType>& clBuffer,
    size_t num_requests) {
#ifdef HAVE_GPU_DEVICE_MEM_SUPPORT
    slog::info << "Device memory will be used for input and output blobs" << slog::endl;
    if (inputFiles.size()) {
        slog::warn << "Device memory supports only random data at this moment, input images will be ignored"
                   << slog::endl;
    }

    std::map<std::string, ov::TensorVector> remoteTensors;
    auto context = compiledModel.get_context();
    auto& oclContext = static_cast<ov::intel_gpu::ocl::ClContext&>(context);
    auto oclInstance = std::make_shared<gpu::OpenCL>(oclContext.get());

    for (size_t i = 0; i < num_requests; i++) {
        for (auto& inputs_info : app_inputs_info) {
            for (auto& input : inputs_info) {
                // Fill random
                slog::info << "Prepare remote blob for input '" << input.first << "' with random values ("
                           << std::string((input.second.is_image() ? "image" : "some binary data")) << " is expected)"
                           << slog::endl;

                // Creating and filling shared buffers
                cl_int err;
                auto elementsNum = std::accumulate(begin(input.second.dataShape),
                                                   end(input.second.dataShape),
                                                   1,
                                                   std::multiplies<size_t>());
                auto inputSize = elementsNum * input.second.type.bitwidth() / 8;

                clBuffer.push_back(
                    cl::Buffer(oclInstance->_context, CL_MEM_READ_WRITE, (cl::size_type)inputSize, NULL, &err));

                void* mappedPtr = oclInstance->_queue.enqueueMapBuffer(clBuffer.back(),
                                                                       CL_TRUE,
                                                                       CL_MEM_READ_WRITE,
                                                                       0,
                                                                       (cl::size_type)inputSize);

                auto tensor =
                    oclContext.create_tensor(input.second.type, input.second.dataShape, clBuffer.back().get());
                remoteTensors[input.first].push_back(tensor);

                if (inputFiles.empty()) {
                    // Filling in random data
                    fill_buffer(mappedPtr, elementsNum, input.second.type);
                } else {
                    // TODO: add filling with real image data
                }
                oclInstance->_queue.enqueueUnmapMemObject(clBuffer.back(), mappedPtr);
            }
        }
    }
    return remoteTensors;
#else
    OPENVINO_THROW("Device memory requested for GPU device, but OpenCL was not linked");
#endif
}

ov::Shape get_static_shape(const ov::Output<const ov::Node>& compiled_output) {
    // FIXME: this is a WA for case when original model has internal dynamism (NonMaxSuppression)
    // and runtime has static output due to conversions to legacy op and lack of dynamism support
    // to be removed along with dynamism support
    const auto& compiled_pshape = compiled_output.get_partial_shape();
    if (compiled_pshape.is_static())
        return compiled_pshape.to_shape();
    else if (compiled_pshape.rank().is_dynamic())
        OPENVINO_THROW("Benchmark App - NOT IMPLEMENTED - Output of dynamic rank is not supported for remote tensor. ",
                       "Output: ",
                       compiled_output);
    ov::Shape shape;
    for (const auto& dimension : compiled_pshape) {
        if (dimension.get_interval().has_upper_bound())
            shape.push_back(static_cast<ov::Shape::value_type>(dimension.get_max_length()));
        else
            OPENVINO_THROW("Benchmark App - NOT IMPLEMENTED - Fully dynamic output dimensions are not supported "
                           "for remote tensor. ",
                           "Output: ",
                           compiled_output);
    }
    return shape;
}

std::map<std::string, ov::Tensor> get_remote_output_tensors(const ov::CompiledModel& compiledModel,
                                                            std::map<std::string, ::gpu::BufferType>& clBuffer) {
#ifdef HAVE_GPU_DEVICE_MEM_SUPPORT
    std::map<std::string, ov::Tensor> outputTensors;
    std::shared_ptr<const ov::Model> runtime_model = nullptr;
    for (auto& output : compiledModel.outputs()) {
        auto context = compiledModel.get_context();
        auto& oclContext = static_cast<ov::intel_gpu::ocl::ClContext&>(context);
        auto oclInstance = std::make_shared<OpenCL>(oclContext.get());
        ov::Shape shape = get_static_shape(output);
        cl_int err;
        auto elementsNum = shape_size(shape);
        auto inputSize = elementsNum * output.get_element_type().bitwidth() / 8;

        cl::size_type bufferSize = 0;
        if (clBuffer.find(output.get_any_name()) == clBuffer.end()) {
            clBuffer[output.get_any_name()] =
                cl::Buffer(oclInstance->_context, CL_MEM_READ_WRITE, (cl::size_type)inputSize, NULL, &err);
        } else {
            auto& buff = clBuffer[output.get_any_name()];
            buff.getInfo(CL_MEM_SIZE, &bufferSize);
            if (inputSize != bufferSize) {
                buff = cl::Buffer(oclInstance->_context, CL_MEM_READ_WRITE, (cl::size_type)inputSize, NULL, &err);
            }
        }
        outputTensors[output.get_any_name()] =
            oclContext.create_tensor(output.get_element_type(), shape, clBuffer[output.get_any_name()].get());
    }

    return outputTensors;
#else
    OPENVINO_THROW("Device memory requested for GPU device, but OpenCL was not linked");
#endif
}
}  // namespace gpu

namespace npu {

std::map<std::string, ov::TensorVector> get_remote_input_tensors(
    const std::map<std::string, std::vector<std::string>>& inputFiles,
    const std::vector<benchmark_app::InputsInfo>& app_inputs_info,
    const ov::CompiledModel& compiledModel,
    size_t num_requests) {
    slog::info << "Device memory will be used for input blobs" << slog::endl;

    std::map<std::string, ov::TensorVector> remoteTensors;
    auto context = compiledModel.get_context();
    auto& zeroContext = static_cast<ov::intel_npu::level_zero::ZeroContext&>(context);

    for (size_t i = 0; i < num_requests; i++) {
        for (auto& inputs_info : app_inputs_info) {
            for (auto& input : inputs_info) {
                auto tensor = zeroContext.create_l0_host_tensor(input.second.type,
                                                                input.second.dataShape,
                                                                ov::intel_npu::TensorType::INPUT);
                remoteTensors[input.first].push_back(tensor);

                if (inputFiles.empty()) {
                    // Filling in random data
                    slog::info << "Prepare remote blob for input '" << input.first << "' with random values ("
                               << std::string((input.second.is_image() ? "image" : "some binary data"))
                               << " is expected)" << slog::endl;

                    auto elementsNum = std::accumulate(begin(input.second.dataShape),
                                                       end(input.second.dataShape),
                                                       1,
                                                       std::multiplies<size_t>());

                    fill_buffer(tensor.get(), elementsNum, input.second.type);
                } else {
                    OPENVINO_THROW(
                        "[NPU] Device memory supports only random data at this moment, input images will be ignored");
                }
            }
        }
    }
    return remoteTensors;
}
}  // namespace npu
