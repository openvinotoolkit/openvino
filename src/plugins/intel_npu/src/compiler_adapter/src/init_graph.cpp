// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "init_graph.hpp"

#include "openvino/core/type/element_iterator.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace intel_npu {

InitGraph::InitGraph(const std::shared_ptr<ZeGraphExtWrappers>& zeGraphExt,
                     const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                     ze_graph_handle_t graphHandle,
                     NetworkMetadata metadata,
                     std::unique_ptr<BlobContainer> blobPtr,
                     const Config& config,
                     const ov::SoPtr<ICompiler>& compiler)
    : Graph(zeGraphExt, zeroInitStruct, graphHandle, metadata, std::move(blobPtr), config, compiler) {}

InitInputData InitGraph::allocateInputs(const std::vector<std::shared_ptr<ov::op::v0::Constant>>& constants,
                                        const ov::SoPtr<ov::IRemoteContext>& context,
                                        const Config& config) {
    std::vector<std::vector<std::shared_ptr<ov::ITensor>>> inputTensors;

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    std::chrono::steady_clock::time_point begin_memcpy;
    std::chrono::steady_clock::time_point end_memcpy;
    std::chrono::steady_clock::time_point begin_tensor_creation;
    std::chrono::steady_clock::time_point end_tensor_creation;
    long long memcpy_duration = 0;

    begin = std::chrono::steady_clock::now();
    size_t initInputsByteSize = 0;

    for (const IODescriptor& descriptor : get_metadata().inputs) {
        initInputsByteSize +=
            ov::element::get_memory_size(descriptor.precision, shape_size(descriptor.shapeFromCompiler.to_shape()));
    }

    begin_tensor_creation = std::chrono::steady_clock::now();
    const ov::SoPtr<ov::ITensor> initInputsTensor = {
        std::make_shared<ZeroHostTensor>(context._ptr,
                                         _zeroInitStruct,
                                         ov::element::Type_t::u8,
                                         ov::Shape({initInputsByteSize}),
                                         ov::intel_npu::TensorType::INPUT)};
    end_tensor_creation = std::chrono::steady_clock::now();
    std::cout
        << "init inputs tensor creation "
        << std::chrono::duration_cast<std::chrono::microseconds>(end_tensor_creation - begin_tensor_creation).count()
        << "[microseconds]" << std::endl;

    size_t offset = 0;
    for (const IODescriptor& descriptor : get_metadata().inputs) {
        const size_t id = std::stoi(descriptor.nameFromCompiler);
        auto currentInputBufferLocation = static_cast<unsigned char*>(initInputsTensor->data()) + offset;
        const size_t currentInputSize =
            ov::element::get_memory_size(descriptor.precision, shape_size(descriptor.shapeFromCompiler.to_shape()));

        OPENVINO_ASSERT(id < constants.size(), "Mismatch between weights IDs and parsed inputs");
        const auto& constant = constants[id];
        OPENVINO_ASSERT(constant->get_byte_size() == currentInputSize,
                        "Byte size mismatch for ",
                        descriptor.nameFromCompiler);
        OPENVINO_ASSERT(constant->get_element_type() == descriptor.precision,
                        "Precision mismatch for ",
                        descriptor.nameFromCompiler);
        OPENVINO_ASSERT(constant->get_shape() == descriptor.shapeFromCompiler.to_shape(),
                        "Shape mismatch for ",
                        descriptor.nameFromCompiler);

        begin_memcpy = std::chrono::steady_clock::now();
        // TODO: should we copy the constant acknowledging strides? (if there
        // are strides, we risk copying bogus data here)
        std::memcpy(currentInputBufferLocation, constant->get_data_ptr(), currentInputSize);
        end_memcpy = std::chrono::steady_clock::now();
        memcpy_duration =
            memcpy_duration + std::chrono::duration_cast<std::chrono::microseconds>(end_memcpy - begin_memcpy).count();

        inputTensors.push_back(
            {ov::make_tensor(constant->get_element_type(), constant->get_shape(), currentInputBufferLocation)});
        offset += currentInputSize;
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Creating input tensors " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
              << "[microseconds]" << std::endl;
    std::cout << "Memcpy duration " << memcpy_duration << "[microseconds]" << std::endl;

    return {inputTensors, initInputsTensor};
}

InitOutputData InitGraph::allocateOutputs(const ov::SoPtr<ov::IRemoteContext>& context, const Config& config) {
    std::vector<std::shared_ptr<ov::ITensor>> outputTensors;
    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> outputTensorsMap;

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    std::chrono::steady_clock::time_point begin_tensor_creation;
    std::chrono::steady_clock::time_point end_tensor_creation;

    begin = std::chrono::steady_clock::now();
    size_t initOutputsByteSize = 0;

    for (const IODescriptor& descriptor : get_metadata().outputs) {
        initOutputsByteSize +=
            ov::element::get_memory_size(descriptor.precision, shape_size(descriptor.shapeFromCompiler.to_shape()));
    }

    begin_tensor_creation = std::chrono::steady_clock::now();
    const ov::SoPtr<ov::ITensor> initOutputsTensor = {
        std::make_shared<ZeroHostTensor>(context._ptr,
                                         _zeroInitStruct,
                                         ov::element::Type_t::u8,
                                         ov::Shape({initOutputsByteSize}),
                                         ov::intel_npu::TensorType::BINDED)};
    end_tensor_creation = std::chrono::steady_clock::now();
    std::cout
        << "init outputs tensor creation "
        << std::chrono::duration_cast<std::chrono::microseconds>(end_tensor_creation - begin_tensor_creation).count()
        << "[microseconds]" << std::endl;

    size_t offset = 0;
    for (const IODescriptor& descriptor : get_metadata().outputs) {
        const auto currentOutputBufferLocation = static_cast<unsigned char*>(initOutputsTensor->data()) + offset;

        const ov::SoPtr<ov::ITensor> hostTensor =
            ov::make_tensor(descriptor.precision, descriptor.shapeFromCompiler.to_shape(), currentOutputBufferLocation);

        outputTensors.push_back(hostTensor._ptr);
        outputTensorsMap.emplace(descriptor.nameFromCompiler, hostTensor._ptr);
        offset +=
            ov::element::get_memory_size(descriptor.precision, shape_size(descriptor.shapeFromCompiler.to_shape()));
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Creating output tensors "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]"
              << std::endl;

    return {outputTensors, initOutputsTensor, outputTensorsMap};
}

void InitGraph::createPipeline(const Config& config,
                               const std::vector<std::vector<std::shared_ptr<ov::ITensor>>>& input_tensors,
                               const std::vector<std::shared_ptr<ov::ITensor>>& output_tensors) {
    _pipeline = std::make_unique<Pipeline>(config, _zeroInitStruct, shared_from_this(), input_tensors, output_tensors);
}

void InitGraph::runPipeline() {
    OPENVINO_ASSERT(_pipeline != nullptr);
    _pipeline->push();
    _pipeline->pull();
    _pipeline = nullptr;
}

}  // namespace intel_npu
