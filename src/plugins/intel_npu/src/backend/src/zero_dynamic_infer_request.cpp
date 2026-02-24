// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_dynamic_infer_request.hpp"

#include "intel_npu/common/itt.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/utils.hpp"
#include "openvino/runtime/make_tensor.hpp"

using namespace intel_npu;

ZeroDynamicInferRequest::ZeroDynamicInferRequest(const std::shared_ptr<ZeroInitStructsHolder>& initStructs,
                                                 const std::shared_ptr<const ICompiledModel>& compiledModel,
                                                 const Config& config)
    : ZeroInferRequest(initStructs, compiledModel, config) {
    _logger.setName("ZeroDynamicInferRequest");
}

void ZeroDynamicInferRequest::construct_pipeline() {
    _logger.debug("create_pipeline - constructing pipeline");
    auto batchSize = _graph->get_batch_size();
    // Construct pipeline
    _pipeline =
        std::make_unique<DynamicPipeline>(_config,
                                          _initStructs,
                                          _graph,
                                          _levelZeroInputTensors,
                                          _levelZeroOutputTensors,
                                          batchSize.has_value() ? batchSize.value() : utils::DEFAULT_BATCH_SIZE);

    _logger.debug("create_pipeline - completed");
}

void ZeroDynamicInferRequest::update_command_list_for_tensor(SyncInferRequest::FoundPort& foundPort,
                                                             const ov::SoPtr<ov::ITensor>& tensor) {
    OV_ITT_TASK_CHAIN(ZERO_SET_TENSOR, itt::domains::LevelZeroBackend, "set_tensor", "update_command_list_for_tensor");
    if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0)) {
        auto& levelZeroTensor =
            foundPort.is_input() ? get_level_zero_input(foundPort.idx) : _levelZeroOutputTensors.at(foundPort.idx);

        try {
            _logger.debug("set_tensor - create zero tensor");
            OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "create zero tensor");
            // Try to use the user tensor directly if its underlying data is already allocated in the same Level Zero
            // context.
            levelZeroTensor = std::make_shared<ZeroTensor>(_initStructs, _config, tensor);
        } catch (const ZeroMemException& exception) {
            _logger.debug("set_tensor - exception caught while trying to create a Level Zero tensor "
                          "from the user tensor: %s",
                          exception.what());

            // Check if the current Level Zero tensor was previously shared with the user. If so, it cannot be reused;
            // allocate a new tensor to back up the user tensor (which cannot be imported or used directly).
            if (_dynamicBatchValueChanged || levelZeroTensor == nullptr || !levelZeroTensor->can_be_reused() ||
                (levelZeroTensor != nullptr && (levelZeroTensor->get_byte_size() < tensor->get_byte_size()))) {
                _logger.debug("set_tensor - allocate locally L0 tensor");
                OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "allocate tensor");

                auto batch = _graph->get_batch_size();
                levelZeroTensor = allocate_tensor(foundPort.idx, foundPort.is_input(), batch);
            } else {
                _logger.debug("set_tensor - reusing the level zero tensor since it is not shared "
                              "with the user, and old L0 tensor is large enough");
            }
        }

        if (_pipelineIsCreated && !_dynamicBatchValueChanged) {
            _logger.debug("infer_async - update command list");

            OPENVINO_ASSERT(levelZeroTensor->data(), "Empty buffer");

            OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "update_graph_arguments");
            if (levelZeroTensor->get_byte_size() != tensor->get_byte_size()) {
                _pipeline->update_graph_arguments(foundPort.is_input()
                                                      ? _metadata.inputs.at(foundPort.idx).indexUsedByDriver
                                                      : _metadata.outputs.at(foundPort.idx).indexUsedByDriver,
                                                  levelZeroTensor,
                                                  tensor._ptr);
                _isTensorChanged = true;
            } else {
                // This L0 tensor shall have same info with user tensor
                _pipeline->update_graph_arguments(foundPort.is_input()
                                                      ? _metadata.inputs.at(foundPort.idx).indexUsedByDriver
                                                      : _metadata.outputs.at(foundPort.idx).indexUsedByDriver,
                                                  levelZeroTensor);
            }
        }
    }
    if (!_pipelineIsCreated) {
        // If pipeline is not created, need to predict real output shape
        _isTensorChanged = true;
    }
    // If command list updates are not supported, fallback to copying tensors every time.
}

void ZeroDynamicInferRequest::update_command_list_for_tensors(SyncInferRequest::FoundPort& foundPort,
                                                              const std::vector<ov::SoPtr<ov::ITensor>>& tensors,
                                                              std::optional<size_t> batchSizeCandidate) {
    OV_ITT_TASK_CHAIN(ZERO_SET_TENSORS,
                      itt::domains::LevelZeroBackend,
                      "set_tensors",
                      "update_command_list_for_tensors");
    if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0) && batchSizeCandidate.has_value()) {
        get_level_zero_inputs(foundPort.idx).resize(tensors.size());

        for (size_t i = 0; i < tensors.size(); i++) {
            try {
                _logger.debug("set_tensors - create zero tensor");
                OV_ITT_TASK_NEXT(ZERO_SET_TENSORS, "create zero tensor");
                get_level_zero_input(foundPort.idx, i) =
                    std::make_shared<ZeroTensor>(_initStructs, _config, tensors.at(i));
            } catch (const ZeroMemException& exception) {
                _logger.debug("set_tensors - exception caught while trying to create a Level "
                              "Zero tensor "
                              "from the user tensor: %s",
                              exception.what());

                _logger.debug("set_tensors - allocate locally L0 tensor");
                OV_ITT_TASK_NEXT(ZERO_SET_TENSORS, "allocate tensor");
                get_level_zero_input(foundPort.idx, i) = allocate_tensor(foundPort.idx, INPUT, batchSizeCandidate);
            }

            if (_pipelineIsCreated && !_dynamicBatchValueChanged) {
                OPENVINO_ASSERT(get_level_zero_input(foundPort.idx, i)->data(), "Empty buffer");
                OV_ITT_TASK_NEXT(ZERO_SET_TENSORS, "updateCommandList");
                _pipeline->update_graph_arguments(_metadata.inputs.at(foundPort.idx).indexUsedByDriver,
                                                  get_level_zero_input(foundPort.idx, i),
                                                  i,
                                                  tensors.at(i)._ptr);
                _isTensorChanged = true;
            }
        }
    }
    if (!_pipelineIsCreated) {
        // If pipeline is not created, need to predict real output shape
        _isTensorChanged = true;
    }
    // If command list updates are not supported, fallback to copying tensors every time.
}

std::shared_ptr<ZeroTensor> ZeroDynamicInferRequest::allocate_tensor(const size_t index,
                                                                     const bool isInput,
                                                                     const std::optional<std::size_t> batchSize) const {
    IODescriptor descriptor = isInput ? _metadata.inputs.at(index) : _metadata.outputs.at(index);
    // Create new IODescriptor based on user input|output and descriptor
    if (isInput && get_user_input(index) != nullptr) {
        _logger.debug("Update input descriptor with shape from user input : %s instead of %s",
                      get_user_input(index)->get_shape().to_string().c_str(),
                      descriptor.shapeFromCompiler.to_string().c_str());
        descriptor.shapeFromCompiler = get_user_input(index)->get_shape();
    } else if (!isInput && _userOutputTensors.at(index) != nullptr) {
        _logger.debug("Update output descriptor with shape from user output : %s instead of %s",
                      _userOutputTensors.at(index)->get_shape().to_string().c_str(),
                      descriptor.shapeFromCompiler.to_string().c_str());
        descriptor.shapeFromCompiler = _userOutputTensors.at(index)->get_shape();
    }

    check_network_precision(descriptor.precision);

    ov::Shape allocatedTensorShape = descriptor.shapeFromCompiler.get_max_shape();

    if (batchSize.has_value()) {
        allocatedTensorShape[utils::BATCH_AXIS] = *batchSize;
    }

    auto tensor =
        std::make_shared<ZeroTensor>(_initStructs, _config, descriptor.precision, allocatedTensorShape, isInput);

    if (isInput) {
        if (get_user_input(index) == nullptr) {
            get_user_input(index) = tensor;
        }
    } else if (_userOutputTensors.at(index) == nullptr) {
        _userOutputTensors.at(index) = tensor;
    }

    return tensor;
}

void ZeroDynamicInferRequest::infer_async() {
    _logger.debug("infer_async started");
    OV_ITT_TASK_CHAIN(ZERO_INFER, itt::domains::LevelZeroBackend, "infer_async", "start");
    // Store the predicted output shapes
    std::vector<IDynamicGraph::MemRefType> outputPros;
    predict_shapes(outputPros);
    check_tensor_and_predicted_shapes(outputPros);
    prepare_inputs();
    update_tensor(outputPros);

    OV_ITT_TASK_NEXT(ZERO_INFER, "push");
    _pipeline->push();
}

void ZeroDynamicInferRequest::predict_shapes(std::vector<IDynamicGraph::MemRefType>& outputProps) {
    // TODO: If current output tensor is not large enough to be compatible with input tensor, need recreate pipeline
    // But reshape ZeroTensor can be used to avoid recreate pipeline now
    // bool reCreatePipeline = false;
    // Predict output shapes based on current inputs
    intel_npu::IDynamicGraph* dynamicGraph = dynamic_cast<intel_npu::IDynamicGraph*>(_graph.get());
    if (dynamicGraph && _isTensorChanged) {
        IDynamicGraph::GraphArguments graphArgs;
        // Need change to use arguments in pipeline
        dynamicGraph->getBinding(graphArgs);
        std::vector<IDynamicGraph::MemRefType> inputPros = graphArgs._inputs;
        outputProps = graphArgs._outputs;

        // TODO: Support Batch later
        // Update input Info
        // TENTATIVE CODE TO ALLOCATE a memref handle
        for (size_t i = 0; i < inputPros.size(); ++i) {
            auto& levelZeroTensor = get_level_zero_input(i);
            auto& userTensor = get_user_input(i);
            if (userTensor != nullptr) {
                // If userTensor is set, use userTensor to update memref handle
                inputPros[i].set(get_tensor_data_ptr(userTensor._ptr), 0, userTensor._ptr);
            } else if (levelZeroTensor != nullptr) {
                // If userTensor is not set, use levelZeroTensor to update memref handle
                inputPros[i].set(get_tensor_data_ptr(levelZeroTensor), 0, levelZeroTensor);
            } else {
                // If all tensors are not set, use metadata
                inputPros[i].setArg(nullptr);
                inputPros[i]._offset = 0;
                // TODO : BatchSize not checked here
                inputPros[i].setSize(_metadata.inputs.at(i).shapeFromCompiler.get_max_shape());
                inputPros[i].updateStride();
            }
        }

        // Update output Info
        for (size_t i = 0; i < outputProps.size(); ++i) {
            auto& levelZeroTensor = _levelZeroOutputTensors.at(i);
            auto& userTensor = _userOutputTensors.at(i);
            if (userTensor != nullptr) {
                // If userTensor is set, use userTensor to update memref handle
                outputProps[i].set(get_tensor_data_ptr(userTensor._ptr), 0, userTensor._ptr);
            } else if (levelZeroTensor != nullptr) {
                // If userTensor is not set, use levelZeroTensor to update memref handle
                outputProps[i].set(get_tensor_data_ptr(levelZeroTensor), 0, levelZeroTensor);
            } else {
                // If all tensors are not set, use metadata
                outputProps[i].setArg(nullptr);
                outputProps[i]._offset = 0;
                // TODO : BatchSize not checked here
                outputProps[i].setSize(_metadata.outputs.at(i).shapeFromCompiler.get_max_shape());
                outputProps[i].updateStride();
            }
        }

        auto originalOutputProps = outputProps;

        dynamicGraph->predict_output_shape(inputPros, outputProps);

        for (size_t i = 0; i < outputProps.size(); i++) {
            if (!originalOutputProps[i].compare(outputProps[i])) {
                _logger.debug("Output shape change detected");
                break;
            }
        }
    }
}

void ZeroDynamicInferRequest::check_tensor_and_predicted_shapes(
    const std::vector<IDynamicGraph::MemRefType>& outputProps) {
    // check_tensor in set_tensor already checked the input tensor and output tensor with metadata
    // Check again here to see if the shape is right compared with predicted shape
    // If user set output tensor, need check if the tensor is large enough
    for (size_t i = 0; i < _userOutputTensors.size(); i++) {
        auto& userTensor = _userOutputTensors.at(i);
        auto zeroTensor = std::dynamic_pointer_cast<ZeroTensor>(userTensor._ptr);
        auto& levelZeroTensor = _levelZeroOutputTensors.at(i);
        if (levelZeroTensor != nullptr && zeroTensor != nullptr && zeroTensor == levelZeroTensor) {
            // If user output tensor is ZeroTensor, no need check size here
            // These tensors are allocated by plugin and reshape will used later to resize tensor
            _logger.debug("Output tensor %zu is ZeroTensor, skip size check", i);
            continue;
        }

        ov::Shape predictedShape;
        for (int64_t j = 0; j < outputProps[i]._dimsCount; j++) {
            predictedShape.push_back(outputProps[i]._sizes[j]);
        }
        if (userTensor != nullptr) {
            // User set output tensor, need check size and throw exception if not large enough
            if (shape_size(userTensor->get_shape()) < shape_size(predictedShape)) {
                _logger.error("User output tensor %zu shape %s is different from predicted shape %s, can not "
                              "run inference",
                              i,
                              userTensor->get_shape().to_string().c_str(),
                              predictedShape.to_string().c_str());
                OPENVINO_THROW("User output tensor shape is smaller than predicted shape.");
            }
        }

        if (levelZeroTensor != nullptr) {
            if (shape_size(levelZeroTensor->get_shape()) < shape_size(predictedShape)) {
                // Local levelZero output tensor is not large enough, reshape will solve issue
                _logger.debug("LevelZero output tensor %zu shape %s is smaller than predicted shape %s, need "
                              "recreate pipeline",
                              i,
                              levelZeroTensor->get_shape().to_string().c_str(),
                              predictedShape.to_string().c_str());
            }
        }
    }
}

void ZeroDynamicInferRequest::update_tensor(const std::vector<IDynamicGraph::MemRefType>& outputProps) {
    // Update local level zero buffer shape with predicted shape to prepare for comparasion
    if (outputProps.size() > 0 && _isTensorChanged) {
        for (size_t i = 0; i < _levelZeroOutputTensors.size(); i++) {
            auto& levelZeroTensor = _levelZeroOutputTensors.at(i);
            if (levelZeroTensor == nullptr) {
                // Do not need to update user output tensor
                continue;
            }
            ov::Shape predictedShape;
            for (int64_t j = 0; j < outputProps[i]._dimsCount; j++) {
                predictedShape.push_back(outputProps[i]._sizes[j]);
            }
            if (levelZeroTensor->get_shape() != predictedShape) {
                _logger.info("Reshape output tensor %d from %s to predicted shape %s",
                             i,
                             levelZeroTensor->get_shape().to_string().c_str(),
                             predictedShape.to_string().c_str());
                levelZeroTensor->set_shape(predictedShape);
                _pipeline->update_graph_arguments(_metadata.outputs.at(i).indexUsedByDriver, levelZeroTensor);
            }
        }
    }
    _isTensorChanged = false;
}
