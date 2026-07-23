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

void ZeroDynamicInferRequest::create_pipeline_impl() {
    _logger.debug("create_pipeline_impl - constructing pipeline");
    auto batchSize = _graph->get_batch_size();
    // Construct pipeline. The pipeline owns the VM execution context shared by shape prediction and execution.
    _pipeline =
        std::make_unique<DynamicPipeline>(_initStructs,
                                          _graph,
                                          _config,
                                          _levelZeroInputTensors,
                                          _levelZeroOutputTensors,
                                          batchSize.has_value() ? batchSize.value() : utils::DEFAULT_BATCH_SIZE);

    _logger.debug("create_pipeline_impl - completed");
}

void ZeroDynamicInferRequest::sync_zero_tensor_with_graph(const ZeroInferRequest::FoundPort& foundPort,
                                                          const ov::SoPtr<ov::ITensor>& tensor) {
    OV_ITT_TASK_CHAIN(ZERO_SET_TENSOR,
                      itt::domains::LevelZeroBackend,
                      "ZeroDynamicInferRequest",
                      "sync_zero_tensor_with_graph");
    if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0)) {
        auto& levelZeroTensor =
            foundPort.is_input() ? get_level_zero_input(foundPort.idx) : _levelZeroOutputTensors.at(foundPort.idx);

        auto originallevelZeroTensor = levelZeroTensor;

        try {
            _logger.debug("sync_zero_tensor_with_graph - create zero tensor");
            OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "create zero tensor");
            // Try to use the user tensor directly if its underlying data is already allocated in the same Level Zero
            // context.
            levelZeroTensor = std::make_shared<ZeroTensor>(_initStructs, tensor);
        } catch (const ZeroMemException& exception) {
            _logger.debug("sync_zero_tensor_with_graph - exception caught while trying to create a Level Zero tensor "
                          "from the user tensor: %s",
                          exception.what());

            // Check if the current Level Zero tensor was previously shared with the user. If so, it cannot be reused;
            // allocate a new tensor to back up the user tensor (which cannot be imported or used directly).
            if (_dynamicBatchValueChanged || levelZeroTensor == nullptr || !levelZeroTensor->can_be_reused() ||
                (levelZeroTensor != nullptr && (levelZeroTensor->get_byte_size() < tensor->get_byte_size()))) {
                _logger.debug("sync_zero_tensor_with_graph - allocate locally L0 tensor");
                OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "allocate tensor");

                auto batch = _graph->get_batch_size();
                levelZeroTensor = allocate_tensor(foundPort.idx, foundPort.is_input(), batch);
            } else {
                _logger.debug("sync_zero_tensor_with_graph - reusing the level zero tensor since it is not shared with "
                              "the user, and old L0 tensor is large enough");
            }
        }

        if (_pipelineIsCreated && !_dynamicBatchValueChanged) {
            _logger.debug("sync_zero_tensor_with_graph - update graph arguments");

            OPENVINO_ASSERT(levelZeroTensor->data(), "Empty buffer");

            OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "update_graph_arguments");
            if (originallevelZeroTensor != nullptr && originallevelZeroTensor->get_shape() != tensor->get_shape()) {
                _logger.debug("sync_zero_tensor_with_graph - update graph arguments with user tensor pointer since "
                              "shape is changed");
                _pipeline->update_graph_arguments(foundPort.is_input()
                                                      ? _metadata.inputs.at(foundPort.idx).indexUsedByDriver
                                                      : _metadata.outputs.at(foundPort.idx).indexUsedByDriver,
                                                  levelZeroTensor,
                                                  tensor._ptr);
                _isTensorChanged = true;
            } else {
                // This L0 tensor shall have same info with user tensor
                _logger.debug("sync_zero_tensor_with_graph - update graph arguments without user tensor pointer since "
                              "shape is not changed");
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

void ZeroDynamicInferRequest::sync_zero_tensors_with_graph(const ZeroInferRequest::FoundPort& foundPort,
                                                           const std::vector<ov::SoPtr<ov::ITensor>>& tensors,
                                                           const std::optional<size_t>& batchSize) {
    OV_ITT_TASK_CHAIN(ZERO_SET_TENSORS,
                      itt::domains::LevelZeroBackend,
                      "ZeroDynamicInferRequest",
                      "sync_zero_tensors_with_graph");
    if (_initStructs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0) && batchSize.has_value()) {
        get_level_zero_inputs(foundPort.idx).resize(tensors.size());

        for (size_t i = 0; i < tensors.size(); i++) {
            auto originalLevelZeroTensor = get_level_zero_input(foundPort.idx, i);
            try {
                _logger.debug("sync_zero_tensors_with_graph - create zero tensor");
                OV_ITT_TASK_NEXT(ZERO_SET_TENSORS, "create_zero_tensor");
                get_level_zero_input(foundPort.idx, i) = std::make_shared<ZeroTensor>(_initStructs, tensors.at(i));
            } catch (const ZeroMemException& exception) {
                _logger.debug("sync_zero_tensors_with_graph - exception caught while trying to create a Level Zero "
                              "tensor from the user tensor: %s",
                              exception.what());

                _logger.debug("sync_zero_tensors_with_graph - allocate locally L0 tensor");
                OV_ITT_TASK_NEXT(ZERO_SET_TENSORS, "allocate_tensor");
                get_level_zero_input(foundPort.idx, i) = allocate_tensor(foundPort.idx, INPUT, batchSize);
            }

            if (_pipelineIsCreated && !_dynamicBatchValueChanged) {
                OPENVINO_ASSERT(get_level_zero_input(foundPort.idx, i)->data(), "Empty buffer");
                OV_ITT_TASK_NEXT(ZERO_SET_TENSORS, "update_graph_arguments");
                if (originalLevelZeroTensor != nullptr &&
                    originalLevelZeroTensor->get_shape() != tensors.at(i)->get_shape()) {
                    _logger.debug(
                        "set_tensors - update graph arguments with user tensor pointer since shape is changed");
                    _pipeline->update_graph_arguments(_metadata.inputs.at(foundPort.idx).indexUsedByDriver,
                                                      get_level_zero_input(foundPort.idx, i),
                                                      i,
                                                      tensors.at(i)._ptr);
                    _isTensorChanged = true;
                } else {
                    _logger.debug(
                        "set_tensors - update graph arguments without user tensor pointer since shape is not changed");
                    _pipeline->update_graph_arguments(_metadata.inputs.at(foundPort.idx).indexUsedByDriver,
                                                      get_level_zero_input(foundPort.idx, i),
                                                      i);
                }
            }
        }
    }
    if (!_pipelineIsCreated) {
        // If pipeline is not created, need to predict real output shape
        _isTensorChanged = true;
    }
    // If command list updates are not supported, fallback to copying tensors every time.
}

std::shared_ptr<ZeroTensor> ZeroDynamicInferRequest::allocate_tensor(
    const size_t index,
    const bool isInput,
    const std::optional<std::size_t>& batchSize) const {
    IODescriptor descriptor = isInput ? _metadata.inputs.at(index) : _metadata.outputs.at(index);
    // Create new IODescriptor based on user input|output and descriptor
    if (isInput && get_user_input(index) != nullptr) {
        _logger.debug("allocate_tensor - update input descriptor with shape from user input: %s instead of %s",
                      get_user_input(index)->get_shape().to_string().c_str(),
                      descriptor.shapeFromCompiler.to_string().c_str());
        descriptor.shapeFromCompiler = get_user_input(index)->get_shape();
    } else if (!isInput && _userOutputTensors.at(index) != nullptr) {
        _logger.debug("allocate_tensor - update output descriptor with shape from user output: %s instead of %s",
                      _userOutputTensors.at(index)->get_shape().to_string().c_str(),
                      descriptor.shapeFromCompiler.to_string().c_str());
        descriptor.shapeFromCompiler = _userOutputTensors.at(index)->get_shape();
    }

    check_network_precision(descriptor.precision);

    ov::Shape allocatedTensorShape = descriptor.shapeFromCompiler.get_max_shape();

    if (batchSize.has_value()) {
        allocatedTensorShape[utils::BATCH_AXIS] = *batchSize;
    }

    auto tensor = std::make_shared<ZeroTensor>(_initStructs, descriptor.precision, allocatedTensorShape, isInput);

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
    _logger.debug("infer_async - started");
    OV_ITT_TASK_CHAIN(ZERO_INFER, itt::domains::LevelZeroBackend, "infer_async", "start");
    // Create (or reuse) the dynamic pipeline first. Shape prediction now runs through the pipeline instance and
    // shares the pipeline-owned VM execution context with execution.
    prepare_inputs();

    // Predict output shapes and validate user output tensors; prepare_outputs() then allocates and resizes the
    // Level Zero output buffers to the predicted shapes.
    predict_output_shapes(_predictedShapes);
    check_tensor_and_predicted_shapes(_predictedShapes);
    prepare_outputs();

    OV_ITT_TASK_NEXT(ZERO_INFER, "push");
    _pipeline->push();
}

void ZeroDynamicInferRequest::predict_output_shapes(std::vector<ov::Shape>& predictedShapes) {
    // TODO: If current output tensor is not large enough to be compatible with input tensor, need recreate pipeline
    // But reshape ZeroTensor can be used to avoid recreate pipeline now

    // Predict output shapes based on current inputs. The infer request only deals with tensors/OV shapes;
    // MemRef packing is done inside the pipeline layer.
    predictedShapes.clear();

    if (_graph->get_handle() != nullptr && _isTensorChanged) {
        std::vector<std::shared_ptr<ov::ITensor>> inputTensors(_metadata.inputs.size());
        std::vector<std::shared_ptr<ov::ITensor>> outputTensors(_metadata.outputs.size());

        // TODO: Support Batch later
        // Update input Info
        // A null entry lets the pipeline fall back to the graph metadata max shape.
        for (size_t i = 0; i < inputTensors.size(); ++i) {
            const auto& userTensor = get_user_input(i);
            if (userTensor != nullptr) {
                // If userTensor is set, use userTensor to update memref handle in prediction
                inputTensors[i] = userTensor._ptr;
            } else {
                // If userTensor is not set, use levelZeroTensor
                inputTensors[i] = get_level_zero_input(i);
            }
        }
        // Update output Info
        for (size_t i = 0; i < outputTensors.size(); ++i) {
            const auto& userTensor = _userOutputTensors.at(i);
            if (userTensor != nullptr) {
                // If userTensor is set, use userTensor to update memref handle in prediction
                outputTensors[i] = userTensor._ptr;
            } else {
                // If userTensor is not set, use levelZeroTensor
                outputTensors[i] = _levelZeroOutputTensors.at(i);
            }
        }

        OPENVINO_ASSERT(_pipeline != nullptr, "Dynamic pipeline must be created before predicting output shapes");
        // ZeroDynamicInferRequest always constructs a DynamicPipeline in create_pipeline_impl,
        // so this downcast is safe.
        predictedShapes =
            static_cast<DynamicPipeline*>(_pipeline.get())->predict_output_shapes(inputTensors, outputTensors);
    }
}

void ZeroDynamicInferRequest::check_tensor_and_predicted_shapes(const std::vector<ov::Shape>& predictedShapes) {
    if (predictedShapes.empty()) {
        _logger.debug("check_tensor_and_predicted_shapes - no output props to check, skip check");
        return;
    }
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
            _logger.debug("check_tensor_and_predicted_shapes - output tensor %zu is ZeroTensor, skip size check", i);
            continue;
        }

        const ov::Shape& predictedShape = predictedShapes[i];
        if (userTensor != nullptr) {
            // User set output tensor, need check size and throw exception if not large enough
            if (shape_size(userTensor->get_shape()) < shape_size(predictedShape)) {
                _logger.error("check_tensor_and_predicted_shapes - user output tensor %zu shape %s is different from "
                              "predicted shape %s, can not run inference",
                              i,
                              userTensor->get_shape().to_string().c_str(),
                              predictedShape.to_string().c_str());
                OPENVINO_THROW("User output tensor shape is smaller than predicted shape.");
            }
        }

        if (levelZeroTensor != nullptr) {
            if (shape_size(levelZeroTensor->get_shape()) < shape_size(predictedShape)) {
                // Local levelZero output tensor is not large enough, reshape will solve issue
                _logger.debug("check_tensor_and_predicted_shapes - LevelZero output tensor %zu shape %s is smaller "
                              "than predicted shape %s, need recreate pipeline",
                              i,
                              levelZeroTensor->get_shape().to_string().c_str(),
                              predictedShape.to_string().c_str());
            }
        }
    }
}

void ZeroDynamicInferRequest::prepare_outputs() {
    OV_ITT_TASK_CHAIN(ZERO_INFER, itt::domains::LevelZeroBackend, "infer_async", "prepare_outputs");
    // Resize outputs to the predicted shapes only when a tensor change triggered a fresh prediction.
    const bool reshapeToPredicted = !_predictedShapes.empty() && _isTensorChanged;

    for (size_t outputIndex = 0; outputIndex < _levelZeroOutputTensors.size(); ++outputIndex) {
        auto& levelZeroTensor = _levelZeroOutputTensors.at(outputIndex);
        OPENVINO_ASSERT(levelZeroTensor, "Output zero tensor is not allocated.");

        bool graphArgsNeedUpdate = false;

        // Ensure the output buffer is allocated (former base prepare_outputs behavior).
        if (levelZeroTensor->data() == nullptr) {
            levelZeroTensor->allocate_data();
            graphArgsNeedUpdate = true;
        }

        // Resize the output buffer to the predicted shape (former update_tensor behavior). set_shape may
        // reallocate the buffer, so the single graph-argument refresh below must run after it.
        if (reshapeToPredicted) {
            const ov::Shape& predictedShape = _predictedShapes[outputIndex];
            if (levelZeroTensor->get_shape() != predictedShape) {
                _logger.info("prepare_outputs - reshape output tensor %zu from %s to predicted shape %s",
                             outputIndex,
                             levelZeroTensor->get_shape().to_string().c_str(),
                             predictedShape.to_string().c_str());
                levelZeroTensor->set_shape(predictedShape);
                graphArgsNeedUpdate = true;
            }
        }

        if (graphArgsNeedUpdate) {
            _pipeline->update_graph_arguments(_metadata.outputs.at(outputIndex).indexUsedByDriver, levelZeroTensor);
        }
    }

    _isTensorChanged = false;
}
