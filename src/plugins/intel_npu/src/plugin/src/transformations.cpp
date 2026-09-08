// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations.hpp"

#include <algorithm>
#include <map>
#include <sstream>

#include "intel_npu/utils/utils.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace intel_npu {

void enable_host_compile_if_needed(const std::shared_ptr<const ov::Model>& model,
                                   FilteredConfig& config,
                                   const Logger& logger) {
    if (model == nullptr || config.get<COMPILER_TYPE>() != ov::intel_npu::CompilerType::PLUGIN ||
        config.has<COMPILATION_MODE>() || config.get<DYNAMIC_SHAPE_TO_STATIC>()) {
        return;
    }

    // HostCompile allocates dynamic buffers from I/O upper bounds, so every dynamic dimension must be bounded.
    const auto hasFiniteUpperBounds = [](const auto& port) {
        const auto& shape = port.get_partial_shape();
        const auto rank = shape.rank();
        return rank.is_static() && std::all_of(shape.begin(), shape.end(), [](const ov::Dimension& dimension) {
                   return dimension.get_interval().has_upper_bound();
               });
    };

    // Detect a bounded dynamic 4D I/O port that makes the model a HostCompile candidate.
    const auto isDynamicHostCompilePort = [&hasFiniteUpperBounds](const auto& port) {
        const auto& shape = port.get_partial_shape();
        const auto rank = shape.rank();

        if (!(shape.is_dynamic() && rank.is_static() && rank.get_length() == 4 && hasFiniteUpperBounds(port))) {
            return false;
        }

        // Assume N,C,H,W order for each spatial dimension absent from the layout. Explicit layout dimensions (e.g.
        // set by compile_tool's -il/-iol/-iml/-ioml options or model preprocessing) take precedence.
        const auto layout = ov::layout::get_layout(port);
        const int64_t heightIdx = ov::layout::has_height(layout) ? ov::layout::height_idx(layout) : 2;
        const int64_t widthIdx = ov::layout::has_width(layout) ? ov::layout::width_idx(layout) : 3;

        // Only height (H) and width (W) determine candidacy; channel (C) and batch (N) do not. Considering N/H/W,
        // accepted dynamic patterns are H, W, HW, NH, NW and NHW, while N alone is rejected.
        // Note: When the batch size is greater than 1, using "compiler batch + host compiler interpreter" may cause
        // ConvertBatchedLayerTo1N and AdjustScaleShiftForDWConv to fail on certain models (e.g., maxpool models)
        // because their internal reshape operations do not support dynamic batch shapes. Alternatively, when the
        // batch size is greater than 1, combining "plugin batch + host compiler interpreter" may work but result in an
        // inference output shape that does not match the input shape on certain models (e.g., maxpool models).
        return shape[heightIdx].is_dynamic() || shape[widthIdx].is_dynamic();
    };

    const auto& modelInputs = model->inputs();
    const auto& modelOutputs = model->outputs();
    const bool inputsDynamic = std::any_of(modelInputs.begin(), modelInputs.end(), isDynamicHostCompilePort);
    const bool outputsDynamic = std::any_of(modelOutputs.begin(), modelOutputs.end(), isDynamicHostCompilePort);

    // Candidate detection above uses any_of; validate every I/O separately because one unrelated unbounded port
    // still prevents HostCompile from allocating all dynamic buffers.
    const bool allPortsHaveFiniteUpperBounds =
        std::all_of(modelInputs.begin(), modelInputs.end(), hasFiniteUpperBounds) &&
        std::all_of(modelOutputs.begin(), modelOutputs.end(), hasFiniteUpperBounds);

    if (inputsDynamic && outputsDynamic && allPortsHaveFiniteUpperBounds) {
        logger.info("NPU_COMPILATION_MODE not set; selecting 'HostCompile_Interpreter' for a bounded dynamic 4D "
                    "model with dynamic spatial input and output dimensions");
        config.update(ov::intel_npu::compilation_mode.name(), "HostCompile_Interpreter");
    }
}

namespace batch_helpers {

bool hasOtherDynamicDims(const ov::PartialShape& shape) {
    for (size_t dim_idx = 1; dim_idx < shape.size(); dim_idx++) {
        if (shape[dim_idx].is_dynamic()) {
            return true;  // Found dynamic dimension other than batch
        }
    }
    return false;
}

bool checkModelDynamicDims(const std::shared_ptr<const ov::Model>& model) {
    // Check parameters (inputs)
    const auto& params = model->get_parameters();
    for (const auto& param : params) {
        const auto& shape = param->get_partial_shape();
        if (hasOtherDynamicDims(shape)) {
            return true;
        }
    }

    // Check results (outputs)
    const auto& results = model->get_results();
    for (const auto& result : results) {
        const auto& shape = result->get_output_partial_shape(0);
        if (hasOtherDynamicDims(shape)) {
            return true;
        }
    }

    return false;
}

bool validateModelBatch(const std::shared_ptr<const ov::Model>& model, Logger logger) {
    std::set<ov::Output<const ov::Node>> batchedInputs;
    std::set<ov::Output<const ov::Node>> batchedOutputs;
    std::set<size_t> sBatchSize;

    // Limitation: Plugin batching is not supported when there are dynamic
    // dimensions other than the batch dimension.
    if (checkModelDynamicDims(model) && model->is_dynamic()) {
        return false;
    }

    const auto& params = model->get_parameters();
    for (size_t input_id = 0; input_id < params.size(); input_id++) {
        const auto& input = params[input_id];
        const auto& shape = input->get_partial_shape();
        ov::Layout layout = ov::layout::get_layout(input);

        // Batching on plugin is working only when batching is found on 0th dimension
        if ((shape.size() &&
             shape[intel_npu::utils::BATCH_AXIS].get_max_length() != intel_npu::utils::DEFAULT_BATCH_SIZE) ||
            (ov::layout::has_batch(layout) && ov::layout::batch_idx(layout) == intel_npu::utils::BATCH_AXIS)) {
            const auto& staticShape = shape.is_dynamic() ? shape.get_max_shape() : input->get_shape();
            batchedInputs.insert(params[input_id]->output(0));

            if (shape.rank().is_dynamic()) {
                OPENVINO_THROW("Shapes with dynamic rank are not supported.");
            } else {
                sBatchSize.insert(staticShape[intel_npu::utils::BATCH_AXIS]);
            }
        } else {
            // gather some diagnostic info
            std::optional<size_t> batch_dim_index_detected;
            for (size_t i = 1; i < shape.size(); i++) {
                if (shape[i].has_symbol()) {
                    batch_dim_index_detected = i;
                    break;
                }
            }
            std::stringstream sstream;
            sstream << "Only networks with inputs batched by 0th dimension are supported. ";
            if (batch_dim_index_detected.has_value()) {
                sstream << "The batch has been detected on: " << batch_dim_index_detected.value()
                        << " dimension instead. ";
            } else {
                sstream << "The batch hasn't been detected at all. ";
            }
            sstream << "Please check input id: " << input_id << " by the name: " << input->get_friendly_name()
                    << ", layout: " << layout.to_string() << ", is_dynamic: " << shape.is_dynamic();
            logger.info("%s", sstream.str().c_str());
            return false;
        }
    }
    for (const auto& output : model->get_results()) {
        const auto& shape = output->get_output_partial_shape(0);
        ov::Layout layout = ov::layout::get_layout(output);

        // Batching on plugin is working only when batching is found on 0th dimension
        if ((shape.size() &&
             shape[intel_npu::utils::BATCH_AXIS].get_max_length() != intel_npu::utils::DEFAULT_BATCH_SIZE) ||
            (ov::layout::has_batch(layout) && ov::layout::batch_idx(layout) == intel_npu::utils::BATCH_AXIS)) {
            const auto& node = output->input_value(0);
            const auto& staticShape = shape.is_dynamic() ? shape.get_max_shape() : output->get_shape();
            batchedOutputs.insert(ov::Output<const ov::Node>(node.get_node(), node.get_index()));

            if (shape.rank().is_dynamic()) {
                OPENVINO_THROW("Shapes with dynamic rank are not supported.");
            } else {
                sBatchSize.insert(staticShape[intel_npu::utils::BATCH_AXIS]);
            }
        } else {
            logger.info("Only networks with outputs batched by 0th dimension are supported. Please check an output by "
                        "the name: %s, layout: %s",
                        output->get_friendly_name().c_str(),
                        layout.to_string().c_str());
            return false;
        }
    }
    if (!batchedInputs.size() || !batchedOutputs.size()) {
        logger.info(
            "Only networks with inputs/outputs featuring batched dim are supported! Got inputs: %ld, outputs: %ld",
            batchedInputs.size(),
            batchedOutputs.size());
        return false;
    }

    if (sBatchSize.size() != 1) {
        logger.info("Batching size shall have same value for all tensors! Got unique batch sizes number: %ld",
                    sBatchSize.size());
        return false;
    }

    if (*sBatchSize.begin() == intel_npu::utils::DEFAULT_BATCH_SIZE) {
        logger.info("PLUGIN batch won't be applied, got default batch value : %ld", *sBatchSize.begin());
        return false;
    }

    auto node_info_printer = [&logger](const auto& ov_node, std::string nodeType) {
        logger.info("%s: %s has shape value: %s",
                    nodeType.c_str(),
                    ov_node.get_any_name().c_str(),
                    ov_node.get_partial_shape().to_string().c_str());
    };

    for (const auto& ov_node : batchedInputs) {
        node_info_printer(ov_node, "Input");
    }
    for (const auto& ov_node : batchedOutputs) {
        node_info_printer(ov_node, "Output");
    }

    return true;
}

bool deBatchModel(std::shared_ptr<ov::Model>& model,
                  ov::Dimension newBatch,
                  std::optional<ov::Dimension>& originalBatch) {
    try {
        std::map<std::string, ov::PartialShape> newShapes;
        auto shapeChanged = false;
        for (auto&& item : model->get_parameters()) {
            auto layout = item->get_layout();
            auto partShape = item->get_partial_shape();
            if (ov::layout::has_batch(layout)) {
                shapeChanged = true;
                originalBatch = partShape[ov::layout::batch_idx(layout)];
                partShape[ov::layout::batch_idx(layout)] = newBatch;
            }
            newShapes.emplace(item->get_friendly_name(), partShape);
        }
        model->reshape(newShapes);
        return shapeChanged;
    } catch (const std::exception&) {
        // Don't throw - let caller handle the failure
        return false;
    }
}

std::tuple<std::shared_ptr<ov::Model>, bool> handlePluginBatching(
    std::shared_ptr<const ov::Model> model,
    const std::function<void(ov::intel_npu::BatchMode)>& updateBatchMode,
    std::optional<ov::intel_npu::BatchMode> batchMode,
    std::optional<ov::Dimension>& originalBatch,
    Logger logger) {
    // Keep the original model for all no-op/early-return paths.
    // A mutable clone is created only when plugin batching is actually about to be applied.
    auto resultModel = std::const_pointer_cast<ov::Model>(model);
    auto successfullyDebatched = false;

    auto batchModeIsAvailable = batchMode.has_value();
    ov::intel_npu::BatchMode effectiveBatchMode =
        batchModeIsAvailable ? batchMode.value() : ov::intel_npu::BatchMode::AUTO;

    if (batchModeIsAvailable) {
        const auto isAutoOrPluginBatch = (effectiveBatchMode == ov::intel_npu::BatchMode::PLUGIN ||
                                          effectiveBatchMode == ov::intel_npu::BatchMode::AUTO);

        if (!isAutoOrPluginBatch) {
            return {resultModel, successfullyDebatched};
        }
    }

    try {
        const auto pluginBatchingIsSupported = validateModelBatch(model, logger);

        if (!pluginBatchingIsSupported) {
            if (batchModeIsAvailable && effectiveBatchMode == ov::intel_npu::BatchMode::AUTO) {
                logger.info("Batching will be handled by compiler.");
                updateBatchMode(ov::intel_npu::BatchMode::COMPILER);
            }
            return {resultModel, successfullyDebatched};
        }

        logger.info("Attempting to handle batching on the plugin side.");
        // Clone right before mutation to avoid extra memory usage when batching is skipped.
        resultModel = model->clone();

        try {
            originalBatch = ov::get_batch(resultModel);
            ov::set_batch(resultModel, ov::Dimension(1));
            successfullyDebatched = true;
        } catch (const std::exception& ex) {
            logger.warning("The plugin couldn't resize a batched model due to exception: %s.\n"
                           "Trying to debatch it...",
                           ex.what());

            if (!deBatchModel(resultModel, ov::Dimension(1), originalBatch)) {
                OPENVINO_THROW("Cannot debatch a model");
            }
            logger.info("The model has been debatched successfully");
            successfullyDebatched = true;
        }
        if (batchModeIsAvailable) {
            // If we have successfully debatched the model on the PLUGIN side, we should
            // avoid repeating the same in the compiler by resetting the batch mode
            logger.info("The model was reshaped to batch size 1 on the plugin side.");
            updateBatchMode(ov::intel_npu::BatchMode::COMPILER);
        }
    } catch (const std::exception& ex) {
        // If plugin-side transformation failed, keep the original model and drop the clone
        resultModel = std::const_pointer_cast<ov::Model>(model);
        if (effectiveBatchMode == ov::intel_npu::BatchMode::AUTO) {
            logger.info("Couldn't validate and reshape the model. Batching will be handled by compiler. Error: %s",
                        ex.what());
            if (batchModeIsAvailable) {
                // If we failed to handle batching on the plugin side, we should reset the batch mode to default
                // COMPILER but only if the batch mode is available, otherwise we might be running on an older compiler
                // which doesn't support batch mode at all
                updateBatchMode(ov::intel_npu::BatchMode::COMPILER);
            }
        } else {
            OPENVINO_THROW("Couldn't validate and reshape the model for PLUGIN batch mode. Error: ", ex.what());
        }
    }

    return {resultModel, successfullyDebatched};
}

}  // namespace batch_helpers
}  // namespace intel_npu
