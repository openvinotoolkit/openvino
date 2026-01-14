// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations.hpp"

#include <map>
#include <sstream>

#include "intel_npu/utils/utils.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace intel_npu {
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
    FilteredConfig& localConfig,
    const std::function<void(ov::intel_npu::BatchMode)>& updateBatchMode,
    std::optional<ov::Dimension>& originalBatch,
    Logger logger) {
    auto reshapedModel = model->clone();
    auto successfullyDebatched = false;

    auto batchModeIsAvailable = localConfig.isAvailable(ov::intel_npu::batch_mode.name());
    ov::intel_npu::BatchMode batchMode;
    if (batchModeIsAvailable) {
        batchMode = localConfig.get<BATCH_MODE>();
        const auto isAutoOrPluginBatch =
            (batchMode == ov::intel_npu::BatchMode::PLUGIN || batchMode == ov::intel_npu::BatchMode::AUTO);

        if (!isAutoOrPluginBatch) {
            return {reshapedModel, successfullyDebatched};
        }
    } else {
        // If the compiler doesn't support BATCH_MODE, we can still try using the PLUGIN batch
        batchMode = ov::intel_npu::BatchMode::PLUGIN;
    }

    try {
        const auto pluginBatchingIsSupported = validateModelBatch(reshapedModel, logger);

        if (!pluginBatchingIsSupported) {
            if (batchModeIsAvailable && batchMode == ov::intel_npu::BatchMode::AUTO) {
                logger.info("Batching will be handled by compiler.");
                updateBatchMode(ov::intel_npu::BatchMode::COMPILER);
            }
            return {reshapedModel, successfullyDebatched};
        }

        logger.info("Attempting to handle batching on the plugin side.");

        try {
            originalBatch = ov::get_batch(reshapedModel);
            ov::set_batch(reshapedModel, ov::Dimension(1));
            successfullyDebatched = true;
        } catch (const std::exception& ex) {
            logger.warning("The plugin couldn't resize a batched model due to exception: %s.\n"
                           "Trying to debatch it...",
                           ex.what());

            if (!deBatchModel(reshapedModel, ov::Dimension(1), originalBatch)) {
                OPENVINO_THROW("Cannot debatch a model");
            }
            logger.info("The model has been debatched successfully");
            successfullyDebatched = true;
        }
        if (batchModeIsAvailable) {
            // If we have successfully debatched the model on the PLUGIN side, we should
            // avoid repeating the same in the compiler by resetting the batch mode
            updateBatchMode(ov::intel_npu::BatchMode::COMPILER);
        }
    } catch (const std::exception& ex) {
        logger.info("Couldn't validate and reshape the model. Batching will be handled by compiler. Error: %s",
                    ex.what());
    }

    return {reshapedModel, successfullyDebatched};
}

}  // namespace batch_helpers
}  // namespace intel_npu
