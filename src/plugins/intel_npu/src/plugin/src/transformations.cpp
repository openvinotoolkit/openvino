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

bool should_use_host_compile_interpreter(const std::shared_ptr<const ov::Model>& model,
                                         ov::intel_npu::CompilerType compilerType,
                                         bool compilationModeSet,
                                         bool dynamicShapeToStatic) {
    if (compilerType != ov::intel_npu::CompilerType::PLUGIN || compilationModeSet || dynamicShapeToStatic) {
        return false;
    }

    // HostCompile supports dynamic dimensions only; a single dynamic-rank input rules the model out.
    const auto hasStaticRank = [](const auto& port) {
        return port.get_partial_shape().rank().is_static();
    };

    // Any dynamic dimension other than batch, bounded or unbounded, makes the model a HostCompile model: HostCompile
    // sizes dynamic buffers at run time from the bound tensor or the predicted output shape. A dynamic batch alone is
    // not supported well by HostCompile yet, so such models are left to plugin batching.
    const auto isDynamicPort = [](const auto& port) {
        return batch_helpers::hasOtherDynamicDims(port.get_partial_shape());
    };

    // Only the inputs decide: shape inference does not reach the outputs of some models, which then look fully
    // dynamic even though the model is a HostCompile one.
    const auto& modelInputs = model->inputs();
    if (!std::all_of(modelInputs.begin(), modelInputs.end(), hasStaticRank)) {
        return false;
    }

    return std::any_of(modelInputs.begin(), modelInputs.end(), isDynamicPort);
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
        // The tensors are not required to have names, "get_any_name" throws in such a case
        const std::string name =
            ov_node.get_names().empty() ? ov_node.get_node()->get_friendly_name() : ov_node.get_any_name();
        logger.info("%s: %s has shape value: %s",
                    nodeType.c_str(),
                    name.c_str(),
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

        // Only the ports of the model are validated below, not the operations between them: the plugin assumes that
        // the graph treats the batch slices independently, as it runs the schedule once per slice. An operation acting
        // along the batch axis which keeps the shapes of the ports (e.g. a Softmax on the axis 0) is not detected.
        //
        // validateModelBatch is heuristic (0th dim != 1), so a port which does not depend on the batch is wrongly
        // considered batched, e.g.:
        // - a ShapeOf result of shape [2] while the batch is 2;
        // - an input without the batch layout, as "ov::set_batch" reshapes only the inputs which have one.
        // Make sure every input and output actually ended up with batch 1 after reshaping, otherwise the compiled
        // schedule would not match the batch size expected by the plugin. A port which became a scalar (e.g. a Squeeze
        // removing the batch dimension of an input reshaped to batch 1) has no batch dimension at all, thus it is not
        // batched either. A batch dimension which is still dynamic was not set to 1 either (e.g. an input without the
        // batch layout, which keeps its bounds), so the schedule would not be compiled with a batch of 1.
        const auto isNotDebatched = [](const ov::PartialShape& shape) {
            if (shape.rank().is_dynamic()) {
                return false;
            }
            if (shape.size() == 0) {
                return true;
            }
            const auto& batchDimension = shape[intel_npu::utils::BATCH_AXIS];
            return batchDimension.is_dynamic() ||
                   static_cast<size_t>(batchDimension.get_length()) != intel_npu::utils::DEFAULT_BATCH_SIZE;
        };

        const auto& parameters = resultModel->get_parameters();
        const auto notDebatchedInput =
            std::find_if(parameters.begin(), parameters.end(), [&isNotDebatched](const auto& parameter) {
                return isNotDebatched(parameter->get_partial_shape());
            });
        if (notDebatchedInput != parameters.end()) {
            OPENVINO_THROW("The input \"",
                           (*notDebatchedInput)->get_friendly_name(),
                           "\" has the shape ",
                           (*notDebatchedInput)->get_partial_shape().to_string(),
                           " after the model was reshaped to batch size 1, thus its batch was not changed.");
        }

        const auto& results = resultModel->get_results();
        const auto notDebatchedOutput =
            std::find_if(results.begin(), results.end(), [&isNotDebatched](const auto& result) {
                return isNotDebatched(result->get_output_partial_shape(0));
            });
        if (notDebatchedOutput != results.end()) {
            OPENVINO_THROW("The output \"",
                           (*notDebatchedOutput)->get_friendly_name(),
                           "\" has the shape ",
                           (*notDebatchedOutput)->get_output_partial_shape(0).to_string(),
                           " after the model was reshaped to batch size 1, thus it is not batched.");
        }

        // Only the batch dimension of an output is expected to change when the inputs are reshaped. If anything else
        // changed, the output does not have the batch as an independent leading dimension (e.g. MatMul(A, B^T) of two
        // batched inputs, where [4, 4] becomes [1, 1]), so running the schedule once per batch slice and stacking the
        // results does not give the output of the original model.
        const auto& originalResults = model->get_results();
        OPENVINO_ASSERT(originalResults.size() == results.size(),
                        "The debatched model has a different number of outputs than the original one");
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& originalShape = originalResults[i]->get_output_partial_shape(0);
            const auto& debatchedShape = results[i]->get_output_partial_shape(0);
            if (originalShape.rank().is_dynamic() || debatchedShape.rank().is_dynamic()) {
                continue;
            }

            bool onlyBatchChanged = originalShape.size() == debatchedShape.size();
            for (size_t dim = 1; onlyBatchChanged && dim < originalShape.size(); ++dim) {
                onlyBatchChanged = originalShape[dim] == debatchedShape[dim];
            }
            if (!onlyBatchChanged) {
                OPENVINO_THROW("The output \"",
                               results[i]->get_friendly_name(),
                               "\" has the shape ",
                               debatchedShape.to_string(),
                               " after the model was reshaped to batch size 1, while it was ",
                               originalShape.to_string(),
                               ", thus it is not batched independently.");
            }
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
        // The model is left untouched, so the caller must not consider it debatched
        successfullyDebatched = false;
        originalBatch = std::nullopt;
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
