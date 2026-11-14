// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/custom_layer_source.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/simple_math.hpp"
#include "intel_gpu/primitives/custom_gpu_primitive.hpp"
#include "intel_gpu/primitives/reorder.hpp"

namespace ov::intel_gpu {

void CreateCustomOp(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op, CustomLayerPtr customLayer) {
    if (!p.use_new_shape_infer()) {
        OPENVINO_ASSERT(op->get_output_size() == 1u, "Custom OP limitation: static model only support one output.");
    }
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    const auto kernel_sources = make_custom_layer_source(op, *customLayer);

    // reserve
    std::vector<cldnn::input_info> reordered_inputs;
    reordered_inputs.resize(inputs.size());

    // Handle kernel parameters
    std::vector<cldnn::custom_gpu_primitive::arg_desc> kernelParameters;
    std::vector<cldnn::format> outputFormats;
    for (const auto& param : customLayer->KernelParams()) {
        switch (param.type) {
        case CustomLayer::ParamType::Input: {
            kernelParameters.resize(kernelParameters.size() > size_t(param.paramIndex + 1) ? kernelParameters.size() : size_t(param.paramIndex + 1));
            kernelParameters[param.paramIndex].type = cldnn::custom_gpu_primitive::arg_input;
            kernelParameters[param.paramIndex].index =
                static_cast<cldnn::custom_gpu_primitive::arg_index>((param.portIndex >= static_cast<int>(inputs.size())) ? -1 : param.portIndex);

            // Handle input reorder
            if (param.portIndex < static_cast<int>(inputs.size()) && reordered_inputs[param.portIndex].pid.empty()) {
                if (param.format != cldnn::format::any) {
                    auto reorderPrimName = inputs[param.portIndex].pid + "_" + op->get_friendly_name() +
                                            "_" + std::to_string(param.portIndex) + ProgramBuilder::m_preCustomLayerTag;
                    auto preprocessPrim = cldnn::reorder(
                        reorderPrimName,
                        inputs[param.portIndex],
                        param.format,
                        cldnn::element_type_to_data_type(op->get_input_element_type(param.portIndex)));

                    p.add_primitive(*op, preprocessPrim);
                    reordered_inputs[param.portIndex] = cldnn::input_info(reorderPrimName);
                } else {
                    reordered_inputs[param.portIndex] = inputs[param.portIndex];
                }
            }
            break;
        }
        case CustomLayer::ParamType::Output: {
            kernelParameters.resize(kernelParameters.size() > size_t(param.paramIndex + 1) ? kernelParameters.size() : size_t(param.paramIndex + 1));
            kernelParameters[param.paramIndex].type = cldnn::custom_gpu_primitive::arg_output;
            kernelParameters[param.paramIndex].index =
                static_cast<cldnn::custom_gpu_primitive::arg_index>((param.portIndex >= static_cast<int>(inputs.size())) ? -1 : param.portIndex);
            outputFormats.push_back(param.format);
            break;
        }
        case CustomLayer::ParamType::Internal: {
           kernelParameters.resize(kernelParameters.size() > size_t(param.paramIndex + 1) ? kernelParameters.size() : size_t(param.paramIndex + 1));
           kernelParameters[param.paramIndex].type = cldnn::custom_gpu_primitive::arg_internal;
           kernelParameters[param.paramIndex].index = static_cast<cldnn::custom_gpu_primitive::arg_index>(param.portIndex);
           kernelParameters[param.paramIndex].size_expr = param.size_expr;
           break;
        }
        default:
            OPENVINO_THROW("Invalid custom layer param type: ", param.type, " in operation: ", op->get_friendly_name());
        }
    }
    int iidx = customLayer->InputDimSourceIndex();
    OPENVINO_ASSERT(outputFormats.size() == op->get_output_size(), "The number of outputFormats should be same as op->get_output_size().");

    std::vector<cldnn::layout> outputLayouts(op->get_output_size());
    for (size_t i = 0; i < op->get_output_size(); i++) {
        auto dims = op->get_output_partial_shape(i);

        // format::any needs the shape-based ctor for the same reason a dynamic shape does:
        // the tensor ctor stores an `any` layout's sizes in raw internal order at full
        // internal rank, transposing the WorkSizes resolution and the downstream shape.
        if (dims.is_dynamic() || outputFormats[i] == cldnn::format::any) {
            outputLayouts[i] = cldnn::layout(dims, cldnn::element_type_to_data_type(op->get_output_element_type(i)), outputFormats[i]);
        } else {
            size_t N = (dims.size() > 0) ? dims[0].get_length() : 1;
            size_t C = (dims.size() > 1) ? dims[1].get_length() : 1;
            size_t H = (dims.size() > 2) ? dims[2].get_length() : 1;
            size_t W = (dims.size() > 3) ? dims[3].get_length() : 1;
            cldnn::tensor outputTensor = cldnn::tensor(cldnn::batch(N), cldnn::feature(C), cldnn::spatial(W, H));
            outputLayouts[i] = cldnn::layout(cldnn::element_type_to_data_type(op->get_output_element_type(i)), outputFormats[i], outputTensor);
        }
    }

    std::vector<size_t> gws, lws;

    // if input index is greater than -1, take dimension from input
    if (iidx >= 0) {
        if (static_cast<size_t>(iidx) >= op->get_input_size()) {
            OPENVINO_THROW("Invalid input tensor for index: ", iidx);
        }
        auto inputDims = op->get_input_shape(iidx);
        // Regardless of whether the model has one or more outputs, only the first output is used to update gws and lws.
        cldnn::custom_gpu_primitive::update_work_group_size(op->get_output_partial_shape(0),
                                                            iidx,
                                                            inputDims,
                                                            customLayer->GlobalSizeRules(),
                                                            customLayer->LocalSizeRules(),
                                                            gws,
                                                            lws);
    } else {
        // Regardless of whether the model has one or more outputs, only the first output is used to update gws and lws.
        cldnn::custom_gpu_primitive::update_work_group_size(op->get_output_partial_shape(0),
                                                            iidx,
                                                            ov::PartialShape(),
                                                            customLayer->GlobalSizeRules(),
                                                            customLayer->LocalSizeRules(),
                                                            gws,
                                                            lws);
    }

    std::string genericLayerName = layer_type_name_ID(op);

    // Clone a new op to make sure original model can be released.
    ov::OutputVector new_inputs;
    for (size_t i = 0; i < op->get_input_size(); i++) {
        auto input = std::make_shared<ov::op::v0::Parameter>(op->get_input_element_type(i), op->get_input_partial_shape(i));
        new_inputs.emplace_back(input);
    }
    std::shared_ptr<ov::Node> op_bk = op->clone_with_new_inputs(new_inputs);

    auto customPrim = cldnn::custom_gpu_primitive(genericLayerName,
                                                  reordered_inputs,
                                                  kernel_sources,
                                                  customLayer->KernelEntry(),
                                                  kernelParameters,
                                                  customLayer->CompilerOptions(),
                                                  outputLayouts,
                                                  gws,
                                                  lws,
                                                  op_bk,
                                                  iidx,
                                                  customLayer->GlobalSizeRules(),
                                                  customLayer->LocalSizeRules());
    p.add_primitive(*op, customPrim);

    for (size_t i = 0; i < outputLayouts.size(); i++) {
        if (outputLayouts[i].format != cldnn::format::any) {
            auto default_format = cldnn::format::get_default_format(op->get_output_partial_shape(i).size());
            if (outputLayouts.size() > 1) {
                OPENVINO_ASSERT(default_format == outputLayouts[i].format,
                                "Multiple outputs with non-default formats are not supported because a reorder primitive cannot be inserted; the subsequent "
                                "nodes would fail to retrieve the correct output port index during shape inference after the graph transformation.");
            } else {
                // Handle output reorder
                auto reorderPrimName = genericLayerName + ProgramBuilder::m_postCustomLayerTag + std::string("_output_") + std::to_string(i);
                auto reorderPrim = cldnn::reorder(reorderPrimName,
                                                  cldnn::input_info(genericLayerName, static_cast<int>(i)),
                                                  default_format,
                                                  customPrim.output_layouts[i].data_type);
                p.add_primitive(*op, reorderPrim);
            }
        }
    }
}

}  // namespace ov::intel_gpu
