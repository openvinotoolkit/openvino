// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/simple_math.hpp"
#include "intel_gpu/primitives/custom_gpu_primitive.hpp"
#include "intel_gpu/primitives/reorder.hpp"

namespace ov::intel_gpu {

template<typename T>
static inline std::string vecToString(std::vector<T> vec) {
    if (vec.empty())
        return "";

    std::string res = std::to_string(vec[0]);
    for (size_t i = 1; i < vec.size(); i++) {
        res += "," + std::to_string(vec[i]);
    }
    return res;
}

template<>
inline std::string vecToString<std::string>(std::vector<std::string> vec) {
    if (vec.empty())
        return "";

    std::string res = vec[0];
    for (size_t i = 1; i < vec.size(); i++) {
        res += "," + vec[i];
    }
    return res;
}

class CustomLayerAttributeVisitor : public ov::AttributeVisitor {
public:
    CustomLayerAttributeVisitor() : m_values({}) { }

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        OPENVINO_THROW("Attribute ", name, " can't be processed\n");
    }
    // The remaining adapter methods fall back on the void adapter if not implemented
    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) override {
        m_values[name] = adapter.get();
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override {
        m_values[name] = std::to_string(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override {
        m_values[name] = std::to_string(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override {
        m_values[name] = std::to_string(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<double>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int8_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int16_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int32_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint8_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint16_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint32_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }

    std::map<std::string, std::string> get_parameters() const {
        return m_values;
    }

protected:
    std::map<std::string, std::string> m_values;
};

void CreateCustomOp(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op, CustomLayerPtr customLayer) {
    if (!p.use_new_shape_infer()) {
        OPENVINO_ASSERT(op->get_output_size() == 1u, "Custom OP limitation: static model only support one output.");
    }
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    CustomLayerAttributeVisitor visitor;
    op->visit_attributes(visitor);
    auto params = visitor.get_parameters();

    // Handle defines
    std::string layerDefines;
    for (const auto& def : customLayer->Defines()) {
        std::string singleDefine("#define " + def.name + " " + def.prefix);
        if (params.find(def.param) != params.end()) {
            singleDefine += params.at(def.param);
        } else {
            singleDefine += def.default_value;
        }
        singleDefine += def.postfix + "\n";
        layerDefines.append(singleDefine);
    }

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
                // todo: add support for multiple reorders of the same input? (read as bfyx for one arg and yxfb for another)
                if (param.format != cldnn::format::any) {
                    auto reorderPrimName = inputs[param.portIndex].pid + "_" + op->get_friendly_name() + ProgramBuilder::m_preCustomLayerTag;
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
        default:
            OPENVINO_THROW("Invalid custom layer param type: ", param.type, " in operation: ", op->get_friendly_name());
        }
    }
    const std::string layerTitle("\n// Layer " + op->get_friendly_name() + " using Custom Layer " + customLayer->Name() + "\n");
    const std::string defineTitle("// Custom Layer User Defines\n");

    int iidx = customLayer->InputDimSourceIndex();
    OPENVINO_ASSERT(outputFormats.size() == op->get_output_size(), "The number of outputFormats should be same as op->get_output_size().");

    std::vector<cldnn::layout> outputLayouts(op->get_output_size());
    for (size_t i = 0; i < op->get_output_size(); i++) {
        auto dims = op->get_output_partial_shape(i);

        constexpr size_t kDynamic = std::numeric_limits<size_t>::max();
        size_t N = (dims.size() > 0) ? dims[0].is_dynamic() ? kDynamic : dims[0].get_length() : 1;
        size_t C = (dims.size() > 1) ? dims[1].is_dynamic() ? kDynamic : dims[1].get_length() : 1;
        size_t H = (dims.size() > 2) ? dims[2].is_dynamic() ? kDynamic : dims[2].get_length() : 1;
        size_t W = (dims.size() > 3) ? dims[3].is_dynamic() ? kDynamic : dims[3].get_length() : 1;

        if (dims.is_dynamic()) {
            outputLayouts[i] = cldnn::layout(dims, cldnn::element_type_to_data_type(op->get_output_element_type(i)), outputFormats[i]);
        } else {
            cldnn::tensor outputTensor = cldnn::tensor(cldnn::batch(N), cldnn::feature(C), cldnn::spatial(W, H));
            outputLayouts[i] = cldnn::layout(cldnn::element_type_to_data_type(op->get_output_element_type(i)), outputFormats[i], outputTensor);
        }
    }

    std::vector<size_t> gws, lws;

    // if input index is greater than -1, take dimension from input
    if (iidx >= 0) {
        if (static_cast<size_t>(iidx) >= op->get_input_size())
            OPENVINO_THROW("Invalid input tensor for index: ", iidx);
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
                                                  {layerTitle, defineTitle, layerDefines, customLayer->KernelSource()},
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
