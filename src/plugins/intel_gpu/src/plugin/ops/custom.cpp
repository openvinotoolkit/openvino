// Copyright (C) 2018-2025 Intel Corporation
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
    cldnn::format outputFormat(cldnn::format::any);
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
            outputFormat = param.format;
            break;
        }
        default:
            OPENVINO_THROW("Invalid custom layer param type: ", param.type, " in operation: ", op->get_friendly_name());
        }
    }
    const std::string layerTitle("\n// Layer " + op->get_friendly_name() + " using Custom Layer " + customLayer->Name() + "\n");
    const std::string defineTitle("// Custom Layer User Defines\n");

    auto dims = op->get_output_partial_shape(0);
    int iidx = customLayer->InputDimSourceIndex();

    constexpr size_t kDynamic = std::numeric_limits<size_t>::max();
    size_t N = (dims.size() > 0) ? dims[0].is_dynamic() ? kDynamic : dims[0].get_length() : 1;
    size_t C = (dims.size() > 1) ? dims[1].is_dynamic() ? kDynamic : dims[1].get_length() : 1;
    size_t H = (dims.size() > 2) ? dims[2].is_dynamic() ? kDynamic : dims[2].get_length() : 1;
    size_t W = (dims.size() > 3) ? dims[3].is_dynamic() ? kDynamic : dims[3].get_length() : 1;

    cldnn::layout outputLayout;
    if (dims.is_dynamic()) {
        outputLayout = cldnn::layout(dims, cldnn::element_type_to_data_type(op->get_output_element_type(0)), outputFormat);
    } else {
        cldnn::tensor outputTensor = cldnn::tensor(cldnn::batch(N), cldnn::feature(C), cldnn::spatial(W, H));
        outputLayout = cldnn::layout(cldnn::element_type_to_data_type(op->get_output_element_type(0)), outputFormat, outputTensor);
    }

    std::vector<size_t> gws, lws;

    // if input index is greater than -1, take dimension from input
    if (iidx >= 0) {
        if (static_cast<size_t>(iidx) >= op->get_input_size())
            OPENVINO_THROW("Invalid input tensor for index: ", iidx);
        auto inputDims = op->get_input_shape(iidx);
        cldnn::custom_gpu_primitive::update_work_group_size(dims, iidx, inputDims, customLayer->GlobalSizeRules(), customLayer->LocalSizeRules(), gws, lws);
    } else {
        cldnn::custom_gpu_primitive::update_work_group_size(dims,
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
                                                  outputLayout,
                                                  gws,
                                                  lws,
                                                  op_bk,
                                                  iidx,
                                                  customLayer->GlobalSizeRules(),
                                                  customLayer->LocalSizeRules());
    p.add_primitive(*op, customPrim);

    auto prevLayerName = genericLayerName;
    if (outputLayout.format != cldnn::format::any) {
        // Handle output reorder
        auto reorderPrimName = genericLayerName + ProgramBuilder::m_postCustomLayerTag;
        p.add_primitive(*op, cldnn::reorder(reorderPrimName,
                                            cldnn::input_info(genericLayerName),
                                            cldnn::format::get_default_format(op->get_output_partial_shape(0).size()),
                                            customPrim.output_layout.data_type));
        prevLayerName = reorderPrimName;
    }
}

}  // namespace ov::intel_gpu
