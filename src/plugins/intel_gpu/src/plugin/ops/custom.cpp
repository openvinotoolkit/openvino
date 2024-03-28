// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/simple_math.hpp"
#include "intel_gpu/primitives/custom_gpu_primitive.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"

namespace ov {
namespace intel_gpu {

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

void CreatePagedAttention(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op) {
    validate_inputs_count(op, {13});
    auto inputs = p.GetInputInfo(op);
    auto prim = cldnn::paged_attention(layer_type_name_ID(op), inputs);

    // These parameters should be obtained from PA inputs, but currently inputs have fully dynamic shapes
    // query_shape = [batch_size, seq_len, heads_num * head_size]
    // const auto query_shape = query_layout.get_shape();
    // key_cache_shape = [num_blocks, kv_heads_num, head_size / x_size, block_size, x_size]
    // const auto key_cache_shape = key_cache_layout.get_shape();
    // value_cache_shape = [num_blocks, kv_heads_num, head_size, block_size]
    // const auto value_cache_shape = value_cache_layout.get_shape();
    // const size_t hidden_size = query_shape[2];
    // const size_t kv_heads_num = value_cache_shape[1];
    // const size_t head_size = value_cache_shape[2];
    // const size_t heads_num = hidden_size / head_size;
    // const size_t block_size = value_cache_shape[3];
    // const size_t x_size = key_cache_shape[4];

    prim.head_size = 128;
    prim.heads_num = 32;
    prim.kv_heads_num = 32;
    prim.block_size = 16;
    prim.x_block_size = 8;

    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);
    prim.output_paddings = get_output_paddings(op);

    OPENVINO_ASSERT(prim.num_outputs == 1, "[GPU] Unexpected outputs number");

    p.add_primitive(*op, prim);
}

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

    auto dims = op->get_output_shape(0);
    size_t N = (dims.size() > 0) ? dims[0] : 1;
    size_t C = (dims.size() > 1) ? dims[1] : 1;
    size_t H = (dims.size() > 2) ? dims[2] : 1;
    size_t W = (dims.size() > 3) ? dims[3] : 1;
    cldnn::tensor outputTensor = cldnn::tensor(cldnn::batch(N), cldnn::feature(C), cldnn::spatial(W, H));

    cldnn::layout outputLayout = cldnn::layout(cldnn::element_type_to_data_type(op->get_output_element_type(0)), outputFormat, outputTensor);

    // evaluate work sizes rules
    std::vector<size_t> gws, lws;

    // assume output tensor is dimension source by default
    int batchDim = outputTensor.batch[0];
    int featureDim = outputTensor.feature[0];
    int yDim = outputTensor.spatial[1];
    int xDim = outputTensor.spatial[0];
    int iidx = customLayer->InputDimSourceIndex();

    std::string genericLayerName = layer_type_name_ID(op);
    // if input index is greater than -1, take dimension from input
    if (iidx >= 0) {
        if (static_cast<size_t>(iidx) >= op->get_input_size())
            OPENVINO_THROW("Invalid input tensor for index: ", iidx);
        auto inputDims = op->get_input_shape(iidx);

        xDim = static_cast<int>(inputDims[inputDims.size() - 1]);
        yDim = dims.size() > 1 ? static_cast<int>(inputDims[inputDims.size() - 2]) : 0;
        featureDim = dims.size() > 2 ? static_cast<int>(inputDims[inputDims.size() - 3]) : 0;
        batchDim = dims.size() > 3 ? static_cast<int>(inputDims[inputDims.size() - 4]) : 0;
    }
    const std::map<char, int> vars = {
        { 'b', batchDim }  , { 'B', batchDim },
        { 'f', featureDim }, { 'F', featureDim },
        { 'y', yDim },       { 'Y', yDim },
        { 'x', xDim },       { 'X', xDim },
    };
    for (const auto& rule : customLayer->GlobalSizeRules()) {
        SimpleMathExpression expr;
        expr.SetVariables(vars);
        expr.SetExpression(rule);
        gws.push_back(expr.Evaluate());
    }
    for (const auto& rule : customLayer->LocalSizeRules()) {
        SimpleMathExpression expr;
        expr.SetVariables(vars);
        expr.SetExpression(rule);
        lws.push_back(expr.Evaluate());
    }

    auto customPrim = cldnn::custom_gpu_primitive(genericLayerName,
                                                  reordered_inputs,
                                                  { layerTitle, defineTitle, layerDefines, customLayer->KernelSource() },
                                                  customLayer->KernelEntry(),
                                                  kernelParameters,
                                                  customLayer->CompilerOptions(),
                                                  outputLayout,
                                                  gws,
                                                  lws);
    p.add_primitive(*op, customPrim);

    auto prevLayerName = genericLayerName;
    if (outputLayout.format != cldnn::format::any) {
        // Handle output reorder
        auto reorderPrimName = genericLayerName + ProgramBuilder::m_postCustomLayerTag;
        p.add_primitive(*op, cldnn::reorder(reorderPrimName,
                                            cldnn::input_info(genericLayerName),
                                            cldnn::format::get_default_format(op->get_output_shape(0).size()),
                                            customPrim.output_layout.data_type));
        prevLayerName = reorderPrimName;
    }
}

}  // namespace intel_gpu
}  // namespace ov
