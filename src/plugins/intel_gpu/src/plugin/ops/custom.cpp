// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"

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

template <typename T>
T convert_to(const std::string &str) {
    std::istringstream ss(str);
    T res;
    ss >> res;
    return res;
}

template <>
std::string convert_to(const std::string &str) {
    return str;
}

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

    if (const auto env_var = std::getenv("PA_HEAD_SIZE")) {
        prim.head_size = convert_to<size_t>(env_var);
    }

    if (const auto env_var = std::getenv("PA_HEADS_NUM")) {
        prim.heads_num = convert_to<size_t>(env_var);
    }

    if (const auto env_var = std::getenv("PA_KV_HEADS_NUM")) {
        prim.kv_heads_num = convert_to<size_t>(env_var);
    }

    if (const auto env_var = std::getenv("PA_BLOCK_SIZE")) {
        prim.block_size = convert_to<size_t>(env_var);
    }

    if (const auto env_var = std::getenv("PA_X_BLOCK_SIZE")) {
        prim.x_block_size = convert_to<size_t>(env_var);
    }

    auto key_cache_ps = op->get_input_partial_shape(3);
    prim.head_size = key_cache_ps[2].get_length();
    prim.heads_num = key_cache_ps[1].get_length();
    prim.kv_heads_num = key_cache_ps[1].get_length();

    std::shared_ptr<ov::op::v0::Constant> scale_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(9));
    OPENVINO_ASSERT(scale_const != nullptr);
    OPENVINO_ASSERT(ov::shape_size(scale_const->get_output_shape(0)) == 1);

    prim.scale_val = scale_const->cast_vector<float>()[0];

    // Q - f16:bfyx:?x768:nopad
    // K - f16:bfyx:?x768:nopad
    // V - f16:bfyx:?x768:nopad
    // key_cache - f16:bfyx:?x12x?x64:nopad
    // value_cache - f16:bfyx:?x12x?x64:nopad

    // past_lens - i32:bfyx:?:nopad
    // subsequence_begins - i32:bfyx:?:nopad
    // block_indices - i32:bfyx:?:nopad
    // block_indices_begins - i32:bfyx:?:nopad
    // scale - f16:bfyx::nopad

    // sliding_window - i32:bfyx::nopad
    // alibi_slopes - f16:bfyx:0:nopad
    // max_context_len - i32:bfyx::nopad


    // update shape dep [0] : reshape:Reshape_8733 was: f16:bfyx:?x768:nopad now: f16:bfyx:87x768:nopad
    // update shape dep [1] : reshape:Reshape_8737 was: f16:bfyx:?x768:nopad now: f16:bfyx:87x768:nopad
    // update shape dep [2] : reshape:Reshape_8739 was: f16:bfyx:?x768:nopad now: f16:bfyx:87x768:nopad
    // update shape dep [3] : parameter:key_cache.0 was: f16:bfyx:?x12x?x64:nopad now: f16:bfyx:3640x12x32x64:nopad
    // update shape dep [4] : parameter:value_cache.0 was: f16:bfyx:?x12x?x64:nopad now: f16:bfyx:3640x12x32x64:nopad

    // update shape dep [5] : parameter:past_lens was: i32:bfyx:?:nopad now: i32:bfyx:2:nopad
    // update shape dep [6] : parameter:subsequence_begins was: i32:bfyx:?:nopad now: i32:bfyx:3:nopad
    // update shape dep [7] : parameter:block_indices was: i32:bfyx:?:nopad now: i32:bfyx:4:nopad
    // update shape dep [8] : parameter:block_indices_begins was: i32:bfyx:?:nopad now: i32:bfyx:3:nopad



    // Test with 2 prompts: 81 and 6 tokens
    // update shape dep [0] : reshape:Reshape_8733 was: f16:bfyx:?x768:nopad now: f16:bfyx:87x768:nopad
    // update shape dep [1] : reshape:Reshape_8737 was: f16:bfyx:?x768:nopad now: f16:bfyx:87x768:nopad
    // update shape dep [2] : reshape:Reshape_8739 was: f16:bfyx:?x768:nopad now: f16:bfyx:87x768:nopad
    // update shape dep [3] : parameter:key_cache.0 was: f16:bfyx:?x12x?x64:nopad now: f16:bfyx:3640x12x32x64:nopad
    // update shape dep [4] : parameter:value_cache.0 was: f16:bfyx:?x12x?x64:nopad now: f16:bfyx:3640x12x32x64:nopad
    // update shape dep [5] : parameter:past_lens was: i32:bfyx:?:nopad now: i32:bfyx:2:nopad
    // update shape dep [6] : parameter:subsequence_begins was: i32:bfyx:?:nopad now: i32:bfyx:3:nopad
    // update shape dep [7] : parameter:block_indices was: i32:bfyx:?:nopad now: i32:bfyx:4:nopad
    // update shape dep [8] : parameter:block_indices_begins was: i32:bfyx:?:nopad now: i32:bfyx:3:nopad
    // Input #5:  Array (len=2) content: 0, 0,
    // Input #6:  Array (len=3) content: 0, 81, 87,
    // Input #7:  Array (len=4) content: 3639, 3638, 3637, 3636,
    // Input #8:  Array (len=3) content: 0, 3, 4,  <= ranges of block_indices: 1st prompt uses: block_indices[0], block_indices[1], block_indices[2] blocks; 2nd block_indices[3]
    // Input #12: Array (len=1) content: 81,
    // GWS should be: (96 + 16) / 16 = 7
    // Additional input for Q:  i32:bfyx:7:nopad with content {0, 16, 32, 48, 64, 80, 81, 87}
    // Additional input for KV: i32:bfyx:7:nopad with content {0,  0,  0,  0,  0,  0,  1,  1}



    // Test with 4 prompts:
    // 1st infer context lens: 81, 6, 31, 33
    // sdpa kernel block_size = 16
    // GWS calculation?
    // Input0 size: 151x768
    // Input5 size: 4
    // GWS should be: (96 + 16 + 32 + 48) / 16 = 12

    // Additional input: i32:bfyx:12:nopad with content {}


    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);
    prim.output_paddings = get_output_paddings(op);

    GPU_DEBUG_TRACE_DETAIL << "PA op->get_output_size(): " << op->get_output_size() << "\n";
    GPU_DEBUG_TRACE_DETAIL << "PA op->get_input_size(): " << op->get_input_size() << "\n";

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
