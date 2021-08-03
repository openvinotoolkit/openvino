// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>

#include "cldnn_program.h"
#include "cldnn_common_utils.h"
#include "simple_math.h"

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/node.hpp"

#include "cldnn/primitives/custom_gpu_primitive.hpp"
#include "cldnn/primitives/generic_primitive.hpp"
#include "cldnn/primitives/reorder.hpp"

using namespace InferenceEngine;

namespace CLDNNPlugin {

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

class CustomLayerAttributeVisitor : public ngraph::AttributeVisitor {
public:
    CustomLayerAttributeVisitor() : m_values({}) { }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override {
        IE_THROW() << "Attribute " << name << " can't be processed\n";
    }
    // The remaining adapter methods fall back on the void adapter if not implemented
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::string>& adapter) override {
        m_values[name] = adapter.get();
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<bool>& adapter) override {
        m_values[name] = std::to_string(adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) override {
        m_values[name] = std::to_string(adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<double>& adapter) override {
        m_values[name] = std::to_string(adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<std::string>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<float>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<double>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int8_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int16_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int32_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint8_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint16_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint32_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }

    std::map<std::string, std::string> get_parameters() const {
        return m_values;
    }

protected:
    std::map<std::string, std::string> m_values;
};

void CreateCustomOp(Program& p, const std::shared_ptr<ngraph::Node>& op, CLDNNCustomLayerPtr customLayer) {
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
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
    std::vector<cldnn::primitive_id> reorderedInputs;
    reorderedInputs.resize(inputPrimitives.size());

    // Handle kernel parameters
    std::vector<cldnn::custom_gpu_primitive::arg_desc> kernelParameters;
    cldnn::format outputFormat(cldnn::format::any);
    for (const auto& param : customLayer->KernelParams()) {
        switch (param.type) {
        case CLDNNCustomLayer::ParamType::Input: {
            kernelParameters.resize(kernelParameters.size() > size_t(param.paramIndex + 1) ? kernelParameters.size() : size_t(param.paramIndex + 1));
            kernelParameters[param.paramIndex].type = cldnn::custom_gpu_primitive::arg_input;
            kernelParameters[param.paramIndex].index =
                static_cast<cldnn::custom_gpu_primitive::arg_index>((param.portIndex >= inputPrimitives.size()) ? -1 : param.portIndex);

            // Handle input reorder
            if (param.portIndex < inputPrimitives.size() && reorderedInputs[param.portIndex].empty()) {
                // todo: add support for multiple reorders of the same input? (read as bfyx for one arg and yxfb for another)
                if (param.format != cldnn::format::any) {
                    auto reorderPrimName = inputPrimitives[param.portIndex] + "_" + op->get_friendly_name() + Program::m_preCustomLayerTag;
                    auto preprocessPrim = cldnn::reorder(
                        reorderPrimName,
                        inputPrimitives[param.portIndex],
                        param.format,
                        DataTypeFromPrecision(op->get_input_element_type(param.portIndex)));

                    p.AddPrimitive(preprocessPrim);
                    p.AddInnerPrimitiveToProfiler(reorderPrimName, layer_type_name_ID(op), op);
                    reorderedInputs[param.portIndex] = (reorderPrimName);
                } else {
                    reorderedInputs[param.portIndex] = inputPrimitives[param.portIndex];
                }
            }
            break;
        }
        case CLDNNCustomLayer::ParamType::Output: {
            kernelParameters.resize(kernelParameters.size() > size_t(param.paramIndex + 1) ? kernelParameters.size() : size_t(param.paramIndex + 1));
            kernelParameters[param.paramIndex].type = cldnn::custom_gpu_primitive::arg_output;
            kernelParameters[param.paramIndex].index =
                static_cast<cldnn::custom_gpu_primitive::arg_index>((param.portIndex >= inputPrimitives.size()) ? -1 : param.portIndex);
            outputFormat = param.format;
            break;
        }
        default:
            IE_THROW() << "Invalid custom layer param type: " << param.type << " in operation: " << op->get_friendly_name();
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

    cldnn::layout outputLayout = cldnn::layout(DataTypeFromPrecision(op->get_output_element_type(0)), outputFormat, outputTensor);

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
        if (iidx >= op->get_input_size())
            IE_THROW() << "Invalid input tensor for index: " << iidx;
        auto inputDims = op->get_input_shape(iidx);

        xDim = inputDims[inputDims.size() - 1];
        yDim = dims.size() > 1 ? inputDims[inputDims.size() - 2] : 0;
        featureDim = dims.size() > 2 ? inputDims[inputDims.size() - 3] : 0;
        batchDim = dims.size() > 3 ? inputDims[inputDims.size() - 4]: 0;
    }
    const std::map<char, int> vars = {
        { 'b', batchDim }  , { 'B', batchDim },
        { 'f', featureDim }, { 'F', featureDim },
        { 'y', yDim },       { 'Y', yDim },
        { 'x', xDim },       { 'X', xDim },
    };
    for (auto rule : customLayer->GlobalSizeRules()) {
        SimpleMathExpression expr;
        expr.SetVariables(vars);
        expr.SetExpression(rule);
        gws.push_back(expr.Evaluate());
    }
    for (auto rule : customLayer->LocalSizeRules()) {
        SimpleMathExpression expr;
        expr.SetVariables(vars);
        expr.SetExpression(rule);
        lws.push_back(expr.Evaluate());
    }

    auto customPrim = cldnn::custom_gpu_primitive(genericLayerName,
                                                  reorderedInputs,
                                                  { layerTitle, defineTitle, layerDefines, customLayer->KernelSource() },
                                                  customLayer->KernelEntry(),
                                                  kernelParameters,
                                                  customLayer->CompilerOptions(),
                                                  outputLayout,
                                                  gws,
                                                  lws);

    auto prevLayerName = genericLayerName;
    if (outputLayout.format != cldnn::format::any) {
        // Handle output reorder
        auto reorderPrimName = genericLayerName + Program::m_postCustomLayerTag;
        p.AddPrimitive(
            cldnn::reorder(reorderPrimName,
                           genericLayerName,
                           DefaultFormatForDims(op->get_output_shape(0).size()),
                           customPrim.output_layout.data_type));
        prevLayerName = reorderPrimName;
        p.AddInnerPrimitiveToProfiler(reorderPrimName, layer_type_name_ID(op), op);
    }
    p.AddPrimitive(customPrim);
    p.AddPrimitiveToProfiler(genericLayerName, op);
    p.primitiveIDs[genericLayerName] = prevLayerName;
}

TensorDesc IETensorFromClnnLayout(const cldnn::layout &layout) {
    auto dataType = layout.data_type;
    Precision p;
    switch(dataType) {
        case cldnn::data_types::u8: p = Precision::U8; break;
        case cldnn::data_types::i8: p = Precision::I8; break;
        case cldnn::data_types::i32: p = Precision::I32; break;
        case cldnn::data_types::i64: p = Precision::I64; break;
        case cldnn::data_types::f16: p = Precision::FP16; break;
        case cldnn::data_types::f32: p = Precision::FP32; break;
        default:
            IE_THROW() << "The plugin does not support input " << cldnn::data_type_traits::name(dataType) << " precision";
    }

    auto const & tensor = layout.size;
    return TensorDesc{
        InferenceEngine::Precision::U8,
        {tensor.batch[0], tensor.feature[0], tensor.spatial[0], tensor.spatial[1] },
        Layout::NCHW // TODO: this is determined by what?
    };
}

void CreateGenericPrimitiveOp(Program &p, const std::shared_ptr<ngraph::Node>& op,
                              InferenceEngine::ILayerExecImpl::Ptr impl) {
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string genericLayerName = layer_type_name_ID(op);
    auto context = p.GetContextPtr();

    // TODO: Handle reordering input/output dims... Requires a way to describe input/output layout
    cldnn::generic_primitive::execute_function f = [genericLayerName, impl, context](
            const std::vector<cldnn::event::ptr>& dependent_events,
            const std::vector<cldnn::memory::ptr>& inputs,
            const std::vector<cldnn::memory::ptr>& outputs) {
        cldnn::stream &stream = inputs[0]->get_engine()->get_program_stream();

        // TODO: we don't want to wait for the events, we just want the user to submit onto the queue
        for (auto& ev : dependent_events) {
            ev->wait();
        }

        // TODO: how do we have the user return an event?
        cldnn::event::ptr ev = stream.create_user_event(false);

        std::vector<InferenceEngine::Blob::Ptr> inputBlobs(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto& input = inputs[i];
            // Construct a TensorDesc from memory information:
            auto tensor = input->get_layout().size;
            TensorDesc desc = IETensorFromClnnLayout(input->get_layout());

            auto params = input->get_internal_params();
            assert(params.mem_type == cldnn::shared_mem_type::shared_mem_buffer);

            inputBlobs[i] = InferenceEngine::gpu::make_shared_blob(desc, context, static_cast<cl_mem>(params.mem));
        }

        std::vector<InferenceEngine::Blob::Ptr> outputBlobs(outputs.size());
        for (size_t i = 0; i < outputBlobs.size(); ++i) {
            auto& output = outputs[i];
            TensorDesc desc = IETensorFromClnnLayout(output->get_layout());

            auto params = output->get_internal_params();
            assert(params.mem_type == cldnn::shared_mem_type::shared_mem_buffer);

            outputBlobs[i] = InferenceEngine::gpu::make_shared_blob(desc, context, static_cast<cl_mem>(params.mem));
        }

        InferenceEngine::ResponseDesc resp;
        InferenceEngine::StatusCode rc = impl->execute(inputBlobs, outputBlobs, &resp);

        if (rc != InferenceEngine::OK) {
            IE_THROW() << "clDNN: " << genericLayerName << ": " << resp.msg;
        }

        ev->set();
        return ev;
    };

    auto outSize = CldnnTensorFromIEDims(op->get_output_shape(0));
    cldnn::layout outputLayout = cldnn::layout(DataTypeFromPrecision(op->get_output_element_type(0)),
                                               cldnn::format::any, // TODO: allow for user-defined input/output formats?
                                               outSize);

    auto prim = cldnn::generic_primitive(genericLayerName, inputPrimitives, f, outputLayout);

    p.AddPrimitive(prim);
    p.AddPrimitiveToProfiler(op->get_friendly_name(), op);
    p.primitiveIDs[genericLayerName] = genericLayerName;
}

}  // namespace CLDNNPlugin
