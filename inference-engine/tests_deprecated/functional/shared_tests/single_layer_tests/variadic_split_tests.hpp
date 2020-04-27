// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>
#include <cmath>
#include <string>

#include "tests_common.hpp"
#include "single_layer_common.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace std;

struct variadic_split_params
{
    std::string device_name;
    int axis;
    std::vector<int> variadic_lenghts;
    SizeVector input_dims;
    std::vector<SizeVector> output_dims;
};

class VariadicSplitTests : public TestsCommon, public WithParamInterface<variadic_split_params> {
    std::string model_base = R"V0G0N(
    <net name="Activation" version="10">
        <layers>
            <layer id="0" name="in1" type="Parameter"  version="opset1">
                <data element_type="f32" shape="_IB_,_IC_,_IH_,_IW_"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>_IB_</dim>
                        <dim>_IC_</dim>
                        <dim>_IH_</dim>
                        <dim>_IW_</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="const1" type="Const" version="opset1">
			    <data offset="0" size="8"/>
                <output>
                    <port id="0" precision="I64"/>
                </output>
            </layer>
            <layer id="2" name="const2" type="Const" version="opset1">
			    <data offset="8" size="_VARIADIC_LENGHTS_BYTE_SIZE_"/>
                <output>
                    <port id="0" precision="I64">
                        <dim>_VARIADIC_LENGHTS_SIZE_</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="split" type="VariadicSplit" version="opset1">
                <input>
                    <port id="0" precision="FP32">
                        <dim>_IB_</dim>
                        <dim>_IC_</dim>
                        <dim>_IH_</dim>
                        <dim>_IW_</dim>
                    </port>
                    <port id="1" precision="I64"/>
                    <port id="2" precision="I64">
                        <dim>_VARIADIC_LENGHTS_SIZE_</dim>
                    </port>
                </input>
                <output>
                    _VARIADIC_OUTPUTS_
                </output>
            </layer>
            _OUTPUT_LAYERS_
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
            <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
            _OUTPUT_PORTS_
        </edges>
    </net>
    )V0G0N";

    std::string getModel(variadic_split_params p) {
        std::string variadic_outputs, output_layers, output_ports;

        size_t variadic_port_id = 3;
        for (auto& size_vector : p.output_dims) {
            variadic_outputs += "<port id=\"" + std::to_string(variadic_port_id) + "\" precision=\"FP32\">\n";
            variadic_outputs += "<dim>" + std::to_string(size_vector[0]) + "</dim>\n";
            variadic_outputs += "<dim>" + std::to_string(size_vector[1]) + "</dim>\n";
            variadic_outputs += "<dim>" + std::to_string(size_vector[2]) + "</dim>\n";
            variadic_outputs += "<dim>" + std::to_string(size_vector[3]) + "</dim>\n";
            variadic_outputs += "</port>\n";
            variadic_port_id++;
        }

        size_t layer_id = 4;
        size_t layer_name_id = 1;
        for (auto& size_vector : p.output_dims) {
            output_layers += "<layer name=\"output" + std::to_string(layer_name_id) +  "\" type=\"Result\" id=\"" + std::to_string(layer_id) + "\" version=\"opset1\">\n";
            output_layers += "<input>\n";
            output_layers += "<port id=\"0\" precision=\"FP32\">\n";
            output_layers += "<dim>" + std::to_string(size_vector[0]) + "</dim>\n";
            output_layers += "<dim>" + std::to_string(size_vector[1]) + "</dim>\n";
            output_layers += "<dim>" + std::to_string(size_vector[2]) + "</dim>\n";
            output_layers += "<dim>" + std::to_string(size_vector[3]) + "</dim>\n";
            output_layers += "</port>\n";
            output_layers += "</input>\n";
            output_layers += "</layer>\n";
            layer_id++;
            layer_name_id++;
        }

        for (int id = 3; id < p.variadic_lenghts.size() + 3; id++) {
            output_ports += "<edge from-layer=\"3\" from-port=\"" + std::to_string(id) + "\" to-layer=\"" + std::to_string(id + 1) + "\" to-port=\"0\"/>\n";
        }

        REPLACE_WITH_STR(model_base, "_IB_", std::to_string(p.input_dims[0]));
        REPLACE_WITH_STR(model_base, "_IC_", std::to_string(p.input_dims[1]));
        REPLACE_WITH_STR(model_base, "_IH_", std::to_string(p.input_dims[2]));
        REPLACE_WITH_STR(model_base, "_IW_", std::to_string(p.input_dims[3]));

        REPLACE_WITH_STR(model_base, "_VARIADIC_LENGHTS_BYTE_SIZE_", std::to_string(p.variadic_lenghts.size() * sizeof(int64_t)));
        REPLACE_WITH_STR(model_base, "_VARIADIC_LENGHTS_SIZE_", std::to_string(p.variadic_lenghts.size()));
        REPLACE_WITH_STR(model_base, "_VARIADIC_OUTPUTS_", variadic_outputs);
        REPLACE_WITH_STR(model_base, "_OUTPUT_LAYERS_", output_layers);
        REPLACE_WITH_STR(model_base, "_OUTPUT_PORTS_", output_ports);

        return model_base;
    }

    size_t get_index_bfhw(SizeVector tensor, size_t b, size_t f, size_t h, size_t w)
    {
        size_t res = 0;
        res += b * (tensor[1] * tensor[2] * tensor[3]);
        res += f * (tensor[2] * tensor[3]);
        res += h * (tensor[3]);
        res += w;
        return res;
    }

    void check_buffers_after_split(InferRequest& inf_req, InputsDataMap& inputs, OutputsDataMap& outputs, variadic_split_params vs_params){
        Blob::Ptr inputBlob = inf_req.GetBlob(inputs.begin()->first);
        float* src_ptr = inputBlob->buffer().as<float*>();

        size_t outputs_number = outputs.size();
        std::vector<const float*> output_ptrs(outputs_number);

        // Getting raw output pointers
        OutputsDataMap::iterator output_it = outputs.begin();
        for (size_t index = 0; index < outputs_number; ++index) {
            Blob::Ptr temp_blob = inf_req.GetBlob(output_it->first);
            output_ptrs[index] = temp_blob->buffer().as<float*>();
            output_it++;
        }

        // Getting number of elements inside buffer
        auto input_tensor = vs_params.input_dims;
        size_t input_tensor_size = input_tensor[0] * input_tensor[1] * input_tensor[2] * input_tensor[3];
        std::vector<size_t> output_tensor_sizes(outputs_number);
        for (size_t output_id = 0; output_id < outputs_number; ++output_id) {
            auto output_tensors = vs_params.output_dims;
            output_tensor_sizes[output_id] =
                output_tensors[output_id][0] * output_tensors[output_id][1] * output_tensors[output_id][2] * output_tensors[output_id][3];
        }

        // Comparing input and output buffers
        SizeVector input_it_tensor = { 0, 0, 0, 0 };
        SizeVector output_tensor = { 0, 0, 0, 0 };
        for (size_t output_id = 0; output_id < outputs_number; ++output_id) {
            // Tensor iteration
            for (size_t b = input_it_tensor[0]; b < input_it_tensor[0] + vs_params.output_dims[output_id][0]; b++) {
                for (size_t f = input_it_tensor[1]; f < input_it_tensor[1] + vs_params.output_dims[output_id][1]; f++) {
                    for (size_t h = input_it_tensor[2]; h < input_it_tensor[2] + vs_params.output_dims[output_id][2]; h++) {
                        for (size_t w = input_it_tensor[3]; w < input_it_tensor[3] + vs_params.output_dims[output_id][3]; w++) {
                            ASSERT_EQ(
                                src_ptr[get_index_bfhw(vs_params.input_dims, b, f, h, w)],
                                output_ptrs[output_id][get_index_bfhw(vs_params.output_dims[output_id], output_tensor[0], output_tensor[1], output_tensor[2], output_tensor[3])]
                            );
                            output_tensor[3]++;
                        }
                        output_tensor[3] = 0;
                        output_tensor[2]++;
                    }
                    output_tensor[2] = 0;
                    output_tensor[1]++;
                }
                output_tensor[1] = 0;
                output_tensor[0]++;
            }
            output_tensor = { 0, 0, 0, 0 };
            input_it_tensor[vs_params.axis] += vs_params.variadic_lenghts[output_id];
        }
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            variadic_split_params p = ::testing::WithParamInterface<variadic_split_params>::GetParam();

            // Fill weights data
            auto fillBlob = [p](Blob::Ptr& weights) {
                auto* data = weights->buffer().as<int64_t*>();
                data[0] = p.axis;
                size_t id = 1;
                for (auto& variadic_lenght : p.variadic_lenghts)
                {
                    data[id] = variadic_lenght;
                    id++;
                }
            };

            // Allocate weights data for axis + variadic_lenghts vector
            Blob::Ptr weights;
            weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, { (1 + p.variadic_lenghts.size()) * sizeof(int64_t) }, Layout::C));
            weights->allocate();
            fill_data((float*)weights->buffer(), weights->size() / sizeof(float));
            fillBlob(weights);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(getModel(p), weights);
            InputsDataMap in_info_map = net.getInputsInfo();
            OutputsDataMap out_info_map = net.getOutputsInfo();

            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name );
            InferRequest infer_request = executable_network.CreateInferRequest();

            // Generate input data
            Blob::Ptr inputBlob = infer_request.GetBlob(in_info_map.begin()->first);
            float* src_ptr = inputBlob->buffer().as<float*>();
            fill_data(src_ptr, inputBlob->size());
         
            infer_request.Infer();

            check_buffers_after_split(infer_request, in_info_map, out_info_map, p);
        }
        catch (const InferenceEngine::details::InferenceEngineException & e) {
            FAIL() << e.what();
        }
    }
};
