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

struct select_params
{
    std::string device_name;
    SizeVector input1_tensor;
    SizeVector input2_tensor;
    SizeVector mask_tensor;
    std::string auto_broadcast;
    bool fail_expected;
};

class SelectTests : public TestsCommon, public WithParamInterface<select_params> {
    std::string model_base = R"V0G0N(
    <net name="Select_net" version="7">
        <layers>
            <layer name="cond" type="Input" id="0" version="opset1">
                <data element_type="boolean" shape="_MASK_SHAPE_"/>
                <output>
                    <port id="0" precision="BOOL">_MASK_DIMS_</port>
                </output>
            </layer>
            <layer name="input1" type="Input" id="1" version="opset1">
                <data element_type="f32" shape="_INPUT1_SHAPE_"/>
                <output>
                    <port id="0" precision="FP32">_INPUT1_DIMS_</port>
                </output>
            </layer>
            <layer name="input2" type="Input" id="2" version="opset1">
                <data element_type="f32" shape="_INPUT2_SHAPE_"/>
                <output>
                    <port id="0" precision="FP32">_INPUT2_DIMS_</port>
                </output>
            </layer>
            <layer name="select" id="3" type="Select" version="opset1">
                <data auto_broadcast="_AUTO_BROADCAST_"/>
                <input>
                    <port id="0">_MASK_DIMS_</port>
                    <port id="1">_INPUT1_DIMS_</port>
                    <port id="2">_INPUT2_DIMS_</port>
                </input>
                <output>
                    <port id="3" precision="FP32">_OUTPUT_DIMS_</port>
                </output>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
            <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
        </edges>
    </net>
    )V0G0N";

    SizeVector get_output_tensor(const SizeVector& cond_dims, const SizeVector& input1_dims, const SizeVector& input2_dims)
    {
        auto max_in_size = std::max({cond_dims.size(), input1_dims.size(), input2_dims.size()});
        auto out_size = std::max(max_in_size, (size_t)4);

        SizeVector cond_dims_extended = cond_dims;
        SizeVector in1_dims_extended = input1_dims;
        SizeVector in2_dims_extended = input2_dims;

        cond_dims_extended.insert(cond_dims_extended.begin(), out_size - cond_dims_extended.size(), 1);
        in1_dims_extended.insert(in1_dims_extended.begin(), out_size - in1_dims_extended.size(), 1);
        in2_dims_extended.insert(in2_dims_extended.begin(), out_size - in2_dims_extended.size(), 1);

        SizeVector output_tensor(out_size, 1);

        for (size_t i = 0; i < output_tensor.size(); i++) {
            output_tensor[i] = std::max({ cond_dims_extended[i], in1_dims_extended[i], in2_dims_extended[i] });
        }

        return output_tensor;
    }

    std::string getModel(select_params p) {
        std::string mask_shape_str = "";
        std::string mask_dims_str = "";

        for (size_t i=0; i<p.mask_tensor.size(); i++) {
            mask_shape_str += std::to_string(p.mask_tensor[i]);
            mask_dims_str += "\n                        ";
            mask_dims_str += "<dim>" + std::to_string(p.mask_tensor[i]) + "</dim>";
            if (i < p.mask_tensor.size() - 1) {
                mask_shape_str += ",";
            } else {
                mask_dims_str += "\n                    ";
            }
        }

        std::string input1_shape_str = "";
        std::string input1_dims_str = "";

        for (size_t i=0; i<p.input1_tensor.size(); i++) {
            input1_shape_str += std::to_string(p.input1_tensor[i]);
            input1_dims_str += "\n                        ";
            input1_dims_str += "<dim>" + std::to_string(p.input1_tensor[i]) + "</dim>";
            if (i < p.input1_tensor.size() - 1) {
                input1_shape_str += ",";
            } else {
                input1_dims_str += "\n                    ";
            }
        }

        std::string input2_shape_str = "";
        std::string input2_dims_str = "";

        for (size_t i=0; i<p.input2_tensor.size(); i++) {
            input2_shape_str += std::to_string(p.input2_tensor[i]);
            input2_dims_str += "\n                        ";
            input2_dims_str += "<dim>" + std::to_string(p.input2_tensor[i]) + "</dim>";
            if (i < p.input2_tensor.size() - 1) {
                input2_shape_str += ",";
            } else {
                input2_dims_str += "\n                    ";
            }
        }

        SizeVector output_tensor = get_output_tensor(p.mask_tensor, p.input1_tensor, p.input2_tensor);

        std::string output_shape_str = "";
        std::string output_dims_str = "";

        for (size_t i=0; i<output_tensor.size(); i++) {
            output_shape_str += std::to_string(output_tensor[i]);
            output_dims_str += "\n                        ";
            output_dims_str += "<dim>" + std::to_string(output_tensor[i]) + "</dim>";
            if (i < output_tensor.size() - 1) {
                output_shape_str += ",";
            } else {
                output_dims_str += "\n                    ";
            }
        }

        REPLACE_WITH_STR(model_base, "_MASK_SHAPE_", mask_shape_str);
        REPLACE_WITH_STR(model_base, "_MASK_DIMS_", mask_dims_str);

        REPLACE_WITH_STR(model_base, "_INPUT1_SHAPE_", input1_shape_str);
        REPLACE_WITH_STR(model_base, "_INPUT1_DIMS_", input1_dims_str);

        REPLACE_WITH_STR(model_base, "_INPUT2_SHAPE_", input2_shape_str);
        REPLACE_WITH_STR(model_base, "_INPUT2_DIMS_", input2_dims_str);

        REPLACE_WITH_STR(model_base, "_OUTPUT_SHAPE_", output_shape_str);
        REPLACE_WITH_STR(model_base, "_OUTPUT_DIMS_", output_dims_str);

        REPLACE_WITH_STR(model_base, "_AUTO_BROADCAST_", p.auto_broadcast);

        return model_base;
    }

    size_t get_index_bfhw(SizeVector tensor, size_t b, size_t f, size_t h, size_t w)
    {
        if ((tensor.size() < 4) || (b >= tensor[tensor.size() - 4])) b = 0;
        if ((tensor.size() < 3) || (f >= tensor[tensor.size() - 3])) f = 0;
        if ((tensor.size() < 2) || (h >= tensor[tensor.size() - 2])) h = 0;
        if ((tensor.size() < 1) || (w >= tensor[tensor.size() - 1])) w = 0;

        size_t res = 0;

        size_t b_multiplier = 1;
        if (tensor.size() >= 3) {
            b_multiplier = std::accumulate(std::end(tensor) - 3, std::end(tensor), 1, std::multiplies<size_t>());
        } else {
            b_multiplier = std::accumulate(std::begin(tensor), std::end(tensor), 1, std::multiplies<size_t>());
        }
        res += b * b_multiplier;

        size_t f_multiplier = 1;
        if (tensor.size() >= 2) {
            f_multiplier = std::accumulate(std::end(tensor) - 2, std::end(tensor), 1, std::multiplies<size_t>());
        } else {
            f_multiplier = std::accumulate(std::begin(tensor), std::end(tensor), 1, std::multiplies<size_t>());
        }
        res += f * f_multiplier;

        size_t h_multiplier = 1;
        if (tensor.size() >= 1) {
            h_multiplier = std::accumulate(std::end(tensor) - 1, std::end(tensor), 1, std::multiplies<size_t>());
        }
        res += h * h_multiplier;

        res += w;
        return res;
    }

    void check_output(const float* input1, const float* input2, const uint8_t* mask, const float* output, select_params p) {

        SizeVector output_tensor = get_output_tensor(p.mask_tensor, p.input1_tensor, p.input2_tensor);

        size_t b_max = (output_tensor.size() > 0) ? output_tensor[0] : 1;
        size_t f_max = (output_tensor.size() > 1) ? output_tensor[1] : 1;
        size_t h_max = (output_tensor.size() > 2) ? output_tensor[2] : 1;
        size_t w_max = (output_tensor.size() > 3) ? output_tensor[3] : 1;

        for (size_t b = 0; b < b_max; b++) {
            for (size_t f = 0; f < f_max; f++) {
                for (size_t h = 0; h < h_max; h++) {
                    for (size_t w = 0; w < w_max; w++) {
                        if (mask[get_index_bfhw(p.mask_tensor, b, f, h, w)] == 0)
                        {
                            EXPECT_EQ(output[get_index_bfhw(output_tensor, b, f, h, w)],
                                      input2[get_index_bfhw(p.input2_tensor, b, f, h, w)]);
                        }
                        else
                        {
                            EXPECT_EQ(output[get_index_bfhw(output_tensor, b, f, h, w)],
                                      input1[get_index_bfhw(p.input1_tensor, b, f, h, w)]);
                        }
                    }
                }
            }
        }
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        bool fail_expected = false;
        try {
            select_params p = ::testing::WithParamInterface<select_params>::GetParam();
            fail_expected = p.fail_expected;

            Core ie;
            CNNNetwork net = ie.ReadNetwork(getModel(p), Blob::Ptr());
            InputsDataMap in_info_map = net.getInputsInfo();
            OutputsDataMap out_info_map = net.getOutputsInfo();

            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest infer_request = executable_network.CreateInferRequest();

            uint8_t* mask;
            float* input1_ptr, *input2_ptr;
            auto input_iterator = in_info_map.begin();
            size_t input1_buffer_size = std::accumulate(std::begin(p.input1_tensor), std::end(p.input1_tensor), 1, std::multiplies<size_t>());
            size_t input2_buffer_size = std::accumulate(std::begin(p.input2_tensor), std::end(p.input2_tensor), 1, std::multiplies<size_t>());

            // Creating mask buffer
            // If true, take value from first buffer, if false, take from second
            Blob::Ptr maskBlob = infer_request.GetBlob(input_iterator->first);
            mask = maskBlob->buffer().as<uint8_t*>();
            for (size_t id = 0; id < maskBlob->size(); id++) {
                mask[id] = (id % 2);
            }
            input_iterator++;

            // Inputs random generator
            Blob::Ptr input1Blob = infer_request.GetBlob(input_iterator->first);
            input_iterator++;
            Blob::Ptr input2Blob = infer_request.GetBlob(input_iterator->first);
            input1_ptr = input1Blob->buffer().as<float*>();
            input2_ptr = input2Blob->buffer().as<float*>();
            for (int index = 0; index < input1_buffer_size; index++) {
                input1_ptr[index] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            }
            for (int index = 0; index < input2_buffer_size; index++) {
                input2_ptr[index] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            }

            // Output allocation
            SizeVector output_tensor = get_output_tensor(p.mask_tensor, p.input1_tensor, p.input2_tensor);
 
            Blob::Ptr outputBlob = infer_request.GetBlob(out_info_map.begin()->first);
            TBlob<float> dst_ref({ Precision::FP32, output_tensor, Layout::NCHW });
            dst_ref.allocate();

            infer_request.Infer();

            // Output buffer
            outputBlob = infer_request.GetBlob(out_info_map.begin()->first);
            const float* output_ptr = outputBlob->buffer().as<float*>();

            check_output(input1_ptr, input2_ptr, mask, output_ptr, p);
        }
        catch (const InferenceEngine::details::InferenceEngineException & e) {
            if (!fail_expected) {
                FAIL() << e.what();
            }
        }
    }
};
