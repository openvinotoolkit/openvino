// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <tests_common.hpp>
#include <tests_common_func.hpp>
#include <memory>
#include "xml_helper.hpp"
#include <ie_ir_reader.hpp>
#include <ie_core.hpp>

#define XBYAK_NO_OP_NAMES
#define XBYAK_UNDEF_JNL

using namespace ::testing;
using namespace InferenceEngine;

struct ngraph_network_param {
    std::string modelFile;
    std::string imageName;
    std::string ngraphModel;

    std::string model() {
        ModelsPath result;
        result += kPathSeparator;
        result += modelFile;
        return result;
    }

    std::string weights() {
        ModelsPath result;
        result += kPathSeparator;
        result += FileUtils::fileNameNoExt(modelFile);
        result += ".bin";
        return result;
    }

    std::string image() {
        std::string result = TestDataHelpers::get_data_path();
        result += kPathSeparator;
        result += imageName;
        return result;
    }

    std::string v7model() {
        ModelsPath result;
        result += kPathSeparator;
        result += ngraphModel;
        return result;
    }
};

class smoke_NGraphNetworkTest : public TestsCommon, public TestsCommonFunc {
protected:
    Blob::Ptr classifyV7(ngraph_network_param p, size_t batch_size = 1, float threshold = 0.005f) {
        IRReader reader;
        auto ngraph = reader.read(p.v7model());

        auto network = CNNNetwork(ngraph);

        Core ie;
        ExecutableNetwork exeNetwork = ie.LoadNetwork(network, "CPU");
        InferRequest inferRequest = exeNetwork.CreateInferRequest();

        Blob::Ptr src = readInput(p.image(), batch_size);

        OutputsDataMap outInfo = network.getOutputsInfo();
        InputsDataMap inputInfo = network.getInputsInfo();

        auto dst = make_shared_blob<float>(outInfo.begin()->second->getTensorDesc());
        dst->allocate();
        inferRequest.SetBlob(inputInfo.begin()->first, src);
        inferRequest.SetBlob(outInfo.begin()->first, dst);
        inferRequest.Infer();

        return dst;
    }

    Blob::Ptr classifyV5(ngraph_network_param p, size_t batch_size = 1, float threshold = 0.005f) {
        Core ie;
        CNNNetwork network = ie.ReadNetwork(p.model(), p.weights());
        if (batch_size != 1)
            network.setBatchSize(batch_size);

        ExecutableNetwork exeNetwork = ie.LoadNetwork(network, "CPU");
        InferRequest inferRequest = exeNetwork.CreateInferRequest();

        Blob::Ptr src = readInput(p.image(), batch_size);

        OutputsDataMap outInfo;
        outInfo = network.getOutputsInfo();

        auto dst = make_shared_blob<float>(outInfo.begin()->second->getTensorDesc());
        dst->allocate();
        inferRequest.SetBlob(network.getInputsInfo().begin()->first, src);
        inferRequest.SetBlob(outInfo.begin()->first, dst);
        inferRequest.Infer();

        return dst;
    }

    void classify(ngraph_network_param p) {
        try {
            auto v7blb = classifyV7(p);
            auto v5blb = classifyV5(p);

            auto* v7data = v7blb->buffer().as<float *>();
            auto* v5data = v5blb->buffer().as<float *>();

            ASSERT_EQ(v7blb->size(), v5blb->size());
            for (size_t i = 0; i < v7blb->size(); i++) {
                ASSERT_EQ(v7data[i], v5data[i]);
            }
        } catch (const InferenceEngine::details::InferenceEngineException& e) {
            FAIL() << e.what();
        }
    }
};

/*************************************************
 * !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!!
 * All ref values was obtained from Caffe scoring
 * !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!!
 *************************************************/
#ifndef ENABLE_MKL_DNN
 #include "disable_tests.hpp"
#endif

TEST_F(smoke_NGraphNetworkTest, reshapeLoadTest) {
    std::string model = R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="test" precision="FP32" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,1,28,28"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>28</dim>
                    <dim>28</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="13/Output_0/Data__const" type="Const" version="opset1">
            <data offset="0" size="2000"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>20</dim>
                    <dim>1</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="conv1" type="Convolution" version="opset1">
            <data dilations="1,1" group="1" output="20" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>28</dim>
                    <dim>28</dim>
                </port>
                <port id="1">
                    <dim>20</dim>
                    <dim>1</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>20</dim>
                    <dim>24</dim>
                    <dim>24</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="conv1/Dims215/copy_const" type="Const" version="opset1">
            <data offset="2000" size="80"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>20</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="conv1/Bias" type="Add" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>20</dim>
                    <dim>24</dim>
                    <dim>24</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>20</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>20</dim>
                    <dim>24</dim>
                    <dim>24</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="pool1" type="MaxPool" version="opset1">
            <data kernel="2,2" pads_begin="0,0" pads_end="0,0" rounding_type="ceil" strides="2,2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>20</dim>
                    <dim>24</dim>
                    <dim>24</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>20</dim>
                    <dim>12</dim>
                    <dim>12</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="11/Output_0/Data__const" type="Const" version="opset1">
            <data offset="2080" size="100000"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>50</dim>
                    <dim>20</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer id="7" name="conv2" type="Convolution" version="opset1">
            <data dilations="1,1" group="1" output="50" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>20</dim>
                    <dim>12</dim>
                    <dim>12</dim>
                </port>
                <port id="1">
                    <dim>50</dim>
                    <dim>20</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>50</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </output>
        </layer>
        <layer id="8" name="conv2/Dims209/copy_const" type="Const" version="opset1">
            <data offset="102080" size="200"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>50</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="9" name="conv2/Bias" type="Add" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>50</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>50</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>50</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </output>
        </layer>
        <layer id="10" name="pool2" type="MaxPool" version="opset1">
            <data kernel="2,2" pads_begin="0,0" pads_end="0,0" rounding_type="ceil" strides="2,2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>50</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>50</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="11" name="prob/sink_port_0" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>50</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="4" to-port="0"/>
        <edge from-layer="3" from-port="1" to-layer="4" to-port="1"/>
        <edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
        <edge from-layer="5" from-port="1" to-layer="7" to-port="0"/>
        <edge from-layer="6" from-port="1" to-layer="7" to-port="1"/>
        <edge from-layer="7" from-port="2" to-layer="9" to-port="0"/>
        <edge from-layer="8" from-port="1" to-layer="9" to-port="1"/>
        <edge from-layer="9" from-port="2" to-layer="10" to-port="0"/>
        <edge from-layer="10" from-port="1" to-layer="11" to-port="0"/>
    </edges>
</net>)V0G0N";
    InferenceEngine::Blob::Ptr weights = make_shared_blob<uint8_t>({InferenceEngine::Precision::U8, {1724336}, InferenceEngine::C});
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    std::map<std::string, std::vector<size_t>> shape;
    shape["data"] = {1, 1, 28, 28};

    Core ie;
    CNNNetwork network = ie.ReadNetwork(model, weights);
    for (size_t i = 0; i < 10; i++) {
        network.reshape(shape);
        ExecutableNetwork exeNetwork = ie.LoadNetwork(network, "CPU");
    }
}

