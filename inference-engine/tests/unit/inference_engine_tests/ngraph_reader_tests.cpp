// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_ir_reader.hpp>
#include "tests_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <transform/transformations/conv_bias_fusion.hpp>
#include <transform/transformations/matmul_bias_fusion.hpp>
#include <transform/transformations/quantizeconv_dequantize_fusion.hpp>
#include <transform/transformations/convert_quantize_conv_elimination.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/frontend/onnx_import/onnx.hpp>

using namespace testing;
using namespace InferenceEngine;

class NGraphReaderTests : public TestsCommon {
protected:
    void TearDown() override {}
    void SetUp() override {}

    void compareICNNNetworks(ICNNNetwork::Ptr newNetwork, const CNNNetwork& oldNetwork) {
        auto compareParamVal = [](const std::string& val1, const std::string& val2) -> bool {
            std::vector<std::string> vals1, vals2;
            std::stringstream ss1(val1);
            std::string field;
            while (getline(ss1, field, ',' )) {
                std::stringstream fs(field);
                std::string value;
                fs >> value;
                vals1.emplace_back(value);
            }

            std::stringstream ss2(val2);
            while (getline(ss2, field, ',' )) {
                std::stringstream fs(field);
                std::string value;
                fs >> value;
                vals2.emplace_back(value);
            }

            if (vals1.size() != vals2.size())
                return false;

            for (size_t i = 0; i < vals1.size(); i++) {
                try {
                    float v1 = std::stof(vals1[i]);
                    float v2 = std::stof(vals2[i]);
                    if (abs(v2 - v1) > 0.00001f)
                        return false;
                } catch (...) {
                    if (vals1[i] != vals2[i])
                        return false;
                }
            }
            return true;
        };
        std::vector<std::string> err_log;
        CNNNetwork network(newNetwork);
        CNNNetwork refNetwork(oldNetwork);
        if (newNetwork->layerCount() != oldNetwork.layerCount())
            THROW_IE_EXCEPTION << "ICNNNetworks have different numbers of layers! " + std::to_string(newNetwork->layerCount()) + " and " + std::to_string(oldNetwork.layerCount());
        auto newIterator = network.begin();
        auto oldIterator = refNetwork.begin();
        for (; newIterator != network.end() && oldIterator != refNetwork.end(); newIterator++, oldIterator++) {
            CNNLayerPtr layer = *newIterator;
            CNNLayerPtr oldLayer = *oldIterator;
            if (layer->type != oldLayer->type) {
                err_log.push_back("Layer " + oldLayer->name + " and old layer " + oldLayer->name + " have different type: " + layer->type + " and " + oldLayer->type);
            }
            if (layer->precision != oldLayer->precision) {
                err_log.push_back("Layer " + oldLayer->name + " and old layer " + oldLayer->name + " have different precision");
            }
            bool success = layer->type == oldLayer->type &&
                layer->insData.size() == oldLayer->insData.size() &&
                layer->outData.size() == oldLayer->outData.size() &&
                layer->precision == oldLayer->precision;

            for (size_t i = 0; i < layer->insData.size() && success; i++) {
                auto lockedOldData = oldLayer->insData[i].lock();
                auto lockedData = layer->insData[i].lock();
                success = success && lockedOldData->getTensorDesc() == lockedData->getTensorDesc();
                if (lockedOldData->getTensorDesc() != lockedData->getTensorDesc()) {
                    err_log.push_back("Layer " + oldLayer->name + " and old layer " + oldLayer->name + " have different tensor desc for locked data");
                }
            }
            for (size_t i = 0; i < layer->outData.size() && success; i++) {
                if (oldLayer->outData[i]->getTensorDesc() != layer->outData[i]->getTensorDesc()) {
                    err_log.push_back("Layer " + oldLayer->name + " and old layer " + oldLayer->name + " have different tensor desc");
                }
                success = success && oldLayer->outData[i]->getTensorDesc() == layer->outData[i]->getTensorDesc();
            }
            for (const auto& item : layer->params) {
                if (!success)
                    break;
                if (oldLayer->params.find(item.first) != oldLayer->params.end()) {
                    if (!compareParamVal(oldLayer->params[item.first], item.second)) {
                        success = false;
                        err_log.push_back("Layer " + layer->name + " in new network differ from reference parameter " + item.first + " (new, old): " + item.second + ", " + oldLayer->params[item.first]);
                    }
                } else {
                    success = false;
                    err_log.push_back("Layer " + oldLayer->name + " in old net has no " + item.first + " attribute.");
                }
            }

            if (!success) {
                for (auto & it: err_log) {
                    std::cout << "ERROR: " << it << std::endl;
                }
                THROW_IE_EXCEPTION << "ICNNNetworks have different layers!";
            }
        }

        InputsDataMap newInput;
        OutputsDataMap newOutput;
        newNetwork->getInputsInfo(newInput);
        newNetwork->getOutputsInfo(newOutput);
        InputsDataMap oldInput = oldNetwork.getInputsInfo();
        OutputsDataMap oldOutput = oldNetwork.getOutputsInfo();

        bool success = newInput.size() == oldInput.size();
        for (const auto& it : newInput) {
            if (!success)
                break;
            success = success && oldInput.find(it.first) != oldInput.end();
        }
        if (!success)
            THROW_IE_EXCEPTION << "ICNNNetworks have different inputs!";

        success = newOutput.size() == oldOutput.size();
        for (const auto& it : newOutput) {
            if (!success)
                break;
            success = success && oldOutput.find(it.first) != oldOutput.end();
        }
        if (!success)
            THROW_IE_EXCEPTION << "ICNNNetworks have different outputs!";
    }
};

TEST_F(NGraphReaderTests, ReadScalarNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32" />
            </output>
        </layer>
        <layer name="prior" id="2" type="ReLU" >
            <input>
                <port id="1" precision="FP32" />
            </input>
            <output>
                <port id="2" precision="FP32" />
            </output>
        </layer>
        <layer name="output" type="Result" id="3">
            <input>
                <port id="0" precision="FP32" />
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);
    CNNNetwork cnetwork(network);
    cnetwork.begin();
}

TEST_F(NGraphReaderTests, ReadIncorrectNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="4" type="ShapeOf">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="prior" id="2" type="ReLU">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="3">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="4" from-port="2" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::CPtr blob;
    ASSERT_THROW(reader.read(model, blob), InferenceEngine::details::InferenceEngineException);
}

TEST_F(NGraphReaderTests, ReadPriorBoxClusteredNetwork) {
std::string model = R"V0G0N(
<net name="PriorBoxClusteredNet" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>768</dim>
                    <dim>30</dim>
                    <dim>30</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>512</dim>
                    <dim>512</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="ch_concat_mixed_7_chconcat_anchors/0_port" type="ShapeOf">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>768</dim>
                    <dim>30</dim>
                    <dim>30</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="1344813449_const" type="Const">
            <data offset="0" size="8"/>
            <output>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="1345813459_const" type="Const">
            <data offset="8" size="8"/>
            <output>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="9" name="1345813459_const" type="Const">
            <data offset="16" size="8"/>
            <output>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="ch_concat_mixed_7_chconcat_anchors/ss_0_port" type="StridedSlice">
            <data begin_mask="1" ellipsis_mask="0" end_mask="0" new_axis_mask="0" shrink_axis_mask="0"/>
            <input>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>1</dim>
                </port>
                <port id="3" precision="I64">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="ch_concat_mixed_7_chconcat_anchors/1_port" type="ShapeOf">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>512</dim>
                    <dim>512</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="7" name="ch_concat_mixed_7_chconcat_anchors/ss_1_port" type="StridedSlice">
            <data begin_mask="1" ellipsis_mask="0" end_mask="0" new_axis_mask="0" shrink_axis_mask="0"/>
            <input>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>1</dim>
                </port>
                <port id="3" precision="I64">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="8" name="prior" type="PriorBoxClustered">
            <data clip="0" flip="0" height="44.0,10.0,30.0,19.0,94.0,32.0,61.0,53.0,17.0" offset="0.5" step="16.0" variance="0.1,0.1,0.2,0.2"
                width="86.0,13.0,57.0,39.0,68.0,34.0,142.0,50.0,23.0"/>
            <input>
                <port id="0" precision="I64">
                    <dim>2</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>32400</dim>
                </port>
            </output>
        </layer>
        <layer id="10" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>32400</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="6" to-port="0"/>
        <edge from-layer="2" from-port="1" to-layer="5" to-port="0"/>
        <edge from-layer="6" from-port="1" to-layer="7" to-port="0"/>
        <edge from-layer="3" from-port="1" to-layer="5" to-port="1"/>
        <edge from-layer="3" from-port="1" to-layer="7" to-port="1"/>
        <edge from-layer="4" from-port="1" to-layer="5" to-port="2"/>
        <edge from-layer="4" from-port="1" to-layer="7" to-port="2"/>
        <edge from-layer="9" from-port="1" to-layer="5" to-port="3"/>
        <edge from-layer="9" from-port="1" to-layer="7" to-port="3"/>
        <edge from-layer="5" from-port="4" to-layer="8" to-port="0"/>
        <edge from-layer="7" from-port="4" to-layer="8" to-port="1"/>
        <edge from-layer="8" from-port="2" to-layer="10" to-port="0"/>
    </edges>
</net>
)V0G0N";
std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="in1" type="Input" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>768</dim>
                    <dim>30</dim>
                    <dim>30</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" type="Input" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>512</dim>
                    <dim>512</dim>
                </port>
            </output>
        </layer>
        <layer name="prior" id="2" type="PriorBoxClustered" precision="FP32">
            <data clip="0" step_h="16.000000" step_w="16.000000" flip="0" height="44,10,30,19,94,32,61,53,17" offset="0.500000" step="16.000000" variance="0.1,0.1,0.2,0.2" width="86,13,57,39,68,34,142,50,23"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>768</dim>
                    <dim>30</dim>
                    <dim>30</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>512</dim>
                    <dim>512</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>32400</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="2"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {30}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    // Set shape constant for broadcast1
    int64_t * buffer = (int64_t *)(weights->buffer());
    buffer[0] = 2;
    buffer[1] = 4;
    buffer[1] = 1;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);
    auto nGraph = reader.read(model, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, DISABLED_ReadProposalNetwork) {
std::string model = R"V0G0N(
<net name="PriorBoxNet" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>12</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="in3" type="Const">
            <data offset="0" size="48"/>
            <output>
                <port id="0" precision="I64">
                    <dim>1</dim>
                    <dim>6</dim>
                </port>
            </output>
        </layer>
        <layer name="proposal" type="Proposal" precision="FP32" id="3">
            <data feat_stride="16" base_size="16" min_size="16" ratio="2.669000" scale="4.000000,6.000000,9.000000,16.000000,24.000000,32.000000" pre_nms_topn="6000" post_nms_topn="200" nms_thresh="0.600000"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>12</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>6</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>200</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>200</dim>
                    <dim>5</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="2"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="3"/>
        <edge from-layer="3" from-port="4" to-layer="4" to-port="0"/>
    </edges>
</net>
)V0G0N";
std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="in1" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>12</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="in3" precision="I64" type="Const">
            <data offset="0" size="48"/>
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>6</dim>
                </port>
            </output>
        </layer>
        <layer name="proposal" type="Proposal" precision="FP32" id="3">
            <data feat_stride="16" base_size="16" min_size="16" ratio="2.669000" scale="4.000000,6.000000,9.000000,16.000000,24.000000,32.000000" pre_nms_topn="6000" post_nms_topn="200" nms_thresh="0.600000"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>12</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>6</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>200</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="2"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="3"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {48}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);
    auto nGraph = reader.read(model, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadPriorBoxNetwork) {
std::string model = R"V0G0N(
<net name="PriorBoxNet" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>768</dim>
                    <dim>30</dim>
                    <dim>30</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>512</dim>
                    <dim>512</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="ch_concat_mixed_7_chconcat_anchors/0_port" type="ShapeOf">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>768</dim>
                    <dim>30</dim>
                    <dim>30</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="1344813449_const" type="Const">
            <data offset="0" size="8"/>
            <output>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="1345813459_const" type="Const">
            <data offset="8" size="8"/>
            <output>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="9" name="1345813459_const" type="Const">
            <data offset="16" size="8"/>
            <output>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="ch_concat_mixed_7_chconcat_anchors/ss_0_port" type="StridedSlice">
            <data begin_mask="1" ellipsis_mask="0" end_mask="0" new_axis_mask="0" shrink_axis_mask="0"/>
            <input>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>1</dim>
                </port>
                <port id="3" precision="I64">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="ch_concat_mixed_7_chconcat_anchors/1_port" type="ShapeOf">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>512</dim>
                    <dim>512</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="7" name="ch_concat_mixed_7_chconcat_anchors/ss_1_port" type="StridedSlice">
            <data begin_mask="1" ellipsis_mask="0" end_mask="0" new_axis_mask="0" shrink_axis_mask="0"/>
            <input>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>1</dim>
                </port>
                <port id="3" precision="I64">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="8" name="prior" type="PriorBox">
            <data aspect_ratio="1.0,2.0,0.5" clip="0" flip="0" img_h="0" img_size="0" img_w="0" max_size="" min_size="51.2,72.407552" offset="0.5" scale_all_sizes="0" step="17.066666666666666" step_h="0" step_w="0" variance="0.100000,0.100000,0.200000,0.200000"/>
            <input>
                <port id="0" precision="I64">
                    <dim>2</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>14400</dim>
                </port>
            </output>
        </layer>
        <layer id="10" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>14400</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="6" to-port="0"/>
        <edge from-layer="2" from-port="1" to-layer="5" to-port="0"/>
        <edge from-layer="6" from-port="1" to-layer="7" to-port="0"/>
        <edge from-layer="3" from-port="1" to-layer="5" to-port="1"/>
        <edge from-layer="3" from-port="1" to-layer="7" to-port="1"/>
        <edge from-layer="4" from-port="1" to-layer="5" to-port="2"/>
        <edge from-layer="4" from-port="1" to-layer="7" to-port="2"/>
        <edge from-layer="9" from-port="1" to-layer="5" to-port="3"/>
        <edge from-layer="9" from-port="1" to-layer="7" to-port="3"/>
        <edge from-layer="5" from-port="4" to-layer="8" to-port="0"/>
        <edge from-layer="7" from-port="4" to-layer="8" to-port="1"/>
        <edge from-layer="8" from-port="2" to-layer="10" to-port="0"/>
    </edges>
</net>
)V0G0N";
std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="in1" type="Input" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>768</dim>
                    <dim>30</dim>
                    <dim>30</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" type="Input" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>512</dim>
                    <dim>512</dim>
                </port>
            </output>
        </layer>
        <layer name="prior" id="2" type="PriorBox" precision="FP32">
            <data aspect_ratio="1,2,0.5" clip="0" flip="0" img_h="0" img_size="0" img_w="0" max_size="" min_size="51.200001,72.407555" offset="0.500000" scale_all_sizes="0" step="17.066668" step_h="0" step_w="0" variance="0.1,0.1,0.2,0.2"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>768</dim>
                    <dim>30</dim>
                    <dim>30</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>512</dim>
                    <dim>512</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>14400</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="2"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {30}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    // Set shape constant for broadcast1
    int64_t * buffer = (int64_t *)(weights->buffer());
    buffer[0] = 2;
    buffer[1] = 4;
    buffer[1] = 1;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);
    auto nGraph = reader.read(model, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadSplitNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="split" id="2" type="Split">
            <data axis="1"/>
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output1" type="Result" id="1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
        <layer name="output" type="Result" id="3">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
        <edge from-layer="2" from-port="3" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="split" id="2" type="Split" precision="FP32">
            <data axis="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, DISABLED_ReadDetectionOutputNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>38360</dim>
                </port>
            </output>
        </layer>
        <layer name="in2" type="Parameter" id="1">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>19180</dim>
                </port>
            </output>
        </layer>
        <layer name="in3" type="Parameter" id="4">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>38360</dim>
                </port>
            </output>
        </layer>
        <layer name="detectionOut" id="2" type="DetectionOutput" precision="FP32">
            <data num_classes="2" share_location="1" background_label_id="0" nms_threshold="0.450000" top_k="400" input_height="1" input_width="1" code_type="caffe.PriorBoxParameter.CENTER_SIZE" variance_encoded_in_target="0" keep_top_k="200" confidence_threshold="0.010000"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>38360</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>19180</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>38360</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>38360</dim>
                </port>
                <port id="5">
                    <dim>1</dim>
                    <dim>19180</dim>
                </port>
            </input>
            <output>
                <port id="6" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>200</dim>
                    <dim>7</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="3">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>200</dim>
                    <dim>7</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="2"/>
        <edge from-layer="4" from-port="0" to-layer="2" to-port="3"/>
        <edge from-layer="2" from-port="6" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>38360</dim>
                </port>
            </output>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="1">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>19180</dim>
                </port>
            </output>
        </layer>
        <layer name="in3" type="Input" precision="FP32" id="4">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>38360</dim>
                </port>
            </output>
        </layer>
        <layer name="detectionOut" id="2" type="DetectionOutput" precision="FP32">
            <data num_classes="2" clip_after_nms="0" clip_before_nms="0" decrease_label_id="0" normalized="0" objectness_score="0.000000" share_location="1" background_label_id="0" nms_threshold="0.450000" top_k="400" input_height="1" input_width="1" code_type="caffe.PriorBoxParameter.CENTER_SIZE" variance_encoded_in_target="0" keep_top_k="200" confidence_threshold="0.010000"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>38360</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>19180</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>38360</dim>
                </port>
            </input>
            <output>
                <port id="6" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>200</dim>
                    <dim>7</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="2"/>
        <edge from-layer="4" from-port="0" to-layer="2" to-port="3"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadConcatNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="in2" type="Parameter" id="1">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="concat" id="2" type="Concat">
            <data axis="1"/>
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="3">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="2"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="1">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="concat" id="2" type="Concat" precision="FP32">
            <data axis="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>6</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="2"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, DISABLED_ReadTopKNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="1345813459_const" type="Const">
            <data offset="0" size="8"/>
            <output>
                <port id="1" precision="I64" />
            </output>
        </layer>
        <layer name="activation" id="1" type="TopK">
            <data axis="1" mode="max" sorting="value"/>
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="2" precision="I64" />
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="4" from-port="1" to-layer="1" to-port="2"/>
        <edge from-layer="1" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ArgMax" precision="FP32">
            <data alpha="0.000000" beta="0.750000" local-size="5" region="across" />
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadMVNNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="MVN">
            <data across_channels="1" eps="0.75" normalize_variance="1" />
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="MVN" precision="FP32">
            <data across_channels="1" eps="0.75" normalize_variance="1" />
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadLrnNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="LRN">
            <data alpha="0" beta="0.75" local-size="5" bias="1" />
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="Norm" precision="FP32">
            <data alpha="0.000000" beta="0.750000" local-size="5" region="across" />
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, DISABLED_ReadLrnNetwork2) {
    std::string model = R"V0G0N(
<net name="LRN" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="lrn" id="1" type="LRN">
            <data alpha="0" beta="0.75" local-size="5" bias="1" />
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="in1" type="Input" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="lrn" type="LRN" precision="FP32">
            <data alpha="0.000000" beta="0.750000" local-size="5" region="across" />
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="data1" precision="FP32" type="Const">
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="5808"/>
            </blobs>
        </layer>
        <layer id="3" name="add" precision="FP32" type="Eltwise">
            <data operation="sum"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="3" to-port="0"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}


TEST_F(NGraphReaderTests, ReadClampNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="Clamp">
            <data max="6" min="0" />"
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="Clamp" precision="FP32">
            <data max="6.000000" min="0.000000" />"
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadSigmoidNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="Sigmoid">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="Sigmoid" precision="FP32">
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadPReLUNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="data" type="Const">
            <data offset="0" size="256"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="PRelU">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="3" precision="FP32">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="3" from-port="1" to-layer="1" to-port="3"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="PReLU" precision="FP32">
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
            <blobs>
                <weights offset="0" size="256"/>
            </blobs>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {256}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadELUNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ELU">
            <data alpha="1"/>
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ELU" precision="FP32">
            <data alpha="1.000000"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadShapeOfNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ShapeOf">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ShapeOf" precision="I64">
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadLeakyReLUNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="LeakyReLU">
            <data negative_slope="0.1" />
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ReLU" precision="FP32">
            <data negative_slope="0.1" />
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadTanhNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="Tanh">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="TanH" precision="FP32">
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadExpNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="Exp">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="Exp" precision="FP32">
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadReLUNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ReLU">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ReLU" precision="FP32">
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadBroadcastNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer id="0" name="data_add_5451_const" type="Const">
            <data offset="0" size="256"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Add1_/Fused_Add_/Broadcast/Shape7264_const" type="Const">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="Add1_/Fused_Add_/Broadcast/Axis7265_const" type="Const">
            <data offset="288" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="Add1_/Fused_Add_/Broadcast/" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="4">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="1" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="2"/>
        <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;
    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {320}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    auto nGraph = reader.read(model, weights);
}

TEST_F(NGraphReaderTests, ReadSoftMaxNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer name="softmax" id="1" type="SoftMax">
            <data axis="1"/>
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer name="softmax" id="1" type="SoftMax" precision="FP32">
            <data axis="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadMaxPoolNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="pool" id="1" type="MaxPool">
            <data kernel="3,3" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>11</dim>
                    <dim>11</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>11</dim>
                    <dim>11</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="pool" id="1" type="Pooling" precision="FP32">
            <data kernel="3,3" rounding_type="floor" pads_begin="1,1" pads_end="1,1" strides="2,2" pool-method="max"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>11</dim>
                    <dim>11</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}
TEST_F(NGraphReaderTests, ReadAvgPoolNetwork) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="pool" id="1" type="AvgPool">
            <data kernel="3,3" pads_begin="1,1" pads_end="1,1" strides="2,2" exclude-pad="true"/>
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>11</dim>
                    <dim>11</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>11</dim>
                    <dim>11</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="pool" id="1" type="Pooling" precision="FP32">
            <data kernel="3,3" pads_begin="1,1" pads_end="1,1" strides="2,2" pool-method="avg" exclude-pad="true" rounding_type="floor"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>11</dim>
                    <dim>11</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
};

TEST_F(NGraphReaderTests, ReadReLUNetworkWithoutTopologicalOrder) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
        <layer name="activation" id="1" type="ReLU">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="in1" type="Parameter" id="0">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Activation" version="5" precision="FP32" batch="1">
    <layers>
        <layer name="activation" id="1" type="ReLU" precision="FP32">
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;

    Blob::CPtr blob;
    auto nGraph = reader.read(model, blob);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadTileNetwork) {
    std::string model = R"V0G0N(
<net name="Transpose" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" type="Const">
            <data offset="0" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="tile" type="Tile">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Transpose" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="tile" precision="FP32" type="Tile">
        <data axis="1" tiles="2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {32}, Layout::C));
    weights->allocate();
    auto *data = weights->buffer().as<int64_t *>();
    data[0] = 1;
    data[1] = 2;
    data[2] = 1;
    data[3] = 1;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadTileNetwork2) {
    std::string model = R"V0G0N(
<net name="Transpose" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>10</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" type="Const">
            <data offset="0" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="tile" type="Tile">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>10</dim>
                    <dim>10</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>2</dim>
                    <dim>64</dim>
                    <dim>30</dim>
                    <dim>40</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>64</dim>
                    <dim>30</dim>
                    <dim>40</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Transpose" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>10</dim>
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="tile" precision="FP32" type="Tile">
        <data axis="3" tiles="4"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>10</dim>
                    <dim>10</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>10</dim>
                    <dim>40</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="tile_3" precision="FP32" type="Tile">
        <data axis="2" tiles="3"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>10</dim>
                    <dim>40</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>30</dim>
                    <dim>40</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="tile_3_2" precision="FP32" type="Tile">
        <data axis="0" tiles="2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>30</dim>
                    <dim>40</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>2</dim>
                    <dim>64</dim>
                    <dim>30</dim>
                    <dim>40</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="3" to-layer="2" to-port="0"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {32}, Layout::C));
    weights->allocate();
    auto *data = weights->buffer().as<int64_t *>();
    data[0] = 2;
    data[1] = 1;
    data[2] = 3;
    data[3] = 4;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadTransposeNetwork) {
    std::string model = R"V0G0N(
<net name="Transpose" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" type="Const">
            <data offset="0" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="transp" type="Transpose">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>4</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>4</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Transpose" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="transp" precision="FP32" type="Permute">
        <data order="3,2,1,0"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>4</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {32}, Layout::C));
    weights->allocate();
    auto *data = weights->buffer().as<int64_t *>();
    data[0] = 3;
    data[1] = 2;
    data[2] = 1;
    data[3] = 0;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadReshapeNetwork) {
    std::string model = R"V0G0N(
<net name="Reshape" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" type="Const">
            <data offset="0" size="16"/>
            <output>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="reshape1" type="Reshape">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>2</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="16"/>
            </blobs>
        </layer>
        <layer id="3" name="reshape1" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {16}, Layout::C));
    weights->allocate();
    auto *data = weights->buffer().as<int64_t *>();
    data[0] = 1;
    data[1] = 2048;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadSqueeze) {
    std::string model = R"V0G0N(
<net name="Reshape" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
		<layer id="1" name="const1" precision="I64" type="Const">
			<data offset="0" size="8"/>
			<output>
				<port id="1">
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="squeeze" precision="FP32" type="Squeeze">
			<input>
				<port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
				</port>
				<port id="1">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
				</port>
			</output>
		</layer>
        <layer name="output" type="Result" id="3">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="8"/>
            </blobs>
        </layer>
        <layer id="3" name="squeeze" precision="FP32" type="Squeeze">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {24}, Layout::C));
    weights->allocate();
    auto *data = weights->buffer().as<int64_t *>();
    data[0] = 3;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadInterpolateNetwork) {
    std::string model = R"V0G0N(
<net name="Reshape" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>48</dim>
                    <dim>80</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" type="Const" precision="I64">
            <data offset="0" size="16"/>
            <output>
                <port id="1">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="interpolate" type="Interpolate" precision="FP32">
            <data axes="2,3" align_corners="0" pads_begin="0" pads_end="0" mode="linear"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>48</dim>
                    <dim>80</dim>
                </port>
                <port id="1">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>50</dim>
                    <dim>60</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" precision="FP32">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>50</dim>
                    <dim>60</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>48</dim>
                    <dim>80</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="interpolate" precision="FP32" type="Interp">
            <data align_corners="0" pad_beg="0" pad_end="0" mode="linear" width="60" height="50"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>48</dim>
                    <dim>80</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>50</dim>
                    <dim>60</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {16}, Layout::C));
    weights->allocate();
    auto *data = weights->buffer().as<int64_t *>();
    data[0] = 50;
    data[1] = 60;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadMatMulNetwork) {
    std::string model = R"V0G0N(
<net name="Convolution" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="embedded_input__const" type="Const">
            <data offset="0" size="8192000"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>2048</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" type="MatMul">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>2048</dim>
                    <dim>1000</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" precision="FP32" type="FullyConnected">
            <data alpha="0" beta="0" out-size="1000"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
            <weights offset="0" size="8192000" />
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {8192000}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadDeconvolution3DNetwork) {
    std::string model = R"V0G0N(
<net name="Convolution" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>512</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="embedded_input__const" type="Const">
            <data offset="0" size="33554432"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>512</dim>
                    <dim>256</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="deconv1" precision="FP32" type="Deconvolution">
            <data auto_pad="same_upper" kernel="4,4,4" output="256" pads_begin="1,1,1" pads_end="1,1,1" strides="2,2,2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>512</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
                <port id="1">
                    <dim>512</dim>
                    <dim>256</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>512</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="deconv1" precision="FP32" type="Deconvolution">
            <data dilations="1,1,1" auto_pad="same_upper" kernel="4,4,4" output="256" pads_begin="1,1,1" pads_end="1,1,1" strides="2,2,2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>512</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </output>
            <weights offset="0" size="33554432" />
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {33554432}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadDeconvolution2DNetwork) {
    std::string model = R"V0G0N(
<net name="Convolution" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>512</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="embedded_input__const" type="Const">
            <data offset="0" size="8388608"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>512</dim>
                    <dim>256</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="deconv1" precision="FP32" type="Deconvolution">
            <data auto_pad="same_upper" kernel="4,4" output="256" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>512</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
                <port id="1">
                    <dim>512</dim>
                    <dim>256</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>512</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="deconv1" precision="FP32" type="Deconvolution">
            <data dilations="1,1" auto_pad="same_upper" kernel="4,4" output="256" pads_begin="1,1" pads_end="1,1" strides="2,2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>512</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </output>
            <weights offset="0" size="8388608" />
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {33554432}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadConvolutionNetwork) {
    std::string model = R"V0G0N(
<net name="Convolution" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="embedded_input__const" type="Const">
            <data offset="0" size="139392"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>96</dim>
                    <dim>3</dim>
                    <dim>11</dim>
                    <dim>11</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="conv1" type="Convolution">
            <data dilations="1,1" group="1" kernel="11,11" output="96" pads_begin="0,0" pads_end="0,0" strides="4,4"/>
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>96</dim>
                    <dim>3</dim>
                    <dim>11</dim>
                    <dim>11</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="conv1" precision="FP32" type="Convolution">
            <data dilations="1,1" group="1" kernel="11,11" output="96" pads_begin="0,0" pads_end="0,0" strides="4,4"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
            <weights offset="0" size="139392" />
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {139392}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadMaximumNetwork) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data1" type="Const">
            <data offset="0" size="3211264"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="div" type="Maximum">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="data1" precision="FP32" type="Const">
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="3211264"/>
            </blobs>
        </layer>
        <layer id="3" name="div" precision="FP32" type="Eltwise">
            <data operation="max"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {3211264}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadDivideNetwork) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data1" type="Const">
            <data offset="0" size="3211264"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="div" type="Divide">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="data1" precision="FP32" type="Const">
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="3211264"/>
            </blobs>
        </layer>
        <layer id="3" name="div" precision="FP32" type="Eltwise">
            <data operation="div"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {3211264}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadPowNetwork) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data1" type="Const">
            <data offset="0" size="3211264"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="pow" type="Pow">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="data1" precision="FP32" type="Const">
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="3211264"/>
            </blobs>
        </layer>
        <layer id="3" name="pow" precision="FP32" type="Eltwise">
            <data operation="pow"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {3211264}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadMultiplyNetwork) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data1" type="Const">
            <data offset="0" size="3211264"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="mul" type="Multiply">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="data1" precision="FP32" type="Const">
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="3211264"/>
            </blobs>
        </layer>
        <layer id="3" name="mul" precision="FP32" type="Eltwise">
            <data operation="prod"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {3211264}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ReadAddNoBroadcastNetwork) {
    std::string model = R"V0G0N(
<net name="Add" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data1" type="Const">
            <data offset="0" size="3211264"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" type="Add">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="data1" precision="FP32" type="Const">
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="3211264"/>
            </blobs>
        </layer>
        <layer id="3" name="add" precision="FP32" type="Eltwise">
            <data operation="sum"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {3211264}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, DISABLED_ReadAddNetwork) {
    std::string model = R"V0G0N(
<net name="Add" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data1" type="Const">
            <data offset="0" size="256"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" type="Add">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5">
    <layers>
        <layer id="0" name="data" type="Input">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="data1" type="Const">
            <output>
                <port id="3" precision="FP32">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="256"/>
            </blobs>
        </layer>
        <layer id="3" name="add" type="Eltwise">
            <data operation="sum"/>
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {139392}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ConvBiasFusion) {
    std::string model = R"V0G0N(
<net name="ConvBias" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="conv_weights" type="Const">
            <data offset="0" size="139392"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>96</dim>
                    <dim>3</dim>
                    <dim>11</dim>
                    <dim>11</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="conv" type="Convolution">
            <data dilations="1,1" group="1" kernel="11,11" output="96" pads_begin="0,0" pads_end="0,0" strides="4,4"/>
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>96</dim>
                    <dim>3</dim>
                    <dim>11</dim>
                    <dim>11</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="data_add_5451_const" type="Const">
            <data offset="139392" size="384"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>96</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="Add1_/Fused_Add_/Broadcast/Shape7264_const" type="Const">
            <data offset="139776" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="Add1_/Fused_Add_/Broadcast/Axis7265_const" type="Const">
            <data offset="139808" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="7" name="Add1_/Fused_Add_/Broadcast/" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>96</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" type="Add">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
        </layer>
        <layer  id="8" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="4" from-port="1" to-layer="7" to-port="0"/>
        <edge from-layer="5" from-port="1" to-layer="7" to-port="1"/>
        <edge from-layer="6" from-port="1" to-layer="7" to-port="2"/>
        <edge from-layer="7" from-port="3" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
        <edge from-layer="3" from-port="3" to-layer="8" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" precision="FP32" type="Convolution">
            <data dilations="1,1" group="1" kernel="11,11" output="96" pads_begin="0,0" pads_end="0,0" strides="4,4"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
            <weights offset="0" size="139392" />
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;
    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {139840}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    auto nGraph = reader.read(model, weights);
    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ConvBiasFusionFP16) {
    std::string model = R"V0G0N(
<net name="ConvBias" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP16">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="conv_weights" type="Const">
            <data offset="0" size="69696"/>
            <output>
                <port id="1" precision="FP16">
                    <dim>96</dim>
                    <dim>3</dim>
                    <dim>11</dim>
                    <dim>11</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="conv" type="Convolution">
            <data dilations="1,1" group="1" kernel="11,11" output="96" pads_begin="0,0" pads_end="0,0" strides="4,4"/>
            <input>
                <port id="0" precision="FP16">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
                <port id="1" precision="FP16">
                    <dim>96</dim>
                    <dim>3</dim>
                    <dim>11</dim>
                    <dim>11</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP16">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="data_add_5451_const" type="Const">
            <data offset="139392" size="384"/>
            <output>
                <port id="1" precision="FP16">
                    <dim>96</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="Add1_/Fused_Add_/Broadcast/Shape7264_const" type="Const">
            <data offset="139776" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="Add1_/Fused_Add_/Broadcast/Axis7265_const" type="Const">
            <data offset="139808" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="7" name="Add1_/Fused_Add_/Broadcast/" type="Broadcast">
            <input>
                <port id="0" precision="FP16">
                    <dim>96</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP16">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" type="Add">
            <input>
                <port id="0" precision="FP16">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
                <port id="1" precision="FP16">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP16">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
        </layer>
        <layer  id="8" name="output" type="Result">
            <input>
                <port id="0" precision="FP16">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="4" from-port="1" to-layer="7" to-port="0"/>
        <edge from-layer="5" from-port="1" to-layer="7" to-port="1"/>
        <edge from-layer="6" from-port="1" to-layer="7" to-port="2"/>
        <edge from-layer="7" from-port="3" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
        <edge from-layer="3" from-port="3" to-layer="8" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP16" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP16" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" precision="FP16" type="Convolution">
            <data dilations="1,1" group="1" kernel="11,11" output="96" pads_begin="0,0" pads_end="0,0" strides="4,4"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
            <weights offset="0" size="69696" />
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;
    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {139840}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    auto nGraph = reader.read(model, weights);
    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, DISABLED_MatMulBiasFusion) {
    std::string model = R"V0G0N(
<net name="MatMulBias" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="weights" type="Const">
            <data offset="0" size="8192000"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>2048</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="fc" type="MatMul">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>2048</dim>
                    <dim>1000</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="b_input" type="Const">
            <data offset="8192000" size="4000"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="b_shape" type="Const">
            <data offset="8196000" size="8"/>
            <output>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="b_axis" type="Const">
            <data offset="8196008" size="16"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="broadcast" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>2</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="7" name="add" type="Add">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer  id="8" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="3" from-port="1" to-layer="6" to-port="0"/>
        <edge from-layer="4" from-port="1" to-layer="6" to-port="1"/>
        <edge from-layer="5" from-port="1" to-layer="6" to-port="2"/>
        <edge from-layer="6" from-port="3" to-layer="7" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="7" to-port="0"/>
        <edge from-layer="7" from-port="2" to-layer="8" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="MatMulBias" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" precision="FP32" type="FullyConnected">
            <data alpha="0" beta="0" out-size="1000"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
            <weights offset="0" size="8192000"/>
            <biases offset="8192000" size="4000"/>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;
    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {8196024}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    auto nGraph = reader.read(model, weights);
    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, MatMulBiasFusionNoBroadcast) {
    std::string model = R"V0G0N(
<net name="MatMulBias" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="weights" type="Const">
            <data offset="0" size="8192000"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>2048</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="fc" type="MatMul">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>2048</dim>
                    <dim>1000</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="b_input" type="Const">
            <data offset="8192000" size="4000"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="7" name="add" type="Add">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer  id="8" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="3" from-port="1" to-layer="7" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="7" to-port="0"/>
        <edge from-layer="7" from-port="2" to-layer="8" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="MatMulBias" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" precision="FP32" type="FullyConnected">
            <data alpha="0" beta="0" out-size="1000"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
            <weights offset="0" size="8192000"/>
            <biases offset="8192000" size="4000"/>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    IRReader reader;
    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {8196024}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    auto nGraph = reader.read(model, weights);
    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ConvertMulAddToScaleShift) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="14" name="broadcast1_data" type="Const">
            <data offset="0" size="256"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="16" name="broadcast1_axis" type="Const">
            <data offset="288" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="mul" type="Multiply">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="24" name="broadcast2_data" type="Const">
            <data offset="320" size="256"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="25" name="broadcast2_shape" type="Const">
            <data offset="576" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="26" name="broadcast2_axis" type="Const">
            <data offset="608" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="27" name="broadcast_2" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="add" type="Add">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="16" from-port="1" to-layer="17" to-port="2"/>
        <edge from-layer="24" from-port="1" to-layer="27" to-port="0"/>
        <edge from-layer="25" from-port="1" to-layer="27" to-port="1"/>
        <edge from-layer="26" from-port="1" to-layer="27" to-port="2"/>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="17" from-port="3" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="5" to-port="0"/>
        <edge from-layer="27" from-port="3" to-layer="5" to-port="1"/>
        <edge from-layer="5" from-port="3" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" precision="FP32" type="ScaleShift">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
            <weights offset="0" size="256"/>
            <biases offset="256" size="256"/>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {6422528}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    // Set shape constant for broadcast1
    int64_t * broadcast1_shape = (int64_t *)((int8_t *) weights->buffer() + 256);
    broadcast1_shape[0] = 1;
    broadcast1_shape[1] = 64;
    broadcast1_shape[2] = 112;
    broadcast1_shape[3] = 112;

    // Set shape constant for broadcast1
    int64_t * broadcast2_shape = (int64_t *)((int8_t *) weights->buffer() + 576);
    broadcast2_shape[0] = 1;
    broadcast2_shape[1] = 64;
    broadcast2_shape[2] = 112;
    broadcast2_shape[3] = 112;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);
    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ConvertMulAddToPower) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="14" name="broadcast1_data" type="Const">
            <data offset="0" size="4"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const">
            <data offset="4" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="16" name="broadcast1_axis" type="Const">
            <data offset="36" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="mul" type="Multiply">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="24" name="broadcast2_data" type="Const">
            <data offset="68" size="4"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="25" name="broadcast2_shape" type="Const">
            <data offset="72" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="26" name="broadcast2_axis" type="Const">
            <data offset="104" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="27" name="broadcast_2" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="add" type="Add">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="16" from-port="1" to-layer="17" to-port="2"/>
        <edge from-layer="24" from-port="1" to-layer="27" to-port="0"/>
        <edge from-layer="25" from-port="1" to-layer="27" to-port="1"/>
        <edge from-layer="26" from-port="1" to-layer="27" to-port="2"/>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="17" from-port="3" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="5" to-port="0"/>
        <edge from-layer="27" from-port="3" to-layer="5" to-port="1"/>
        <edge from-layer="5" from-port="3" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" precision="FP32" type="Power">
            <data power="1.000000" scale="127.500000" shift="0.820000"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {6422528}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    // Set shape constant for broadcast1
    int64_t * broadcast1_shape = (int64_t *)((int8_t *) weights->buffer() + 4);
    broadcast1_shape[0] = 1;
    broadcast1_shape[1] = 64;
    broadcast1_shape[2] = 112;
    broadcast1_shape[3] = 112;

    // Set shape constant for broadcast1
    int64_t * broadcast2_shape = (int64_t *)((int8_t *) weights->buffer() + 72);
    broadcast2_shape[0] = 1;
    broadcast2_shape[1] = 64;
    broadcast2_shape[2] = 112;
    broadcast2_shape[3] = 112;

    // Set scale/shift constants
    float* scale = (float *)((int8_t *) weights->buffer() + 0);
    scale[0] = 127.5;

    float* shift= (float *)((int8_t *) weights->buffer() + 68);
    shift[0] = 0.82;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ConvertMulToPower) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="14" name="broadcast1_data" type="Const">
            <data offset="0" size="4"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const">
            <data offset="4" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="16" name="broadcast1_axis" type="Const">
            <data offset="36" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="mul" type="Multiply">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="16" from-port="1" to-layer="17" to-port="2"/>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="17" from-port="3" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="mul" precision="FP32" type="Power">
            <data power="1.000000" scale="127.500000" shift="0.000000"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {6422528}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    // Set shape constant for broadcast1
    int64_t *broadcast1_shape = (int64_t *) ((int8_t *) weights->buffer() + 4);
    broadcast1_shape[0] = 1;
    broadcast1_shape[1] = 64;
    broadcast1_shape[2] = 112;
    broadcast1_shape[3] = 112;

    // Set scale/shift constants
    float *scale = (float *) ((int8_t *) weights->buffer() + 0);
    scale[0] = 127.5;

    float *shift = (float *) ((int8_t *) weights->buffer() + 68);
    shift[0] = 0;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ConvertAddToPower) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="14" name="broadcast1_data" type="Const">
            <data offset="0" size="4"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const">
            <data offset="4" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="16" name="broadcast1_axis" type="Const">
            <data offset="36" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" type="Add">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="16" from-port="1" to-layer="17" to-port="2"/>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="17" from-port="3" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" precision="FP32" type="Power">
            <data power="1.000000" scale="1.000000" shift="1.000000"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {6422528}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    // Set shape constant for broadcast1
    int64_t *broadcast1_shape = (int64_t *) ((int8_t *) weights->buffer() + 4);
    broadcast1_shape[0] = 1;
    broadcast1_shape[1] = 64;
    broadcast1_shape[2] = 112;
    broadcast1_shape[3] = 112;

    // Set scale/shift constants
    float *scale = (float *) ((int8_t *) weights->buffer() + 0);
    scale[0] = 1;

    float *shift = (float *) ((int8_t *) weights->buffer() + 68);
    shift[0] = 127;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ConvertMulToScaleShift) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="14" name="broadcast1_data" type="Const">
            <data offset="0" size="256"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="16" name="broadcast1_axis" type="Const">
            <data offset="288" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="mul" type="Multiply">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="16" from-port="1" to-layer="17" to-port="2"/>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="17" from-port="3" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="mul" precision="FP32" type="ScaleShift">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
            <weights offset="0" size="256"/>
            <biases offset="256" size="256"/>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {6422528}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    // Set shape constant for broadcast1
    int64_t * broadcast1_shape = (int64_t *)((int8_t *) weights->buffer() + 256);
    broadcast1_shape[0] = 1;
    broadcast1_shape[1] = 64;
    broadcast1_shape[2] = 112;
    broadcast1_shape[3] = 112;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ConvertAddToScaleShift) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="14" name="broadcast1_data" type="Const">
            <data offset="0" size="256"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="16" name="broadcast1_axis" type="Const">
            <data offset="288" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" type="Add">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="16" from-port="1" to-layer="17" to-port="2"/>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="17" from-port="3" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" precision="FP32" type="ScaleShift">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
            <weights offset="0" size="256"/>
            <biases offset="256" size="256"/>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {6422528}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    // Set shape constant for broadcast1
    int64_t * broadcast1_shape = (int64_t *)((int8_t *) weights->buffer() + 256);
    broadcast1_shape[0] = 1;
    broadcast1_shape[1] = 64;
    broadcast1_shape[2] = 112;
    broadcast1_shape[3] = 112;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ConvertMulToEltwise) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="14" name="broadcast1_data" type="Const">
            <data offset="0" size="448"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="16" name="broadcast1_axis" type="Const">
            <data offset="288" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="mul" type="Multiply">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="16" from-port="1" to-layer="17" to-port="2"/>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="17" from-port="3" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="14" name="broadcast1_data" precision="FP32" type="Const">
            <data offset="0" size="448"/>
            <output>
                <port id="1">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="mul" precision="FP32" type="Eltwise">
            <data operation="prod" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
            <weights offset="0" size="256"/>
            <biases offset="256" size="256"/>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="14" from-port="1" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {6422528}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    // Set shape constant for broadcast1
    int64_t * broadcast1_shape = (int64_t *)((int8_t *) weights->buffer() + 256);
    broadcast1_shape[0] = 1;
    broadcast1_shape[1] = 64;
    broadcast1_shape[2] = 112;
    broadcast1_shape[3] = 112;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ConvertAddToEltwise) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="14" name="broadcast1_data" type="Const">
            <data offset="0" size="448"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="16" name="broadcast1_axis" type="Const">
            <data offset="288" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" type="Add">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="16" from-port="1" to-layer="17" to-port="2"/>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="17" from-port="3" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="14" name="broadcast1_data" precision="FP32" type="Const">
            <data offset="0" size="448"/>
            <output>
                <port id="1">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="add" precision="FP32" type="Eltwise">
            <data operation="sum" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
            <weights offset="0" size="256"/>
            <biases offset="256" size="256"/>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="14" from-port="1" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {6422528}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    // Set shape constant for broadcast1
    int64_t * broadcast1_shape = (int64_t *)((int8_t *) weights->buffer() + 256);
    broadcast1_shape[0] = 1;
    broadcast1_shape[1] = 64;
    broadcast1_shape[2] = 112;
    broadcast1_shape[3] = 112;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ConvertBroadcastToTiles1) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="14" name="data" type="Parameter">
            <output>
                <port id="1" precision="FP32">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="16" name="broadcast1_axis" type="Const">
            <data offset="288" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="16" from-port="1" to-layer="17" to-port="2"/>
        <edge from-layer="17" from-port="3" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Constant_107" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="32"/>
            </blobs>
        </layer>
        <layer id="2" name="DynReshape_108" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="broadcast_1" precision="FP32" type="Tile">
            <data axis="3" tiles="112"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="broadcast_1_3" precision="FP32" type="Tile">
            <data axis="1" tiles="64"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
        <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {6422528}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    // Set shape constant for broadcast1
    int64_t * broadcast1_shape = (int64_t *)((int8_t *) weights->buffer() + 256);
    broadcast1_shape[0] = 1;
    broadcast1_shape[1] = 64;
    broadcast1_shape[2] = 112;
    broadcast1_shape[3] = 112;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

//    std::vector<std::shared_ptr<ngraph::Function>> g2{nGraph};
//    ngraph::pass::VisualizeTree("after.png").run_on_module(g2);
//
    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

   compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ConvertBroadcastToTiles2) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="14" name="data" type="Parameter">
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="16" name="broadcast1_axis" type="Const">
            <data offset="288" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="16" from-port="1" to-layer="17" to-port="2"/>
        <edge from-layer="17" from-port="3" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Constant_107" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="32"/>
            </blobs>
        </layer>
        <layer id="2" name="DynReshape_108" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>1</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="broadcast_1" precision="FP32" type="Tile">
            <data axis="3" tiles="112"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="broadcast_1_3" precision="FP32" type="Tile">
            <data axis="2" tiles="112"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="broadcast_1_3_2" precision="FP32" type="Tile">
            <data axis="1" tiles="64"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
        <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
        <edge from-layer="4" from-port="3" to-layer="5" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {6422528}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    // Set shape constant for broadcast1
    int64_t * broadcast1_shape = (int64_t *)((int8_t *) weights->buffer() + 256);
    broadcast1_shape[0] = 1;
    broadcast1_shape[1] = 64;
    broadcast1_shape[2] = 112;
    broadcast1_shape[3] = 112;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

   compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ConvertBroadcastToTiles3) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="14" name="data" type="Parameter">
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="15" name="broadcast1_shape" type="Const">
            <data offset="256" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="16" name="broadcast1_axis" type="Const">
            <data offset="288" size="32"/>
            <output>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="17" name="broadcast_1" type="Broadcast">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
                <port id="2" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="14" from-port="1" to-layer="17" to-port="0"/>
        <edge from-layer="15" from-port="1" to-layer="17" to-port="1"/>
        <edge from-layer="16" from-port="1" to-layer="17" to-port="2"/>
        <edge from-layer="17" from-port="3" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="broadcast_1" precision="FP32" type="Tile">
            <data axis="2" tiles="112"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {6422528}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    // Set shape constant for broadcast1
    int64_t * broadcast1_shape = (int64_t *)((int8_t *) weights->buffer() + 256);
    broadcast1_shape[0] = 1;
    broadcast1_shape[1] = 64;
    broadcast1_shape[2] = 112;
    broadcast1_shape[3] = 112;

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

//    std::vector<std::shared_ptr<ngraph::Function>> g2{nGraph};
//    ngraph::pass::VisualizeTree("after.png").run_on_module(g2);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(modelV5.data(), modelV5.length());
    net_reader.SetWeights(tWeights);

   compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, DISABLED_ConvertMulAddToScaleShiftTest) {
    std::string model = R"V0G0N(
<net name="Multiply" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="weights" type="Const">
            <data offset="0" size="256"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="mul" type="Multiply">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="biases" type="Const">
            <data offset="256" size="256"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="add" type="Add">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="output" type="Result">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="5" to-port="0"/>
        <edge from-layer="4" from-port="0" to-layer="5" to-port="1"/>
        <edge from-layer="5" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="data1" precision="FP32" type="Const">
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="3211264"/>
            </blobs>
        </layer>
        <layer id="3" name="mul" precision="FP32" type="Eltwise">
            <data operation="prod"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {6422528}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);

    auto nGraph = reader.read(model, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

   InferenceEngine::CNNNetReader net_reader;
   net_reader.ReadNetwork(modelV5.data(), modelV5.length());
   net_reader.SetWeights(tWeights);

   compareICNNNetworks(network, net_reader.getNetwork());
}
