// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ROIPoolingNetwork){
    std::string model_v10 = R"V0G0N(
    <net name="ROIPoolingNet" version="10">
        <layers>
            <layer id="0" name="in1" type="Parameter">
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>1024</dim>
                        <dim>100</dim>
                        <dim>100</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="in2" type="Parameter">
                <output>
                    <port id="0" precision="FP32">
                        <dim>100</dim>
                        <dim>5</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="CropAndResize/CropAndResize" precision="FP32" type="ROIPooling">
                <data method="bilinear" pooled_h="14" pooled_w="14" spatial_scale="1"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>1024</dim>
                        <dim>100</dim>
                        <dim>100</dim>
                    </port>
                    <port id="1">
                        <dim>100</dim>
                        <dim>5</dim>
                    </port>
                </input>
                <output>
                    <port id="2">
                        <dim>100</dim>
                        <dim>1024</dim>
                        <dim>14</dim>
                        <dim>14</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="output" type="Result">
                <input>
                    <port id="0" precision="FP32">
                        <dim>100</dim>
                        <dim>1024</dim>
                        <dim>14</dim>
                        <dim>14</dim>
                    </port>
                </input>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
            <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
        </edges>
    </net>
    )V0G0N";
    std::string model_v6  = R"V0G0N(
    <net name="ROIPoolingNet" version="6" batch="1">
       <layers>
          <layer name="in2" type="Input" precision="FP32" id="0">
             <output>
                <port id="0">
                   <dim>100</dim>
                   <dim>5</dim>
                </port>
             </output>
          </layer>
          <layer name="in1" type="Input" precision="FP32" id="1">
             <output>
                <port id="0">
                   <dim>1</dim>
                   <dim>1024</dim>
                   <dim>100</dim>
                   <dim>100</dim>
                </port>
             </output>
          </layer>
          <layer name="CropAndResize/CropAndResize" type="ROIPooling" precision="FP32" id="2">
             <data method="bilinear" pooled_h="14" pooled_w="14" spatial_scale="1" />
             <input>
                <port id="0">
                   <dim>1</dim>
                   <dim>1024</dim>
                   <dim>100</dim>
                   <dim>100</dim>
                </port>
                <port id="1">
                   <dim>100</dim>
                   <dim>5</dim>
                </port>
             </input>
             <output>
                <port id="2">
                   <dim>100</dim>
                   <dim>1024</dim>
                   <dim>14</dim>
                   <dim>14</dim>
                </port>
             </output>
          </layer>
       </layers>
       <edges>
          <edge from-layer="0" from-port="0" to-layer="2" to-port="1" />
          <edge from-layer="1" from-port="0" to-layer="2" to-port="0" />
       </edges>
       <statistics />
    </net>
    )V0G0N";

    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {48}, Layout::C));
    weights->allocate();
    fill_data(static_cast<float*>(weights->buffer()), weights->size() / sizeof(float));

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);
    auto nGraph   = reader.read(model_v10, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(model_v6.data(), model_v6.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}

TEST_F(NGraphReaderTests, ROIPoolingNetwork_2){
    std::string model_v10 = R"V0G0N(
    <net name="ROIPoolingNet" version="10">
        <layers>
            <layer id="0" name="in1" type="Parameter">
                <output>
                    <port id="0" precision="FP16">
                         <dim>1</dim>
                         <dim>1024</dim>
                         <dim>14</dim>
                         <dim>14</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="in2" type="Parameter">
                <output>
                    <port id="0" precision="FP16">
                         <dim>300</dim>
                         <dim>5</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="roi_pool5" precision="FP16" type="ROIPooling">
               <data pooled_h="14" pooled_w="14" spatial_scale="0.0625" />
               <input>
                  <port id="0">
                     <dim>1</dim>
                     <dim>1024</dim>
                     <dim>14</dim>
                     <dim>14</dim>
                  </port>
                  <port id="1">
                     <dim>300</dim>
                     <dim>5</dim>
                  </port>
               </input>
               <output>
                  <port id="2">
                     <dim>300</dim>
                     <dim>1024</dim>
                     <dim>14</dim>
                     <dim>14</dim>
                  </port>
               </output>
            </layer>
            <layer id="3" name="output" type="Result">
                <input>
                    <port id="0" precision="FP16">
                         <dim>300</dim>
                         <dim>1024</dim>
                         <dim>14</dim>
                         <dim>14</dim>
                    </port>
                </input>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
            <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
        </edges>
    </net>
    )V0G0N";
    std::string model_v6  = R"V0G0N(
    <net name="ROIPoolingNet" version="6" batch="1">
       <layers>
          <layer name="in2" type="Input" precision="FP16" id="0">
             <output>
                <port id="0">
                   <dim>300</dim>
                   <dim>5</dim>
                </port>
             </output>
          </layer>
          <layer name="in1" type="Input" precision="FP16" id="1">
             <output>
                <port id="0">
                   <dim>1</dim>
                   <dim>1024</dim>
                   <dim>14</dim>
                   <dim>14</dim>
                </port>
             </output>
          </layer>
          <layer name="roi_pool5" type="ROIPooling" precision="FP16" id="2">
             <data method="max" pooled_h="14" pooled_w="14" spatial_scale="0.0625" />
             <input>
                <port id="0">
                   <dim>1</dim>
                   <dim>1024</dim>
                   <dim>14</dim>
                   <dim>14</dim>
                </port>
                <port id="1">
                   <dim>300</dim>
                   <dim>5</dim>
                </port>
             </input>
             <output>
                <port id="2">
                   <dim>300</dim>
                   <dim>1024</dim>
                   <dim>14</dim>
                   <dim>14</dim>
                </port>
             </output>
          </layer>
       </layers>
       <edges>
          <edge from-layer="0" from-port="0" to-layer="2" to-port="1" />
          <edge from-layer="1" from-port="0" to-layer="2" to-port="0" />
       </edges>
       <statistics />
    </net>
    )V0G0N";

    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {48}, Layout::C));
    weights->allocate();
    fill_data(static_cast<float*>(weights->buffer()), weights->size() / sizeof(float));

    auto tWeights = std::dynamic_pointer_cast<TBlob<uint8_t>>(weights);
    auto nGraph   = reader.read(model_v10, weights);

    ICNNNetwork::Ptr network = convertFunctionToICNNNetwork(nGraph);
    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(model_v6.data(), model_v6.length());

    compareICNNNetworks(network, net_reader.getNetwork());
}