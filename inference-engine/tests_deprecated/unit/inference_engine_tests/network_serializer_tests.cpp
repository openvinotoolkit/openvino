// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <cpp/ie_cnn_network.h>
#include <network_serializer.h>


static const auto model = R"_(
<?xml version="1.0" encoding="UTF-8"?>
<net name="model" version="6" batch="1">
   <layers>
      <layer name="relu1" type="Input" precision="FP16" id="1">
         <output>
            <port id="0" precision="FP32">
               <dim>1</dim>
               <dim>96</dim>
               <dim>55</dim>
               <dim>55</dim>
            </port>
         </output>
      </layer>
      <layer name="norm1" type="Norm" precision="FP16" id="2">
         <data alpha="0.000100" beta="0.750000" local-size="5" region="across" />
         <input>
            <port id="0">
               <dim>1</dim>
               <dim>96</dim>
               <dim>55</dim>
               <dim>55</dim>
            </port>
         </input>
         <output>
            <port id="1" precision="FP16">
               <dim>1</dim>
               <dim>96</dim>
               <dim>55</dim>
               <dim>55</dim>
            </port>
         </output>
      </layer>
      <layer name="pool1" type="Pooling" precision="FP16" id="3">
         <data kernel="3,3" kernel-x="3" kernel-y="3" pad-x="0" pad-y="0" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding-type="ceil" stride-x="2" stride-y="2" strides="2,2" />
         <input>
            <port id="0">
               <dim>1</dim>
               <dim>96</dim>
               <dim>55</dim>
               <dim>55</dim>
            </port>
         </input>
         <output>
            <port id="1" precision="FP16">
               <dim>1</dim>
               <dim>96</dim>
               <dim>27</dim>
               <dim>27</dim>
            </port>
         </output>
      </layer>
      <layer name="conv2_split" type="Split" precision="FP16" id="4">
         <data axis="1" out_sizes="48,48" />
         <input>
            <port id="0">
               <dim>1</dim>
               <dim>96</dim>
               <dim>27</dim>
               <dim>27</dim>
            </port>
         </input>
         <output>
            <port id="1" precision="FP32">
               <dim>1</dim>
               <dim>48</dim>
               <dim>27</dim>
               <dim>27</dim>
            </port>
            <port id="2" precision="FP32">
               <dim>1</dim>
               <dim>48</dim>
               <dim>27</dim>
               <dim>27</dim>
            </port>
         </output>
      </layer>
   </layers>
   <edges>
      <edge from-layer="1" from-port="0" to-layer="2" to-port="0" />
      <edge from-layer="2" from-port="1" to-layer="3" to-port="0" />
      <edge from-layer="3" from-port="1" to-layer="4" to-port="0" />
   </edges>
   <statistics />
</net>
)_";

TEST(NetworkSerializerTest, TopoSortResultUnique) {
    auto reader = InferenceEngine::CreateCNNNetReaderPtr();

    InferenceEngine::ResponseDesc resp;

    ASSERT_EQ(InferenceEngine::StatusCode::OK, reader->ReadNetwork(model, std::strlen(model), &resp)) << resp.msg;

    auto network = InferenceEngine::CNNNetwork(reader);

    auto sorted = InferenceEngine::Serialization::TopologicalSort(network);

    std::vector<std::string> actualLayerNames;
    for (auto&& layer : sorted) {
        actualLayerNames.emplace_back(layer->name);
    }

    std::vector<std::string> expectedLayerNames = {
        "relu1", "norm1", "pool1", "conv2_split"
    };

    ASSERT_EQ(expectedLayerNames, actualLayerNames);
}
