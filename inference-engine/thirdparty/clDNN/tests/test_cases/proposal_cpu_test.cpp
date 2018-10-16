/*
// Copyright (c) 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <fstream>

#include <gtest/gtest.h>
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include <api/CPP/proposal.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include "test_utils/float16.h"

namespace cldnn
{
template<> struct type_to_data_type<FLOAT16> { static const data_types value = data_types::f16; };
}

using namespace cldnn;
using namespace tests;
using namespace std;

extern float cls_scores_data[];
extern size_t cls_scores_data_size;
extern float bbox_pred_data[];
extern size_t bbox_pred_data_size;
extern float proposal_ref[];
extern size_t proposal_ref_size;

const float epsilon = 0.00025f;
const float epsilon_fp16 = 0.125f;

// !!!!!!!!
// The data for this test (input and ref) was generated in clCaffe using the zf truncated prototxt with the following modifications:
// input height: 420 -> 210
// input width: 700 -> 350
// max proposals: 300 -> 50
// post nms topn: 150 -> 25
// !!!!!!!!

const primitive_id cls_scores_name = "cls_scores";
const primitive_id bbox_pred_name = "bbox_pred";
const primitive_id image_info_name = "image_info";
const primitive_id layer_name = "proposal";

const int max_proposals = 50;
const float iou_threshold = 0.7f;
const int min_bbox_size = 16;
const int feature_stride = 16;
const int pre_nms_topn = 6000;
const int post_nms_topn = 25;
const int image_w = 350;
const int image_h = 210;
const int image_z = 1;
const std::vector<float> ratios = { 0.5f, 1.0f, 2.0f };
const std::vector<float> scales = { 8.0f, 16.0f, 32.0f };

template <typename Dtype>
class TestRunnerProposal
{
    public:
        TestRunnerProposal();

        memory Run(std::vector<Dtype>& data, 
                   std::vector<Dtype>& rois);

    private:
        engine _engine;
        layout _cls_scores_layout;
        layout _bbox_pred_layout;
        layout _image_info_layout;
        topology _topology;
        proposal _test_layer;
        std::unique_ptr<network> _network;
};

template <typename Dtype>
TestRunnerProposal<Dtype>::TestRunnerProposal() :
                            _cls_scores_layout(cldnn::type_to_data_type<Dtype>::value, format::bfyx, { 1, 18, 23, 14 } ),
                            _bbox_pred_layout(cldnn::type_to_data_type<Dtype>::value, format::bfyx, { 1, 36, 23, 14 } ),
                            _image_info_layout(cldnn::type_to_data_type<Dtype>::value, format::bfyx, { 1, 1, 3, 1 } ),
                            _test_layer(layer_name, 
                                        cls_scores_name, 
                                        bbox_pred_name,
                                        image_info_name,
                                        max_proposals,
                                        iou_threshold,
                                        min_bbox_size,
                                        feature_stride,
                                        pre_nms_topn,
                                        post_nms_topn,
                                        ratios,
                                        scales,
                                        padding())
{    
    _topology.add(input_layout(cls_scores_name, _cls_scores_layout));
    _topology.add(input_layout(bbox_pred_name, _bbox_pred_layout));
    _topology.add(input_layout(image_info_name, _image_info_layout));

    _topology.add(_test_layer);

    _network.reset(new network(_engine, _topology));
}

template <typename Dtype>
memory TestRunnerProposal<Dtype>::Run(std::vector<Dtype>& cls_scores_vals, 
                              std::vector<Dtype>& bbox_pred_vals)
{
    memory cls_scores = memory::attach(_cls_scores_layout, cls_scores_vals.data(), cls_scores_vals.size());
    memory bbox_pred  = memory::attach(_bbox_pred_layout, bbox_pred_vals.data(), bbox_pred_vals.size());

    Dtype image_info_vals[] = { (Dtype)((float)image_h - 0.0000001f), // check fp robustness of the layer
                                (Dtype)((float)image_w + 0.0000001f), // check fp robustness of the layer 
                                (Dtype)((float)image_z) };
    memory image_info = memory::attach(_image_info_layout, &image_info_vals[0], 3);
   
    _network->set_input_data(cls_scores_name, cls_scores);
    _network->set_input_data(bbox_pred_name, bbox_pred);
    _network->set_input_data(image_info_name, image_info);

    std::map<primitive_id, network_output> network_output = _network->execute();
    EXPECT_EQ(network_output.begin()->first, layer_name);
    return network_output.at(layer_name).get_memory();    
}

TEST(proposal, basic) {
    std::vector<float> cls_scores(&cls_scores_data[0], &cls_scores_data[cls_scores_data_size]);
    std::vector<float> bbox_pred(&bbox_pred_data[0], &bbox_pred_data[bbox_pred_data_size]);

    TestRunnerProposal<float> t;

    const memory& output = t.Run(cls_scores, bbox_pred);
    ASSERT_EQ(output.get_layout().count(), proposal_ref_size);

    auto f = output.pointer<float>();

    for (size_t i = 0; i < proposal_ref_size; i++) {
        EXPECT_NEAR(f[i], proposal_ref[i], epsilon);
    }
}

TEST(proposal, fp16) {
    std::vector<FLOAT16> cls_scores(&cls_scores_data[0], &cls_scores_data[cls_scores_data_size]);
    std::vector<FLOAT16> bbox_pred(&bbox_pred_data[0], &bbox_pred_data[bbox_pred_data_size]);
    
    TestRunnerProposal<FLOAT16> t;

    const memory& output = t.Run(cls_scores, bbox_pred);
    ASSERT_EQ(output.get_layout().count(), proposal_ref_size);

    auto d = output.pointer<FLOAT16>();

    for (size_t i = 0; i < proposal_ref_size; i++) {
        FLOAT16 ref(proposal_ref[i]);
        EXPECT_NEAR((float)d[i], (float)ref, epsilon_fp16);
    }
}

