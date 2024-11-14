// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/proposal.hpp>

#include <fstream>

using namespace cldnn;
using namespace ::tests;
using namespace std;

extern float cls_scores_data[];
extern size_t cls_scores_data_size;
extern float bbox_pred_data[];
extern size_t bbox_pred_data_size;
extern float proposal_ref[];
extern size_t proposal_ref_size;

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

template <typename Dtype, typename ImInfoType = Dtype>
class TestRunnerProposal
{
    public:
        explicit TestRunnerProposal(cldnn::tensor image_info_size, bool is_caching_test = false);

        memory::ptr Run(std::vector<Dtype>& data,
                        std::vector<Dtype>& rois);

    private:
        layout _cls_scores_layout;
        layout _bbox_pred_layout;
        layout _image_info_layout;
        topology _topology;
        proposal _test_layer;
        std::unique_ptr<network> _network;
};

template <typename Dtype, typename ImInfoType>
TestRunnerProposal<Dtype, ImInfoType>::TestRunnerProposal(cldnn::tensor image_info_size, bool is_caching_test) :
                            _cls_scores_layout(ov::element::from<Dtype>(), format::bfyx, { 1, 18, 23, 14 } ),
                            _bbox_pred_layout(ov::element::from<Dtype>(), format::bfyx, { 1, 36, 23, 14 } ),
                            _image_info_layout(ov::element::from<ImInfoType>(), format::bfyx, image_info_size),
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
                                        scales)
{
    _topology.add(input_layout(cls_scores_name, _cls_scores_layout));
    _topology.add(input_layout(bbox_pred_name, _bbox_pred_layout));
    _topology.add(input_layout(image_info_name, _image_info_layout));

    _topology.add(_test_layer);

    if (is_caching_test) {
        membuf mem_buf;
        {
            std::ostream out_mem(&mem_buf);
            BinaryOutputBuffer ob = BinaryOutputBuffer(out_mem);
            ob.set_stream(get_test_stream_ptr().get());
            program::build_program(get_test_engine(), _topology, {})->save(ob);
        }
        {
            std::istream in_mem(&mem_buf);
            BinaryInputBuffer ib = BinaryInputBuffer(in_mem, get_test_engine());
            auto imported_prog = std::make_shared<cldnn::program>(get_test_engine());
            imported_prog->load(ib);
            _network.reset(new network(imported_prog));
        }
    } else {
        _network.reset(new network(get_test_engine(), _topology));
    }
}

template <typename Dtype, typename ImInfoType>
memory::ptr TestRunnerProposal<Dtype, ImInfoType>::Run(std::vector<Dtype>& cls_scores_vals,
                                                       std::vector<Dtype>& bbox_pred_vals)
{
    auto& engine = get_test_engine();
    memory::ptr cls_scores = engine.attach_memory(_cls_scores_layout, cls_scores_vals.data());
    memory::ptr bbox_pred  = engine.attach_memory(_bbox_pred_layout, bbox_pred_vals.data());

    std::vector<ImInfoType> image_info_vals = { (ImInfoType)((float)image_h - 0.0000001f), // check fp robustness of the layer
                                                (ImInfoType)((float)image_w + 0.0000001f), // check fp robustness of the layer
                                                (ImInfoType)((float)image_z) };
    memory::ptr image_info = engine.allocate_memory(_image_info_layout);
    tests::set_values(image_info, image_info_vals);

    std::vector<event::ptr> events;
    events.push_back(_network->set_input_data(cls_scores_name, cls_scores));
    events.push_back(_network->set_input_data(bbox_pred_name, bbox_pred));
    events.push_back(_network->set_input_data(image_info_name, image_info));
    _network->get_stream().wait_for_events(events);

    std::map<primitive_id, network_output> network_output = _network->execute();
    EXPECT_EQ(network_output.begin()->first, layer_name);
    return network_output.at(layer_name).get_memory();
}

template <typename Dtype>
void test_proposal_basic(cldnn::tensor image_info_size, bool is_caching_test) {
    std::vector<Dtype> cls_scores(&cls_scores_data[0], &cls_scores_data[cls_scores_data_size]);
    std::vector<Dtype> bbox_pred(&bbox_pred_data[0], &bbox_pred_data[bbox_pred_data_size]);

    TestRunnerProposal<Dtype> t(image_info_size, is_caching_test);

    memory::ptr output = t.Run(cls_scores, bbox_pred);
    ASSERT_EQ(output->get_layout().count(), proposal_ref_size);

    cldnn::mem_lock<Dtype> f(output, get_test_stream());

    for (size_t i = 0; i < proposal_ref_size; i++) {
        Dtype ref(proposal_ref[i]);
        ASSERT_NEAR((float)f[i], (float)ref, epsilon_fp16);
    }
}

TEST(proposal, basic) {
    test_proposal_basic<float>({ 1, 3, 1, 1 }, false);
}

TEST(proposal, fp16) {
    test_proposal_basic<ov::float16>({ 1, 3, 1, 1 }, false);
}

TEST(proposal, img_info_batched) {
    test_proposal_basic<float>({ 2, 3, 1, 1 }, false);
}

TEST(proposal, img_info_batch_only) {
    test_proposal_basic<float>({ 3, 1, 1, 1 }, false);
}

template <typename Dtype, typename ImInfoType>
void test_proposal_basic_two_types(cldnn::tensor image_info_size, bool is_caching_test) {
    std::vector<Dtype> cls_scores(&cls_scores_data[0], &cls_scores_data[cls_scores_data_size]);
    std::vector<Dtype> bbox_pred(&bbox_pred_data[0], &bbox_pred_data[bbox_pred_data_size]);

    TestRunnerProposal<Dtype, ImInfoType> t(image_info_size, is_caching_test);

    memory::ptr output = t.Run(cls_scores, bbox_pred);
    ASSERT_EQ(output->get_layout().count(), proposal_ref_size);

    cldnn::mem_lock<Dtype> d(output, get_test_stream());

    for (size_t i = 0; i < proposal_ref_size; i++) {
        Dtype ref(proposal_ref[i]);
        ASSERT_NEAR((float)d[i], (float)ref, epsilon_fp16);
    }
}

TEST(proposal, scores_fp16_im_info_fp32) {
    test_proposal_basic_two_types<ov::float16, float>({ 1, 3, 1, 1 }, false);
}

TEST(proposal, scores_fp32_im_info_fp16) {
    test_proposal_basic_two_types<float, ov::float16>({ 1, 3, 1, 1 }, false);
}
#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST(proposal, basic_cached) {
    test_proposal_basic<float>({ 1, 3, 1, 1 }, true);
}

TEST(proposal, fp16_cached) {
    test_proposal_basic<ov::float16>({ 1, 3, 1, 1 }, true);
}

TEST(proposal, img_info_batched_cached) {
    test_proposal_basic<float>({ 2, 3, 1, 1 }, true);
}

TEST(proposal, img_info_batch_only_cached) {
    test_proposal_basic<float>({ 3, 1, 1, 1 }, true);
}

TEST(proposal, scores_fp16_im_info_fp32_cached) {
    test_proposal_basic_two_types<ov::float16, float>({ 1, 3, 1, 1 }, true);
}
#endif // RUN_ALL_MODEL_CACHING_TESTS
TEST(proposal, scores_fp32_im_info_fp16_cached) {
    test_proposal_basic_two_types<float, ov::float16>({ 1, 3, 1, 1 }, true);
}
