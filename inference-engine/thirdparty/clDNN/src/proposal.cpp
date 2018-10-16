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

#include "proposal_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"

#include <cmath>

namespace cldnn
{

static void generate_anchors(unsigned base_size, const std::vector<float>& ratios, const std::vector<float>& scales,    // input
                             std::vector<proposal_inst::anchor>& anchors);                                              // output


primitive_type_id proposal_type_id()
{
    static primitive_type_base<proposal> instance;
    return &instance;
}


layout proposal_inst::calc_output_layout(proposal_node const& node)
{
    auto desc = node.get_primitive();
    layout input_layout = node.get_dependency(cls_scores_index).get_output_layout();

    return layout(input_layout.data_type, format::bfyx, { desc->post_nms_topn, 1, CLDNN_ROI_VECTOR_SIZE, 1 });
}

static inline std::string stringify_vector(std::vector<float> v)
{
    std::stringstream s;

    s << "{ ";

    for (size_t i = 0; i < v.size(); ++i)
    {
        s << v.at(i);
        if (i + 1 < v.size()) s << ", ";
    }

    s << " }";

    return s.str();
}

//TODO: rename to?
static std::string stringify_port(const program_node & p)
{
    std::stringstream res;
    auto node_info = p.desc_to_json();
    node_info.dump(res);

    return res.str();
}


std::string proposal_inst::to_string(proposal_node const& node)
{
    auto desc         = node.get_primitive();
    auto node_info    = node.desc_to_json();
    auto scales_parm  = desc->scales;

    std::stringstream primitive_description;

    json_composite proposal_info;
    proposal_info.add("cls score", stringify_port(node.cls_score()));
    proposal_info.add("box pred", stringify_port(node.bbox_pred()));
    proposal_info.add("image info", stringify_port(node.image_info()));

    json_composite params;
    params.add("max proposals", desc->max_proposals);
    params.add("iou threshold", desc->iou_threshold);
    params.add("min bbox size", desc->min_bbox_size);
    params.add("pre nms topn", desc->pre_nms_topn);
    params.add("post nms topn", desc->post_nms_topn);
    params.add("ratios", stringify_vector(desc->ratios));
    params.add("scales", stringify_vector(desc->scales));
    proposal_info.add("params", params);

    node_info.add("proposal info", proposal_info);
    node_info.dump(primitive_description);

    return primitive_description.str();
}

proposal_inst::typed_primitive_inst(network_impl& network, proposal_node const& node)
    :parent(network, node)
{
//    std::vector<float> default_ratios = { 0.5f, 1.0f, 2.0f };
    int default_size = 16;
    generate_anchors(default_size, argument.ratios, argument.scales, _anchors);
}

static void calc_basic_params(
        const proposal_inst::anchor& base_anchor,                       // input
        float& width, float& height, float& x_center, float& y_center)  // output
{
    width  = base_anchor.end_x - base_anchor.start_x + 1.0f;
    height = base_anchor.end_y - base_anchor.start_y + 1.0f;

    x_center = base_anchor.start_x + 0.5f * (width - 1.0f);
    y_center = base_anchor.start_y + 0.5f * (height - 1.0f);
}

static std::vector<proposal_inst::anchor> make_anchors(
        const std::vector<float>& ws,
        const std::vector<float>& hs,
        float x_center,
        float y_center)
{
    size_t len = ws.size();
    assert(hs.size() == len);

    std::vector<proposal_inst::anchor> anchors(len);

    for (size_t i = 0 ; i < len ; i++)
    {
        // transpose to create the anchor
        anchors[i].start_x = x_center - 0.5f * (ws[i] - 1.0f);
        anchors[i].start_y = y_center - 0.5f * (hs[i] - 1.0f);
        anchors[i].end_x   = x_center + 0.5f * (ws[i] - 1.0f);
        anchors[i].end_y   = y_center + 0.5f * (hs[i] - 1.0f);
    }

    return anchors;
}

static std::vector<proposal_inst::anchor> calc_anchors(
        const proposal_inst::anchor& base_anchor,
        const std::vector<float>& scales)
{
    float width = 0.0f, height = 0.0f, x_center = 0.0f, y_center = 0.0f;

    calc_basic_params(base_anchor, width, height, x_center, y_center);

    size_t num_scales = scales.size();
    std::vector<float> ws(num_scales), hs(num_scales);

    for (unsigned int i = 0 ; i < num_scales ; i++)
    {
        ws[i] = width * scales[i];
        hs[i] = height * scales[i];
    }

    return make_anchors(ws, hs, x_center, y_center);
}

static std::vector<proposal_inst::anchor> calc_ratio_anchors(
        const proposal_inst::anchor& base_anchor,
        const std::vector<float>& ratios)
{
    float width = 0.0f, height = 0.0f, x_center = 0.0f, y_center = 0.0f;

    calc_basic_params(base_anchor, width, height, x_center, y_center);

    float size = width * height;

    size_t num_ratios = ratios.size();

    std::vector<float> ws(num_ratios), hs(num_ratios);

    for (unsigned int i = 0 ; i < num_ratios ; i++)
    {
        float new_size = size / ratios[i];
        ws[i] = round(sqrt(new_size));
        hs[i] = round(ws[i] * ratios[i]);
    }

    return make_anchors(ws, hs, x_center, y_center);
}

static void generate_anchors(unsigned int base_size, const std::vector<float>& ratios, const std::vector<float>& scales,   // input
                     std::vector<proposal_inst::anchor>& anchors)                                                       // output
{
    float end = (float)(base_size - 1);        // because we start at zero

    proposal_inst::anchor base_anchor(0.0f, 0.0f, end, end);

    std::vector<proposal_inst::anchor> ratio_anchors = calc_ratio_anchors(base_anchor, ratios);

    for (auto& ratio_anchor : ratio_anchors)
    {
        std::vector<proposal_inst::anchor> tmp = calc_anchors(ratio_anchor, scales);
        anchors.insert(anchors.end(), tmp.begin(), tmp.end());
    }
}
}
