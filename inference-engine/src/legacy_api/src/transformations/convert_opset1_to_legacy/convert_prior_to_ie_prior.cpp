// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/transformations/convert_opset1_to_legacy/convert_prior_to_ie_prior.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset1.hpp>

#include <legacy/ngraph_ops/prior_box_ie.hpp>
#include <legacy/ngraph_ops/prior_box_clustered_ie.hpp>
#include <ngraph/rt_info.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertPriorBox, "ConvertPriorBox", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertPriorBoxToLegacy, "ConvertPriorBoxToLegacy", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertPriorBoxClusteredToLegacy, "ConvertPriorBoxClusteredToLegacy", 0);

ngraph::pass::ConvertPriorBoxToLegacy::ConvertPriorBoxToLegacy() {
    auto data = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});
    auto axes = ngraph::opset1::Constant::create(element::i64, Shape{1}, {0});
    auto image = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});

    ngraph::op::PriorBoxAttrs attr;
    attr.min_size = {162.0f};
    attr.max_size = {213.0f};
    attr.aspect_ratio = {2.0f, 3.0f};
    attr.variance = {0.1f, 0.1f, 0.2f, 0.2f};
    attr.step = 64.0f;
    attr.offset = 0.5f;
    attr.clip = 0;
    attr.flip = 1;
    attr.scale_all_sizes = true;

    auto prior_box = std::make_shared<ngraph::opset1::PriorBox>(data, image, attr);
    auto unsqueeze = std::make_shared<ngraph::opset1::Unsqueeze> (prior_box, axes);

    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto unsqueeze = std::dynamic_pointer_cast<ngraph::opset1::Unsqueeze> (m.get_match_root());
        if (!unsqueeze) {
            return false;
        }
        auto prior_box_node = std::dynamic_pointer_cast<ngraph::opset1::PriorBox> (unsqueeze->input_value(0).get_node_shared_ptr());

        if (!prior_box_node || transformation_callback(prior_box_node)) {
            return false;
        }

        // vector of nGraph nodes that will be replaced
        ngraph::NodeVector ops_to_replace{unsqueeze, prior_box_node};

        std::shared_ptr<Node> input_1(prior_box_node->input_value(0).get_node_shared_ptr());
        std::shared_ptr<Node> input_2(prior_box_node->input_value(1).get_node_shared_ptr());

        auto convert1 = std::dynamic_pointer_cast<ngraph::opset1::Convert> (input_1);
        auto convert2 = std::dynamic_pointer_cast<ngraph::opset1::Convert> (input_2);

        if (convert1 && convert2) {
            ops_to_replace.push_back(convert1);
            ops_to_replace.push_back(convert2);
            input_1 = convert1->input_value(0).get_node_shared_ptr();
            input_2 = convert2->input_value(0).get_node_shared_ptr();
        }

        auto strided_slice1 = std::dynamic_pointer_cast<ngraph::opset1::StridedSlice> (input_1);
        auto strided_slice2 = std::dynamic_pointer_cast<ngraph::opset1::StridedSlice> (input_2);

        if (!strided_slice1 || !strided_slice2) {
            return false;
        }

        ops_to_replace.push_back(strided_slice1);
        ops_to_replace.push_back(strided_slice2);

        //  Check that StridedSlice1 cuts H,W dims for PriorBox
        auto begin = std::dynamic_pointer_cast<ngraph::opset1::Constant> (strided_slice1->input_value(1).get_node_shared_ptr());
        auto end = std::dynamic_pointer_cast<ngraph::opset1::Constant> (strided_slice1->input_value(2).get_node_shared_ptr());
        auto stride = std::dynamic_pointer_cast<ngraph::opset1::Constant> (strided_slice1->input_value(3).get_node_shared_ptr());

        if (!begin || !end || !stride) {
            return false;
        }

        auto begin_val = begin->get_vector<int64_t>();
        auto end_val = end->get_vector<int64_t>();
        auto stride_val = stride->get_vector<int64_t>();

        if (begin_val.size() != 1 && begin_val[0] != 2) {
            return false;
        }

        if (end_val.size() != 1 && end_val[0] != 4) {
            return false;
        }

        if (stride_val.size() != 1 && stride_val[0] != 1) {
            return false;
        }

        // TODO: should we check second StridedSlice?
        input_1 = strided_slice1->input_value(0).get_node_shared_ptr();
        input_2 = strided_slice2->input_value(0).get_node_shared_ptr();

        convert1 = std::dynamic_pointer_cast<ngraph::opset1::Convert> (input_1);
        convert2 = std::dynamic_pointer_cast<ngraph::opset1::Convert> (input_2);

        if (convert1 && convert2) {
            ops_to_replace.push_back(convert1);
            ops_to_replace.push_back(convert2);
            input_1 = convert1->input_value(0).get_node_shared_ptr();
            input_2 = convert2->input_value(0).get_node_shared_ptr();
        }

        // the input can be either ShapeOf-1 or ShapeOf-3
        std::shared_ptr<ngraph::op::Op> shape_of1 = std::dynamic_pointer_cast<ngraph::opset1::ShapeOf> (input_1);
        std::shared_ptr<ngraph::op::Op> shape_of2 = std::dynamic_pointer_cast<ngraph::opset1::ShapeOf> (input_2);

        if (!shape_of1 || !shape_of2) {
            shape_of1 = std::dynamic_pointer_cast<ngraph::opset3::ShapeOf>(input_1);
            shape_of2 = std::dynamic_pointer_cast<ngraph::opset3::ShapeOf>(input_2);
        }
        if (!shape_of1 || !shape_of2) {
            return false;
        }
        // keep this code for a while if will decide to run this transformation again in the opset1->legacy
        // the input can be either ShapeOf or Convert(ShapeOf)
//        if (!shape_of1 || !shape_of2) {
//            auto shapeof1_convert = std::dynamic_pointer_cast<ngraph::opset1::Convert> (input_1);
//            auto shapeof2_convert = std::dynamic_pointer_cast<ngraph::opset1::Convert> (input_2);
//            if (!shapeof1_convert || !shapeof2_convert)
//                return false;
//            shape_of1 = std::dynamic_pointer_cast<ngraph::opset1::ShapeOf>(shapeof1_convert->input_value(0).get_node_shared_ptr());
//            shape_of2 = std::dynamic_pointer_cast<ngraph::opset1::ShapeOf>(shapeof2_convert->input_value(0).get_node_shared_ptr());
//            if (!shape_of1 || !shape_of2)
//                return false;
//            ops_to_replace.push_back(shapeof1_convert);
//            ops_to_replace.push_back(shapeof2_convert);
//        }

        ops_to_replace.push_back(shape_of1);
        ops_to_replace.push_back(shape_of2);

        auto prior_box_ie = std::make_shared<ngraph::op::PriorBoxIE> (shape_of1->input_value(0),
                                                                      shape_of2->input_value(0),
                                                                      prior_box_node->get_attrs());

        prior_box_ie->set_friendly_name(unsqueeze->get_friendly_name());

        // Nodes in copy runtime info function should be in topological order
        std::reverse(ops_to_replace.begin(), ops_to_replace.end());
        ngraph::copy_runtime_info(ops_to_replace, prior_box_ie);
        ngraph::replace_node(m.get_match_root(), prior_box_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(unsqueeze, "ConvertPriorBoxToLegacy");
    register_matcher(m, callback);
}

ngraph::pass::ConvertPriorBoxClusteredToLegacy::ConvertPriorBoxClusteredToLegacy() {
    auto data = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});
    auto axes = ngraph::opset1::Constant::create(element::i64, Shape{1}, {0});
    auto image = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});

    ngraph::op::PriorBoxClusteredAttrs attr;
    attr.widths = {0.1f, 0.1f, 0.2f, 0.2f};
    attr.heights = {0.1f, 0.1f, 0.2f, 0.2f};
    attr.variances = {0.1f, 0.1f, 0.2f, 0.2f};
    attr.step_widths = 64.0f;
    attr.step_heights = 64.0f;
    attr.offset = 0.5f;
    attr.clip = false;

    auto prior_box = std::make_shared<ngraph::opset1::PriorBoxClustered>(data, image, attr);
    auto unsqueeze = std::make_shared<ngraph::opset1::Unsqueeze> (prior_box, axes);

    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto unsqueeze = std::dynamic_pointer_cast<ngraph::opset1::Unsqueeze> (m.get_match_root());
        if (!unsqueeze) {
            return false;
        }
        auto prior_box_node = std::dynamic_pointer_cast<ngraph::opset1::PriorBoxClustered>(unsqueeze->input_value(0).get_node_shared_ptr());

        if (!prior_box_node || transformation_callback(prior_box_node)) {
            return false;
        }

        // vector of nGraph nodes that will be replaced
        ngraph::NodeVector ops_to_replace{unsqueeze, prior_box_node};

        std::shared_ptr<Node> input_1(prior_box_node->input_value(0).get_node_shared_ptr());
        std::shared_ptr<Node> input_2(prior_box_node->input_value(1).get_node_shared_ptr());

        auto convert1 = std::dynamic_pointer_cast<ngraph::opset1::Convert> (input_1);
        auto convert2 = std::dynamic_pointer_cast<ngraph::opset1::Convert> (input_2);

        if (convert1 && convert2) {
            ops_to_replace.push_back(convert1);
            ops_to_replace.push_back(convert2);
            input_1 = convert1->input_value(0).get_node_shared_ptr();
            input_2 = convert2->input_value(0).get_node_shared_ptr();
        }

        auto strided_slice1 = std::dynamic_pointer_cast<ngraph::opset1::StridedSlice> (input_1);
        auto strided_slice2 = std::dynamic_pointer_cast<ngraph::opset1::StridedSlice> (input_2);

        if (!strided_slice1 || !strided_slice2) {
            return false;
        }

        ops_to_replace.push_back(strided_slice1);
        ops_to_replace.push_back(strided_slice2);

        //  Check that StridedSlice1 cuts H,W dims for PriorBox
        auto begin = std::dynamic_pointer_cast<ngraph::opset1::Constant> (strided_slice1->input_value(1).get_node_shared_ptr());
        auto end = std::dynamic_pointer_cast<ngraph::opset1::Constant> (strided_slice1->input_value(2).get_node_shared_ptr());
        auto stride = std::dynamic_pointer_cast<ngraph::opset1::Constant> (strided_slice1->input_value(3).get_node_shared_ptr());

        if (!begin || !end || !stride) {
            return false;
        }

        auto begin_val = begin->get_vector<int64_t>();
        auto end_val = end->get_vector<int64_t>();
        auto stride_val = stride->get_vector<int64_t>();

        if (begin_val.size() != 1 && begin_val[0] != 2) {
            return false;
        }

        if (end_val.size() != 1 && end_val[0] != 4) {
            return false;
        }

        if (stride_val.size() != 1 && stride_val[0] != 1) {
            return false;
        }

        // TODO: should we check second StridedSlice?
        input_1 = strided_slice1->input_value(0).get_node_shared_ptr();
        input_2 = strided_slice2->input_value(0).get_node_shared_ptr();

        convert1 = std::dynamic_pointer_cast<ngraph::opset1::Convert> (input_1);
        convert2 = std::dynamic_pointer_cast<ngraph::opset1::Convert> (input_2);

        if (convert1 && convert2) {
            ops_to_replace.push_back(convert1);
            ops_to_replace.push_back(convert2);
            input_1 = convert1->input_value(0).get_node_shared_ptr();
            input_2 = convert2->input_value(0).get_node_shared_ptr();
        }

        // the input can be either ShapeOf-1 or ShapeOf-3
        std::shared_ptr<ngraph::op::Op> shape_of1 = std::dynamic_pointer_cast<ngraph::opset1::ShapeOf> (input_1);
        std::shared_ptr<ngraph::op::Op> shape_of2 = std::dynamic_pointer_cast<ngraph::opset1::ShapeOf> (input_2);

        if (!shape_of1 || !shape_of2) {
            shape_of1 = std::dynamic_pointer_cast<ngraph::opset3::ShapeOf>(input_1);
            shape_of2 = std::dynamic_pointer_cast<ngraph::opset3::ShapeOf>(input_2);
        }
        if (!shape_of1 || !shape_of2) {
            return false;
        }
        // keep this code for a while if will decide to run this transformation again in the opset1->legacy
        // the input can be either ShapeOf or Convert(ShapeOf)
//        if (!shape_of1 || !shape_of2) {
//            auto shapeof1_convert = std::dynamic_pointer_cast<ngraph::opset1::Convert> (input_1);
//            auto shapeof2_convert = std::dynamic_pointer_cast<ngraph::opset1::Convert> (input_2);
//            if (!shapeof1_convert || !shapeof2_convert)
//                return false;
//            shape_of1 = std::dynamic_pointer_cast<ngraph::opset1::ShapeOf>(shapeof1_convert->input_value(0).get_node_shared_ptr());
//            shape_of2 = std::dynamic_pointer_cast<ngraph::opset1::ShapeOf>(shapeof2_convert->input_value(0).get_node_shared_ptr());
//            if (!shape_of1 || !shape_of2)
//                return false;
//            ops_to_replace.push_back(shapeof1_convert);
//            ops_to_replace.push_back(shapeof2_convert);
//        }

        ops_to_replace.push_back(shape_of1);
        ops_to_replace.push_back(shape_of2);

        auto prior_box_ie = std::make_shared<ngraph::op::PriorBoxClusteredIE> (shape_of1->input_value(0),
                                                                               shape_of2->input_value(0),
                                                                               prior_box_node->get_attrs());
        prior_box_ie->set_friendly_name(unsqueeze->get_friendly_name());

        // Nodes in copy runtime info function should be in topological order
        std::reverse(ops_to_replace.begin(), ops_to_replace.end());
        ngraph::copy_runtime_info(ops_to_replace, prior_box_ie);
        ngraph::replace_node(unsqueeze, prior_box_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(unsqueeze, "ConvertPriorBoxClusteredToLegacy");
    register_matcher(m, callback);
}
