// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_extract_image_patches_to_reorg_yolo.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::pass::ConvertExtractImagePatchesToReorgYolo::ConvertExtractImagePatchesToReorgYolo() {
    auto image = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 1});
    auto eip = std::make_shared<ngraph::opset3::ExtractImagePatches>(image, ngraph::Shape{1, 1}, ngraph::Strides{1, 1}, ngraph::Shape{1, 1},
            ngraph::op::PadType::VALID);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto extract_image_patches =  std::dynamic_pointer_cast<ngraph::opset3::ExtractImagePatches>(m.get_match_root());

        if (!extract_image_patches || m_transformation_callback(extract_image_patches) {
            return false;
        }

        if (extract_image_patches->get_auto_pad() != ngraph::op::PadType::VALID) {
            return false;
        }

        if (extract_image_patches->get_strides() != extract_image_patches->get_sizes()) {
            return false;
        }

        auto p_shape_input = extract_image_patches->get_input_partial_shape(0);
        auto sizes = extract_image_patches->get_sizes();
        auto strides = extract_image_patches->get_strides();
        auto rates = extract_image_patches->get_rates();

        // Check that ExtractImagePatches input have static shape
        if (!p_shape_input.rank().is_static()) {
            return false;
        }

        if (p_shape_input.rank().get_length() != 4) {
            return false;
        }

        if (p_shape_input[2].is_dynamic() || p_shape_input[3].is_dynamic()) {
            return false;
        }

        if (p_shape_input[2].get_length() % strides[0] != 0 || p_shape_input[3].get_length() % strides[1] != 0) {
            return false;
        }

        if (sizes[0] != strides[0] || sizes[1] != strides[1]) {
            return false;
        }

        if (rates[0] != 1 || rates[1] != 1) {
            return false;
        }

        auto reorg_yolo = std::make_shared<ngraph::opset3::ReorgYolo>(extract_image_patches->input(0).get_source_output(),
            ngraph::Strides{extract_image_patches->get_strides()});

        reorg_yolo->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(pattern_to_output.at(eip).get_node_shared_ptr(), reorg_yolo);
        ngraph::replace_node(m.get_match_root(), reorg_yolo);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(eip, "ConvertExtractImagePatchesToReorgYolo");
    register_matcher(m, callback);
}
