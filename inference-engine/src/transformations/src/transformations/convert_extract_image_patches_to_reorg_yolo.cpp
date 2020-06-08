// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_extract_image_patches_to_reorg_yolo.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>

void ngraph::pass::ConvertExtractImagePatchesToReorgYolo::convert_extract_image_patches_to_reorg_yolo() {
    auto image = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto extract_image_patches = std::make_shared<ngraph::opset3::ExtractImagePatches>(image, Shape{1, 1}, Strides{1, 1}, Shape{1, 1}, ngraph::op::PadType);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto extract_image_patches = std::dynamic_pointer_cast<ngraph::opset3::ExtractImagePatches>(m.get_match_root());

        if (!extract_image_patches) {
            return false;
        }

        if (extract_image_patches->get_strides() != extract_image_patches->get_sizes()) {
            return false;
        }

        auto input_shape = extract_image_patches->get_input_shape(0);
        auto strides = extract_image_patches->get_strides();

        if (input_shape[2] % strides[1] != 0 || input_shape[3] % strides[2] != 0) {
            return false;
        }

        if (extract_image_patches->get_auto_pad() != ngraph::op::PadType::VALID) {
            return false;
        }

        auto reorg_yolo = std::make_shared<ngraph::opset3::ReorgYolo>(extract_image_patches->input(0).get_source_output(),
                                                                      Strides{extract_image_patches->get_strides()});

        reorg_yolo->set_friendly_name(extract_image_patches->get_friendly_name());
        ngraph::copy_runtime_info(extract_image_patches, reorg_yolo);
        ngraph::replace_node(extract_image_patches, reorg_yolo);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(extract_image_patches, "ConvertExtractImagePatchesToReorgYolo");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
