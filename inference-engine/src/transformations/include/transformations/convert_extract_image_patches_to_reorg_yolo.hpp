// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

        class INFERENCE_ENGINE_API_CLASS(ConvertExtractImagePatchesToReorgYolo);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertExtractImagePatchesToReorgYolo : public ngraph::pass::GraphRewrite {
public:
    ConvertExtractImagePatchesToReorgYolo() : GraphRewrite() {
        convert_extract_image_patches_to_reorg_yolo();
    }

private:
    void convert_extract_image_patches_to_reorg_yolo();
};
