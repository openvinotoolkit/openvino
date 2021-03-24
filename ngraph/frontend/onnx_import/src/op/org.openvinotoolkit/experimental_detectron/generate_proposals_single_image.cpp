// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector experimental_detectron_generate_proposals(const Node& node)
                {
                    using GenerateProposalsSingleImage =
                        ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage;

                    const auto inputs = node.get_ng_inputs();
                    NGRAPH_CHECK(inputs.size() == 4,
                                 "ExperimentalDetectronGenerateProposalsSingleImage expects 4 "
                                 "inputs, received: ",
                                 inputs.size());

                    auto im_info = inputs[0];
                    auto anchors = inputs[1];
                    auto deltas = inputs[2];
                    auto scores = inputs[3];

                    GenerateProposalsSingleImage::Attributes attrs{};
                    attrs.min_size = node.get_attribute_value<float>("min_size", 0.0);
                    attrs.nms_threshold = node.get_attribute_value<float>("nms_threshold", 0.7);
                    attrs.post_nms_count =
                        node.get_attribute_value<std::int64_t>("post_nms_count", 1000);
                    attrs.pre_nms_count =
                        node.get_attribute_value<std::int64_t>("pre_nms_count", 1000);
                    auto generate_proposals_single_image =
                        std::make_shared<GenerateProposalsSingleImage>(
                            im_info, anchors, deltas, scores, attrs);
                    return {generate_proposals_single_image->output(0),
                            generate_proposals_single_image->output(1)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
