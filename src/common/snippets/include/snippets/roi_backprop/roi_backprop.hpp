// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>

#include <ngraph/runtime/host_tensor.hpp>
#include <openvino/core/core.hpp>
#include <openvino/core/node.hpp>

namespace ov {
namespace snippets {

class ROIBackprop {
public:
    std::vector<ov::PartialShape> shapes;
    std::vector<ov::Shape> strides;
};

void roi_backprop(ov::Node* op,
                  const std::vector<ov::PartialShape>& input_shapes,
                  const std::vector<ov::PartialShape>& cur_roi,
                  const std::vector<ov::Shape>& cur_roi_strides,
                  std::vector<ov::PartialShape>& new_roi,
                  std::vector<ov::Shape>& new_roi_strides);

using roi_map = std::map<ov::Node*, ROIBackprop>;
roi_map get_roi_from_function(const std::shared_ptr<ov::Model>& m, const std::vector<ov::PartialShape>& start_roi);

class BaseROIBackprop {
public:
    BaseROIBackprop(std::shared_ptr<ov::Node> node) : node(node) {}
    virtual ROIBackprop infer_roi(
            const std::vector<ov::PartialShape>& input_shapes,
            const std::vector<ov::PartialShape>& cur_roi,
            const std::vector<ov::Shape>& cur_strides) = 0;

protected:
    std::shared_ptr<ov::Node> node;
};

std::shared_ptr<BaseROIBackprop> make_roi_backprop(const std::shared_ptr<ngraph::Node>& op);

}   // namespace snippets
}   // namespace ov
