// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/roi_backprop/roi_backprop.hpp"

#include <openvino/core/node.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>

//#include "snippets/roi_backprop/gather_roi_backprop.hpp"
#include "snippets/roi_backprop/convolution.hpp"
//#include "snippets/roi_backprop/max_pool.hpp"
#include "snippets/roi_backprop/utils.hpp"

namespace ov {
namespace snippets {

void roi_backprop(ov::Node* op,
                  const std::vector<ov::PartialShape>& input_shapes,
                  const std::vector<ov::PartialShape>& cur_roi,
                  const std::vector<ov::Shape>& cur_strides,
                  std::vector<ov::PartialShape>& new_roi,
                  std::vector<ov::Shape>& new_strides) {
    auto infer_roi = make_roi_backprop(op->shared_from_this());
    auto roi = infer_roi->infer_roi(input_shapes, cur_roi, cur_strides);
    new_roi = roi.shapes;
    new_strides = roi.strides;
}

namespace {
template <class ShapeType>
ShapeType intersect_shapes(const ShapeType& lhs, const ShapeType& rhs) {
    auto max_dimension = [](const ov::Dimension& lhs, const ov::Dimension& rhs) {
        if (lhs.is_static() && rhs.is_static()) {
            return ov::Dimension(std::max(lhs.get_length(), rhs.get_length()));
        } else if (lhs.is_dynamic() && rhs.is_dynamic()) {
            auto min_length = std::min(lhs.get_min_length(), rhs.get_min_length());
            auto max_length = std::min(lhs.get_max_length(), rhs.get_max_length());
            return ov::Dimension(min_length, max_length);
        } else {
            return lhs.is_dynamic() ? lhs : rhs;
        }
    };

    const auto intersected_rank = max_dimension(lhs.rank(), rhs.rank());
    if (intersected_rank.is_dynamic())
        return ShapeType::dynamic();

    ShapeType intersected_shape;
    const auto common_size = std::min(lhs.size(), rhs.size());
    for (size_t i = 0; i < common_size; ++i) {
        intersected_shape.push_back(max_dimension(lhs[i], rhs[i]));
    }

    if (lhs.size() > common_size) {
        for (size_t i = 0; i < lhs.size() - common_size; ++i) {
            intersected_shape.push_back(lhs[i]);
        }
    }

    if (rhs.size() > common_size) {
        for (size_t i = 0; i < rhs.size() - common_size; ++i) {
            intersected_shape.push_back(rhs[i]);
        }
    }

    return intersected_shape;
}
}  // namespace

roi_map get_roi_from_function(const std::shared_ptr<ov::Model>& m, const std::vector<ov::PartialShape>& start_roi) {
    size_t result_idx = 0;
    roi_map result;

    auto get_roi_after = [&](const std::shared_ptr<ov::Node>& n) -> std::vector<ov::PartialShape> {
        if (ov::as_type_ptr<ov::opset1::Result>(n)) {
            return {start_roi[result_idx++]};
        }

        std::vector<ov::PartialShape> roi_after;
        for (const auto& output : n->outputs()) {
            ov::PartialShape out_roi;
            for (const auto& target_input : output.get_target_inputs()) {
                const auto node = target_input.get_node();
                size_t idx = 0;

                for (size_t i = 0; i < node->get_input_size(); ++i) {
                    if (node->get_input_node_ptr(i) == n.get()) {
                        idx = i;
                    }
                }

                // TODO: backprop: should be fixed: Node has two inputs but one output
                auto& roi = result[node];
                if (roi.shapes.size() >= (idx + 1ul)) {
                    const auto& roi_value = result[node].shapes[idx];
                    out_roi = intersect_shapes(out_roi, roi_value);
                }
            }
            roi_after.push_back(out_roi);
        }
        return roi_after;
    };

    auto get_input_shapes = [&](const std::shared_ptr<ov::Node>& n) -> std::vector<ov::PartialShape> {
        std::vector<ov::PartialShape> input_shapes;
        for (const auto& input : n->input_values())
            input_shapes.push_back(input.get_partial_shape());
        return input_shapes;
    };

    auto prepare_roi_before = [&](const std::shared_ptr<ov::Node>& n) -> std::vector<ov::PartialShape> {
        return std::vector<ov::PartialShape>(n->get_input_size());
    };

    auto get_strides_before = [&](const std::shared_ptr<Node>& node) -> std::vector<ov::Shape> {
        if (is_type<opset1::Result>(node)) {
            return {};
        }
        auto child = node->get_output_target_inputs(0).begin()->get_node();
        auto it = result.find(child);

        std::vector<ov::Shape> cur_strides;
        if (it != result.end()) {
            cur_strides = it->second.strides;
        }

        if (cur_strides.empty()) {
            std::vector<ov::PartialShape> input_shapes;
            for (const auto& input : child->input_values()) {
                input_shapes.push_back(input.get_partial_shape());
            }

            cur_strides = { ov::Shape(std::vector<size_t>(input_shapes[0].size(), 1ul)) };
        }
        return cur_strides;
    };

    const auto& nodes = m->get_ordered_ops();
    for (auto iter = nodes.rbegin(); iter != nodes.rend(); ++iter) {
        if (!ov::as_type_ptr<ov::opset1::Parameter>(*iter)) {
            const auto cur_roi = get_roi_after(*iter);
            const auto in_shapes = get_input_shapes(*iter);

            // TODO: backprop: workaround
            auto node = *iter;
            std::vector<ov::Shape> cur_strides = get_strides_before(node);

            auto new_roi = prepare_roi_before(*iter);

            std::vector<ov::Shape> new_strides;

            const auto node_ptr = (*iter).get();
            roi_backprop(node_ptr, in_shapes, cur_roi, cur_strides, new_roi, new_strides);
#ifdef CPU_DEBUG_CAPS
            std::cout <<
                      "get_roi_from_function = " << node_ptr->get_type_name() << ":" << node_ptr->get_friendly_name() <<
                      ", in_shapes = " << (in_shapes.empty() ? PartialShape{} : in_shapes[0]) <<
                      ", cur_shapes = " << (cur_roi.empty() ? PartialShape{} : cur_roi[0]) << " (" << cur_roi.size() << ")" <<
                      ", cur_strides = " << (cur_strides.empty() ? Shape{} : cur_strides[0]) << " (" << cur_strides.size() << ")" <<
                      ", new_shapes = " << (new_roi.empty() ? PartialShape{} : new_roi[0]) << " (" << new_roi.size() << ")" <<
                      ", new_strides = " << (new_strides.empty() ? Shape{} : new_strides[0]) << " (" << new_strides.size() << ")" <<
                      std::endl;
#endif
            if (result.count(node_ptr))
                //OPENVINO_UNREACHABLE("node already exist in roi_map");
                throw Exception("node already exist in roi_map");
            result[node_ptr] = ROIBackprop{new_roi, new_strides};
        } else {
            result[(*iter).get()] = ROIBackprop{get_roi_after((*iter)), get_strides_before(*iter)};
        }
    }

    return result;
}

class TransparentROIBackprop : public BaseROIBackprop {
public:
    TransparentROIBackprop(std::shared_ptr<ov::Node> node) : BaseROIBackprop(node) {}

    ROIBackprop infer_roi(
            const std::vector<ov::PartialShape>& input_shapes,
            const std::vector<ov::PartialShape>& cur_roi,
            const std::vector<ov::Shape>& cur_strides) override {
        auto op = node.get();
        std::vector<ov::PartialShape> roi_shapes(op->get_input_size());
        std::vector<ov::Shape> roi_strides;
        transparent_roi_backprop(op, input_shapes, cur_roi, cur_strides, roi_shapes, roi_strides);
        return ROIBackprop{roi_shapes, roi_strides};
    }
};

template <typename OP>
class GatherROIBackprop : public BaseROIBackprop {
public:
    GatherROIBackprop(std::shared_ptr<OP> node) : BaseROIBackprop(node) {}

    ROIBackprop infer_roi(
            const std::vector<ov::PartialShape>& input_shapes,
            const std::vector<ov::PartialShape>& cur_roi,
            const std::vector<ov::Shape>& cur_strides) override {
        auto op = static_cast<OP*>(node.get());
        std::vector<ov::PartialShape> roi_shapes = cur_roi;
        std::vector<ov::Shape> roi_strides = cur_strides;
        //TODO: uncomment
        //roi_backprop(op, input_shapes, roi_shapes, roi_strides);
        return ROIBackprop{roi_shapes, roi_strides};
    }
};

std::shared_ptr<BaseROIBackprop> make_roi_backprop(const std::shared_ptr<ngraph::Node>& op) {
    if (auto gather = ov::as_type_ptr<ov::opset8::Gather>(op)) {
        return std::make_shared<GatherROIBackprop<ov::opset8::Gather>>(gather);
    } else if (auto max_pool = ov::as_type_ptr<ov::opset1::MaxPool>(op)) {
        return std::make_shared<GatherROIBackprop<ov::opset1::MaxPool>>(max_pool);
    } else if (auto convolution = ov::as_type_ptr<ov::opset1::Convolution>(op)) {
        return std::make_shared<GatherROIBackprop<ov::opset1::Convolution>>(convolution);
    } else {
        return std::make_shared<TransparentROIBackprop>(op);
    }
}

}   // namespace snippets
}   // namespace ov
