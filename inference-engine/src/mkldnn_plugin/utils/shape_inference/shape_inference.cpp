// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_inference.hpp"

#include <ngraph/runtime/host_tensor.hpp>
#include <openvino/core/node.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>

#include "assign_shape_inference.hpp"
#include "convolution_shape_inference.hpp"
#include "experimental_detectron_detection_output_shape_inference.hpp"
#include "experimental_detectron_prior_grid_generator_shape_inference.hpp"
#include "lstm_cell_shape_inference.hpp"
#include "read_value_shape_inference.hpp"
#include "reduce_shape_inference.hpp"
#include "shape_nodes.hpp"
#include "static_shape.hpp"
#include "tile_shape_inference.hpp"

void shape_inference(ov::Node* op,
                     const std::vector<ov::StaticShape>& input_shapes,
                     std::vector<ov::StaticShape>& output_shapes,
                     const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    if (auto node = ov::as_type<ov::opset8::Convolution>(op)) {
        ov::CoordinateDiff pads_begin, pads_end;
        bool status = resolve_auto_pad_for_shape(node, pads_begin, pads_end, input_shapes, 2, 2);
        OPENVINO_ASSERT(status,
                        "Convolution shape inference doesn't have enough information to calculate static shapes");
        shape_infer(node, pads_begin, pads_end, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::op::util::ArithmeticReductionKeepDims>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::op::util::LogicalReductionKeepDims>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::Reshape>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::Squeeze>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::Unsqueeze>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::ShapeOf>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset3::ShapeOf>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::ExperimentalDetectronDetectionOutput>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset3::Assign>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::Assign>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::ExperimentalDetectronPriorGridGenerator>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::LSTMCell>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::LSTMCell>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset3::ReadValue>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::ReadValue>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::Tile>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else {
        ngraph::OutputVector new_inputs;
        for (size_t i = 0; i < op->get_input_size(); ++i) {
            if (constant_data.count(i)) {
                new_inputs.push_back(std::make_shared<ov::opset1::Constant>(constant_data.at(i)));
            } else {
                new_inputs.push_back(std::make_shared<ov::opset1::Parameter>(op->get_input_element_type(i),
                                                                             input_shapes[i].to_partial_shape()));
            }
        }
        const auto local_op = op->clone_with_new_inputs(new_inputs);
        local_op->validate_and_infer_types();

        output_shapes.resize(op->get_output_size());
        for (size_t i = 0; i < output_shapes.size(); ++i) {
            const auto& partial_shape = local_op->get_output_partial_shape(i);
            OPENVINO_ASSERT(
                partial_shape.is_static(),
                "On device shape infer shouldn't support default shape infer for nodes with internal dynamism");
            output_shapes[i] = ov::StaticShape(partial_shape.to_shape());
        }
    }
}