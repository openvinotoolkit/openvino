// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/runtime/host_tensor.hpp>
#include <openvino/core/core.hpp>
#include <openvino/core/node.hpp>

#include "static_shape.hpp"

void shape_inference(ov::Node* op,
                     const std::vector<ov::StaticShape>& input_shapes,
                     std::vector<ov::StaticShape>& output_shapes,
                     const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {});

class IShapeInfer {
public:
    virtual void infer(const std::vector<ov::StaticShape>& input_shapes,
                       std::vector<ov::StaticShape>& output_shapes,
                       const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) = 0;

    // infer may generate padding as by-product, these APIs is designed to retrieve them back
    virtual const ov::CoordinateDiff& get_pads_begin();
    virtual const ov::CoordinateDiff& get_pads_end();

    virtual ngraph::Node * get_op();
};

std::shared_ptr<IShapeInfer> make_shape_inference(const std::shared_ptr<ngraph::Node>& op);
