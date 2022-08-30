// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/runtime/host_tensor.hpp>
#include <openvino/core/core.hpp>
#include <openvino/core/node.hpp>

#include "static_shape.hpp"

namespace ov {
namespace intel_cpu {

void shape_inference(ov::Node* op,
                     const std::vector<StaticShape>& input_shapes,
                     std::vector<StaticShape>& output_shapes,
                     const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {});

class IShapeInfer {
public:
    virtual std::vector<StaticShape> infer(
        const std::vector<StaticShape>& input_shapes,
        const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) = 0;

    // infer may generate padding as by-product, these APIs is designed to retrieve them back
    virtual const ov::CoordinateDiff& get_pads_begin() = 0;
    virtual const ov::CoordinateDiff& get_pads_end() = 0;

    virtual const std::vector<int64_t>& get_input_ranks() = 0;
};

std::shared_ptr<IShapeInfer> make_shape_inference(const std::shared_ptr<ngraph::Node>& op);

}   // namespace intel_cpu
}   // namespace ov
