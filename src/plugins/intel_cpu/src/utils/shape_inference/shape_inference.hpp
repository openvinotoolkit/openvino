// Copyright (C) 2018-2023 Intel Corporation
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
                     const std::map<size_t, HostTensorPtr>& constant_data = {});

class IShapeInferCommon {
public:
    virtual std::vector<StaticShape> infer(const std::vector<StaticShape>& input_shapes,
                                           const std::map<size_t, HostTensorPtr>& constant_data) = 0;

    // infer may generate padding as by-product, these APIs is designed to retrieve them back
    virtual const ov::CoordinateDiff& get_pads_begin() = 0;
    virtual const ov::CoordinateDiff& get_pads_end() = 0;

    virtual const std::vector<int64_t>& get_input_ranks() = 0;
};

class IStaticShapeInfer : public IShapeInferCommon {
public:
    virtual std::vector<StaticShape> infer(
        const std::vector<StaticShape>& input_shapes,
        const std::map<size_t, std::reference_wrapper<const Tensor>>& constant_data) = 0;
};

template <class TShapeInferIface = IShapeInferCommon>
std::shared_ptr<TShapeInferIface> make_shape_inference(std::shared_ptr<ov::Node> op);

template <>
std::shared_ptr<IShapeInferCommon> make_shape_inference<IShapeInferCommon>(std::shared_ptr<ov::Node> op);

template <>
std::shared_ptr<IStaticShapeInfer> make_shape_inference<IStaticShapeInfer>(std::shared_ptr<ov::Node> op);

}  // namespace intel_cpu
}  // namespace ov
