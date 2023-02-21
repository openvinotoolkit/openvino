// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/runtime/host_tensor.hpp>
#include <openvino/core/core.hpp>
#include <openvino/core/node.hpp>

#include "static_shape.hpp"
#include "tensor_data_accessor.hpp"

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
    using port_mask_t = uint32_t;  //!< Operator's port mask to indicate input data dependency

    virtual std::vector<StaticShape> infer(const std::vector<StaticShape>& input_shapes,
                                           const ITensorAccessor& tensor_accessor) = 0;

    /**
     * @brief Some shape inference implementation may require input data stored inside the input tensors. To define
     * which inputs data are required, the port mask is used. Each set bit corresponds to the specific input port
     * number.
     *
     * @return port_mask_t a bit mask where each bit corresponds to an input port number.
     */
    virtual port_mask_t get_port_mask() const = 0;
};

template <class TShapeInferIface = IShapeInferCommon>
std::shared_ptr<TShapeInferIface> make_shape_inference(std::shared_ptr<ov::Node> op);

template <>
std::shared_ptr<IShapeInferCommon> make_shape_inference<IShapeInferCommon>(std::shared_ptr<ov::Node> op);

template <>
std::shared_ptr<IStaticShapeInfer> make_shape_inference<IStaticShapeInfer>(std::shared_ptr<ov::Node> op);

}  // namespace intel_cpu
}  // namespace ov
