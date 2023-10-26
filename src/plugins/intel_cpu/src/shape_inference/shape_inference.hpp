// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/runtime/host_tensor.hpp>
#include <openvino/core/core.hpp>
#include <openvino/core/node.hpp>

#include "ov_optional.hpp"
#include "shape_inference_status.hpp"
#include "static_shape.hpp"
#include "tensor_data_accessor.hpp"

namespace ov {
namespace intel_cpu {

class IStaticShapeInfer {
public:
    using port_mask_t = uint32_t;  //!< Operator's port mask to indicate input data dependency

    /**
     * @brief Do shape inference.
     *
     * @param input_shapes     Input shapes vector of static shape reference adapter.
     * @param tensor_accessor  Accessor to CPU constant data specific for operator.
     * @return Optionally return vector of static shape adapters holding CPU dimensions.
     */
    virtual ov::optional<std::vector<StaticShape>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                         const ov::ITensorAccessor& tensor_accessor) = 0;

    /**
     * @brief Some shape inference implementation may require input data stored inside the input tensors. To define
     * which inputs data are required, the port mask is used. Each set bit corresponds to the specific input port
     * number.
     *
     * @return port_mask_t a bit mask where each bit corresponds to an input port number.
     */
    virtual port_mask_t get_port_mask() const = 0;

    // infer may generate padding as by-product, these APIs is designed to retrieve them back
    virtual const ov::CoordinateDiff& get_pads_begin() = 0;
    virtual const ov::CoordinateDiff& get_pads_end() = 0;

    virtual const std::vector<int64_t>& get_input_ranks() = 0;
};

std::shared_ptr<IStaticShapeInfer> make_shape_inference(std::shared_ptr<ov::Node> op);
}  // namespace intel_cpu
}  // namespace ov
