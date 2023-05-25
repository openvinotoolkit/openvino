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
#include <ie_common.h>

namespace ov {
namespace intel_cpu {

class IShapeInferCommon {
public:
    struct Result {
        std::vector<StaticShape> shapes;
        ShapeInferStatus status;
    };

public:
    virtual Result infer(const std::vector<StaticShape>& input_shapes,
                         const std::map<size_t, HostTensorPtr>& constant_data) = 0;

    // infer may generate padding as by-product, these APIs is designed to retrieve them back
    virtual const ov::CoordinateDiff& get_pads_begin() = 0;
    virtual const ov::CoordinateDiff& get_pads_end() = 0;

    virtual const std::vector<int64_t>& get_input_ranks() = 0;
};

class IStaticShapeInfer : public IShapeInferCommon {
public:
    using port_mask_t = uint32_t;  //!< Operator's port mask to indicate input data dependency
    using IShapeInferCommon::infer;

    virtual Result infer(const std::vector<StaticShape>& input_shapes, const ov::ITensorAccessor& tensor_accessor) = 0;
    Result infer(const std::vector<StaticShape>& input_shapes,
                     const std::map<size_t, HostTensorPtr>& constant_data) override {
        IE_THROW(Unexpected)  << "should not come here, this function should not be call or should be override";
    }

    /**
     * @brief Do shape inference.
     *
     * @param input_shapes     Input shapes vector of static shape reference adapter.
     * @param tensor_accessor  Accessor to CPU constant data specific for operator.
     * @return Optionally return vector of static shape adapters holding CPU dimensions.
     */
    virtual ov::optional<std::vector<StaticShapeCon>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                            const ov::ITensorAccessor& tensor_accessor) = 0;

    /**
     * @brief Some shape inference implementation may require input data stored inside the input tensors. To define
     * which inputs data are required, the port mask is used. Each set bit corresponds to the specific input port
     * number.
     *
     * @return port_mask_t a bit mask where each bit corresponds to an input port number.
     */
    virtual port_mask_t get_port_mask() const = 0;

    /**
     * @brief For backward compatibility, IShapeInferCommon didn't have fucntion get_port_mask()
     *
     * @param port_mask_t  a bit mask where each bit corresponds to an input port number.
     */
    virtual void set_port_mask(port_mask_t) {}


    /**
     * @brief this function is only used temporarily, after all implementations use accessor, will remove it.
     *
     * @returns if implement ITensorAccess return true, otherwise return false.
     */
    virtual bool is_implemented_accessor(void) {
        return true;
    }
};

template <class TShapeInferIface = IStaticShapeInfer>
std::shared_ptr<TShapeInferIface> make_shape_inference(std::shared_ptr<ov::Node> op);

template <>
std::shared_ptr<IShapeInferCommon> make_shape_inference<IShapeInferCommon>(std::shared_ptr<ov::Node> op);

template <>
std::shared_ptr<IStaticShapeInfer> make_shape_inference<IStaticShapeInfer>(std::shared_ptr<ov::Node> op);

}  // namespace intel_cpu
}  // namespace ov
