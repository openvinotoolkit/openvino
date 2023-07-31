// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/core.hpp>

namespace ov {
namespace snippets {

enum class ShapeInferStatus {
    success, ///< shapes were successfully calculated
    skip ///< shape inference was skipped.
    ///< This status is used when the implementation was expectedly not able to compute defined output shape
    ///< e.g. in the case of internal dynamism.
};
/**
 * This is Snippets specific shape inference interface.
 *
 */
class IShapeInferSnippets {
public:
    enum {DYNAMIC_DIMENSION = 0xffffffffffffffff};
    using VectorDims = std::vector<size_t>;
    struct Result {
        std::vector<VectorDims> dims;
        ShapeInferStatus status;
    };

    virtual ~IShapeInferSnippets() = default;

    /**
     * @brief This method actually performs all the necessary shape inference computations
     *
     * @param input_shapes are the input tensors shapes
     * @param data_dependency are the input tensors data, which are required by the shape inference algorithm. To define
     * which inputs data are actually required, get_port_mask() is used
     * @return ShapeInferResult which contains resulting array of calculated shapes (per each output port) plus status of the shape infer call
     */
    virtual Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes) = 0;
};

/**
 * \brief Shape infer factory
 *
 * \tparam R     Result type of created interface object.
 * \tparam TKey  Type of Maker map key.
 * \tparam Args  TypesInference object ctor args.
 */
class IShapeInferSnippetsFactory {
public:
    // Helper type to define specific Makers map values.
    using ShapeInferPtr = std::shared_ptr<IShapeInferSnippets>;
    // Helper type to define specific Makers map type.
    using TRegistry = std::unordered_map<ov::DiscreteTypeInfo, std::function<ShapeInferPtr (std::shared_ptr<ov::Node>)>>;

    /**
     * \brief  Creates the shape inference object.
     *
     * \param key   Key value to get specified shape inference object maker.
     * \param args  Inference object args.
     *
     * \return The shape inference object or R{} if not found in the map.
     */
    ShapeInferPtr make(const ov::DiscreteTypeInfo& key, const std::shared_ptr<ov::Node>& op);
    virtual ~IShapeInferSnippetsFactory() = default;

private:
    /** \brief Factory makers registry which can be specialized for key and value. */
    static const TRegistry registry;

protected:
    /**
    * @brief get shape infer instances for operations from backend-specific opset
    * @return register ShapeInferPtr
    */
    virtual ShapeInferPtr get_specific_op_shape_infer(const ov::DiscreteTypeInfo& key, const std::shared_ptr<ov::Node>& op) const;
};
std::shared_ptr<IShapeInferSnippets> make_shape_inference(const std::shared_ptr<ov::Node>& op,
                                                          const std::shared_ptr<IShapeInferSnippetsFactory>& factory);
} // namespace snippets
} // namespace ov
