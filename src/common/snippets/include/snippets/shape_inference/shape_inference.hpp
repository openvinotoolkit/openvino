// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/core.hpp>
#include "snippets/shape_types.hpp"

namespace ov {
namespace snippets {

enum class ShapeInferStatus {
    success, ///< shapes were successfully calculated
    skip     ///< shape inference was skipped.
};
/**
 * This is Snippets specific shape inference interface.
 *
 */
class IShapeInferSnippets {
public:
    enum {DYNAMIC_DIMENSION = std::numeric_limits<size_t>::max()};
    struct Result {
        std::vector<VectorDims> dims;
        ShapeInferStatus status;
    };

    virtual ~IShapeInferSnippets() = default;

    /**
     * @brief This method actually performs all the necessary shape inference computations
     *
     * @param input_shapes are the input tensors shapes
     * @return Result instance that contains an array of calculated shapes (per each output port) and a status of the shape infer call
     */
    virtual Result infer(const std::vector<VectorDimsRef>& input_shapes) = 0;
};

/**
 * Shape inference class for Subgraph node (both openvino and Linear IRs).
 * It stores the result of the last shape inference, so it can be reused in optimization pipeline.
 *
 */
class ShapeInferSnippetsNode : public IShapeInferSnippets {
public:
    const Result& get_last_result() {return m_last_result; }
protected:
    Result m_last_result{{}, ShapeInferStatus::success};
};

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
     * \return Pointer to shape inference object or nullptr if failed to construct the object.
     */
    ShapeInferPtr make(const ov::DiscreteTypeInfo& key, const std::shared_ptr<ov::Node>& op);
    virtual ~IShapeInferSnippetsFactory() = default;

private:
    /** \brief Factory makers registry which can be specialized for key and value. */
    static const TRegistry registry;

protected:
    /**
    * @brief get shape infer instances for operations from backend-specific opset
    * @return Pointer to shape inference object or nullptr if failed to construct the object.
    */
    virtual ShapeInferPtr get_specific_op_shape_infer(const ov::DiscreteTypeInfo& key, const std::shared_ptr<ov::Node>& op) const;
};
std::shared_ptr<IShapeInferSnippets> make_shape_inference(const std::shared_ptr<ov::Node>& op,
                                                          const std::shared_ptr<IShapeInferSnippetsFactory>& factory);
} // namespace snippets
} // namespace ov
