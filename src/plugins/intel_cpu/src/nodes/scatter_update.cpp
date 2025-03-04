// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_update.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "common/cpu_memcpy.h"
#include "dnnl_extension_utils.h"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "selective_build.h"

using namespace dnnl;

namespace ov::intel_cpu::node {

bool ScatterUpdate::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                         std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                    ov::op::v3::ScatterElementsUpdate::get_type_info_static(),
                    ov::op::v12::ScatterElementsUpdate::get_type_info_static(),
                    ov::op::v3::ScatterUpdate::get_type_info_static(),
                    ov::op::v3::ScatterNDUpdate::get_type_info_static(),
                    ov::op::v15::ScatterNDUpdate::get_type_info_static())) {
            const std::string opType = op->get_type_name();
            errorMessage = std::string("Type ") + opType + " is not supported.";
            return false;
        }
        if (const auto node_element = ov::as_type_ptr<const ov::op::v12::ScatterElementsUpdate>(op)) {
            using Reduction = ov::op::v12::ScatterElementsUpdate::Reduction;
            if (!one_of(node_element->get_reduction(),
                        Reduction::MAX,
                        Reduction::MEAN,
                        Reduction::MIN,
                        Reduction::NONE,
                        Reduction::PROD,
                        Reduction::SUM)) {
                errorMessage = "ScatterElementsUpdate CPU does not support reduction mode: " +
                               ov::as_string(node_element->get_reduction());
                return false;
            }
        } else if (const auto node_element = ov::as_type_ptr<const ov::op::v15::ScatterNDUpdate>(op)) {
            using Reduction = ov::op::v15::ScatterNDUpdate::Reduction;
            if (!one_of(node_element->get_reduction(),
                        Reduction::MAX,
                        Reduction::MIN,
                        Reduction::NONE,
                        Reduction::PROD,
                        Reduction::SUM,
                        Reduction::SUB)) {
                errorMessage = "ScatterNDUpdate CPU does not support reduction mode: " +
                               ov::as_string(node_element->get_reduction());
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

bool ScatterUpdate::neverExecute() const {
    return getSelectedPrimitiveDescriptor()->hasZeroInputDimsAtPort(DATA_ID);
}

bool ScatterUpdate::isExecutable() const {
    return !isInputTensorAtPortEmpty(DATA_ID);
}

ScatterUpdate::ScatterUpdate(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)),
      dataSize(0lu),
      indicesSize(0lu),
      axisSize(0lu),
      dataPrec(ov::element::dynamic),
      indicesPrec(ov::element::dynamic),
      axisPrec(ov::element::dynamic) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    // In ov::PartialShape with rank 0 (scalars) is converted to ov::intel_cpu::Shape with rank 1.
    // Create workaround by extracting information about rank directly from OV node.
    bool is_not_supported_input =
        ov::is_scalar(op->get_input_partial_shape(0)) || ov::is_scalar(op->get_input_partial_shape(1));
    Type scatterUpdateType = getType();
    if (scatterUpdateType == Type::ScatterUpdate) {
        scatterUpdateMode = ScatterUpdateMode::ScatterUpdate;
        axisRelaxed = true;
        is_not_supported_input = is_not_supported_input || ov::is_scalar(op->get_input_partial_shape(2));
    } else if (scatterUpdateType == Type::ScatterElementsUpdate) {
        scatterUpdateMode = ScatterUpdateMode::ScatterElementsUpdate;
        axisRelaxed = true;
        is_not_supported_input = is_not_supported_input || ov::is_scalar(op->get_input_partial_shape(2));
    } else if (scatterUpdateType == Type::ScatterNDUpdate) {
        scatterUpdateMode = ScatterUpdateMode::ScatterNDUpdate;
        axisRelaxed = false;
        isUpdateScalar = ov::is_scalar(op->get_input_partial_shape(2));
    } else {
        THROW_CPU_NODE_ERR("is not supported");
    }
    if (is_not_supported_input) {
        THROW_CPU_NODE_ERR("do not support scalar input");
    }

    reduction_type = ScatterUpdate::Reduction::NONE;
    if (const auto node_element = ov::as_type_ptr<const ov::op::v12::ScatterElementsUpdate>(op)) {
        using OpReduction = ov::op::v12::ScatterElementsUpdate::Reduction;
        switch (node_element->get_reduction()) {
        case OpReduction::SUM:
            reduction_type = ScatterUpdate::Reduction::SUM;
            break;
        case OpReduction::PROD:
            reduction_type = ScatterUpdate::Reduction::PROD;
            break;
        case OpReduction::MEAN:
            reduction_type = ScatterUpdate::Reduction::MEAN;
            break;
        case OpReduction::MAX:
            reduction_type = ScatterUpdate::Reduction::MAX;
            break;
        case OpReduction::MIN:
            reduction_type = ScatterUpdate::Reduction::MIN;
            break;
        case OpReduction::NONE:
            reduction_type = ScatterUpdate::Reduction::NONE;
            break;
        default:
            THROW_CPU_NODE_ERR("ScatterElementsUpdate CPU does not support reduction mode: ",
                               ov::as_string(node_element->get_reduction()));
        }
        use_init_val = node_element->get_use_init_val();
    } else if (const auto node_element = ov::as_type_ptr<const ov::op::v15::ScatterNDUpdate>(op)) {
        using OpReduction = ov::op::v15::ScatterNDUpdate::Reduction;
        switch (node_element->get_reduction()) {
        case OpReduction::SUM:
            reduction_type = ScatterUpdate::Reduction::SUM;
            break;
        case OpReduction::PROD:
            reduction_type = ScatterUpdate::Reduction::PROD;
            break;
        case OpReduction::SUB:
            reduction_type = ScatterUpdate::Reduction::SUB;
            break;
        case OpReduction::MAX:
            reduction_type = ScatterUpdate::Reduction::MAX;
            break;
        case OpReduction::MIN:
            reduction_type = ScatterUpdate::Reduction::MIN;
            break;
        case OpReduction::NONE:
            reduction_type = ScatterUpdate::Reduction::NONE;
            break;
        default:
            THROW_CPU_NODE_ERR("ScatterNDUpdate CPU does not support reduction mode: ",
                               ov::as_string(node_element->get_reduction()));
        }
    }
}

void ScatterUpdate::getSupportedDescriptors() {
    if ((getParentEdges().size() != 3) && (getParentEdges().size() != 4)) {
        THROW_CPU_NODE_ERR("has incorrect number of input edges");
    }
    if (getChildEdges().empty()) {
        THROW_CPU_NODE_ERR("has incorrect number of output edges");
    }
}

void ScatterUpdate::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    const auto& srcDataDim = getInputShapeAtPort(DATA_ID).getDims();
    const auto& indicesDim = getInputShapeAtPort(INDICES_ID).getDims();
    const auto& updateDim = getInputShapeAtPort(UPDATE_ID).getDims();
    const auto& dstDataDim = getOutputShapeAtPort(0).getDims();

    size_t srcRank = srcDataDim.size();
    size_t indicesRank = indicesDim.size();
    size_t updateRank = updateDim.size();
    size_t dstRank = dstDataDim.size();

    // common check
    if (srcRank != dstRank) {
        THROW_CPU_NODE_ERR("should have same rank for input and output tensor");
    } else {
        for (size_t r = 0; r < srcRank; r++) {
            if (!dimsEqualWeak(srcDataDim[r], dstDataDim[r])) {
                THROW_CPU_NODE_ERR("should have same shape for input and output tensor. The input shape is ",
                                   srcDataDim[r],
                                   ", while output shape is ",
                                   dstDataDim[r],
                                   " for ",
                                   r,
                                   "th dimension");
            }
        }
    }
    // specific check
    switch (scatterUpdateMode) {
    case ScatterUpdateMode::ScatterUpdate: {
        if (updateRank != (srcRank + indicesRank - 1)) {
            THROW_CPU_NODE_ERR("do not have matched tensor rank relationship for input, indices and update");
        }
        break;
    }
    case ScatterUpdateMode::ScatterNDUpdate: {
        if (indicesDim[indicesRank - 1] != Shape::UNDEFINED_DIM) {
            size_t k = indicesDim[indicesRank - 1];
            if (k > srcRank) {
                THROW_CPU_NODE_ERR("do not have an correct indices' last dimension value, ",
                                   "which should be smaller than or equal to input tensor rank");
            }

            size_t tupleRank = indicesRank - 1;
            VectorDims expectUpdateShape(tupleRank + srcRank - k, 0);
            if (isUpdateScalar) {
                // Workaround to properly identify rank of scalar
                updateRank = 0;
            }
            int updateAxisIter = 0;
            for (size_t ri = 0; ri < tupleRank; ri++) {
                expectUpdateShape[updateAxisIter] = indicesDim[ri];
                updateAxisIter++;
            }
            for (size_t rd = k; rd < srcRank; rd++) {
                expectUpdateShape[updateAxisIter] = srcDataDim[rd];
                updateAxisIter++;
            }
            if (expectUpdateShape.size() != updateRank) {
                THROW_CPU_NODE_ERR("do not have matched tensor rank relationship for input, indices and update");
            }
            for (size_t ru = 0; ru < updateRank; ru++) {
                if (!dimsEqualWeak(updateDim[ru], expectUpdateShape[ru])) {
                    THROW_CPU_NODE_ERR("do not have matched tensor shape relationship for input, indices and update");
                }
            }
        }
        break;
    }
    case ScatterUpdateMode::ScatterElementsUpdate: {
        if (srcRank != indicesRank || srcRank != updateRank) {
            THROW_CPU_NODE_ERR("do not have the same tensor rank for input, indices and update");
        }
        for (size_t ri = 0; ri < indicesRank; ri++) {
            if (!dimsEqualWeak(indicesDim[ri], updateDim[ri])) {
                THROW_CPU_NODE_ERR("do not have the same tensor shape for indices and update");
            }
        }
        break;
    }
    default: {
        THROW_CPU_NODE_ERR("is not supported");
    }
    }

    indicesPrec = getOriginalInputPrecisionAtPort(INDICES_ID);
    auto indicesType = DnnlExtensionUtils::ElementTypeToDataType(indicesPrec);
    indicesSize = DnnlExtensionUtils::sizeOfDataType(indicesType);
    if (indicesSize >= 8) {
        indicesPrec = ov::element::i64;
        indicesSize = 8;
    } else {
        indicesPrec = ov::element::i32;
        indicesSize = 4;
    }

    if (axisRelaxed) {
        axisPrec = getOriginalInputPrecisionAtPort(AXIS_ID);
        auto axisType = DnnlExtensionUtils::ElementTypeToDataType(axisPrec);
        axisSize = DnnlExtensionUtils::sizeOfDataType(axisType);
        if (axisSize >= 8) {
            axisPrec = ov::element::i64;
            axisSize = 8;
        } else {
            axisPrec = ov::element::i32;
            axisSize = 4;
        }
    }

    dataPrec = getOriginalInputPrecisionAtPort(DATA_ID);
    if (one_of(scatterUpdateMode, ScatterUpdateMode::ScatterElementsUpdate, ScatterUpdateMode::ScatterNDUpdate) &&
        !one_of(dataPrec,
                ov::element::f32,
                ov::element::i32,
                ov::element::bf16,
                ov::element::f16,
                ov::element::u8,
                ov::element::i8)) {
        dataPrec = ov::element::f32;
    }
    dataSize = dataPrec.size();

    bool canBeInplace = !getParentEdgeAt(DATA_ID)->getParent()->isConstant();

    std::vector<PortConfigurator> inPortConfig{{LayoutType::ncsp, dataPrec, false, canBeInplace ? 0 : -1},
                                               {LayoutType::ncsp, indicesPrec},
                                               {LayoutType::ncsp, dataPrec}};
    if (axisRelaxed) {
        inPortConfig.emplace_back(LayoutType::ncsp, axisPrec);
    }
    addSupportedPrimDesc(inPortConfig,
                         {{LayoutType::ncsp, dataPrec, false, canBeInplace ? 0 : -1}},
                         impl_desc_type::unknown);
}

bool ScatterUpdate::needPrepareParams() const {
    return false;
}

void ScatterUpdate::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

int64_t ScatterUpdate::getIndicesValue(uint8_t* indices, size_t offset) {
    auto* indicesPtr = indices + offset * indicesSize;
    int64_t ret = 0;
    if (indicesSize == 4) {
        auto* indicesPtr32 = reinterpret_cast<int32_t*>(indicesPtr);
        ret = *indicesPtr32;
    } else {
        auto* indicesPtr64 = reinterpret_cast<int64_t*>(indicesPtr);
        ret = *indicesPtr64;
    }
    return ret;
}

// 5D example:
// shapeND: n     c     d     h    w
// blockND: ncdhw cdhw  dhw   hw   w    1
// index  : 0      1    2     3    4    5
static std::vector<size_t> getBlockND(const VectorDims& shape) {
    size_t shapeRank = shape.size();
    std::vector<size_t> blockND(shapeRank + 1, 1);
    for (int i = shapeRank - 1; i >= 0; i--) {
        blockND[i] = shape[i] * blockND[i + 1];
    }
    return blockND;
}

namespace scatter_elements_update {
template <typename T>
static T reduction_neutral_value(const ScatterUpdate::Reduction reduction_type) {
    switch (reduction_type) {
    case ScatterUpdate::Reduction::MAX:
        return std::numeric_limits<T>::lowest();
    case ScatterUpdate::Reduction::MIN:
        return std::numeric_limits<T>::max();
    case ScatterUpdate::Reduction::PROD:
        return T{1};
    case ScatterUpdate::Reduction::SUM:
    case ScatterUpdate::Reduction::MEAN:
    case ScatterUpdate::Reduction::NONE:
        return T{0};
    default:
        OPENVINO_THROW("Neutral value not available for this type of reduction");
        return 0;
    }
}

static inline void getCoordinate(VectorDims& coordinate, size_t offset, const VectorDims& shape) {
    size_t shapeRank = shape.size();
    for (int i = shapeRank - 1; i >= 0; i--) {
        coordinate[i] = offset % shape[i];
        offset /= shape[i];
    }
}

struct TensorIterator {
    TensorIterator(VectorDims squashed_shape, const int64_t squashed_axis)
        : m_squashed_shape(std::move(squashed_shape)),
          m_squashed_axis(squashed_axis) {
        OPENVINO_ASSERT(m_squashed_shape[m_squashed_axis] == 1);
    }

    std::array<size_t, 2> startover(const size_t start,
                                    const std::vector<size_t>& dataBlockND,
                                    const std::vector<size_t>& indicesBlockND) {
        m_tensorIter.resize(m_squashed_shape.size(), 0);
        getCoordinate(m_tensorIter, start, m_squashed_shape);

        size_t i, dst_idx = 0, indices_idx = 0;
        for (i = 0; i < static_cast<size_t>(m_squashed_axis); ++i) {
            dst_idx += m_tensorIter[i] * dataBlockND[i + 1];
            indices_idx += m_tensorIter[i] * indicesBlockND[i + 1];
        }
        for (i++; i < m_squashed_shape.size(); ++i) {
            dst_idx += m_tensorIter[i] * dataBlockND[i + 1];
            indices_idx += m_tensorIter[i] * indicesBlockND[i + 1];
        }

        return {dst_idx, indices_idx};
    }

    void increment(std::array<size_t, 2>& offsets,
                   const std::vector<size_t>& dataBlockND,
                   const std::vector<size_t>& indicesBlockND) {
        for (int64_t j = m_squashed_shape.size() - 1; j >= 0; j--) {
            m_tensorIter[j]++;
            if (m_tensorIter[j] < m_squashed_shape[j]) {  // no need check if (j != axis) as it is squashed
                offsets[0] += dataBlockND[j + 1];
                offsets[1] += indicesBlockND[j + 1];
                break;
            }
            m_tensorIter[j] = 0;
            size_t i = 0;
            for (offsets[0] = 0, offsets[1] = 0; i < m_squashed_axis; ++i) {
                offsets[0] += m_tensorIter[i] * dataBlockND[i + 1];
                offsets[1] += m_tensorIter[i] * indicesBlockND[i + 1];
            }
            for (i++; i < m_squashed_shape.size(); ++i) {
                offsets[0] += m_tensorIter[i] * dataBlockND[i + 1];
                offsets[1] += m_tensorIter[i] * indicesBlockND[i + 1];
            }
        }
    }

    VectorDims m_tensorIter;
    const VectorDims m_squashed_shape;
    const size_t m_squashed_axis;
};

struct ScatterElementsUpdateContext {
    ScatterUpdate* node;
    MemoryPtr dstMemPtr;
    MemoryPtr indicesMemPtr;
    MemoryPtr updateMemPtr;
    int axis;
    ScatterUpdate::Reduction reduction_type;
};

// tier 2 dispatcher with Reduce which follows up DataType.
template <typename PT>
struct ScatterElementsUpdateReduceDispatcher {
    void operator()(ScatterElementsUpdateContext& ctx) {
        using kernel_t = typename PT::second_type;
        using data_t = typename PT::first_type;
        ctx.node->scatterElementsUpdate<data_t>(ctx.dstMemPtr,
                                                ctx.indicesMemPtr,
                                                ctx.updateMemPtr,
                                                ctx.axis,
                                                kernel_t{});
    }
};

// tier 1 dispatcher with DataType
template <typename DataType>
struct ScatterElementsUpdateDispatcher {
    void operator()(ScatterElementsUpdateContext& ctx) {
        scatterElementsUpdate_dispatch(ctx);
    }

private:
    void scatterElementsUpdate_dispatch(ScatterElementsUpdateContext& ctx) {
        using namespace scatter_reductions;
        using DT_NONE = std::pair<DataType, ReduceNone>;
        using DT_SUM = std::pair<DataType, ReduceAdd>;
        using DT_MAX = std::pair<DataType, ReduceMaximum>;
        using DT_MIN = std::pair<DataType, ReduceMinimum>;
        using DT_MUL = std::pair<DataType, ReduceMultiply>;
        using DT_MEAN = std::pair<DataType, ReduceMean>;
        OV_SWITCH(intel_cpu,
                  ScatterElementsUpdateReduceDispatcher,
                  ctx,
                  ctx.reduction_type,
                  OV_CASE(ScatterUpdate::Reduction::NONE, DT_NONE),
                  OV_CASE(ScatterUpdate::Reduction::SUM, DT_SUM),
                  OV_CASE(ScatterUpdate::Reduction::MAX, DT_MAX),
                  OV_CASE(ScatterUpdate::Reduction::MIN, DT_MIN),
                  OV_CASE(ScatterUpdate::Reduction::PROD, DT_MUL),
                  OV_CASE(ScatterUpdate::Reduction::MEAN, DT_MEAN));
    }
};
};  // namespace scatter_elements_update

namespace scatter_nd_update {
struct ScatterNDUpdateContext {
    ScatterUpdate* node;
    MemoryPtr dstMemPtr;
    MemoryPtr indicesMemPtr;
    MemoryPtr updateMemPtr;
    ScatterUpdate::Reduction reduction_type;
};

// tier 2 dispatcher with Reduce which follows up DataType.
template <typename PT>
struct ScatterNDUpdateReduceDispatcher {
    void operator()(ScatterNDUpdateContext& ctx) {
        using kernel_t = typename PT::second_type;
        using data_t = typename PT::first_type;
        ctx.node->scatterNDUpdate<data_t>(ctx.dstMemPtr, ctx.indicesMemPtr, ctx.updateMemPtr, kernel_t{});
    }
};

// tier 2 dispatcher with ReduceNone that does not depend on DatType
template <>
struct ScatterNDUpdateReduceDispatcher<scatter_reductions::ReduceNone> {
    void operator()(ScatterNDUpdateContext& ctx) {
        ctx.node->scatterNDUpdate(ctx.dstMemPtr, ctx.indicesMemPtr, ctx.updateMemPtr, scatter_reductions::ReduceNone{});
    }
};

// tier 1 dispatcher with DataType
template <typename DataType>
struct ScatterNDUpdateDispatcher {
    void operator()(ScatterNDUpdateContext& ctx) {
        scatterNDUpdate_dispatch(ctx);
    }

private:
    void scatterNDUpdate_dispatch(ScatterNDUpdateContext& ctx) {
        using namespace scatter_reductions;
        // ReduceNone does not depend on DataType.
        using DT_NONE = ReduceNone;
        using DT_SUM = std::pair<DataType, ReduceAdd>;
        using DT_SUB = std::pair<DataType, ReduceSub>;
        using DT_MAX = std::pair<DataType, ReduceMaximum>;
        using DT_MIN = std::pair<DataType, ReduceMinimum>;
        using DT_MUL = std::pair<DataType, ReduceMultiply>;
        OV_SWITCH(intel_cpu,
                  ScatterNDUpdateReduceDispatcher,
                  ctx,
                  ctx.reduction_type,
                  OV_CASE(ScatterUpdate::Reduction::NONE, DT_NONE),
                  OV_CASE(ScatterUpdate::Reduction::SUM, DT_SUM),
                  OV_CASE(ScatterUpdate::Reduction::SUB, DT_SUB),
                  OV_CASE(ScatterUpdate::Reduction::MAX, DT_MAX),
                  OV_CASE(ScatterUpdate::Reduction::MIN, DT_MIN),
                  OV_CASE(ScatterUpdate::Reduction::PROD, DT_MUL));
    }
};
};  // namespace scatter_nd_update

// output[indices[i][j][k]][j][k] = updates[i][j][k] if axis = 0,
// output[i][indices[i][j][k]][k] = updates[i][j][k] if axis = 1,
// output[i][j][indices[i][j][k]] = updates[i][j][k] if axis = 2.
template <typename DataType, typename KernelType>
void ScatterUpdate::scatterElementsUpdate(const MemoryPtr& mem_data,
                                          const MemoryPtr& mem_indices,
                                          const MemoryPtr& mem_updates,
                                          int axis,
                                          const KernelType& kernel) {
    using namespace scatter_elements_update;
    auto* dataPtr = mem_data->getDataAs<DataType>();
    auto* updatePtr = mem_updates->getDataAs<DataType>();
    auto* indicesPtr = mem_indices->getDataAs<uint8_t>();

    const auto& data_shape = mem_data->getStaticDims();
    const auto& indices_shape = mem_indices->getStaticDims();
    const size_t updates_rank = indices_shape.size();

    if (axis < 0) {
        axis += updates_rank;
    }
    CPU_NODE_ASSERT(axis >= 0 && axis < static_cast<int>(updates_rank), "Invalid axis.");

    const auto data_dim_size = static_cast<int64_t>(data_shape[axis]);
    const auto index_dim_size = indices_shape[axis];

    VectorDims squashed_indices_shape(indices_shape);
    squashed_indices_shape[axis] = 1;

    const std::vector<size_t> dataBlockND = getBlockND(data_shape);
    const std::vector<size_t> indicesBlockND = getBlockND(indices_shape);
    const size_t dataBlock_axisplus1 = dataBlockND[axis + 1];
    const size_t indicesBlock_axisplus1 = indicesBlockND[axis + 1];

    // process serially along 'axis' dimension because of data dependency brought by duplicated value in indices
    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        splitter(shape_size(squashed_indices_shape), nthr, ithr, start, end);
        scatter_elements_update::TensorIterator tensorItr(squashed_indices_shape, axis);

        // When *use_init_val* attribute is false, we need to substitute the copied values at target locations with
        // values that will not affect the particular reduction algorithms.
        if (!use_init_val) {
            const auto value = reduction_neutral_value<DataType>(reduction_type);
            auto offsets = tensorItr.startover(start, dataBlockND, indicesBlockND);
            for (size_t worker = start; worker < end; worker++) {
                auto indices_offset = offsets[1];
                for (size_t idx = 0; idx < index_dim_size; idx++) {
                    int64_t idxValue = getIndicesValue(indicesPtr, indices_offset);
                    if (idxValue < 0) {
                        idxValue += data_dim_size;
                    }
                    assert(idxValue < data_dim_size && idxValue >= 0);
                    dataPtr[offsets[0] + idxValue * dataBlock_axisplus1] = value;
                    indices_offset += indicesBlock_axisplus1;
                }

                // increment
                tensorItr.increment(offsets, dataBlockND, indicesBlockND);
            }
        }

        // Apply the Reduce function in an element-wise fashion. For better performance,
        // when axis is the last dimension, we iterate along axis in the inner loop; otherwise we iterate axis
        // in the outer loop.
        auto offsets = tensorItr.startover(start, dataBlockND, indicesBlockND);
        if (axis == static_cast<int>(updates_rank - 1)) {
            for (size_t worker = start; worker < end; worker++) {
                auto indices_offset = offsets[1];
                for (size_t idx = 0; idx < index_dim_size; idx++) {
                    int64_t idxValue = getIndicesValue(indicesPtr, indices_offset);
                    if (idxValue < 0) {
                        idxValue += data_dim_size;
                    }
                    assert(idxValue < data_dim_size && idxValue >= 0);
                    auto dst = &dataPtr[offsets[0] + idxValue * dataBlock_axisplus1];
                    auto src = &updatePtr[indices_offset];
                    kernel(dst, src);
                    indices_offset += indicesBlock_axisplus1;
                }
                // increment
                tensorItr.increment(offsets, dataBlockND, indicesBlockND);
            }
        } else {
            // For better performance, the offsets of dst and indices are cached in the first iteration of outer loop,
            // and reused in the remaining iterations.
            std::vector<size_t> dst_offsets(
                end - start + 1,
                offsets[0]);  // one extra to avoid overflow at the last iteration of inner loop
            std::vector<size_t> indices_offsets(end - start + 1, offsets[1]);
            size_t* ptr_dst_offset = &dst_offsets[0];
            size_t* ptr_indices_offset = &indices_offsets[0];
            for (size_t worker = start; worker < end; worker++) {  // idx = 0
                int64_t idxValue = getIndicesValue(indicesPtr, *ptr_indices_offset);
                if (idxValue < 0) {
                    idxValue += data_dim_size;
                }
                assert(idxValue < data_dim_size && idxValue >= 0);
                auto dst = &dataPtr[ptr_dst_offset[0] + idxValue * dataBlock_axisplus1];
                auto src = &updatePtr[ptr_indices_offset[0]];
                kernel(dst, src);

                // increment once for all
                tensorItr.increment(offsets, dataBlockND, indicesBlockND);
                *++ptr_dst_offset = offsets[0];
                *++ptr_indices_offset = offsets[1];
            }
            for (size_t idx = 1; idx < index_dim_size; idx++) {
                ptr_indices_offset = &indices_offsets[0];
                ptr_dst_offset = &dst_offsets[0];
                for (size_t worker = start; worker < end; worker++) {
                    auto indices_offset = *ptr_indices_offset + idx * indicesBlock_axisplus1;
                    int64_t idxValue = getIndicesValue(indicesPtr, indices_offset);
                    if (idxValue < 0) {
                        idxValue += data_dim_size;
                    }
                    assert(idxValue < data_dim_size && idxValue >= 0);
                    auto dst = &dataPtr[ptr_dst_offset[0] + idxValue * dataBlock_axisplus1];
                    auto src = &updatePtr[indices_offset];
                    kernel(dst, src);
                    ptr_indices_offset++;
                    ptr_dst_offset++;
                }
            }
        }
    });
}

// We specialize ReduceMean to avoid spoil performance of other reduce methods, as otherwise condition branch
// were used in loops.
template <typename DataType>
void ScatterUpdate::scatterElementsUpdate(const MemoryPtr& mem_data,
                                          const MemoryPtr& mem_indices,
                                          const MemoryPtr& mem_updates,
                                          int axis,
                                          const scatter_reductions::ReduceMean& kernel) {
    using namespace scatter_elements_update;
    CPU_NODE_ASSERT(reduction_type == ScatterUpdate::Reduction::MEAN, "The reduction type should be MEAN here.");
    auto* dataPtr = mem_data->getDataAs<DataType>();
    auto* updatePtr = mem_updates->getDataAs<DataType>();
    auto* indicesPtr = mem_indices->getDataAs<uint8_t>();

    const auto& data_shape = mem_data->getStaticDims();
    const auto& indices_shape = mem_indices->getStaticDims();
    size_t updates_rank = indices_shape.size();

    if (axis < 0) {
        axis += updates_rank;
    }
    CPU_NODE_ASSERT(axis >= 0 && axis < static_cast<int>(updates_rank), "Invalid axis.");

    const auto data_dim_size = static_cast<int64_t>(data_shape[axis]);
    const auto index_dim_size = indices_shape[axis];

    VectorDims squashed_indices_shape(indices_shape);
    squashed_indices_shape[axis] = 1;

    const std::vector<size_t> dataBlockND = getBlockND(data_shape);
    const std::vector<size_t> indicesBlockND = getBlockND(indices_shape);
    const size_t dataBlock_axisplus1 = dataBlockND[axis + 1];
    const size_t indicesBlock_axisplus1 = indicesBlockND[axis + 1];

    // process serially along 'axis' dimension because of data dependency brought by duplicated value in indices
    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        splitter(shape_size(squashed_indices_shape), nthr, ithr, start, end);
        scatter_elements_update::TensorIterator tensorItr(squashed_indices_shape, axis);

        // When *use_init_val* attribute is false, we need to substitute the copied values at target locations with
        // values that will not affect the particular reduction algorithms.
        if (!use_init_val) {
            const auto value = reduction_neutral_value<DataType>(reduction_type);
            auto offsets = tensorItr.startover(start, dataBlockND, indicesBlockND);
            for (size_t worker = start; worker < end; worker++) {
                auto indices_offset = offsets[1];
                for (size_t idx = 0; idx < index_dim_size; idx++) {
                    int64_t idxValue = getIndicesValue(indicesPtr, indices_offset);
                    if (idxValue < 0) {
                        idxValue += data_dim_size;
                    }
                    assert(idxValue < data_dim_size && idxValue >= 0);
                    dataPtr[offsets[0] + idxValue * dataBlock_axisplus1] = value;
                    indices_offset += indicesBlock_axisplus1;
                }

                // increment
                tensorItr.increment(offsets, dataBlockND, indicesBlockND);
            }
        }

        // Apply the Reduce function in an element-wise fashion. For better performance,
        // when axis is the last dimension, we iterate along axis in the inner loop; otherwise we iterate axis
        // in the outer loop.
        auto offsets = tensorItr.startover(start, dataBlockND, indicesBlockND);
        if (axis == static_cast<int>(updates_rank - 1)) {
            for (size_t worker = start; worker < end; worker++) {
                std::unordered_map<size_t, int64_t> mean_reduction_counters;  // (idxValue, num_sums) for current worker

                auto indices_offset = offsets[1];
                for (size_t idx = 0; idx < index_dim_size; idx++) {
                    int64_t idxValue = getIndicesValue(indicesPtr, indices_offset);
                    if (idxValue < 0) {
                        idxValue += data_dim_size;
                    }
                    assert(idxValue < data_dim_size && idxValue >= 0);
                    auto dst = &dataPtr[offsets[0] + idxValue * dataBlock_axisplus1];
                    auto src = &updatePtr[indices_offset];
                    kernel(dst, src);
                    indices_offset += indicesBlock_axisplus1;

                    mean_reduction_counters[idxValue] += 1;
                }

                // average
                for (const auto& counter : mean_reduction_counters) {
                    auto dst = &dataPtr[offsets[0] + counter.first * dataBlock_axisplus1];
                    const auto N = counter.second + static_cast<int32_t>(use_init_val);
                    *dst = static_cast<DataType>(static_cast<double>(*dst) / N);
                }

                // increment
                tensorItr.increment(offsets, dataBlockND, indicesBlockND);
            }
        } else {
            // For better performance, the offsets of dst and indices are cached in the first iteration of outer loop,
            // and reused in the remaining iterations.
            std::unordered_map<DataType*, int64_t> mean_reduction_counters;  // (dst_addr, num_sums) for all workers

            std::vector<size_t> dst_offsets(
                end - start + 1,
                offsets[0]);  // one extra to avoid overflow at the last iteration of inner loop
            std::vector<size_t> indices_offsets(end - start + 1, offsets[1]);
            size_t* ptr_dst_offset = &dst_offsets[0];
            size_t* ptr_indices_offset = &indices_offsets[0];
            for (size_t worker = start; worker < end; worker++) {  // idx = 0
                int64_t idxValue = getIndicesValue(indicesPtr, *ptr_indices_offset);
                if (idxValue < 0) {
                    idxValue += data_dim_size;
                }
                assert(idxValue < data_dim_size && idxValue >= 0);
                auto dst = &dataPtr[ptr_dst_offset[0] + idxValue * dataBlock_axisplus1];
                auto src = &updatePtr[ptr_indices_offset[0]];
                kernel(dst, src);

                mean_reduction_counters[dst] += 1;

                // increment once for all
                tensorItr.increment(offsets, dataBlockND, indicesBlockND);
                *++ptr_dst_offset = offsets[0];
                *++ptr_indices_offset = offsets[1];
            }
            for (size_t idx = 1; idx < index_dim_size; idx++) {
                ptr_indices_offset = &indices_offsets[0];
                ptr_dst_offset = &dst_offsets[0];
                for (size_t worker = start; worker < end; worker++) {
                    auto indices_offset = *ptr_indices_offset + idx * indicesBlock_axisplus1;
                    int64_t idxValue = getIndicesValue(indicesPtr, indices_offset);
                    if (idxValue < 0) {
                        idxValue += data_dim_size;
                    }
                    assert(idxValue < data_dim_size && idxValue >= 0);
                    auto dst = &dataPtr[ptr_dst_offset[0] + idxValue * dataBlock_axisplus1];
                    auto src = &updatePtr[indices_offset];
                    kernel(dst, src);
                    mean_reduction_counters[dst] += 1;
                    ptr_indices_offset++;
                    ptr_dst_offset++;
                }
            }

            // average
            for (const auto& counter : mean_reduction_counters) {
                auto dst = counter.first;
                const auto N = counter.second + static_cast<int32_t>(use_init_val);
                *dst = static_cast<DataType>(static_cast<double>(*dst) / N);
            }
        }
    });
}

void ScatterUpdate::scatterElementsUpdate(const MemoryPtr& dstMemPtr,
                                          const MemoryPtr& indicesMemPtr,
                                          const MemoryPtr& updateMemPtr,
                                          int axis) {
    using namespace scatter_elements_update;
    ScatterElementsUpdateContext ctx{this, dstMemPtr, indicesMemPtr, updateMemPtr, axis, reduction_type};
    OV_SWITCH(intel_cpu,
              ScatterElementsUpdateDispatcher,
              ctx,
              dataPrec,
              OV_CASE(ov::element::f32, float),
              OV_CASE(ov::element::i32, int32_t),
              OV_CASE(ov::element::bf16, ov::bfloat16),
              OV_CASE(ov::element::f16, ov::float16),
              OV_CASE(ov::element::i8, int8_t),
              OV_CASE(ov::element::u8, uint8_t));
}

void ScatterUpdate::execute(const dnnl::stream& strm) {
    auto srcMemPtr = getSrcMemoryAtPort(DATA_ID);
    auto dstMemPtr = getDstMemoryAtPort(0);
    auto indicesMemPtr = getSrcMemoryAtPort(INDICES_ID);
    auto updateMemPtr = getSrcMemoryAtPort(UPDATE_ID);

    auto* dstPtr = dstMemPtr->getDataAs<uint8_t>();
    auto* srcPtr = srcMemPtr->getDataAs<uint8_t>();
    auto* indicesPtr = indicesMemPtr->getDataAs<uint8_t>();
    auto* updatePtr = updateMemPtr->getDataAs<uint8_t>();

    const auto& srcDataDim = getParentEdgeAt(DATA_ID)->getMemory().getStaticDims();
    const auto& indicesDim = getParentEdgeAt(INDICES_ID)->getMemory().getStaticDims();
    size_t srcRank = srcDataDim.size();

    // 1d short vector scatter update optimized for shape inference subgraph
    if (scatterUpdateMode == ScatterUpdateMode::ScatterUpdate && srcDataDim.size() == 1 && indicesDim.size() <= 1 &&
        indicesPrec == ov::element::i32 && dataPrec == ov::element::i32 && srcDataDim[0] <= 64) {
        auto updateDims = updateMemPtr->getStaticDims();
        if (updateDims.size() <= 1) {
            DEBUG_LOG(getName(), " exec1DCase");
            auto updateCnt = (updateDims.size() == 0) ? 1 : updateDims[0];
            auto srcLength = srcMemPtr->getStaticDims()[0];
            auto* psrc = reinterpret_cast<int32_t*>(srcPtr);
            auto* pdst = reinterpret_cast<int32_t*>(dstPtr);
            for (size_t i = 0; i < srcLength; i++) {
                pdst[i] = psrc[i];
            }
            auto* pindices = reinterpret_cast<int32_t*>(indicesPtr);
            auto* pupdate = reinterpret_cast<int32_t*>(updatePtr);
            for (size_t i = 0; i < updateCnt; i++) {
                pdst[pindices[i]] = pupdate[i];
            }
            return;
        }
    }

    int axis = 0;
    if (axisRelaxed) {
        auto axisMemPtr = getSrcMemoryAtPort(AXIS_ID);
        auto* axisPtr = axisMemPtr->getDataAs<uint8_t>();
        if (axisSize == 4) {
            auto* axisPtr32 = reinterpret_cast<int32_t*>(axisPtr);
            axis = *axisPtr32;
        } else {
            auto* axisPtr64 = reinterpret_cast<int64_t*>(axisPtr);
            axis = *axisPtr64;
        }

        if (axis >= static_cast<int>(srcRank) || axis < (static_cast<int>(srcRank) * -1)) {
            THROW_CPU_NODE_ERR("should have axis value in range [-r, r - 1], where r is the rank of input data");
        }
        axis = axis < 0 ? (axis + srcRank) : axis;

        size_t srcDimAxis = srcDataDim[axis];
        std::vector<size_t> indicesBlockND = getBlockND(indicesDim);
        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0;
            splitter(indicesBlockND[0], nthr, ithr, start, end);
            for (size_t i = start; i < end; i++) {
                int64_t idxValue = getIndicesValue(indicesPtr, i);
                if (idxValue >= static_cast<int64_t>(srcDimAxis) ||
                    (idxValue < 0 && scatterUpdateMode != ScatterUpdateMode::ScatterElementsUpdate)) {
                    THROW_CPU_NODE_ERR("have indices value that points to non-existing output tensor element");
                }
            }
        });

        if (scatterUpdateMode == ScatterUpdateMode::ScatterUpdate) {
            VectorDims indicesDim = getParentEdgeAt(INDICES_ID)->getMemory().getStaticDims();
            VectorDims updateDim = getParentEdgeAt(UPDATE_ID)->getMemory().getStaticDims();
            size_t indicesRank = indicesDim.size();
            size_t updateRank = updateDim.size();
            VectorDims expectUpdateShape(srcRank + indicesRank - 1, 0);
            int axisIter = 0;
            for (size_t rs = 0; rs < srcRank; rs++) {
                if (rs != static_cast<size_t>(axis)) {
                    expectUpdateShape[axisIter] = srcDataDim[rs];
                    axisIter++;
                } else {
                    for (size_t ri = 0; ri < indicesRank; ri++) {
                        expectUpdateShape[axisIter] = indicesDim[ri];
                        axisIter++;
                    }
                }
            }
            if (updateRank > expectUpdateShape.size()) {
                THROW_CPU_NODE_ERR("cannot update shape. New rank: ",
                                   updateRank,
                                   ", expected: ",
                                   expectUpdateShape.size());
            }
            for (size_t ru = 0; ru < updateRank; ru++) {
                if (updateDim[ru] != expectUpdateShape[ru]) {
                    THROW_CPU_NODE_ERR("do not have matched tensor shape relationship for input, indices and update");
                }
            }
        }
    }

    if (srcPtr != dstPtr) {
        std::vector<size_t> srcBlockND = getBlockND(srcDataDim);
        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0;
            splitter(srcBlockND[0], nthr, ithr, start, end);
            size_t size = (end - start) * dataSize;
            start *= dataSize;
            cpu_memcpy(dstPtr + start, srcPtr + start, size);
        });
    }

    if (isInputTensorAtPortEmpty(INDICES_ID)) {
        return;
    }

    switch (scatterUpdateMode) {
    case ScatterUpdateMode::ScatterUpdate: {
        scatterUpdate(indicesPtr, updatePtr, axis, dstPtr);
        break;
    }
    case ScatterUpdateMode::ScatterNDUpdate: {
        scatterNDUpdate(dstMemPtr, indicesMemPtr, updateMemPtr);
        break;
    }
    case ScatterUpdateMode::ScatterElementsUpdate: {
        scatterElementsUpdate(dstMemPtr, indicesMemPtr, updateMemPtr, axis);
        break;
    }
    default: {
        THROW_CPU_NODE_ERR("is not supported");
    }
    }
}

// For the data tensor of shape [d_0, d_1, ..., d_n],
// and indices tensor of shape [i_0, i_1, ..., i_k].
// Updates tensor shape should be [d_0, d_1, ... d_(axis - 1), i_0, i_1, ..., i_k, d_(axis + 1), ..., d_n].
void ScatterUpdate::scatterUpdate(uint8_t* indices, uint8_t* update, int axis, uint8_t* dstData) {
    const auto& srcDataDim = getParentEdgeAt(DATA_ID)->getMemory().getStaticDims();
    const auto& indicesDim = getParentEdgeAt(INDICES_ID)->getMemory().getStaticDims();
    const auto& updateDim = getParentEdgeAt(UPDATE_ID)->getMemory().getStaticDims();
    size_t indicesRank = indicesDim.size();

    std::vector<size_t> srcBlockND = getBlockND(srcDataDim);
    std::vector<size_t> updateBlockND = getBlockND(updateDim);

    const size_t mulIdentity = 1;
    size_t idxLength = mulIdentity;
    for (size_t ri = 0; ri < indicesRank; ri++) {
        idxLength *= indicesDim[ri];
    }
    size_t batchToUpdate = mulIdentity;
    for (int x = 0; x < axis; x++) {
        batchToUpdate *= srcDataDim[x];
    }
    // blockToUpdate is srcBlockND[axis + 1], also is updateBlockND[axis + indicesRank]
    size_t blockToUpdate = srcBlockND[axis + 1];
    size_t blockToUpdateSize = blockToUpdate * dataSize;

    parallel_for2d(batchToUpdate, idxLength, [&](size_t b, size_t idx) {
        int64_t idxValue = getIndicesValue(indices, idx);
        uint8_t* dstEntry = dstData + (b * srcBlockND[axis] + idxValue * blockToUpdate) * dataSize;
        uint8_t* updateEntry = update + (b * updateBlockND[axis] + idx * blockToUpdate) * dataSize;
        cpu_memcpy(dstEntry, updateEntry, blockToUpdateSize);
    });
}

void ScatterUpdate::scatterNDUpdate(const MemoryPtr& dstMemPtr,
                                    const MemoryPtr& indicesMemPtr,
                                    const MemoryPtr& updateMemPtr) {
    using namespace scatter_nd_update;
    ScatterNDUpdateContext ctx{this, dstMemPtr, indicesMemPtr, updateMemPtr, reduction_type};
    OV_SWITCH(intel_cpu,
              ScatterNDUpdateDispatcher,
              ctx,
              dataPrec,
              OV_CASE(ov::element::f32, float),
              OV_CASE(ov::element::i32, int32_t),
              OV_CASE(ov::element::bf16, ov::bfloat16),
              OV_CASE(ov::element::f16, ov::float16),
              OV_CASE(ov::element::i8, int8_t),
              OV_CASE(ov::element::u8, uint8_t));
}

// indices is a (q-1)-dimension tensor of k-tuple,
// k is indices.shape[-1] and should not be greater than rank of input, q is rank of indicies.
// updates is a (q-1)-dimension tensor of replacement-slice-values
void ScatterUpdate::scatterNDUpdate(const MemoryPtr& mem_data,
                                    const MemoryPtr& mem_indices,
                                    const MemoryPtr& mem_updates,
                                    const scatter_reductions::ReduceNone& kernel) {
    auto* indices = mem_indices->getDataAs<uint8_t>();
    auto* update = mem_updates->getDataAs<uint8_t>();
    auto* dstData = mem_data->getDataAs<uint8_t>();
    const auto& srcDataDim = getParentEdgeAt(DATA_ID)->getMemory().getStaticDims();
    const auto elementsCount = getParentEdgeAt(DATA_ID)->getMemory().getShape().getElementsCount();
    const auto& indicesDim = getParentEdgeAt(INDICES_ID)->getMemory().getStaticDims();
    size_t indicesRank = indicesDim.size();

    std::vector<size_t> srcBlockND = getBlockND(srcDataDim);

    size_t k = indicesDim[indicesRank - 1];
    size_t idxTupleNum = 1;
    for (size_t ri = 0; ri < indicesRank - 1; ri++) {
        idxTupleNum *= indicesDim[ri];
    }

    size_t sizeToUpdate = srcBlockND[k] * dataSize;
    parallel_for(idxTupleNum, [&](size_t tupleIdx) {
        size_t indicesOffset = tupleIdx * k;
        size_t dstOffset = 0;
        for (size_t i = 0; i < k; i++) {
            int64_t idxValue = getIndicesValue(indices, indicesOffset + i);
            if (idxValue < 0) {
                // Negative value for indices means counting backwards from the end.
                idxValue += srcDataDim[i];
            }
            dstOffset += idxValue * srcBlockND[i + 1];
        }

        // Exception must be thrown according to the specification
        CPU_NODE_ASSERT(dstOffset < elementsCount,
                        " indices contain values that points to non-existing data tensor element");

        dstOffset *= dataSize;
        size_t updateOffset = tupleIdx * sizeToUpdate;

        cpu_memcpy(dstData + dstOffset, update + updateOffset, sizeToUpdate);
    });
}

template <typename DataType, typename KernelType>
void ScatterUpdate::scatterNDUpdate(const MemoryPtr& mem_data,
                                    const MemoryPtr& mem_indices,
                                    const MemoryPtr& mem_updates,
                                    const KernelType& kernel) {
    CPU_NODE_ASSERT(reduction_type != ScatterUpdate::Reduction::NONE, "The reduction should not be NONE.");
    auto* indices = mem_indices->getDataAs<uint8_t>();
    auto* update = mem_updates->getDataAs<DataType>();
    auto* dstData = mem_data->getDataAs<DataType>();
    const auto& srcDataDim = getParentEdgeAt(DATA_ID)->getMemory().getStaticDims();
    const auto elementsCount = getParentEdgeAt(DATA_ID)->getMemory().getShape().getElementsCount();
    const auto& indicesDim = getParentEdgeAt(INDICES_ID)->getMemory().getStaticDims();
    const auto indicesRank = indicesDim.size();

    std::vector<size_t> srcBlockND = getBlockND(srcDataDim);
    const auto k = indicesDim[indicesRank - 1];
    size_t idxTupleNum = 1;
    for (size_t ri = 0; ri < indicesRank - 1; ri++) {
        idxTupleNum *= indicesDim[ri];
    }
    const auto sizeToUpdate = srcBlockND[k];
    for (size_t tupleIdx = 0; tupleIdx < idxTupleNum; tupleIdx++) {
        size_t indicesOffset = tupleIdx * k;
        size_t dstOffset = 0;
        for (size_t i = 0; i < k; i++) {
            int64_t idxValue = getIndicesValue(indices, indicesOffset + i);
            if (idxValue < 0) {
                // Negative value for indices means counting backwards from the end.
                idxValue += srcDataDim[i];
            }
            dstOffset += idxValue * srcBlockND[i + 1];
        }

        // Exception must be thrown according to the specification
        CPU_NODE_ASSERT(dstOffset < elementsCount,
                        " indices contain values that points to non-existing data tensor element");

        const auto updateOffset = tupleIdx * sizeToUpdate;
        DataType* dstDataWithOffset = dstData + dstOffset;
        const DataType* updateWithOffset = update + updateOffset;
        for (size_t idx = 0; idx < sizeToUpdate; idx++) {
            kernel(dstDataWithOffset + idx, updateWithOffset + idx);
        }
    }
}

bool ScatterUpdate::created() const {
    return getType() == Type::ScatterUpdate || getType() == Type::ScatterElementsUpdate ||
           getType() == Type::ScatterNDUpdate;
}

}  // namespace ov::intel_cpu::node
