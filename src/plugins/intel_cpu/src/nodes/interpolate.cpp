// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate.h"
#include "executors/interpolate_config.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <common/primitive_attr.hpp>
#include <common/primitive_hashing_utils.hpp>
#include <common/utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <utility>
#include <vector>

#include "common/cpu_memcpy.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "eltwise.h"
#include "fake_quantize.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/interpolate_config.hpp"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/enum_names.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/interpolate.hpp"
#include "shape_inference/shape_inference.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/bfloat16.hpp"
#include "utils/general_utils.h"
#include "utils/ngraph_utils.hpp"
#include "utils/precision_support.h"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include <xbyak/xbyak.h>

#    include <common/c_types_map.hpp>
#    include <unordered_map>

#    include "cpu/x64/injectors/jit_uni_depthwise_injector.hpp"
#    include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#    include "cpu/x64/injectors/jit_uni_quantization_injector.hpp"
#    include "cpu/x64/jit_generator.hpp"
#    include "emitters/plugin/x64/jit_emitter.hpp"
#    include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#    include "utils/cpu_utils.hpp"
#endif

using namespace dnnl;

using namespace dnnl::impl;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace ov::intel_cpu::node {



// shapeND: n     c     d     h    w
// blockND: ncdhw cdhw  dhw   hw   w    1
// index  : 0      1    2     3    4    5
// legacy layout helpers removed (handled in executors)

using ngInterpMode = ov::op::v4::Interpolate::InterpolateMode;
using ngInterpCoordTransf = ov::op::v4::Interpolate::CoordinateTransformMode;
using ngInterpNearMode = ov::op::v4::Interpolate::NearestMode;
using ngInterpShapeCalcMode = ov::op::v4::Interpolate::ShapeCalcMode;

bool Interpolate::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (const auto interp = ov::as_type_ptr<const ov::op::v4::Interpolate>(op)) {
            const auto& interpAttr = interp->get_attrs();
            const auto& interpMode = interpAttr.mode;
            if (none_of(interpMode,
                        ngInterpMode::NEAREST,
                        ngInterpMode::LINEAR,
                        ngInterpMode::LINEAR_ONNX,
                        ngInterpMode::CUBIC)) {
                errorMessage = "Interpolate-4 does not support interpolate mode: " + ov::as_string(interpMode);
                return false;
            }

            const auto& interpCoordTransMode = interpAttr.coordinate_transformation_mode;
            if (none_of(interpCoordTransMode,
                        ngInterpCoordTransf::HALF_PIXEL,
                        ngInterpCoordTransf::PYTORCH_HALF_PIXEL,
                        ngInterpCoordTransf::ASYMMETRIC,
                        ngInterpCoordTransf::TF_HALF_PIXEL_FOR_NN,
                        ngInterpCoordTransf::ALIGN_CORNERS)) {
                errorMessage = "Interpolate-4 does not support coordinate transformation mode: " +
                               ov::as_string(interpCoordTransMode);
                return false;
            }

            if (interpMode == ngInterpMode::NEAREST) {
                const auto& interpNearestMode = interpAttr.nearest_mode;
                if (none_of(interpNearestMode,
                            ngInterpNearMode::ROUND_PREFER_FLOOR,
                            ngInterpNearMode::ROUND_PREFER_CEIL,
                            ngInterpNearMode::FLOOR,
                            ngInterpNearMode::CEIL,
                            ngInterpNearMode::SIMPLE)) {
                    errorMessage =
                        "Interpolate-4 does not support nearest round mode: " + ov::as_string(interpNearestMode);
                    return false;
                }
            }

            const auto& interpShapeCalcMode = interpAttr.shape_calculation_mode;
            if (none_of(interpShapeCalcMode, ngInterpShapeCalcMode::SCALES, ngInterpShapeCalcMode::SIZES)) {
                errorMessage =
                    "Interpolate-4 does not support shape_calculation_mode: " + ov::as_string(interpShapeCalcMode);
                return false;
            }

            const size_t dataRank = interp->get_input_partial_shape(DATA_ID).rank().get_length();
            if (dataRank < 1 || dataRank > 5) {
                errorMessage = "Interpolate-4 does not support input tensor of rank : " + std::to_string(dataRank);
                return false;
            }

            if (dataRank == 5 && interpMode == ngInterpMode::CUBIC) {
                errorMessage = "Interpolate-4 doesn't support input tensor with rank: " + std::to_string(dataRank) +
                               " for 'cubic' mode ";
                return false;
            }

            if (!isDynamicNgraphNode(op) && interpShapeCalcMode == ngInterpShapeCalcMode::SCALES &&
                !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(SCALES_ID))) {
                errorMessage = "Only const 'scales' input is supported for static shapes in Interpolate-4";
                return false;
            }

            if (interp->get_input_size() > 3 &&
                ov::as_type_ptr<const ov::op::v0::Constant>(interp->get_input_node_shared_ptr(AXES_ID)) == nullptr) {
                errorMessage = "Only const 'axes' input is supported in Interpolate-4";
                return false;
            }
        } else if (const auto interp = ov::as_type_ptr<const ov::op::v11::Interpolate>(op)) {
            const auto& interpAttr = interp->get_attrs();
            const auto& interpMode = interpAttr.mode;
            if (none_of(interpMode, ngInterpMode::BILINEAR_PILLOW, ngInterpMode::BICUBIC_PILLOW)) {
                errorMessage = "Interpolate-11 does not support interpolate mode: " + ov::as_string(interpMode);
                return false;
            }
            const auto& interpShapeCalcMode = interpAttr.shape_calculation_mode;
            if (none_of(interpShapeCalcMode, ngInterpShapeCalcMode::SCALES, ngInterpShapeCalcMode::SIZES)) {
                errorMessage =
                    "Interpolate-11 does not support shape_calculation_mode: " + ov::as_string(interpShapeCalcMode);
                return false;
            }
            const size_t dataRank = interp->get_input_partial_shape(DATA_ID).rank().get_length();
            if (dataRank < 2 || dataRank > 4) {
                // pillow only resize on H and W. resize on D(depth) is not defined.
                errorMessage = "Interpolate-11 does not support input tensor of rank : " + std::to_string(dataRank);
                return false;
            }
            if (!isDynamicNgraphNode(op) &&
                !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(SIZE_OR_SCALE_ID_V11))) {
                errorMessage = "Only const 'scales_or_sizes' input is supported for static shapes in Interpolate-11";
                return false;
            }
            if (interp->get_input_size() > 2 && ov::as_type_ptr<const ov::op::v0::Constant>(
                                                    interp->get_input_node_shared_ptr(AXES_ID_V11)) == nullptr) {
                errorMessage = "Only const 'axes' input is supported in Interpolate-11";
                return false;
            }
        } else {
            errorMessage = "Only v4 and v11 interpolate operation are supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

namespace {
/**
 * Interpolate shape inference factory. It defines the input mask depending on the shape calculation mode.
 *
 */
class InterpolateShapeInferFactory : public ShapeInferFactory {
public:
    explicit InterpolateShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override {
        if (auto interp4 = ov::as_type_ptr<ov::op::v4::Interpolate>(m_op)) {
            const auto& attr = interp4->get_attrs();
            const auto is_supported_mode = (attr.shape_calculation_mode == ngInterpShapeCalcMode::SCALES) ||
                                           (attr.shape_calculation_mode == ngInterpShapeCalcMode::SIZES);
            OPENVINO_ASSERT(is_supported_mode, "Unsupported interpolate shape calculation mode");
            return make_shape_inference(m_op);
        }
        if (auto interp11 = ov::as_type_ptr<ov::op::v11::Interpolate>(m_op)) {
            return make_shape_inference(m_op);
        }
        OPENVINO_THROW("Shape infer factory cannot be created for ",
                       m_op->get_type_name(),
                       " node with name: ",
                       m_op->get_friendly_name(),
                       ", only versions 4 and 11 are supported.");
    }

private:
    std::shared_ptr<ov::Node> m_op;
};
}  // namespace

Interpolate::Interpolate(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, InterpolateShapeInferFactory(op)) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        const auto& inputDataShape = getInputShapeAtPort(DATA_ID);
        dataRank = inputDataShape.getRank();
        if (const auto interp = ov::as_type_ptr<const ov::op::v4::Interpolate>(op)) {
            is_version11 = false;
            const auto numInputs = inputShapes.size();
            CPU_NODE_ASSERT(numInputs == 3 || numInputs == 4, "has incorrect number of input edges");
            CPU_NODE_ASSERT(outputShapes.size() == 1, "has incorrect number of output edges");
            isAxesSpecified = numInputs != 3;

            const auto& interpAttr = interp->get_attrs();

            const auto& interpMode = interpAttr.mode;
            if (interpMode == ngInterpMode::NEAREST) {
                interpAttrs.mode = InterpolateMode::nearest;
            } else if (interpMode == ngInterpMode::LINEAR) {
                if (dataRank < 5) {
                    interpAttrs.mode = InterpolateMode::linear_onnx;
                } else {
                    interpAttrs.mode = InterpolateMode::linear;
                }
            } else if (interpMode == ngInterpMode::LINEAR_ONNX) {
                interpAttrs.mode = InterpolateMode::linear_onnx;
            } else if (interpMode == ngInterpMode::CUBIC) {
                interpAttrs.mode = InterpolateMode::cubic;
            } else {
                CPU_NODE_THROW("has unsupported interpolate mode");
            }

            const auto& interpCoordTransMode = interpAttr.coordinate_transformation_mode;
            if (interpCoordTransMode == ngInterpCoordTransf::HALF_PIXEL) {
                interpAttrs.coordTransMode = InterpolateCoordTransMode::half_pixel;
            } else if (interpCoordTransMode == ngInterpCoordTransf::PYTORCH_HALF_PIXEL) {
                interpAttrs.coordTransMode = InterpolateCoordTransMode::pytorch_half_pixel;
            } else if (interpCoordTransMode == ngInterpCoordTransf::ASYMMETRIC) {
                interpAttrs.coordTransMode = InterpolateCoordTransMode::asymmetric;
            } else if (interpCoordTransMode == ngInterpCoordTransf::TF_HALF_PIXEL_FOR_NN) {
                interpAttrs.coordTransMode = InterpolateCoordTransMode::tf_half_pixel_for_nn;
            } else if (interpCoordTransMode == ngInterpCoordTransf::ALIGN_CORNERS) {
                interpAttrs.coordTransMode = InterpolateCoordTransMode::align_corners;
            } else {
                CPU_NODE_THROW("has unsupported coordination transformation mode");
            }

            if (interpAttrs.mode == InterpolateMode::nearest) {
                const auto& interpNearestMode = interpAttr.nearest_mode;
                if (interpNearestMode == ngInterpNearMode::ROUND_PREFER_FLOOR) {
                    interpAttrs.nearestMode = InterpolateNearestMode::round_prefer_floor;
                } else if (interpNearestMode == ngInterpNearMode::ROUND_PREFER_CEIL) {
                    interpAttrs.nearestMode = InterpolateNearestMode::round_prefer_ceil;
                } else if (interpNearestMode == ngInterpNearMode::FLOOR) {
                    interpAttrs.nearestMode = InterpolateNearestMode::floor;
                } else if (interpNearestMode == ngInterpNearMode::CEIL) {
                    interpAttrs.nearestMode = InterpolateNearestMode::ceil;
                } else if (interpNearestMode == ngInterpNearMode::SIMPLE) {
                    interpAttrs.nearestMode = InterpolateNearestMode::simple;
                } else {
                    CPU_NODE_THROW("has unsupported nearest mode");
                }
            } else if (interpAttrs.mode == InterpolateMode::cubic) {
                interpAttrs.cubeCoeff = static_cast<float>(interpAttr.cube_coeff);
            }
            interpAttrs.antialias = interpAttr.antialias;

            const auto& interpShapeCalcMode = interpAttr.shape_calculation_mode;
            if (interpShapeCalcMode == ngInterpShapeCalcMode::SCALES) {
                interpAttrs.shapeCalcMode = InterpolateShapeCalcMode::scales;
            } else if (interpShapeCalcMode == ngInterpShapeCalcMode::SIZES) {
                interpAttrs.shapeCalcMode = InterpolateShapeCalcMode::sizes;
            } else {
                CPU_NODE_THROW("has unsupported shape calculation mode");
            }

            if (interpAttr.pads_begin.empty()) {
                interpAttrs.padBegin.resize(dataRank, 0);
            } else {
                interpAttrs.padBegin.resize(interpAttr.pads_begin.size());
                for (size_t i = 0; i < interpAttr.pads_begin.size(); i++) {
                    interpAttrs.padBegin[i] = static_cast<int>(interpAttr.pads_begin[i]);
                }
            }

            if (interpAttr.pads_end.empty()) {
                interpAttrs.padEnd.resize(dataRank, 0);
            } else {
                interpAttrs.padEnd.resize(interpAttr.pads_end.size());
                for (size_t i = 0; i < interpAttr.pads_end.size(); i++) {
                    interpAttrs.padEnd[i] = static_cast<int>(interpAttr.pads_end[i]);
                }
            }

            const auto scalesNode =
                ov::as_type_ptr<const ov::op::v0::Constant>(interp->get_input_node_shared_ptr(SCALES_ID));
            if (scalesNode) {
                scales = scalesNode->cast_vector<float>();
                isScaleConstant = true;
            }

            if (isAxesSpecified) {
                axes = ov::as_type_ptr<const ov::op::v0::Constant>(interp->get_input_node_shared_ptr(AXES_ID))
                           ->cast_vector<int>();
                if (dataRank == 4 && axes.size() == 2 && axes[0] == 1 && axes[1] == 2) {
                    interpAttrs.NCHWAsNHWC = true;
                    axes[0] = 2;
                    axes[1] = 3;
                }
            } else {
                const auto& outputShape = getOutputShapeAtPort(0);
                // For the static shapes, we can avoid reordering NHWC to NCHW
                // if the last dimension of input and output shapes are equal.
                auto avoidReorder = [](const auto& inputDataShape, const auto& outputShape) {
                    auto lastInDim = inputDataShape.getDims().back();

                    // Dynamic shape
                    if (lastInDim == Shape::UNDEFINED_DIM) {
                        return false;
                    }
                    auto lastOutDim = outputShape.getDims().back();
                    return static_cast<bool>(lastOutDim == lastInDim);
                };
                if (dataRank == 4 && avoidReorder(inputDataShape, outputShape)) {
                    interpAttrs.NCHWAsNHWC = true;
                    axes = {0, 2, 3, 1};  // NHWC
                } else {
                    axes.resize(dataRank);
                    for (int i = 0; i < static_cast<int>(dataRank); i++) {
                        axes[i] = i;
                    }
                }
            }
        } else if (const auto interp = ov::as_type_ptr<const ov::op::v11::Interpolate>(op)) {
            is_version11 = true;
            const auto numInputs = inputShapes.size();
            CPU_NODE_ASSERT(numInputs == 2 || numInputs == 3, "has incorrect number of input edges");
            CPU_NODE_ASSERT(outputShapes.size() == 1, "has incorrect number of output edges");
            isAxesSpecified = numInputs != 2;

            const auto& interpAttr = interp->get_attrs();
            const auto& interpMode = interpAttr.mode;
            if (interpMode == ngInterpMode::BILINEAR_PILLOW) {
                interpAttrs.mode = InterpolateMode::bilinear_pillow;
            } else if (interpMode == ngInterpMode::BICUBIC_PILLOW) {
                interpAttrs.mode = InterpolateMode::bicubic_pillow;
                interpAttrs.cubeCoeff = static_cast<float>(interpAttr.cube_coeff);  // fixed to be -0.5
            } else {
                CPU_NODE_THROW("has unsupported interpolate mode");
            }

            // pillow use fixed tf_half_pixel_for_nn style mode for coodinate transformation
            interpAttrs.coordTransMode = InterpolateCoordTransMode::tf_half_pixel_for_nn;
            interpAttrs.antialias = interpAttr.antialias;

            const auto& interpShapeCalcMode = interpAttr.shape_calculation_mode;
            if (interpShapeCalcMode == ngInterpShapeCalcMode::SCALES) {
                interpAttrs.shapeCalcMode = InterpolateShapeCalcMode::scales;
                const auto scalesNode = ov::as_type_ptr<const ov::op::v0::Constant>(
                    interp->get_input_node_shared_ptr(SIZE_OR_SCALE_ID_V11));
                if (scalesNode) {
                    scales = scalesNode->cast_vector<float>();
                    isScaleConstant = true;
                }
            } else if (interpShapeCalcMode == ngInterpShapeCalcMode::SIZES) {
                interpAttrs.shapeCalcMode = InterpolateShapeCalcMode::sizes;
            } else {
                CPU_NODE_THROW("has unsupported shape calculation mode");
            }

            if (interpAttr.pads_begin.empty()) {
                interpAttrs.padBegin.resize(dataRank, 0);
            } else {
                interpAttrs.padBegin.resize(interpAttr.pads_begin.size());
                for (size_t i = 0; i < interpAttr.pads_begin.size(); i++) {
                    interpAttrs.padBegin[i] = static_cast<int>(interpAttr.pads_begin[i]);
                }
            }

            if (interpAttr.pads_end.empty()) {
                interpAttrs.padEnd.resize(dataRank, 0);
            } else {
                interpAttrs.padEnd.resize(interpAttr.pads_end.size());
                for (size_t i = 0; i < interpAttr.pads_end.size(); i++) {
                    interpAttrs.padEnd[i] = static_cast<int>(interpAttr.pads_end[i]);
                }
            }

            if (isAxesSpecified) {
                axes = ov::as_type_ptr<const ov::op::v0::Constant>(interp->get_input_node_shared_ptr(AXES_ID_V11))
                           ->cast_vector<int>();
                if (dataRank == 4 && axes.size() == 2 && axes[0] == 1 && axes[1] == 2) {
                    interpAttrs.NCHWAsNHWC = true;
                    axes[0] = 2;
                    axes[1] = 3;
                }
            } else {
                axes.resize(dataRank);
                for (int i = 0; i < static_cast<int>(dataRank); i++) {
                    axes[i] = i;
                }
            }
        }
    } else {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

void Interpolate::getSupportedDescriptors() {
    // v4: data, target_shape, scale, axis(optional).
    // v11: data, size_or_scale, axis(optional)
    CPU_NODE_ASSERT(getParentEdges().size() == 2 || getParentEdges().size() == 3 || getParentEdges().size() == 4,
                    "has incorrect number of input edges");
    CPU_NODE_ASSERT(!getChildEdges().empty(), "has incorrect number of output edges");

    // get pad
    for (int i : interpAttrs.padBegin) {
        if (i != 0) {
            hasPad = true;
            break;
        }
    }
    for (int i : interpAttrs.padEnd) {
        if (i != 0) {
            hasPad = true;
            break;
        }
    }
    // correct pad
    if (hasPad) {
        interpAttrs.NCHWAsNHWC = false;
        auto correctPad = [&](std::vector<int> pad, int rank) {
            int padLen = pad.size();
            if (padLen == rank) {
                return pad;
            }
            std::vector<int> result;
            if (padLen > rank) {
                result.insert(result.end(), pad.begin(), pad.begin() + rank);
            } else {
                result = pad;
                result.insert(result.end(), rank - padLen, 0);
            }
            return result;
        };

        interpAttrs.padBegin = correctPad(interpAttrs.padBegin, dataRank);
        interpAttrs.padEnd = correctPad(interpAttrs.padEnd, dataRank);
    }
}

void Interpolate::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    interpAttrs.postOps = getPostOps(fusedWith, ov::element::dynamic);
    interpAttrs.hasPad = hasPad;
    interpAttrs.axes = axes;
    interpAttrs.isAxesSpecified = isAxesSpecified;

    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();

    const auto inputPrecision = getOriginalInputPrecisionAtPort(DATA_ID);
    auto dstPrecision = getOriginalOutputPrecisionAtPort(0);
    if (!fusedWith.empty()) {
        dstPrecision = fusedWith.back()->getOriginalOutputPrecisionAtPort(0);
    }

    MemoryDescArgs descs;
    descs[ARG_SRC] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(inputPrecision, getInputShapeAtPort(DATA_ID));
    // map size/scales input if present
    if (is_version11) {
        // v11: data, size_or_scale, axes(optional)
        descs[ARG_SRC_1] = creatorsMap.at(LayoutType::ncsp)
                               ->createSharedDesc(interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::sizes
                                                      ? ov::element::i32
                                                      : ov::element::f32,
                                                  getInputShapeAtPort(SIZE_OR_SCALE_ID_V11));
        if (isAxesSpecified) {
            descs[ARG_SRC_2] =
                creatorsMap.at(LayoutType::ncsp)->createSharedDesc(ov::element::i32, getInputShapeAtPort(AXES_ID_V11));
        } else {
            descs[ARG_SRC_2] = MemoryDescUtils::makeEmptyDesc();
        }
    } else {
        // v4: data, target_shape, scales, axes(optional)
        descs[ARG_SRC_1] = creatorsMap.at(LayoutType::ncsp)
                               ->createSharedDesc(ov::element::i32, getInputShapeAtPort(TARGET_SHAPE_ID));
        descs[ARG_SRC_2] = creatorsMap.at(LayoutType::ncsp)
                               ->createSharedDesc(ov::element::f32, getInputShapeAtPort(SCALES_ID));
        if (isAxesSpecified) {
            descs[ARG_SRC_3] = creatorsMap.at(LayoutType::ncsp)
                                    ->createSharedDesc(ov::element::i32, getInputShapeAtPort(AXES_ID));
        }
    }
    descs[ARG_DST] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(dstPrecision, getOutputShapeAtPort(0));

    auto executionContext = std::make_shared<ExecutorContext>(context, getImplPriority());
    factory = std::make_shared<ExecutorFactory<InterpolateAttrs>>(interpAttrs, executionContext, descs);

    const std::vector<MemoryDescArgs> nodeDescriptorsList = factory->getProperMemoryDescriptors(descs);
    for (const auto& nodeDescriptors : nodeDescriptorsList) {
        NodeConfig nodeConfig;
        nodeConfig.inConfs.resize(getParentEdges().size());

        // set mapping between arguments and ports
        m_atoi.clear();
        m_atoi[ARG_SRC] = DATA_ID;
        if (is_version11) {
            m_atoi[ARG_SRC_1] = SIZE_OR_SCALE_ID_V11;
            if (isAxesSpecified) m_atoi[ARG_SRC_2] = AXES_ID_V11;
        } else {
            m_atoi[ARG_SRC_1] = TARGET_SHAPE_ID;
            m_atoi[ARG_SRC_2] = SCALES_ID;
            if (isAxesSpecified) m_atoi[ARG_SRC_3] = AXES_ID;
        }

        for (const auto& desc : nodeDescriptors) {
            if (auto it = m_atoi.find(desc.first); it != m_atoi.end()) {
                nodeConfig.inConfs[it->second] = PortConfig(desc.second);
            }
        }

        const auto& outDesc = nodeDescriptors.at(ARG_DST);
        nodeConfig.outConfs.emplace_back(outDesc, BlockedMemoryDesc::FULL_MASK, -1);
        supportedPrimitiveDescriptors.emplace_back(nodeConfig, impl_desc_type::undef);
    }
}

bool Interpolate::needShapeInfer() const {
    if (Node::inputShapesModified()) {
        return true;
    }
    if (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::scales) {
        if (lastScales.empty()) {
            return true;
        }
        const auto* scales = getSrcDataAtPortAs<const float>(get_scale_id());
        for (size_t i = 0; i < lastScales.size(); i++) {
            if (lastScales[i] != scales[i]) {
                return true;
            }
        }
    } else {
        if (lastSizes.empty()) {
            return true;
        }
        const auto* sizes = getSrcDataAtPortAs<const int32_t>(TARGET_SHAPE_ID);
        for (size_t i = 0; i < lastSizes.size(); i++) {
            if (sizes[i] != lastSizes[i]) {
                return true;
            }
        }
    }
    return false;
}

void Interpolate::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);

    const size_t port = interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::sizes ? TARGET_SHAPE_ID : get_scale_id();
    const auto& memory = getParentEdgeAt(port)->getMemory();
    if (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::scales) {
        const auto* scales = memory.getDataAs<const float>();
        lastScales.assign(scales, scales + memory.getDesc().getShape().getElementsCount());
    } else {
        const auto* sizes = memory.getDataAs<const int32_t>();
        lastSizes.assign(sizes, sizes + memory.getDesc().getShape().getElementsCount());
    }
}

bool Interpolate::needPrepareParams() const {
    return (inputShapesModified() || lastOutputDims != getChildEdgeAt(0)->getMemory().getStaticDims());
}

inline int Interpolate::get_scale_id() const {
    if (is_version11) {
        return SIZE_OR_SCALE_ID_V11;
    }
    return SCALES_ID;
}
inline int Interpolate::get_axis_id() const {
    if (is_version11) {
        return AXES_ID_V11;
    }
    return AXES_ID;
}

void Interpolate::prepareParams() {
    for (const auto& entry : m_atoi) {
        const auto argId = entry.first;
        const auto portId = entry.second;
        memory[argId] = getSrcMemoryAtPort(portId);
    }
    memory[ARG_DST] = getDstMemoryAtPort(0);

    const auto& exec = executor;
    CPU_NODE_ASSERT(exec, "Executor is not created");
    exec->update(memory);
    getSelectedPrimitiveDescriptor()->setImplementationType(exec->implType());
    lastOutputDims = getChildEdgeAt(0)->getMemory().getStaticDims();
}

void Interpolate::createPrimitive() {
    // map input memories to arguments according to initSupportedPrimitiveDescriptors
    for (const auto& entry : m_atoi) {
        const auto argId = entry.first;
        const auto portId = entry.second;
        memory[argId] = getSrcMemoryAtPort(portId);
    }
    memory[ARG_DST] = getDstMemoryAtPort(0);

    executor = factory->make(memory);
    Node::createPrimitive();
}

inline int clipCoord(int pos, int length) {
    return std::max(0, std::min(pos, length - 1));
}

static inline float triangleCoeff(float x) {
    return (std::max)(0.0F, 1 - std::abs(x));
}

// post-ops handled in executor pipeline

VectorDims Interpolate::getPaddedInputShape(const VectorDims& srcDims,
                                            const std::vector<int>& padBegin,
                                            const std::vector<int>& padEnd) {
    VectorDims paddedShape;
    int dataRank = srcDims.size();
    for (int i = 0; i < dataRank; i++) {
        paddedShape.push_back(srcDims[i] + padBegin[i] + padEnd[i]);
    }
    return paddedShape;
}

// get scales of data rank size
// if "scale" version: set scales with input scales, 1.F for other dims not in axis
// if "size" version: scales = shape[target] / shape[input].pad, 1.F for other dims not in axis
// scales is a required input, but should not use input scales when "size" case, which may added eps or is a dummy
// value, recalculate scales instead.
std::vector<float> Interpolate::getScales(const VectorDims& srcDimPad, const VectorDims& dstDim) {
    std::vector<float> fullScales(dataRank, 1.F);
    const size_t axesRank = axes.size();
    for (size_t i = 0; i < axesRank; i++) {
        int axis = axes[i];
        // pillow always re-generate scales with input and output shape
        if (interpAttrs.mode == InterpolateMode::bilinear_pillow ||
            interpAttrs.mode == InterpolateMode::bicubic_pillow) {
            fullScales[axis] = static_cast<float>(dstDim[axis]) / static_cast<float>(srcDimPad[axis]);
        } else {
            fullScales[axis] = (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::scales)
                                   ? scales[i]
                                   : static_cast<float>(dstDim[axis]) / static_cast<float>(srcDimPad[axis]);
        }
    }
    return fullScales;
}

void Interpolate::execute([[maybe_unused]] const dnnl::stream& strm) {
    executor->execute(memory);
}


size_t Interpolate::getSpatialDimsNum(const std::vector<float>& scales) {
    size_t spatialDims = scales.size();
    for (auto scale : scales) {
        if (scale != 1.0F) {
            break;
        }
        spatialDims--;
    }
    return spatialDims > 0 ? spatialDims : 1;
}

bool Interpolate::canFuse(const NodePtr& node) const {
    // JIT-only fusing policy: allow fusing only when JIT is applicable
    if (!mayiuse(cpu::x64::sse41))
        return false;
    if (interpAttrs.mode == InterpolateMode::linear ||
        interpAttrs.mode == InterpolateMode::bilinear_pillow ||
        interpAttrs.mode == InterpolateMode::bicubic_pillow)
        return false;
    // Only 4D/5D tensors are supported by JIT interpolate
    if (!(dataRank == 4 || dataRank == 5))
        return false;
    return canFuseSimpleOperation(node);
}

bool Interpolate::created() const {
    return getType() == Type::Interpolate;
}

}  // namespace ov::intel_cpu::node
