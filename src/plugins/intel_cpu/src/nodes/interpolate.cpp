// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate.h"
#include "executors/interpolate_config.hpp"
#include "executors/x64/interpolate.hpp"
#include "executors/common/interpolate.hpp"

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <utility>
#include <vector>

#include "common/cpu_memcpy.h"
#include "cpu_types.h"
#include "eltwise.h"
#include "fake_quantize.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/interpolate_list.hpp"
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
#include "utils/general_utils.h"
#include "utils/ngraph_utils.hpp"
#include "utils/precision_support.h"

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
inline VectorDims getBlockND(const VectorDims& shape) {
    int shapeRank = shape.size();
    VectorDims blockND(shapeRank + 1, 1);
    for (int i = shapeRank - 1; i >= 0; i--) {
        blockND[i] = shape[i] * blockND[i + 1];
    }
    return blockND;
}
// w/hw/ncw/nchw/ncdhw to ncdhw
inline VectorDims to5Dim(VectorDims casesDim) {
    size_t caseSize = casesDim.size();
    VectorDims dim5(5, 1lu);
    dim5[4] = casesDim[caseSize - 1];
    if (caseSize > 1) {
        dim5[3] = casesDim[caseSize - 2];
    }
    if (caseSize > 2) {
        dim5[0] = casesDim[0];
    }
    if (caseSize > 3) {
        dim5[1] = casesDim[1];
    }
    if (caseSize > 4) {
        dim5[2] = casesDim[2];
    }
    if (caseSize == 3) {  // nhw -> ncw
        dim5[1] = dim5[3];
        dim5[3] = 1lu;
    }
    return dim5;
}

using ngInterpMode = ov::op::v4::Interpolate::InterpolateMode;
using ngInterpCoordTransf = ov::op::v4::Interpolate::CoordinateTransformMode;
using ngInterpNearMode = ov::op::v4::Interpolate::NearestMode;
using ngInterpShapeCalcMode = ov::op::v4::Interpolate::ShapeCalcMode;

bool Interpolate::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        constexpr size_t DATA_ID = 0;
        constexpr size_t SCALES_ID = 2;
        constexpr size_t AXES_ID = 3;
        constexpr size_t SIZE_OR_SCALE_ID_V11 = 1;
        constexpr size_t AXES_ID_V11 = 2;

        if (const auto interp = ov::as_type_ptr<const ov::op::v4::Interpolate>(op)) {
            const auto& tmpInterpAttr = interp->get_attrs();
            const auto& interpMode = tmpInterpAttr.mode;
            if (!one_of(interpMode,
                        ngInterpMode::NEAREST,
                        ngInterpMode::LINEAR,
                        ngInterpMode::LINEAR_ONNX,
                        ngInterpMode::CUBIC)) {
                errorMessage = "Interpolate-4 does not support interpolate mode: " + ov::as_string(interpMode);
                return false;
            }

            const auto& interpCoordTransMode = tmpInterpAttr.coordinate_transformation_mode;
            if (!one_of(interpCoordTransMode,
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
                const auto& interpNearestMode = tmpInterpAttr.nearest_mode;
                if (!one_of(interpNearestMode,
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

            const auto& interpShapeCalcMode = tmpInterpAttr.shape_calculation_mode;
            if (!one_of(interpShapeCalcMode, ngInterpShapeCalcMode::SCALES, ngInterpShapeCalcMode::SIZES)) {
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
        } else if (const auto interp_v11 = ov::as_type_ptr<const ov::op::v11::Interpolate>(op)) {
            const auto& tmpInterpAttr = interp_v11->get_attrs();
            const auto& interpMode = tmpInterpAttr.mode;
            if (!one_of(interpMode, ngInterpMode::BILINEAR_PILLOW, ngInterpMode::BICUBIC_PILLOW)) {
                errorMessage = "Interpolate-11 does not support interpolate mode: " + ov::as_string(interpMode);
                return false;
            }
            const auto& interpShapeCalcMode = tmpInterpAttr.shape_calculation_mode;
            if (!one_of(interpShapeCalcMode, ngInterpShapeCalcMode::SCALES, ngInterpShapeCalcMode::SIZES)) {
                errorMessage =
                    "Interpolate-11 does not support shape_calculation_mode: " + ov::as_string(interpShapeCalcMode);
                return false;
            }
            const size_t dataRank = interp_v11->get_input_partial_shape(DATA_ID).rank().get_length();
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
            if (interp_v11->get_input_size() > 2 && ov::as_type_ptr<const ov::op::v0::Constant>(
                    interp_v11->get_input_node_shared_ptr(AXES_ID_V11)) == nullptr) {
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
    InterpolateShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}
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
        dataRank = getInputShapeAtPort(interpAttrs.DATA_ID).getRank();
        if (const auto interp_v4 = ov::as_type_ptr<const ov::op::v4::Interpolate>(op)) {
            is_version11 = false;
            const auto numInputs = inputShapes.size();
            if (numInputs != 3 && numInputs != 4) {
                THROW_CPU_NODE_ERR("has incorrect number of input edges");
            }
            if (outputShapes.size() != 1) {
                THROW_CPU_NODE_ERR("has incorrect number of output edges");
            }
            isAxesSpecified = numInputs != 3;

            const auto& interpAttr = interp_v4->get_attrs();

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
                THROW_CPU_NODE_ERR("has unsupported interpolate mode");
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
                THROW_CPU_NODE_ERR("has unsupported coordination transformation mode");
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
                    THROW_CPU_NODE_ERR("has unsupported nearest mode");
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
                THROW_CPU_NODE_ERR("has unsupported shape calculation mode");
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
                ov::as_type_ptr<const ov::op::v0::Constant>(interp_v4->get_input_node_shared_ptr(interpAttrs.SCALES_ID));
            if (scalesNode) {
                scales = scalesNode->cast_vector<float>();
                isScaleConstant = true;
            }

            if (isAxesSpecified) {
                axes = ov::as_type_ptr<const ov::op::v0::Constant>(interp_v4->get_input_node_shared_ptr(interpAttrs.AXES_ID))
                           ->cast_vector<int>();
            } else {
                axes.resize(dataRank);
                for (int i = 0; i < static_cast<int>(dataRank); i++) {
                    axes[i] = i;
                }
            }
        } else if (const auto interp = ov::as_type_ptr<const ov::op::v11::Interpolate>(op)) {
            is_version11 = true;
            const auto numInputs = inputShapes.size();
            if (numInputs != 2 && numInputs != 3) {
                THROW_CPU_NODE_ERR("has incorrect number of input edges");
            }
            if (outputShapes.size() != 1) {
                THROW_CPU_NODE_ERR("has incorrect number of output edges");
            }
            isAxesSpecified = numInputs != 2;

            const auto& interpAttr = interp->get_attrs();
            const auto& interpMode = interpAttr.mode;
            if (interpMode == ngInterpMode::BILINEAR_PILLOW) {
                interpAttrs.mode = InterpolateMode::bilinear_pillow;
            } else if (interpMode == ngInterpMode::BICUBIC_PILLOW) {
                interpAttrs.mode = InterpolateMode::bicubic_pillow;
                interpAttrs.cubeCoeff = static_cast<float>(interpAttr.cube_coeff);  // fixed to be -0.5
            } else {
                THROW_CPU_NODE_ERR("has unsupported interpolate mode");
            }

            // pillow use fixed tf_half_pixel_for_nn style mode for coodinate transformation
            interpAttrs.coordTransMode = InterpolateCoordTransMode::tf_half_pixel_for_nn;
            interpAttrs.antialias = interpAttr.antialias;

            const auto& interpShapeCalcMode = interpAttr.shape_calculation_mode;
            if (interpShapeCalcMode == ngInterpShapeCalcMode::SCALES) {
                interpAttrs.shapeCalcMode = InterpolateShapeCalcMode::scales;
                const auto scalesNode = ov::as_type_ptr<const ov::op::v0::Constant>(
                    interp->get_input_node_shared_ptr(interpAttrs.SIZE_OR_SCALE_ID_V11));
                if (scalesNode) {
                    scales = scalesNode->cast_vector<float>();
                    isScaleConstant = true;
                }
            } else if (interpShapeCalcMode == ngInterpShapeCalcMode::SIZES) {
                interpAttrs.shapeCalcMode = InterpolateShapeCalcMode::sizes;
            } else {
                THROW_CPU_NODE_ERR("has unsupported shape calculation mode");
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
                axes = ov::as_type_ptr<const ov::op::v0::Constant>(interp->get_input_node_shared_ptr(interpAttrs.AXES_ID_V11))
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
    if (getParentEdges().size() != 2 && getParentEdges().size() != 3 && getParentEdges().size() != 4) {
        // v4: data, target_shape, scale, axis(optional).
        // v11: data, size_or_scale, axis(optional)
        THROW_CPU_NODE_ERR("has incorrect number of input edges");
    }
    if (getChildEdges().empty()) {
        THROW_CPU_NODE_ERR("has incorrect number of output edges");
    }

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

    ov::element::Type inputPrecision = getOriginalInputPrecisionAtPort(interpAttrs.DATA_ID);

#if defined(OV_CPU_WITH_ACL)
    bool isInputPrecisionSupported = one_of(inputPrecision, ov::element::i8, ov::element::u8, ov::element::f16);
#else
    bool isInputPrecisionSupported = one_of(inputPrecision, ov::element::i8, ov::element::u8, ov::element::bf16);
#endif
    if (!isInputPrecisionSupported) {
        inputPrecision = ov::element::f32;
    }

    if (!hasHardwareSupport(inputPrecision)) {
        inputPrecision = ov::element::f32;
    }

    // support input with rank<=3 only with float precision and planar layout.
    // Jit for avx2(gather is available) and ref for no-avx2 machine.
    if (!one_of(dataRank, 4u, 5u)) {
        inputPrecision = ov::element::f32;
    }
    ov::element::Type outputPrecision = inputPrecision;

    if (!fusedWith.empty()) {
        outputPrecision = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(interpAttrs.DATA_ID);
    }

#if !defined(OV_CPU_WITH_ACL)
    if (!mayiuse(cpu::x64::sse41)) {
        inputPrecision = outputPrecision = ov::element::f32;
    }
#endif

    auto targetShapeType = ov::element::i32;
    auto scalesType = ov::element::f32;
    auto axesType = ov::element::i32;

    NodeConfig config;
    config.outConfs.resize(1);
    if (is_version11) {
        if (isAxesSpecified) {
            config.inConfs.resize(3);
        } else {
            config.inConfs.resize(2);
        }
    } else {
        if (isAxesSpecified) {
            config.inConfs.resize(4);
        } else {
            config.inConfs.resize(3);
        }
    }
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto pushDesc = [&](LayoutType dataFormat,
                        impl_desc_type implDetail,
                        bool is_version11_desc,
                        bool useAclExecutor = false) {
        config.inConfs[interpAttrs.DATA_ID].setMemDesc(
            creatorsMap.at(dataFormat)->createSharedDesc(inputPrecision, getInputShapeAtPort(interpAttrs.DATA_ID)));
        if (is_version11_desc) {
            if (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::sizes) {
                config.inConfs[interpAttrs.SIZE_OR_SCALE_ID_V11].setMemDesc(
                    creatorsMap.at(LayoutType::ncsp)
                        ->createSharedDesc(targetShapeType, getInputShapeAtPort(interpAttrs.SIZE_OR_SCALE_ID_V11)));
            } else {
                config.inConfs[interpAttrs.SIZE_OR_SCALE_ID_V11].setMemDesc(
                    creatorsMap.at(LayoutType::ncsp)
                        ->createSharedDesc(scalesType, getInputShapeAtPort(interpAttrs.SIZE_OR_SCALE_ID_V11)));
            }

            if (isAxesSpecified) {
                config.inConfs[interpAttrs.AXES_ID_V11].setMemDesc(
                    creatorsMap.at(LayoutType::ncsp)->createSharedDesc(axesType, getInputShapeAtPort(interpAttrs.AXES_ID_V11)));
            }
        } else {
            config.inConfs[interpAttrs.TARGET_SHAPE_ID].setMemDesc(
                creatorsMap.at(LayoutType::ncsp)
                    ->createSharedDesc(targetShapeType, getInputShapeAtPort(interpAttrs.TARGET_SHAPE_ID)));
            config.inConfs[get_scale_id()].setMemDesc(
                creatorsMap.at(LayoutType::ncsp)->createSharedDesc(scalesType, getInputShapeAtPort(get_scale_id())));

            if (isAxesSpecified) {
                config.inConfs[get_axis_id()].setMemDesc(
                    creatorsMap.at(LayoutType::ncsp)->createSharedDesc(axesType, getInputShapeAtPort(get_axis_id())));
            }
        }

        config.outConfs[0].setMemDesc(
            creatorsMap.at(dataFormat)->createSharedDesc(outputPrecision, getOutputShapeAtPort(0)));

        if (useAclExecutor) {
            std::vector<MemoryDescPtr> srcMemoryDescs;
            srcMemoryDescs.reserve(config.inConfs.size());
            for (const auto& inConf : config.inConfs) {
                srcMemoryDescs.push_back(inConf.getMemDesc());
            }
            std::vector<MemoryDescPtr> dstMemoryDescs;
            dstMemoryDescs.reserve(config.outConfs.size());
            for (const auto& outConf : config.outConfs) {
                dstMemoryDescs.push_back(outConf.getMemDesc());
            }

            auto factory = std::make_shared<InterpolateExecutorFactory>(
                interpAttrs,
                srcMemoryDescs,
                dstMemoryDescs,
                std::make_shared<ExecutorContext>(context, getImplPriority()));
            if (!factory->isEmpty()) {
                supportedPrimitiveDescriptors.emplace_back(config, implDetail, factory);
            }
        } else {
            supportedPrimitiveDescriptors.emplace_back(config, implDetail);
        }
    };
    if (is_version11) {
#if defined(OV_CPU_WITH_ACL)
        interpAttrs.hasPad = hasPad;
        pushDesc(LayoutType::nspc, undef, true, true);
        pushDesc(LayoutType::ncsp, undef, true, true);
        canUseAclExecutor = !supportedPrimitiveDescriptors.empty();
        if (canUseAclExecutor) {
            return;
        }
        // fallback to f32 if ref is used
        inputPrecision = outputPrecision = ov::element::f32;
#endif

        if (dataRank == 4) {
            if (mayiuse(cpu::x64::avx512_core)) {
                if (interpAttrs.NCHWAsNHWC) {
                    pushDesc(LayoutType::ncsp, jit_avx512, true);
                } else {
                    pushDesc(LayoutType::nspc, jit_avx512, true);
                }
            } else if (mayiuse(cpu::x64::avx2)) {
                if (interpAttrs.NCHWAsNHWC) {
                    pushDesc(LayoutType::ncsp, jit_avx2, true);
                } else {
                    pushDesc(LayoutType::nspc, jit_avx2, true);
                }
            } else if (mayiuse(cpu::x64::sse41)) {
                if (interpAttrs.NCHWAsNHWC) {
                    pushDesc(LayoutType::ncsp, jit_sse42, true);
                } else {
                    pushDesc(LayoutType::nspc, jit_sse42, true);
                }
            }
        }
        pushDesc(LayoutType::ncsp, ref, true);
    } else {
        const auto& dataMinDims = getInputShapeAtPort(interpAttrs.DATA_ID).getMinDims();
        bool isBlkApplied = dataRank > 1 && dataMinDims[1] != Shape::UNDEFINED_DIM && dataMinDims[1] > 1;

#if defined(OV_CPU_WITH_ACL)
        interpAttrs.hasPad = hasPad;
        pushDesc(LayoutType::nspc, undef, false, true);
        pushDesc(LayoutType::ncsp, undef, false, true);
        canUseAclExecutor = !supportedPrimitiveDescriptors.empty();
        if (canUseAclExecutor) {
            return;
        }
        // fallback to f32 if ref is used
        inputPrecision = outputPrecision = ov::element::f32;
#endif

        if (!mayiuse(cpu::x64::sse41) || interpAttrs.mode == InterpolateMode::linear) {
            pushDesc(LayoutType::ncsp, ref, false);
        } else {
            // blk and by_channel JIT kernel on sse41 or above machine
            if (dataRank == 4 || (dataRank == 5 && interpAttrs.mode != InterpolateMode::cubic)) {
                if (mayiuse(cpu::x64::avx512_core)) {
                    pushDesc(LayoutType::nspc, jit_avx512, false);
                    if (isBlkApplied) {
                        pushDesc(LayoutType::nCsp16c, jit_avx512, false);
                    }
                } else if (mayiuse(cpu::x64::avx2)) {
                    pushDesc(LayoutType::nspc, jit_avx2, false);
                    if (isBlkApplied) {
                        pushDesc(LayoutType::nCsp8c, jit_avx2, false);
                    }
                } else {
                    pushDesc(LayoutType::nspc, jit_sse42, false);
                    if (isBlkApplied) {
                        pushDesc(LayoutType::nCsp8c, jit_sse42, false);
                    }
                }
            }

            // planar is only for float precision.
            // 1.ref on machine w/o avx2(no fuse)
            // 2.JIT kernel for avx2(gatherps is available).(with fuse)
            if (inputPrecision == ov::element::f32) {
                if (mayiuse(cpu::x64::avx2)) {
                    pushDesc(LayoutType::ncsp, jit_avx2, false);
                } else {
                    pushDesc(LayoutType::ncsp, ref, false);
                }
            }
        }
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
        const auto* scales_inf = getSrcDataAtPortAs<const float>(get_scale_id());
        for (size_t i = 0; i < lastScales.size(); i++) {
            if (lastScales[i] != scales_inf[i]) {
                return true;
            }
        }
    } else {
        if (lastSizes.empty()) {
            return true;
        }
        const auto* sizes = getSrcDataAtPortAs<const int32_t>(interpAttrs.TARGET_SHAPE_ID);
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

    const size_t port = interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::sizes ? interpAttrs.TARGET_SHAPE_ID : get_scale_id();
    const auto& memory = getParentEdgeAt(port)->getMemory();
    if (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::scales) {
        const auto* scales_dyn = memory.getDataAs<const float>();
        lastScales.assign(scales_dyn, scales_dyn + memory.getDesc().getShape().getElementsCount());
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
        return interpAttrs.SIZE_OR_SCALE_ID_V11;
    }
    return interpAttrs.SCALES_ID;
}
inline int Interpolate::get_axis_id() const {
    if (is_version11) {
        return interpAttrs.AXES_ID_V11;
    }
    return interpAttrs.AXES_ID;
}

void Interpolate::prepareParams() {
    if (!shapesDefined()) {
        THROW_CPU_NODE_ERR("input/output dims aren't defined");
    }

    auto dstMemPtr = getDstMemoryAtPort(0);
    if (!dstMemPtr || !dstMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("has undefined destination memory");
    }

    auto srcMemPtr = getSrcMemoryAtPort(interpAttrs.DATA_ID);
    if (!srcMemPtr || !srcMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("has undefined input memory");
    }

    if (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::sizes) {
        auto tsMemPtr = getSrcMemoryAtPort(interpAttrs.TARGET_SHAPE_ID);
        if (!tsMemPtr || !tsMemPtr->isDefined()) {
            THROW_CPU_NODE_ERR("has undefined target shape memory");
        }
    } else {
        auto scaleMemPtr = getSrcMemoryAtPort(get_scale_id());
        if (!scaleMemPtr || !scaleMemPtr->isDefined()) {
            THROW_CPU_NODE_ERR("has undefined scales memory");
        }
    }

    if (isAxesSpecified) {
        auto axesMemPtr = getSrcMemoryAtPort(get_axis_id());
        if (!axesMemPtr || !axesMemPtr->isDefined()) {
            THROW_CPU_NODE_ERR("has undefined axes memory");
        }
    }

    const NodeDesc* selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr) {
        THROW_CPU_NODE_ERR("did not set preferable primitive descriptor");
    }

    const auto& srcDimsOrign = srcMemPtr->getStaticDims();
    const auto& dstDimsOrign = dstMemPtr->getStaticDims();

    VectorDims srcDims = srcDimsOrign;
    VectorDims dstDims = dstDimsOrign;

    // layoutAlignment
    if (interpAttrs.NCHWAsNHWC && srcMemPtr->getDesc().hasLayoutType(LayoutType::ncsp)) {
        auto logicalShapeAlign = [](VectorDims& Dims) {
            size_t C = Dims[3];
            Dims[3] = Dims[2];
            Dims[2] = Dims[1];
            Dims[1] = C;
        };
        logicalShapeAlign(srcDims);
        logicalShapeAlign(dstDims);
        interpAttrs.layout = InterpolateLayoutType::by_channel;
    }

    if (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::scales) {
        if (!isScaleConstant) {
            const auto& scalesMem = getParentEdgeAt(get_scale_id())->getMemory();
            const auto* scalesData = scalesMem.getDataAs<const float>();
            scales.assign(scalesData, scalesData + scalesMem.getStaticDims()[0]);
        }
    }

    std::vector<float> dataScales =
        getScales(getPaddedInputShape(srcDims, interpAttrs.padBegin, interpAttrs.padEnd), dstDims);
    if (!interpAttrs.NCHWAsNHWC &&
        (getOutputShapeAtPort(0).getRank() > 2 && (dataScales[0] != 1.f || dataScales[1] != 1.f))) {
        THROW_CPU_NODE_ERR("only supports resize on spatial dimensions(depth, height and width)");
    }

    if (canUseAclExecutor) {
        interpAttrs.dataScales = dataScales;

        std::vector<MemoryDescPtr> srcMemoryDescs;
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            srcMemoryDescs.push_back(getSrcMemoryAtPort(i)->getDescPtr());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        dstMemoryDescs.push_back(getDstMemoryAtPort(0)->getDescPtr());

        auto* selectedPD = getSelectedPrimitiveDescriptor();
        aclExecPtr = selectedPD->getExecutorFactoryAs<InterpolateExecutorFactory>()->makeExecutor(interpAttrs,
                                                                                                  srcMemoryDescs,
                                                                                                  dstMemoryDescs,
                                                                                                  {});
        selectedPD->setImplementationType(aclExecPtr->getImplType());

        return;
    }

    InterpolateKey key = {interpAttrs, srcDims, dstDims, dataScales, dnnl::primitive_attr()};
    setPostOps(key.attr, dstDims);

    auto buildExecutor = [&](const InterpolateKey& key) -> std::shared_ptr<InterpolateExecutorBase> {
        std::shared_ptr<InterpolateExecutorBase> executor;
        bool isNearestLinearOrCubic = key.nodeAttrs.mode == InterpolateMode::nearest ||
                                      key.nodeAttrs.mode == InterpolateMode::linear_onnx ||
                                      key.nodeAttrs.mode == InterpolateMode::cubic;
        bool isPlanarLayourAndSse41 = key.nodeAttrs.layout != InterpolateLayoutType::planar && mayiuse(cpu::x64::sse41);
        bool isAvx2AndF32 = mayiuse(cpu::x64::avx2) && key.nodeAttrs.inPrc == ov::element::f32;
        bool isPillowMode = key.nodeAttrs.mode == InterpolateMode::bilinear_pillow ||
                            key.nodeAttrs.mode == InterpolateMode::bicubic_pillow;
        bool isByChannelLayout = key.nodeAttrs.layout == InterpolateLayoutType::by_channel;
        bool isNearestLinearOrCubicSupported = isNearestLinearOrCubic && (isPlanarLayourAndSse41 || isAvx2AndF32);
        bool isPillowModeSupported = isPillowMode && isByChannelLayout;

        if ((isNearestLinearOrCubicSupported || isPillowModeSupported) && mayiuse(cpu::x64::sse41)) {
            executor = std::make_shared<InterpolateJitExecutor>(key.nodeAttrs,
                                                                key.srcDims,
                                                                key.dstDims,
                                                                key.dataScales,
                                                                key.attr);
        } else {
            executor =
                std::make_shared<InterpolateRefExecutor>(key.nodeAttrs, key.srcDims, key.dstDims, key.dataScales);
        }
        return executor;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, buildExecutor);
    execPtr = result.first;

    lastOutputDims = dstDimsOrign;
}

void Interpolate::createPrimitive() {
    auto srcMemPtr = getSrcMemoryAtPort(interpAttrs.DATA_ID);
    auto dstMemPtr = getDstMemoryAtPort(0);
    if (!srcMemPtr) {
        THROW_CPU_NODE_ERR("has null input memory");
    }
    if (!dstMemPtr) {
        THROW_CPU_NODE_ERR("has null destination memory");
    }

    if (dstMemPtr->getDesc().hasLayoutType(LayoutType::ncsp)) {
        interpAttrs.layout = InterpolateLayoutType::planar;
    } else if (dstMemPtr->getDesc().hasLayoutType(LayoutType::nCsp8c) ||
               dstMemPtr->getDesc().hasLayoutType(LayoutType::nCsp16c)) {
        interpAttrs.layout = InterpolateLayoutType::block;
    } else {
        interpAttrs.layout = InterpolateLayoutType::by_channel;
    }

    interpAttrs.inPrc = srcMemPtr->getDesc().getPrecision();
    interpAttrs.outPrc = dstMemPtr->getDesc().getPrecision();

    if (shapesDefined() && isExecutable()) {
        if (needPrepareParams()) {
            prepareParams();
        }
        updateLastInputDims();
    }
}

void Interpolate::setPostOps(dnnl::primitive_attr& attr, const VectorDims& dims) {
    dnnl::post_ops ops;

    postOpsDataPtrs.clear();
    for (auto& node : fusedWith) {
        int channelAxis = 1;

        auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get());
        if (fakeQuantizeNode) {
            fakeQuantizeNode->appendPostOps(ops, {}, postOpsDataPtrs, channelAxis);
            continue;
        }

        auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get());
        if (eltwiseNode) {
            eltwiseNode->appendPostOps(ops, dims, postOpsDataPtrs, channelAxis);
            continue;
        }

        THROW_CPU_NODE_ERR("Fusing of ",
                           NameFromType(node->getType()),
                           " operation to ",
                           NameFromType(this->getType()),
                           " node is not implemented");
    }

    attr.set_post_ops(ops);
}

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
// if "scale" version: set scales with input scales, 1.f for other dims not in axis
// if "size" version: scales = shape[target] / shape[input].pad, 1.f for other dims not in axis
// scales is a required input, but should not use input scales when "size" case, which may added eps or is a dummy
// value, recalculate scales instead.
std::vector<float> Interpolate::getScales(const VectorDims& srcDimPad, const VectorDims& dstDim) {
    std::vector<float> fullScales(dataRank, 1.f);
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
    auto dstMemPtr = getDstMemoryAtPort(0);
    auto srcMemPtr = getSrcMemoryAtPort(interpAttrs.DATA_ID);

    if (execPtr) {
        auto* dst_data = dstMemPtr->getDataAs<uint8_t>();
        const uint8_t* src_data_origin = srcMemPtr->getDataAs<uint8_t>();
        const uint8_t* src_data = nullptr;
        std::vector<uint8_t> srcPadded;
        if (hasPad) {
            const auto& srcDim = srcMemPtr->getStaticDims();
            auto srcDimPad = execPtr->getSrcDimPad5d();
            size_t dimSize = srcDim.size();

            const auto srcDim5d = to5Dim(srcDim);
            const auto srcDimPad5d = to5Dim(srcDimPad);
            const auto srcDataSize = srcMemPtr->getDesc().getPrecision().size();

            int padB0 = (dimSize > 2) ? interpAttrs.padBegin[0] : 0;
            int padB1 = (dimSize > 2) ? interpAttrs.padBegin[1] : 0;
            int padB2 = (dimSize == 5) ? interpAttrs.padBegin[dimSize - 3] : 0;
            int padB3 = interpAttrs.padBegin[dimSize - 2];
            int padB4 = interpAttrs.padBegin[dimSize - 1];

            VectorDims inShapeBlock = getBlockND(srcDim5d);
            VectorDims inShapePadBlock = getBlockND(srcDimPad5d);

            if (interpAttrs.layout == InterpolateLayoutType::planar) {
                srcPadded.resize(inShapePadBlock[0] * srcDataSize, 0);
                auto* src_data_pad = static_cast<uint8_t*>(&srcPadded[0]);
                parallel_for4d(srcDim5d[0], srcDim5d[1], srcDim5d[2], srcDim5d[3], [&](int n, int c, int d, int h) {
                    const uint8_t* src = src_data_origin + (inShapeBlock[1] * n + inShapeBlock[2] * c +
                                                            inShapeBlock[3] * d + inShapeBlock[4] * h) *
                                                               srcDataSize;
                    uint8_t* srcPad =
                        src_data_pad + (inShapePadBlock[1] * (n + padB0) + inShapePadBlock[2] * (c + padB1) +
                                        inShapePadBlock[3] * (d + padB2) + inShapePadBlock[4] * (h + padB3) + padB4) *
                                           srcDataSize;
                    cpu_memcpy(srcPad, src, srcDim5d[4] * srcDataSize);
                });
                src_data = src_data_pad;
            } else if (interpAttrs.layout == InterpolateLayoutType::by_channel) {
                srcPadded.resize(inShapePadBlock[0] * srcDataSize, 0);
                auto* src_data_pad = static_cast<uint8_t*>(&srcPadded[0]);
                parallel_for4d(srcDim5d[0], srcDim5d[2], srcDim5d[3], srcDim5d[4], [&](int n, int d, int h, int w) {
                    const uint8_t* src =
                        src_data_origin +
                        (inShapeBlock[1] * n +
                         (inShapeBlock[3] * d + inShapeBlock[4] * h + inShapeBlock[5] * w) * srcDim5d[1]) *
                            srcDataSize;
                    uint8_t* srcPad =
                        src_data_pad + (inShapePadBlock[1] * (n + padB0) +
                                        (inShapePadBlock[3] * (d + padB2) + inShapePadBlock[4] * (h + padB3) +
                                         inShapePadBlock[5] * (w + padB4)) *
                                            srcDimPad5d[1] +
                                        padB1) *
                                           srcDataSize;
                    cpu_memcpy(srcPad, src, srcDim5d[1] * srcDataSize);
                });
                src_data = src_data_pad;
            } else if (interpAttrs.layout == InterpolateLayoutType::block) {
                size_t blkSize = mayiuse(cpu::x64::avx512_core) ? 16 : 8;
                size_t CB = div_up(srcDimPad5d[1], blkSize);
                size_t eltsTotal = srcDimPad5d[0] * CB * srcDimPad5d[2] * srcDimPad5d[3] * srcDimPad5d[4] * blkSize;
                srcPadded.resize(eltsTotal * srcDataSize, 0x0);
                auto* src_data_pad = static_cast<uint8_t*>(&srcPadded[0]);
                if ((srcDim5d[0] != srcDimPad5d[0]) || (srcDim5d[1] != srcDimPad5d[1])) {
                    THROW_CPU_NODE_ERR("does not support padding on batch and channel dimensions");
                }
                parallel_for5d(srcDim5d[0],
                               CB,
                               srcDim5d[2],
                               srcDim5d[3],
                               srcDim5d[4],
                               [&](int n, int cb, int d, int h, int w) {
                                   const uint8_t* src =
                                       src_data_origin +
                                       (n * CB * srcDim5d[2] * srcDim5d[3] * srcDim5d[4] * blkSize) * srcDataSize +
                                       (cb * srcDim5d[2] * srcDim5d[3] * srcDim5d[4] * blkSize) * srcDataSize +
                                       (d * srcDim5d[3] * srcDim5d[4] * blkSize) * srcDataSize +
                                       (h * srcDim5d[4] * blkSize) * srcDataSize + (w * blkSize) * srcDataSize;
                                   uint8_t* srcPad =
                                       src_data_pad +
                                       (n * CB * srcDimPad5d[2] * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) *
                                           srcDataSize +
                                       (cb * srcDimPad5d[2] * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) * srcDataSize +
                                       ((d + padB2) * srcDimPad5d[3] * srcDimPad5d[4] * blkSize) * srcDataSize +
                                       ((h + padB3) * srcDimPad5d[4] * blkSize) * srcDataSize +
                                       ((w + padB4) * blkSize) * srcDataSize;
                                   cpu_memcpy(srcPad, src, blkSize * srcDataSize);
                               });
                src_data = src_data_pad;
            }
        } else {
            src_data = src_data_origin;
        }

        execPtr->exec(src_data, dst_data, reinterpret_cast<void*>(postOpsDataPtrs.data()));
    } else if (aclExecPtr) {
        aclExecPtr->exec({srcMemPtr}, {dstMemPtr}, reinterpret_cast<void*>(postOpsDataPtrs.data()));
    } else {
        THROW_CPU_NODE_ERR("Primitive wasn't created");
    }
}

bool Interpolate::canFuse(const NodePtr& node) const {
    if (!mayiuse(cpu::x64::sse41) || interpAttrs.mode == InterpolateMode::linear ||
        interpAttrs.mode == InterpolateMode::bilinear_pillow || interpAttrs.mode == InterpolateMode::bicubic_pillow ||
        (!one_of(dataRank, 4u, 5u) && !mayiuse(cpu::x64::avx2))) {
        return false;
    }

    return canFuseSimpleOperation(node);
}

bool Interpolate::created() const {
    return getType() == Type::Interpolate;
}

}  // namespace ov::intel_cpu::node
