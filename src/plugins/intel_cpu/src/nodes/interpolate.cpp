// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate.h"
#include "executors/interpolate_config.hpp"
#include "executors/x64/interpolate.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/cpu_memcpy.h"
#include "cpu/x64/injectors/jit_uni_depthwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_quantization_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_eltwise.hpp"
#include "dnnl_extension_utils.h"
#include "eltwise.h"
#include "emitters/plugin/x64/jit_bf16_emitters.hpp"
#include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#include "fake_quantize.h"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "openvino/opsets/opset11_decl.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/opsets/opset4_decl.hpp"
#include "shape_inference/shape_inference.hpp"
#include "shape_inference/static_shape.hpp"
#include "utils/bfloat16.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/ngraph_utils.hpp"

using namespace dnnl;

using namespace dnnl::impl;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace ov::intel_cpu::node {

static inline bool isFloatCompatible(ov::element::Type prc) {
    return one_of(prc, ov::element::f32, ov::element::bf16, ov::element::f16, ov::element::f64);
}

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
        if (const auto interp = ov::as_type_ptr<const ov::op::v4::Interpolate>(op)) {
            const auto& interpAttr = interp->get_attrs();
            const auto& interpMode = interpAttr.mode;
            if (!one_of(interpMode,
                        ngInterpMode::NEAREST,
                        ngInterpMode::LINEAR,
                        ngInterpMode::LINEAR_ONNX,
                        ngInterpMode::CUBIC)) {
                errorMessage = "Interpolate-4 does not support interpolate mode: " + ov::as_string(interpMode);
                return false;
            }

            const auto& interpCoordTransMode = interpAttr.coordinate_transformation_mode;
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
                const auto& interpNearestMode = interpAttr.nearest_mode;
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

            const auto& interpShapeCalcMode = interpAttr.shape_calculation_mode;
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
        } else if (const auto interp = ov::as_type_ptr<const ov::op::v11::Interpolate>(op)) {
            const auto& interpAttr = interp->get_attrs();
            const auto& interpMode = interpAttr.mode;
            if (!one_of(interpMode, ngInterpMode::BILINEAR_PILLOW, ngInterpMode::BICUBIC_PILLOW)) {
                errorMessage = "Interpolate-11 does not support interpolate mode: " + ov::as_string(interpMode);
                return false;
            }
            const auto& interpShapeCalcMode = interpAttr.shape_calculation_mode;
            if (!one_of(interpShapeCalcMode, ngInterpShapeCalcMode::SCALES, ngInterpShapeCalcMode::SIZES)) {
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
            errorMessage = "Only opset4 and opset11 interpolate operation are supported";
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
        if (auto interp4 = ov::as_type_ptr<ov::opset4::Interpolate>(m_op)) {
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
        dataRank = getInputShapeAtPort(DATA_ID).getRank();
        if (const auto interp = ov::as_type_ptr<const ov::opset4::Interpolate>(op)) {
            is_version11 = false;
            const auto numInputs = inputShapes.size();
            if (numInputs != 3 && numInputs != 4) {
                THROW_CPU_NODE_ERR("has incorrect number of input edges");
            }
            if (outputShapes.size() != 1) {
                THROW_CPU_NODE_ERR("has incorrect number of output edges");
            }
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
                ov::as_type_ptr<const ov::op::v0::Constant>(interp->get_input_node_shared_ptr(SCALES_ID));
            if (scalesNode) {
                scales = scalesNode->cast_vector<float>();
                isScaleConstant = true;
            }

            if (isAxesSpecified) {
                axes = ov::as_type_ptr<const ov::op::v0::Constant>(interp->get_input_node_shared_ptr(AXES_ID))
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
                    interp->get_input_node_shared_ptr(SIZE_OR_SCALE_ID_V11));
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

    ov::element::Type inputPrecision = getOriginalInputPrecisionAtPort(DATA_ID);

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
        outputPrecision = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(DATA_ID);
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
    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto pushDesc = [&](LayoutType dataFormat,
                        impl_desc_type implDetail,
                        bool is_version11,
                        bool useAclExecutor = false) {
        config.inConfs[DATA_ID].setMemDesc(
            creatorsMap.at(dataFormat)->createSharedDesc(inputPrecision, getInputShapeAtPort(DATA_ID)));
        if (is_version11) {
            if (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::sizes) {
                config.inConfs[SIZE_OR_SCALE_ID_V11].setMemDesc(
                    creatorsMap.at(LayoutType::ncsp)
                        ->createSharedDesc(targetShapeType, getInputShapeAtPort(SIZE_OR_SCALE_ID_V11)));
            } else {
                config.inConfs[SIZE_OR_SCALE_ID_V11].setMemDesc(
                    creatorsMap.at(LayoutType::ncsp)
                        ->createSharedDesc(scalesType, getInputShapeAtPort(SIZE_OR_SCALE_ID_V11)));
            }

            if (isAxesSpecified) {
                config.inConfs[AXES_ID_V11].setMemDesc(
                    creatorsMap.at(LayoutType::ncsp)->createSharedDesc(axesType, getInputShapeAtPort(AXES_ID_V11)));
            }
        } else {
            config.inConfs[TARGET_SHAPE_ID].setMemDesc(
                creatorsMap.at(LayoutType::ncsp)
                    ->createSharedDesc(targetShapeType, getInputShapeAtPort(TARGET_SHAPE_ID)));
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
        if (canUseAclExecutor)
            return;
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
        const auto& dataMinDims = getInputShapeAtPort(DATA_ID).getMinDims();
        bool isBlkApplied = dataRank > 1 && dataMinDims[1] != Shape::UNDEFINED_DIM && dataMinDims[1] > 1;

#if defined(OV_CPU_WITH_ACL)
        interpAttrs.hasPad = hasPad;
        pushDesc(LayoutType::nspc, undef, false, true);
        pushDesc(LayoutType::ncsp, undef, false, true);
        canUseAclExecutor = !supportedPrimitiveDescriptors.empty();
        if (canUseAclExecutor)
            return;
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
    if (!shapesDefined()) {
        THROW_CPU_NODE_ERR("input/output dims aren't defined");
    }

    auto dstMemPtr = getDstMemoryAtPort(0);
    if (!dstMemPtr || !dstMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("has undefined destination memory");
    }

    auto srcMemPtr = getSrcMemoryAtPort(DATA_ID);
    if (!srcMemPtr || !srcMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("has undefined input memory");
    }

    if (interpAttrs.shapeCalcMode == InterpolateShapeCalcMode::sizes) {
        auto tsMemPtr = getSrcMemoryAtPort(TARGET_SHAPE_ID);
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

        auto selectedPD = getSelectedPrimitiveDescriptor();
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
    auto srcMemPtr = getSrcMemoryAtPort(DATA_ID);
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
    auto srcMemPtr = getSrcMemoryAtPort(DATA_ID);

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

void InterpolateRefExecutor::NNRef(const uint8_t* in_ptr_,
                                                uint8_t* out_ptr_,
                                                int B,
                                                int C,
                                                int ID,
                                                int IH,
                                                int IW,
                                                int OD,
                                                int OH,
                                                int OW) {
    auto* index_d = static_cast<int*>(&auxTable[0]);
    auto* index_h = static_cast<int*>(&auxTable[OD]);
    auto* index_w = static_cast<int*>(&auxTable[OD + OH]);

    const auto* in_ptr_f32 = reinterpret_cast<const float*>(in_ptr_);
    auto* out_ptr_f32 = reinterpret_cast<float*>(out_ptr_);

    parallel_for3d(B, C, OD, [&](size_t b, size_t c, size_t od) {
        const float* in_ptr = in_ptr_f32 + (IW * IH * ID * C * b + IW * IH * ID * c + IW * IH * index_d[od]);
        float* out_ptr = out_ptr_f32 + (OW * OH * OD * C * b + OW * OH * OD * c + OW * OH * od);
        for (int oh = 0; oh < OH; oh++) {
            const float* in_ptr_h = in_ptr + (IW * index_h[oh]);
            float* out_ptr_h = out_ptr + (OW * oh);
            for (int ow = 0; ow < OW; ow++) {
                out_ptr_h[ow] = in_ptr_h[index_w[ow]];
            }
        }
    });
}

void InterpolateRefExecutor::linearOnnxRef(const uint8_t* in_ptr_,
                                                        uint8_t* out_ptr_,
                                                        int B,
                                                        int C,
                                                        int ID,
                                                        int IH,
                                                        int IW,
                                                        int OD,
                                                        int OH,
                                                        int OW) {
    std::vector<int*> indexPtr(MAX_INPUT_INTERPOLATE, nullptr);
    std::vector<float*> weightPtr(MAX_INPUT_INTERPOLATE, nullptr);
    // FrontTopLeft:0, FrontTopRight:1, FrontBottomLeft:2, FrontBottomRight:3,
    // EndTopLeft:4,   EndTopRight:5,   EndBottomLeft:6,   EndBottomRight:7
    // weight: Left:0, ritht:1, top:2, bottom:3, front:4, end:5

    int eltInGrid = (spatialDimSize > 2) ? MAX_INPUT_INTERPOLATE : ((spatialDimSize > 1) ? 4 : 2);
    int scratchLen = rnd_up(eltInGrid * OW * OH * OD, 16);

    indexPtr[0] = static_cast<int*>(&auxTable[0]);
    indexPtr[1] = static_cast<int*>(&auxTable[OW * OH * OD]);
    weightPtr[0] = reinterpret_cast<float*>(&auxTable[scratchLen]);
    weightPtr[1] = reinterpret_cast<float*>(&auxTable[scratchLen + OW * OH * OD]);
    if (spatialDimSize > 1) {
        indexPtr[2] = static_cast<int*>(&auxTable[2 * OW * OH * OD]);
        indexPtr[3] = static_cast<int*>(&auxTable[3 * OW * OH * OD]);
        weightPtr[2] = reinterpret_cast<float*>(&auxTable[scratchLen + 2 * OW * OH * OD]);
        weightPtr[3] = reinterpret_cast<float*>(&auxTable[scratchLen + 3 * OW * OH * OD]);
    }
    if (spatialDimSize > 2) {
        indexPtr[4] = static_cast<int*>(&auxTable[4 * OW * OH * OD]);
        indexPtr[5] = static_cast<int*>(&auxTable[5 * OW * OH * OD]);
        indexPtr[6] = static_cast<int*>(&auxTable[6 * OW * OH * OD]);
        indexPtr[7] = static_cast<int*>(&auxTable[7 * OW * OH * OD]);
        weightPtr[4] = reinterpret_cast<float*>(&auxTable[scratchLen + 4 * OW * OH * OD]);
        weightPtr[5] = reinterpret_cast<float*>(&auxTable[scratchLen + 5 * OW * OH * OD]);
    }

    const auto* in_ptr_f32 = reinterpret_cast<const float*>(in_ptr_);
    auto* out_ptr_f32 = reinterpret_cast<float*>(out_ptr_);

    parallel_for2d(B, C, [&](size_t b, size_t c) {
        float* out_ptr_nc = out_ptr_f32 + (OD * OH * OW * C * b + OD * OH * OW * c);
        const float* in_ptr_nc = in_ptr_f32 + (ID * IH * IW * C * b + ID * IH * IW * c);
        // do not combined 1d/2d to 3d unified process to get rid of invalid computing.
        switch (spatialDimSize) {
        case 1:
            for (int i = 0; i < OW; i++) {
                float src0 = in_ptr_nc[indexPtr[0][i]];
                float src1 = in_ptr_nc[indexPtr[1][i]];

                out_ptr_nc[i] = src0 * weightPtr[0][i] + src1 * weightPtr[1][i];
            }
            break;
        case 2:
            for (int i = 0; i < OH * OW; i++) {
                float src00 = in_ptr_nc[indexPtr[0][i]];
                float src01 = in_ptr_nc[indexPtr[1][i]];
                float src10 = in_ptr_nc[indexPtr[2][i]];
                float src11 = in_ptr_nc[indexPtr[3][i]];

                out_ptr_nc[i] = src00 * weightPtr[2][i] * weightPtr[0][i] + src01 * weightPtr[2][i] * weightPtr[1][i] +
                                src10 * weightPtr[3][i] * weightPtr[0][i] + src11 * weightPtr[3][i] * weightPtr[1][i];
            }
            break;
        case 3:
            for (int i = 0; i < OD * OH * OW; i++) {
                float src000 = in_ptr_nc[indexPtr[0][i]];
                float src001 = in_ptr_nc[indexPtr[1][i]];
                float src010 = in_ptr_nc[indexPtr[2][i]];
                float src011 = in_ptr_nc[indexPtr[3][i]];
                float src100 = in_ptr_nc[indexPtr[4][i]];
                float src101 = in_ptr_nc[indexPtr[5][i]];
                float src110 = in_ptr_nc[indexPtr[6][i]];
                float src111 = in_ptr_nc[indexPtr[7][i]];

                // float dstValue =
                // weightPtr[4][i] * weightPtr[2][i] * weightPtr[0][i] * src000 +
                // weightPtr[4][i] * weightPtr[2][i] * weightPtr[1][i] * src001 +
                // weightPtr[4][i] * weightPtr[3][i] * weightPtr[0][i] * src010 +
                // weightPtr[4][i] * weightPtr[3][i] * weightPtr[1][i] * src011 +
                // weightPtr[5][i] * weightPtr[2][i] * weightPtr[0][i] * src100 +
                // weightPtr[5][i] * weightPtr[2][i] * weightPtr[1][i] * src101 +
                // weightPtr[5][i] * weightPtr[3][i] * weightPtr[0][i] * src110 +
                // weightPtr[5][i] * weightPtr[3][i] * weightPtr[1][i] * src111;

                out_ptr_nc[i] =
                    weightPtr[4][i] * (weightPtr[2][i] * (weightPtr[0][i] * src000 + weightPtr[1][i] * src001) +
                                       weightPtr[3][i] * (weightPtr[0][i] * src010 + weightPtr[1][i] * src011)) +
                    weightPtr[5][i] * (weightPtr[2][i] * (weightPtr[0][i] * src100 + weightPtr[1][i] * src101) +
                                       weightPtr[3][i] * (weightPtr[0][i] * src110 + weightPtr[1][i] * src111));
            }
            break;
        default:
            break;
        }
    });
}

void InterpolateRefExecutor::cubicRef(const uint8_t* in_ptr_,
                                                   uint8_t* out_ptr_,
                                                   int B,
                                                   int C,
                                                   int IH,
                                                   int IW,
                                                   int OH,
                                                   int OW) {
    const int idxNum = 1;
    auto* xOrigin = static_cast<int*>(&auxTable[0]);
    auto* xFactor = reinterpret_cast<float*>(&auxTable[OW]);
    auto* yOrigin = static_cast<int*>(&auxTable[(Interpolate::CUBIC_GRID_LEN + idxNum) * OW]);
    auto* yFactor = reinterpret_cast<float*>(&auxTable[(Interpolate::CUBIC_GRID_LEN + idxNum) * OW + OH]);

    const auto* in_ptr_f32 = reinterpret_cast<const float*>(in_ptr_);
    auto* out_ptr_f32 = reinterpret_cast<float*>(out_ptr_);

    parallel_for4d(B, C, OH, OW, [&](size_t n, size_t c, size_t oy, size_t ox) {
        const float* in_ptr_nc = in_ptr_f32 + (IW * IH * C * n + IW * IH * c);
        float* out_ptr_nc = out_ptr_f32 + (OW * OH * C * n + OW * OH * c);

        int iy = yOrigin[oy];
        int ix = xOrigin[ox];

        float retY = 0.f;
        for (int y = iy - 1, i = 0; y <= iy + 2; y++, i++) {
            int yInRange = std::max(0, std::min(y, IH - 1));
            const float* in_ptr_nch = in_ptr_nc + IW * yInRange;
            float retX = 0.f;
            for (int x = ix - 1, j = 0; x <= ix + 2; x++, j++) {
                int xInRange = std::max(0, std::min(x, IW - 1));
                retX += xFactor[ox * Interpolate::CUBIC_GRID_LEN + j] * in_ptr_nch[xInRange];
            }
            retY += yFactor[oy * Interpolate::CUBIC_GRID_LEN + i] * retX;
        }
        out_ptr_nc[oy * OW + ox] = retY;
    });
}

float InterpolateRefExecutor::getValue(const uint8_t* base, size_t offset, ov::element::Type prec) {
    const uint8_t* baseOffset = base + offset;
    switch (prec) {
    case ov::element::u8: {
        return static_cast<float>(*baseOffset);
        break;
    }
    case ov::element::i8: {
        const auto* valuePtr = reinterpret_cast<const int8_t*>(baseOffset);
        return static_cast<float>(*valuePtr);
        break;
    }
    case ov::element::bf16: {
        const auto* valuePtr = reinterpret_cast<const uint16_t*>(baseOffset);
        return bfloat16_t::from_bits(*valuePtr);
        break;
    }
    case ov::element::f32: {
        const auto* valuePtr = reinterpret_cast<const float*>(baseOffset);
        return *valuePtr;
        break;
    }
    default: {
        OPENVINO_THROW("Interpolate layer does not support precision: ", prec);
        break;
    }
    }
}

void InterpolateRefExecutor::setValue(uint8_t* base, size_t offset, float value, ov::element::Type prec) {
    uint8_t* baseOffset = base + offset;
    switch (prec) {
    case ov::element::u8: {
        auto data = static_cast<uint8_t>(value < 0 ? 0 : value);
        cpu_memcpy(baseOffset, &data, 1);
        break;
    }
    case ov::element::i8: {
        auto data = static_cast<int8_t>(value);
        cpu_memcpy(baseOffset, &data, 1);
        break;
    }
    case ov::element::bf16: {
        uint16_t data = bfloat16_t(value).to_bits();
        cpu_memcpy(baseOffset, &data, 2);
        break;
    }
    case ov::element::f32: {
        cpu_memcpy(baseOffset, &value, sizeof(float));
        break;
    }
    default: {
        OPENVINO_THROW("Interpolate layer does not support precision: ", prec);
        break;
    }
    }
}

void InterpolateRefExecutor::linearInterpolation(const uint8_t* in_ptr_,
                                                              uint8_t* out_ptr_,
                                                              int B,
                                                              int C,
                                                              int ID,
                                                              int IH,
                                                              int IW,
                                                              float fx,
                                                              float fy,
                                                              float fz,
                                                              int OD,
                                                              int OH,
                                                              int OW,
                                                              int kernel_width,
                                                              bool antialias) {
    if (IW == OW && IH == OH && ID == OD) {
        size_t spatialDimSize = IW * IH * ID;
        // TODO: enable when fusing into interp with linear mode will support
        if (/*fusedWith.empty() &&*/ inputPrec == outputPrec) {
            size_t size = B * C * spatialDimSize * srcDataSize;
            cpu_memcpy(out_ptr_, in_ptr_, size);
        } else {
            parallel_for2d(B, C, [&](size_t b, size_t c) {
                const uint8_t* in_ptr_nc = in_ptr_ + (spatialDimSize * C * b + spatialDimSize * c) * srcDataSize;
                uint8_t* out_ptr_nc = out_ptr_ + (spatialDimSize * C * b + spatialDimSize * c) * dstDataSize;
                for (size_t i = 0; i < spatialDimSize; i++) {
                    float dstValue = getValue(in_ptr_nc, i * srcDataSize, inputPrec);
                    setValue(out_ptr_nc, i * dstDataSize, dstValue, outputPrec);
                }
            });
        }
        return;
    }

    float ax = antialias ? fx : 1.0f;
    float ay = antialias ? fy : 1.0f;
    float az = antialias ? fz : 1.0f;

    int rx = (fx > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ax));
    int ry = (fy > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ay));
    int rz = (fz > 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / az));

    int diaOD = 2 * rz + 1;
    int diaOH = 2 * ry + 1;
    int diaOW = 2 * rx + 1;
    int sizeOD = OD * diaOD;
    int sizeOH = OH * diaOH;
    int sizeOW = OW * diaOW;

    auto* weightTable = reinterpret_cast<float*>(&auxTable[0]);
    auto* weightOD = static_cast<float*>(&weightTable[0]);
    auto* weightOH = static_cast<float*>(&weightTable[sizeOD]);
    auto* weightOW = static_cast<float*>(&weightTable[sizeOD + sizeOH]);

    auto* idxTable = static_cast<int*>(&auxTable[sizeOD + sizeOH + sizeOW]);
    auto* idxOD = static_cast<int*>(&idxTable[0]);
    auto* idxOH = static_cast<int*>(&idxTable[sizeOD]);
    auto* idxOW = static_cast<int*>(&idxTable[sizeOD + sizeOH]);

    parallel_for2d(B, C, [&](size_t b, size_t c) {
        const uint8_t* in_ptr_nc = in_ptr_ + (IW * IH * ID * C * b + IW * IH * ID * c) * srcDataSize;
        uint8_t* out_ptr_nc = out_ptr_ + (OW * OH * OD * C * b + OW * OH * OD * c) * dstDataSize;
        for (int oz = 0; oz < OD; oz++) {
            uint8_t* out_ptr_ncd = out_ptr_nc + (OW * OH * oz) * dstDataSize;
            for (int oy = 0; oy < OH; oy++) {
                uint8_t* out_ptr_ncdh = out_ptr_ncd + (OW * oy) * dstDataSize;
                for (int ox = 0; ox < OW; ox++) {
                    float sum = 0.f;
                    float wsum = 0.f;

                    // this comment explains the original algo.
                    // for (int z = iz_r - rz; z <= iz_r + rz; z++) {
                    //    for (int y = iy_r - ry; y <= iy_r + ry; y++) {
                    //        for (int x = ix_r - rx; x <= ix_r + rx; x++) {
                    //            bool is_continue =  z < 0                     ||
                    //                                y < 0                     ||
                    //                                x < 0                     ||
                    //                                z >= static_cast<int>(ID) ||
                    //                                y >= static_cast<int>(IH) ||
                    //                                x >= static_cast<int>(IW);
                    //            if (is_continue)
                    //                continue;

                    //            float dx = ix - x;
                    //            float dy = iy - y;
                    //            float dz = iz - z;

                    //            float w = ax * triangleCoeff(ax * dx) *
                    //                      ay * triangleCoeff(ay * dy) *
                    //                      az * triangleCoeff(az * dz);

                    //            sum += w * getValue(in_ptr_nc, (z * IH * IW + y * IW + x) * srcDataSize, inputPrec);
                    //            wsum += w;
                    //        }
                    //    }
                    //}

                    for (int iz = 0; iz < diaOD; iz++) {
                        if (weightOD[oz * diaOD + iz] == 0.f) {
                            continue;
                        }
                        for (int iy = 0; iy < diaOH; iy++) {
                            if (weightOH[oy * diaOH + iy] == 0.f) {
                                continue;
                            }
                            for (int ix = 0; ix < diaOW; ix++) {
                                if (weightOW[ox * diaOW + ix] == 0.f) {
                                    continue;
                                }
                                float w =
                                    weightOD[oz * diaOD + iz] * weightOH[oy * diaOH + iy] * weightOW[ox * diaOW + ix];
                                float value = getValue(in_ptr_nc,
                                                       (idxOD[oz * diaOD + iz] * IH * IW + idxOH[oy * diaOH + iy] * IW +
                                                        idxOW[ox * diaOW + ix]) *
                                                           srcDataSize,
                                                       inputPrec);

                                sum += w * value;
                                wsum += w;
                            }
                        }
                    }

                    if (wsum == 0.0f) {
                        setValue(out_ptr_ncdh, ox * dstDataSize, 0.f, outputPrec);
                    } else {
                        float dst_value = sum / wsum;
                        setValue(out_ptr_ncdh, ox * dstDataSize, dst_value, outputPrec);
                    }
                }
            }
        }
    });
}

void InterpolateRefExecutor::pillowRef(const uint8_t* in_ptr_,
                                                    uint8_t* out_ptr_,
                                                    int B,
                                                    int C,
                                                    int IH,
                                                    int IW,
                                                    int OH,
                                                    int OW) {
    size_t offset = 0;
    int filterLenX = auxTable[offset];
    int filterLenY = auxTable[offset + 1];
    offset += 2;
    auto* weightX = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterLenX * OW;
    auto* weightY = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterLenY * OH;
    auto* indexX = static_cast<int*>(&auxTable[offset]);
    offset += 2 * OW;
    auto* indexY = static_cast<int*>(&auxTable[offset]);

    // workBuffer needed when both pass is true
    bool xPass = IW != OW;
    bool yPass = IH != OH;

    // --------    ----
    // |      |    |  |
    // |      |--> |  |
    // |      |    |  |
    // |      |    |  |
    // --------    ----
    //              \|/
    //             ----
    //             |  |
    //             |  |
    //             ----
    auto bc_loop = [&](size_t b, size_t c) {
        const uint8_t* in_ptr_nc = in_ptr_ + (IW * IH * C * b + IW * IH * c) * srcDataSize;
        uint8_t* out_ptr_nc = out_ptr_ + (OW * OH * C * b + OW * OH * c) * dstDataSize;
        uint8_t* xpass_out_ptr_nc = nullptr;
        const uint8_t* ypass_in_ptr_nc = nullptr;
        if (xPass && yPass) {
            size_t parallel_num = B * C;
            // IH * OW buf needed
            if (parallel_num < m_threads_num) {
                xpass_out_ptr_nc =
                    static_cast<uint8_t*>(&pillow_working_buf[(OW * IH * C * b + OW * IH * c) * srcDataSize]);
                ypass_in_ptr_nc =
                    static_cast<const uint8_t*>(&pillow_working_buf[(OW * IH * C * b + OW * IH * c) * srcDataSize]);
            } else {
                size_t threadsIdx = parallel_get_thread_num();
                auto buffer_size = static_cast<size_t>(OW) * IH;
                xpass_out_ptr_nc = static_cast<uint8_t*>(&pillow_working_buf[threadsIdx * buffer_size * srcDataSize]);
                ypass_in_ptr_nc =
                    static_cast<const uint8_t*>(&pillow_working_buf[threadsIdx * buffer_size * srcDataSize]);
            }
        } else if (xPass && !yPass) {
            xpass_out_ptr_nc = out_ptr_nc;
        } else if (!xPass && yPass) {
            ypass_in_ptr_nc = in_ptr_nc;
        } else if (!xPass && !yPass) {
            cpu_memcpy(out_ptr_nc, in_ptr_nc, OH * OW * dstDataSize);
        }
        float result;
        int f, filterS, filterL;
        float* weight;
        if (xPass) {
            for (size_t ih = 0; ih < static_cast<size_t>(IH); ih++) {
                for (size_t ow = 0; ow < static_cast<size_t>(OW); ow++) {
                    filterS = indexX[ow * 2];
                    filterL = indexX[ow * 2 + 1];
                    weight = reinterpret_cast<float*>(&weightX[ow * filterLenX]);
                    result = 0.f;
                    for (f = 0; f < filterL; f++) {
                        float pixel = getValue(in_ptr_nc, (ih * IW + f + filterS) * srcDataSize, inputPrec);
                        result += pixel * weight[f];
                    }
                    if (!isFloatCompatible(outputPrec)) {
                        result = static_cast<float>(static_cast<int>(result >= 0.0 ? result + 0.5f : result - 0.5f));
                    }
                    setValue(xpass_out_ptr_nc, (ih * OW + ow) * dstDataSize, result, outputPrec);
                }
            }
        }
        if (yPass) {
            for (size_t oh = 0; oh < static_cast<size_t>(OH); oh++) {
                filterS = indexY[oh * 2];
                filterL = indexY[oh * 2 + 1];
                weight = reinterpret_cast<float*>(&weightY[oh * filterLenY]);
                for (size_t ow = 0; ow < static_cast<size_t>(OW); ow++) {
                    result = 0.f;
                    for (f = 0; f < filterL; f++) {
                        float pixel = getValue(ypass_in_ptr_nc, ((f + filterS) * OW + ow) * srcDataSize, inputPrec);
                        result += pixel * weight[f];
                    }
                    if (!isFloatCompatible(outputPrec)) {
                        result = static_cast<float>(static_cast<int>(result >= 0.0 ? result + 0.5f : result - 0.5f));
                    }
                    setValue(out_ptr_nc, (oh * OW + ow) * dstDataSize, result, outputPrec);
                }
            }
        }
    };

    parallel_nt_static(m_threads_num, [&](const int ithr, const int nthr) {
        for_2d(ithr, nthr, B, C, bc_loop);
    });
}

void InterpolateRefExecutor::pillowRefNCHWAsNHWC(const uint8_t* in_ptr_,
                                                              uint8_t* out_ptr_,
                                                              int B,
                                                              int C,
                                                              int IH,
                                                              int IW,
                                                              int OH,
                                                              int OW) {
    size_t offset = 0;
    int filterLenX = auxTable[offset];
    int filterLenY = auxTable[offset + 1];
    offset += 2;
    auto* weightX = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterLenX * OW;
    auto* weightY = reinterpret_cast<float*>(&auxTable[offset]);
    offset += filterLenY * OH;
    auto* indexX = static_cast<int*>(&auxTable[offset]);
    offset += 2 * OW;
    auto* indexY = static_cast<int*>(&auxTable[offset]);

    bool xPass = IW != OW;
    bool yPass = IH != OH;

    auto b_loop = [&](size_t b) {
        const uint8_t* in_ptr_b = in_ptr_ + b * IH * IW * C * srcDataSize;
        uint8_t* out_ptr_b = out_ptr_ + b * OH * OW * C * dstDataSize;

        uint8_t* xpass_out_ptr_b = nullptr;
        const uint8_t* ypass_in_ptr_b = nullptr;

        if (xPass && yPass) {
            size_t parallel_num = B;
            size_t buffer_size = static_cast<size_t>(IH) * OW * C;
            if (parallel_num < m_threads_num) {
                xpass_out_ptr_b = static_cast<uint8_t*>(&pillow_working_buf[b * buffer_size * srcDataSize]);
            } else {
                size_t threadsIdx = parallel_get_thread_num();
                xpass_out_ptr_b = static_cast<uint8_t*>(&pillow_working_buf[threadsIdx * buffer_size * srcDataSize]);
            }
            ypass_in_ptr_b = static_cast<const uint8_t*>(xpass_out_ptr_b);
        } else if (xPass && !yPass) {
            xpass_out_ptr_b = out_ptr_b;
        } else if (!xPass && yPass) {
            ypass_in_ptr_b = in_ptr_b;
        } else if (!xPass && !yPass) {
            cpu_memcpy(out_ptr_b, in_ptr_b, OH * OW * C * dstDataSize);
        }

        float result;
        int f, filterS, filterL;
        float* weight;

        if (xPass) {
            for (size_t ih = 0; ih < static_cast<size_t>(IH); ih++) {
                for (size_t ow = 0; ow < static_cast<size_t>(OW); ow++) {
                    filterS = indexX[ow * 2];
                    filterL = indexX[ow * 2 + 1];
                    weight = reinterpret_cast<float*>(&weightX[ow * filterLenX]);
                    for (size_t c = 0; c < static_cast<size_t>(C); c++) {
                        result = 0.f;
                        for (f = 0; f < filterL; f++) {
                            float pixel =
                                getValue(in_ptr_b, ((ih * IW + (f + filterS)) * C + c) * srcDataSize, inputPrec);
                            result += pixel * weight[f];
                        }
                        if (!isFloatCompatible(outputPrec)) {
                            result =
                                static_cast<float>(static_cast<int>(result >= 0.0 ? result + 0.5f : result - 0.5f));
                        }
                        setValue(xpass_out_ptr_b, ((ih * OW + ow) * C + c) * dstDataSize, result, outputPrec);
                    }
                }
            }
        }

        if (yPass) {
            for (size_t oh = 0; oh < static_cast<size_t>(OH); oh++) {
                filterS = indexY[oh * 2];
                filterL = indexY[oh * 2 + 1];
                weight = reinterpret_cast<float*>(&weightY[oh * filterLenY]);
                for (size_t ow = 0; ow < static_cast<size_t>(OW); ow++) {
                    for (size_t c = 0; c < static_cast<size_t>(C); c++) {
                        result = 0.f;
                        for (f = 0; f < filterL; f++) {
                            float pixel =
                                getValue(ypass_in_ptr_b, (((f + filterS) * OW + ow) * C + c) * srcDataSize, inputPrec);
                            result += pixel * weight[f];
                        }
                        if (!isFloatCompatible(outputPrec)) {
                            result =
                                static_cast<float>(static_cast<int>(result >= 0.0 ? result + 0.5f : result - 0.5f));
                        }
                        setValue(out_ptr_b, ((oh * OW + ow) * C + c) * dstDataSize, result, outputPrec);
                    }
                }
            }
        }
    };

    parallel_nt_static(m_threads_num, [&](const int ithr, const int nthr) {
        for_1d(ithr, nthr, B, b_loop);
    });
}

void InterpolateRefExecutor::exec(const uint8_t* in_ptr_,
                                               uint8_t* out_ptr_,
                                               [[maybe_unused]] const void* post_ops_data_) {
    size_t N = srcDimPad5d[0], C = srcDimPad5d[1], ID = srcDimPad5d[2], IH = srcDimPad5d[3], IW = srcDimPad5d[4];
    size_t OD = dstDim5d[2], OH = dstDim5d[3], OW = dstDim5d[4];

    switch (mode) {
    case InterpolateMode::nearest: {
        NNRef(in_ptr_, out_ptr_, N, C, ID, IH, IW, OD, OH, OW);
        break;
    }
    case InterpolateMode::linear_onnx: {
        linearOnnxRef(in_ptr_, out_ptr_, N, C, ID, IH, IW, OD, OH, OW);
        break;
    }
    case InterpolateMode::cubic: {
        cubicRef(in_ptr_, out_ptr_, N, C, IH, IW, OH, OW);
        break;
    }
    case InterpolateMode::linear: {
        float fz = (dataRank == 5) ? dataScales[dataRank - 3] : 1.f;
        float fy = dataScales[dataRank - 2];
        float fx = dataScales[dataRank - 1];

        bool isDownsample = (fx < 1.f) || (fy < 1.f) || (fz < 1.f);
        int kernel_width = 2;
        linearInterpolation(in_ptr_,
                            out_ptr_,
                            N,
                            C,
                            ID,
                            IH,
                            IW,
                            fx,
                            fy,
                            fz,
                            OD,
                            OH,
                            OW,
                            kernel_width,
                            isDownsample && antialias);
        break;
    }
    case InterpolateMode::bilinear_pillow:
    case InterpolateMode::bicubic_pillow: {
        if (refInterpAttrs.NCHWAsNHWC) {
            pillowRefNCHWAsNHWC(in_ptr_, out_ptr_, N, C, IH, IW, OH, OW);
        } else {
            pillowRef(in_ptr_, out_ptr_, N, C, IH, IW, OH, OW);
        }
        break;
    }
    default: {
        OPENVINO_THROW("Interpolate layer has unsupported interpolate mode: ", mode);
    }
    }
}

size_t Interpolate::getSpatialDimsNum(const Dim rank) {
    switch (rank) {
    case 1:
    case 3:
        return 1;
    case 2:
    case 4:
        return 2;
    case 5:
        return 3;
    default:
        OPENVINO_THROW("Can't define number spatial");
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
