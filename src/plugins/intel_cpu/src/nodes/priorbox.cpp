// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "priorbox.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/prior_box.hpp"
#include "shape_inference/custom/priorbox.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {
namespace {
float clip_great(float x, float threshold) {
    return x < threshold ? x : threshold;
}

float clip_less(float x, float threshold) {
    return x > threshold ? x : threshold;
}

}  // namespace

bool PriorBox::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto priorBox = ov::as_type_ptr<const ov::op::v0::PriorBox>(op);
        if (!priorBox) {
            errorMessage = "Only opset1 PriorBox operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

PriorBox::PriorBox(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, PriorBoxShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto priorBox = ov::as_type_ptr<const ov::op::v0::PriorBox>(op);
    const ov::op::v0::PriorBox::Attributes& attrs = priorBox->get_attrs();
    offset = attrs.offset;
    step = attrs.step;
    min_size = attrs.min_size;
    max_size = attrs.max_size;
    flip = attrs.flip;
    clip = attrs.clip;
    scale_all_sizes = attrs.scale_all_sizes;
    fixed_size = attrs.fixed_size;
    fixed_ratio = attrs.fixed_ratio;
    density = attrs.density;

    bool exist = false;
    aspect_ratio.push_back(1.0F);
    for (float aspect_ratio_item : attrs.aspect_ratio) {
        exist = false;

        CPU_NODE_ASSERT(std::fabs(aspect_ratio_item) >= std::numeric_limits<float>::epsilon(),
                        "has aspect_ratio param can't be equal to zero");

        for (float _aspect_ratio : aspect_ratio) {
            if (std::fabs(aspect_ratio_item - _aspect_ratio) < 1e-6) {
                exist = true;
                break;
            }
        }

        if (exist) {
            continue;
        }

        aspect_ratio.push_back(aspect_ratio_item);
        if (flip) {
            aspect_ratio.push_back(1.0F / aspect_ratio_item);
        }
    }

    number_of_priors = static_cast<int>(ov::op::v0::PriorBox::number_of_priors(attrs));

    if (any_of(attrs.variance.size(), 1U, 4U)) {
        for (float i : attrs.variance) {
            CPU_NODE_ASSERT(i >= 0, "variance must be > 0.");

            variance.push_back(i);
        }
    } else if (attrs.variance.empty()) {
        variance.push_back(0.1F);
    } else {
        CPU_NODE_THROW("has wrong number of variance values. Not less than 1 and more than 4 variance values.");
    }
}

bool PriorBox::needShapeInfer() const {
    auto memory = getDstMemoryAtPort(0);
    if (memory->getShape().isDynamic()) {
        return true;
    }

    const auto& outputShape = memory->getShape().getStaticDims();
    const int* in_data = memory->getDataAs<int>();
    const int h = in_data[0];
    const int w = in_data[1];
    const auto output = static_cast<size_t>(4) * h * w * number_of_priors;

    return outputShape[1] != output;
}

bool PriorBox::needPrepareParams() const {
    return false;
}

void PriorBox::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, ov::element::i32}, {LayoutType::ncsp, ov::element::i32}},
                         {{LayoutType::ncsp, ov::element::f32}},
                         impl_desc_type::ref_any);
}

void PriorBox::createPrimitive() {
    if (inputShapesDefined()) {
        if (needPrepareParams()) {
            prepareParams();
        }
        updateLastInputDims();
    }
}

void PriorBox::execute([[maybe_unused]] const dnnl::stream& strm) {
    const int* in_data = getSrcDataAtPortAs<int>(0);
    const int H = in_data[0];
    const int W = in_data[1];

    const int* in_image = getSrcDataAtPortAs<int>(1);
    const int IH = in_image[0];
    const int IW = in_image[1];

    const int OH = 4 * H * W * number_of_priors;
    const int OW = 1;

    auto* dst_data = getDstDataAtPortAs<float>(0);

    float step_ = step;
    auto min_size_ = min_size;
    if (!scale_all_sizes) {
        // mxnet-like PriorBox
        if (step_ == -1) {
            step_ = 1.F * static_cast<float>(IH) / static_cast<float>(H);
        } else {
            step_ *= static_cast<float>(IH);
        }
        for (auto& size : min_size_) {
            size *= static_cast<float>(IH);
        }
    }

    int64_t idx = 0;
    float center_x = NAN;
    float center_y = NAN;
    float box_width = NAN;
    float box_height = NAN;
    float step_x = NAN;
    float step_y = NAN;
    float IWI = 1.0F / static_cast<float>(IW);
    float IHI = 1.0F / static_cast<float>(IH);

    if (step_ == 0) {
        step_x = static_cast<float>(IW) / static_cast<float>(W);
        step_y = static_cast<float>(IH) / static_cast<float>(H);
    } else {
        step_x = step_;
        step_y = step_;
    }

    auto calculate_data =
        [&dst_data, &IWI, &IHI, &idx](float center_x, float center_y, float box_width, float box_height, bool clip) {
            if (clip) {
                // order: xmin, ymin, xmax, ymax
                dst_data[idx++] = clip_less((center_x - box_width) * IWI, 0);
                dst_data[idx++] = clip_less((center_y - box_height) * IHI, 0);
                dst_data[idx++] = clip_great((center_x + box_width) * IWI, 1);
                dst_data[idx++] = clip_great((center_y + box_height) * IHI, 1);
            } else {
                dst_data[idx++] = (center_x - box_width) * IWI;
                dst_data[idx++] = (center_y - box_height) * IHI;
                dst_data[idx++] = (center_x + box_width) * IWI;
                dst_data[idx++] = (center_y + box_height) * IHI;
            }
        };

    for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
            if (step_ == 0) {
                center_x = (static_cast<float>(w) + 0.5F) * step_x;
                center_y = (static_cast<float>(h) + 0.5F) * step_y;
            } else {
                center_x = (offset + static_cast<float>(w)) * step_;
                center_y = (offset + static_cast<float>(h)) * step_;
            }

            for (size_t s = 0; s < fixed_size.size(); ++s) {
                auto fixed_size_ = static_cast<size_t>(fixed_size[s]);
                box_width = box_height = fixed_size_ * 0.5F;

                if (!fixed_ratio.empty()) {
                    for (float ar : fixed_ratio) {
                        auto density_ = static_cast<int64_t>(density[s]);
                        auto shift = static_cast<int64_t>(fixed_size[s] / static_cast<float>(density_));
                        ar = std::sqrt(ar);
                        float box_width_ratio = fixed_size[s] * 0.5F * ar;
                        float box_height_ratio = fixed_size[s] * 0.5F / ar;
                        for (int64_t r = 0; r < density_; ++r) {
                            for (int64_t c = 0; c < density_; ++c) {
                                float center_x_temp = center_x - static_cast<float>(fixed_size_) / 2.F +
                                                      static_cast<float>(shift) / 2.F +
                                                      static_cast<float>(c) * static_cast<float>(shift);
                                float center_y_temp = center_y - static_cast<float>(fixed_size_) / 2.F +
                                                      static_cast<float>(shift) / 2.F +
                                                      static_cast<float>(r) * static_cast<float>(shift);
                                calculate_data(center_x_temp, center_y_temp, box_width_ratio, box_height_ratio, true);
                            }
                        }
                    }
                } else {
                    if (!density.empty()) {
                        auto density_ = static_cast<int64_t>(density[s]);
                        auto shift = static_cast<int64_t>(fixed_size[s] / static_cast<float>(density_));
                        for (int64_t r = 0; r < density_; ++r) {
                            for (int64_t c = 0; c < density_; ++c) {
                                float center_x_temp = center_x - static_cast<float>(fixed_size_) / 2.F +
                                                      static_cast<float>(shift) / 2.F +
                                                      static_cast<float>(c) * static_cast<float>(shift);
                                float center_y_temp = center_y - static_cast<float>(fixed_size_) / 2.F +
                                                      static_cast<float>(shift) / 2.F +
                                                      static_cast<float>(r) * static_cast<float>(shift);
                                calculate_data(center_x_temp, center_y_temp, box_width, box_height, true);
                            }
                        }
                    }
                    //  Rest of priors
                    for (float ar : aspect_ratio) {
                        if (fabs(ar - 1.) < 1e-6) {
                            continue;
                        }

                        auto density_ = static_cast<int64_t>(density[s]);
                        auto shift = static_cast<int64_t>(fixed_size[s] / static_cast<float>(density_));
                        ar = std::sqrt(ar);
                        float box_width_ratio = fixed_size[s] * 0.5F * ar;
                        float box_height_ratio = fixed_size[s] * 0.5F / ar;
                        for (int64_t r = 0; r < density_; ++r) {
                            for (int64_t c = 0; c < density_; ++c) {
                                float center_x_temp = center_x - static_cast<float>(fixed_size_) / 2.F +
                                                      static_cast<float>(shift) / 2.F +
                                                      static_cast<float>(c) * static_cast<float>(shift);
                                float center_y_temp = center_y - static_cast<float>(fixed_size_) / 2.F +
                                                      static_cast<float>(shift) / 2.F +
                                                      static_cast<float>(r) * static_cast<float>(shift);
                                calculate_data(center_x_temp, center_y_temp, box_width_ratio, box_height_ratio, true);
                            }
                        }
                    }
                }
            }

            for (size_t ms_idx = 0; ms_idx < min_size_.size(); ms_idx++) {
                box_width = min_size_[ms_idx] * 0.5F;
                box_height = min_size_[ms_idx] * 0.5F;
                calculate_data(center_x, center_y, box_width, box_height, false);

                if (max_size.size() > ms_idx) {
                    box_width = box_height = std::sqrt(min_size_[ms_idx] * max_size[ms_idx]) * 0.5F;
                    calculate_data(center_x, center_y, box_width, box_height, false);
                }

                if (scale_all_sizes || (!scale_all_sizes && (ms_idx == min_size_.size() - 1))) {
                    size_t s_idx = scale_all_sizes ? ms_idx : 0;
                    for (float ar : aspect_ratio) {
                        if (std::fabs(ar - 1.0F) < 1e-6) {
                            continue;
                        }

                        ar = std::sqrt(ar);
                        box_width = min_size_[s_idx] * 0.5F * ar;
                        box_height = min_size_[s_idx] * 0.5F / ar;
                        calculate_data(center_x, center_y, box_width, box_height, false);
                    }
                }
            }
        }
    }

    if (clip) {
        parallel_for((H * W * number_of_priors * 4), [&](size_t i) {
            dst_data[i] = (std::min)((std::max)(dst_data[i], 0.0F), 1.0F);
        });
    }

    uint64_t channel_size = OH * OW;
    if (variance.size() == 1) {
        parallel_for(channel_size, [&](size_t i) {
            dst_data[i + channel_size] = variance[0];
        });
    } else {
        parallel_for(H * W * number_of_priors, [&](size_t i) {
            for (size_t j = 0; j < 4; ++j) {
                dst_data[i * 4 + j + channel_size] = variance[j];
            }
        });
    }
}

bool PriorBox::created() const {
    return getType() == Type::PriorBox;
}

}  // namespace ov::intel_cpu::node
