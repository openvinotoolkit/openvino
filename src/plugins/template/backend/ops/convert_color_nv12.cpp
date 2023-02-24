// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/convert_color_nv12.hpp"

#include "evaluates_map.hpp"
#include "openvino/op/i420_to_bgr.hpp"
#include "openvino/op/i420_to_rgb.hpp"
#include "openvino/op/nv12_to_bgr.hpp"
#include "openvino/op/nv12_to_rgb.hpp"

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ov::op::v8::NV12toRGB>& op,
                     const ov::HostTensorVector& outputs,
                     const ov::HostTensorVector& inputs) {
    return ngraph::runtime::reference::color_convert_nv12<ET>(
        op,
        outputs,
        inputs,
        ov::op::util::ConvertColorNV12Base::ColorConversion::NV12_TO_RGB);
}

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ov::op::v8::NV12toBGR>& op,
                     const ov::HostTensorVector& outputs,
                     const ov::HostTensorVector& inputs) {
    return ngraph::runtime::reference::color_convert_nv12<ET>(
        op,
        outputs,
        inputs,
        ov::op::util::ConvertColorNV12Base::ColorConversion::NV12_TO_BGR);
}

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ov::op::v8::I420toRGB>& op,
                     const ov::HostTensorVector& outputs,
                     const ov::HostTensorVector& inputs) {
    return ngraph::runtime::reference::color_convert_i420<ET>(
        op,
        outputs,
        inputs,
        ov::op::util::ConvertColorI420Base::ColorConversion::I420_TO_RGB);
}

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ov::op::v8::I420toBGR>& op,
                     const ov::HostTensorVector& outputs,
                     const ov::HostTensorVector& inputs) {
    return ngraph::runtime::reference::color_convert_i420<ET>(
        op,
        outputs,
        inputs,
        ov::op::util::ConvertColorI420Base::ColorConversion::I420_TO_BGR);
}
