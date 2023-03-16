// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ngraph::op::v8::NV12toRGB>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    return ngraph::runtime::reference::color_convert_nv12<ET>(
        op,
        outputs,
        inputs,
        ov::op::util::ConvertColorNV12Base::ColorConversion::NV12_TO_RGB);
}

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ngraph::op::v8::NV12toBGR>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    return ngraph::runtime::reference::color_convert_nv12<ET>(
        op,
        outputs,
        inputs,
        ov::op::util::ConvertColorNV12Base::ColorConversion::NV12_TO_BGR);
}

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ngraph::op::v8::I420toRGB>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    return ngraph::runtime::reference::color_convert_i420<ET>(
        op,
        outputs,
        inputs,
        ov::op::util::ConvertColorI420Base::ColorConversion::I420_TO_RGB);
}

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ngraph::op::v8::I420toBGR>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    return ngraph::runtime::reference::color_convert_i420<ET>(
        op,
        outputs,
        inputs,
        ov::op::util::ConvertColorI420Base::ColorConversion::I420_TO_BGR);
}