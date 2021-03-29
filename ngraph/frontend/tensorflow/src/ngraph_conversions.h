/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#ifndef NGRAPH_TF_BRIDGE_CONVERSIONS_H_
#define NGRAPH_TF_BRIDGE_CONVERSIONS_H_
#pragma once

#include <string>

//#include "logging/ngraph_log.h"
#include "default_opset.h"
#include "ngraph_builder.h"

namespace tensorflow {
namespace ngraph_bridge {

    // Converts a TensorFlow DataType to an nGraph element::Type. Returns
// errors::Unimplemented if the element type is not supported by nGraph
// Core. Otherwise returns Status::OK().
    Status TFDataTypeToNGraphElementType(DataType tf_dt,
                                         ngraph::element::Type* ng_et);

    Status TFTensorShapeToNGraphShape(const ::tensorflow::TensorShapeProto& tf_shape,
                                      ngraph::PartialShape* ng_shape);

template <size_t a, size_t b, size_t c, size_t d>
void Transpose(ngraph::Output<ngraph::Node>& node) {
  static_assert(a < 4 && b < 4 && c < 4 && d < 4,
                "Number of dimensions cannot exceed 4");
  static_assert(a != b && a != c && a != d && b != c && b != d && c != d,
                "Dimensions indices cannot be equal");
  auto& s = node.get_shape();
  ngraph::Shape reshaped_shape{s[a], s[b], s[c], s[d]};
  ngraph::Shape transpose_order{a, b, c, d};
  NGRAPH_VLOG(3) << "transposing " << ngraph::join(s) << " to "
                 << ngraph::join(reshaped_shape) << " axis-order "
                 << ngraph::join(transpose_order);
  auto input_order = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{transpose_order.size()},
      transpose_order);
  node = std::make_shared<opset::Transpose>(node, input_order);
}

template <size_t a, size_t b, size_t c, size_t d>
void Transpose(std::shared_ptr<ngraph::Node>& node) {
  Transpose<a, b, c, d>(node->get_default_output());
}

template <size_t a, size_t b, size_t c, size_t d, size_t e>
void Transpose3D(ngraph::Output<ngraph::Node>& node) {
  static_assert(a < 5 && b < 5 && c < 5 && d < 5 && e < 5,
                "Number of dimensions cannot exceed 5");
  static_assert(a != b && a != c && a != d && a != e && b != c && b != d &&
                    b != e && c != d && c != e && d != e,
                "Dimensions indices cannot be equal");
  auto& s = node.get_shape();
  ngraph::Shape reshaped_shape{s[a], s[b], s[c], s[d], s[e]};
  ngraph::Shape transpose_order{a, b, c, d, e};
  NGRAPH_VLOG(3) << "transposing " << ngraph::join(s) << " to "
                 << ngraph::join(reshaped_shape) << "axis-order "
                 << ngraph::join(transpose_order);
  auto input_order = std::make_shared<opset::Constant>(
      ngraph::element::u64, ngraph::Shape{transpose_order.size()},
      transpose_order);
  node = std::make_shared<opset::Transpose>(node, input_order);
}

template <size_t a, size_t b, size_t c, size_t d, size_t e>
void Transpose3D(std::shared_ptr<ngraph::Node>& node) {
  Transpose3D<a, b, c, d, e>(node->get_default_output());
}

namespace detail {
template <typename T>
void NHWCtoHW(const std::vector<T>& src, std::vector<size_t>& dst) {
  if (dst.size() >= 2) {
    dst[0] = src[1];
    dst[1] = src[2];
  }
  if (dst.size() >= 3) {
    dst[2] = src[3];
  }
}

template <typename T>
void NCHWtoHW(const std::vector<T>& src, std::vector<size_t>& dst) {
  if (dst.size() >= 2) {
    dst[0] = src[2];
    dst[1] = src[3];
  }
  if (dst.size() >= 3) {
    dst[2] = src[4];
  }
}
}

void NHWCtoNCHW(const std::string& op_name, bool is_nhwc,
                ngraph::Output<ngraph::Node>& ng_input);

void NCHWtoNHWC(const std::string& op_name, bool is_nhwc,
                ngraph::Output<ngraph::Node>& ng_node);

template <typename T>
void NHWCtoHW(bool is_nhwc, const std::vector<T>& src,
              std::vector<size_t>& dst) {
  if (is_nhwc) {
    detail::NHWCtoHW(src, dst);
  } else {
    detail::NCHWtoHW(src, dst);
  }
}

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif  // NGRAPH_TF_BRIDGE_CONVERSIONS_H_
