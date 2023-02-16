// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace util {

/**
 * @brief Make temporay tensor from host tensor pointer.
 *
 * @param t  Input tensor for conversion.
 * @return ov::Tensor which points to host tensor data. Can return not allocated or special dynamic depends on input
 * tensor state.
 */
OPENVINO_API Tensor make_tmp_tensor(const ngraph::HostTensorPtr& t);

/**
 * @brief Make tmp tensor base on output shape and element type.
 *
 * @param output  Node output to make tensor.
 * @return ov::Tensor from output properties.
 */
OPENVINO_API Tensor make_tmp_tensor(const Output<Node>& output);

/**
 * @brief Make vector of ov::Tensor from vector of host tensors.
 *
 * @param tensors  Input vector of host tensor to convert.
 * @return ov::TensorVectors, can contains not allocated or dynamic tensor depends on input tensor properties.
 */
OPENVINO_API TensorVector make_tmp_tensors(const std::vector<ngraph::HostTensorPtr>& tensors);

/**
 * @brief Update output host tensors if they got dynamic shapee before evaluation (not allocated).
 *
 * Other tensor not requires update as they are created from outputs and points to same data blob.
 *
 * @param output_values  Temporary ov::Tensor vector created from outputs for evaluation
 * @param outputs        Output host tensors vector to update.
 */
OPENVINO_API void update_output_host_tensors(const std::vector<ngraph::HostTensorPtr>& output_values,
                                             const ov::TensorVector& outputs);

}  // namespace util
}  // namespace ov
