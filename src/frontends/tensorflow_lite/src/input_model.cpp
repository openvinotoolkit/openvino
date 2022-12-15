// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include <fstream>
#include <iterator>
#include <queue>

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/tensorflow/graph_iterator.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/util/log.hpp"
#include "place.hpp"
#include "utils.hpp"

using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow_lite {

InputModel::InputModel(const ov::frontend::tensorflow::GraphIterator::Ptr& graph_iterator, const std::shared_ptr<TelemetryExtension>& telemetry) {

}
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
