// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/tensorflow/graph_iterator.hpp"
#include "input_model.hpp"
#include "place.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
class OpPlace;
class TensorPlace;
}
namespace tensorflow_lite {
class InputModel : public ov::frontend::InputModel {
    friend class FrontEnd;
public:
    explicit InputModel(const ov::frontend::tensorflow::GraphIterator::Ptr& graph_iterator,
                        const std::shared_ptr<TelemetryExtension>& telemetry = {});

};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
