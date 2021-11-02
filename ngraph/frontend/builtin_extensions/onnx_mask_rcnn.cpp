// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onnx_mask_rcnn.hpp"

bool ONNXMaskRCNN::transform (std::shared_ptr<ov::Function>& function, const nlohmann::json& config) const {
    // TODO: Implement real code here; now it is just a stub

    std::cerr
        << "ONNXMaskRCNN::transform was called with function that contains "
        << function->get_ordered_ops().size() << "\n";
    return false;
}
