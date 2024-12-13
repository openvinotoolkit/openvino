// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/fake_convert.hpp"

#include "intel_gpu/primitives/fake_convert.hpp"

namespace ov {
namespace intel_gpu {

static void CreateFakeConvertOp(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op) {
    validate_inputs_count(op, {2, 3});
    const auto inputs = p.GetInputInfo(op);
    const std::string layerName = layer_type_name_ID(op);
    std::string destination_type = "";
    if (auto fake_convert_v13 = std::dynamic_pointer_cast<ov::op::v13::FakeConvert>(op)) {
        destination_type = fake_convert_v13->get_destination_type();
    } else {
        OPENVINO_THROW("[GPU] Can't cast Broadcast operation to any supported version");
    }
    std::shared_ptr<cldnn::fake_convert> fake_convert_prim = nullptr;
    if (inputs.size() == 2) {
        fake_convert_prim = std::make_shared<cldnn::fake_convert>(layerName,
                                        inputs[0],
                                        inputs[1],
                                        destination_type);
    } else {
        fake_convert_prim = std::make_shared<cldnn::fake_convert>(layerName,
                                        inputs[0],
                                        inputs[1],
                                        inputs[2],
                                        destination_type);
    }

    p.add_primitive(*op, fake_convert_prim);
}

REGISTER_FACTORY_IMPL(v13, FakeConvert);

}  // namespace intel_gpu
}  // namespace ov
