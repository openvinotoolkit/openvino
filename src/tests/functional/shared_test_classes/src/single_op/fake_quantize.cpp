// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/fake_quantize.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace test {
std::string FakeQuantizeLayerTest::getTestCaseName(const testing::TestParamInfo<fqLayerTestParamsSet>& obj) {
    fqSpecificParams fqParams;
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    std::string target_device;
    std::tie(fqParams, model_type, shapes, target_device) = obj.param;
    size_t levels;
    std::vector<size_t> const_shape;
    std::vector<float> fq_direct_args;
    ov::op::AutoBroadcastSpec broadcast;
    std::tie(levels, const_shape, fq_direct_args, broadcast) = fqParams;

    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "CS=" << ov::test::utils::vec2str(const_shape) << "_";
    result << "LEVELS=" << levels << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << target_device;
    if (!fq_direct_args.empty()) {
        result << "_fqArgs=" << fq_direct_args[0] << "_" << fq_direct_args[1] << "_" << fq_direct_args[2] << "_" << fq_direct_args[3];
    }
    result << "_" << broadcast.m_type;
    return result.str();
}

void FakeQuantizeLayerTest::SetUp() {
    fqSpecificParams fqParams;
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    std::tie(fqParams, model_type, shapes, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    std::vector<size_t> kernel, stride, dilation;
    size_t levels;
    std::vector<size_t> const_shape;
    std::vector<float> fq_direct_arg;
    ov::op::AutoBroadcastSpec broadcast;
    std::tie(levels, const_shape, fq_direct_arg, broadcast) = fqParams;
    if (fq_direct_arg.size() != 0) {
        abs_threshold = (fq_direct_arg[3] - fq_direct_arg[2]) / levels;
    }
    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    std::shared_ptr<ov::Node> fq;
    if (fq_direct_arg.empty()) {
        fq = ov::test::utils::make_fake_quantize(param, model_type, levels, const_shape);
    } else {
        fq = ov::test::utils::make_fake_quantize(
            param,
            model_type,
            levels,
            const_shape,
            {fq_direct_arg[0]},
            {fq_direct_arg[1]},
            {fq_direct_arg[2]},
            {fq_direct_arg[3]});
    }

    auto result = std::make_shared<ov::op::v0::Result>(fq);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "fakeQuantize");
}
}  // namespace test
}  // namespace ov
