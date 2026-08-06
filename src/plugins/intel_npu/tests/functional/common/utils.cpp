// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <string>

#include "common_test_utils/data_utils.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/runtime/make_tensor.hpp"

std::string getBackendName(const ov::Core& core) {
    return core.get_property("NPU", ov::intel_npu::backend_name.name()).as<std::string>();
}

std::vector<std::string> getAvailableDevices(const ov::Core& core) {
    return core.get_property("NPU", ov::available_devices.name()).as<std::vector<std::string>>();
}

std::string modelPriorityToString(const ov::hint::Priority priority) {
    std::ostringstream stringStream;

    stringStream << priority;

    return stringStream.str();
}

std::string removeDeviceNameOnlyID(const std::string& device_name_id) {
    std::string::const_iterator first_digit = device_name_id.cend();
    std::string::const_iterator last_digit = device_name_id.cend();
    for (auto&& it = device_name_id.cbegin(); it != device_name_id.cend(); ++it) {
        if (*it >= '0' && *it <= '9') {
            if (first_digit == device_name_id.cend()) {
                first_digit = it;
            }
            last_digit = it;
        }
    }
    if (first_digit == device_name_id.cend()) {
        return std::string("");
    }
    return std::string(first_digit, last_digit + 1);
}

std::shared_ptr<ov::Model> createModelWithStates(ov::element::Type type, const ov::Shape& shape) {
    auto input = std::make_shared<ov::op::v0::Parameter>(type, shape);
    auto mem_i1 = std::make_shared<ov::op::v0::Constant>(type, shape, 0);
    auto mem_r1 = std::make_shared<ov::op::v3::ReadValue>(mem_i1, "r_1-3");
    auto mul1 = std::make_shared<ov::op::v1::Multiply>(mem_r1, input);

    auto mem_i2 = std::make_shared<ov::op::v0::Constant>(type, shape, 0);
    auto mem_r2 = std::make_shared<ov::op::v3::ReadValue>(mem_i2, "c_1-3");
    auto mul2 = std::make_shared<ov::op::v1::Multiply>(mem_r2, mul1);
    auto mem_w2 = std::make_shared<ov::op::v3::Assign>(mul2, "c_1-3");

    auto mem_w1 = std::make_shared<ov::op::v3::Assign>(mul2, "r_1-3");
    auto sigm = std::make_shared<ov::op::v0::Sigmoid>(mul2);
    sigm->set_friendly_name("sigmod_state");
    sigm->get_output_tensor(0).set_names({"sigmod_state"});
    mem_r1->set_friendly_name("Memory_1");
    mem_r1->get_output_tensor(0).set_names({"Memory_1"});
    mem_w1->add_control_dependency(mem_r1);
    sigm->add_control_dependency(mem_w1);

    mem_r2->set_friendly_name("Memory_2");
    mem_r2->get_output_tensor(0).set_names({"Memory_2"});
    mem_w2->add_control_dependency(mem_r2);
    sigm->add_control_dependency(mem_w2);

    auto function = std::make_shared<ov::Model>(ov::OutputVector{sigm}, ov::ParameterVector{input}, "add_output");
    return function;
}
namespace ov {

namespace test {

namespace utils {

std::tuple<ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor> allocate_tensors(
    const std::shared_ptr<ov::Model>& model,
    const ov::element::Type& element_type) {
    auto model_shape = model->get_parameters()[0]->get_shape();
    ov::Coordinate start_coordinate{model_shape};
    ov::Coordinate stop_coordinate{model_shape};
    start_coordinate[0] = 1;
    stop_coordinate[0] = 2;

    ov::Allocator alignedAllocator{::intel_npu::utils::AlignedAllocator{::intel_npu::utils::STANDARD_PAGE_SIZE}};
    ov::Tensor importMemoryBatchedTensor(element_type, model_shape, alignedAllocator);
    ov::Tensor importMemoryTensor_1(importMemoryBatchedTensor, ov::Coordinate{0, 0, 0, 0}, start_coordinate);
    ov::Tensor importMemoryTensor_2(importMemoryBatchedTensor, ov::Coordinate{1, 0, 0, 0}, stop_coordinate);

    ov::Allocator unAlignedAllocator{DefaultAllocatorNotAligned{}};
    ov::Tensor unalignedBatchedTensor(element_type, model_shape, unAlignedAllocator);
    ov::Tensor unalignedTensor_1(unalignedBatchedTensor, ov::Coordinate{0, 0, 0, 0}, start_coordinate);
    ov::Tensor unalignedTensor_2(unalignedBatchedTensor, ov::Coordinate{1, 0, 0, 0}, stop_coordinate);

    ov::test::utils::fill_tensor_random(importMemoryBatchedTensor);
    ov::test::utils::fill_tensor_random(unalignedBatchedTensor);

    return {importMemoryBatchedTensor,
            importMemoryTensor_1,
            importMemoryTensor_2,
            unalignedBatchedTensor,
            unalignedTensor_1,
            unalignedTensor_2};
};

}  // namespace utils

}  // namespace test

}  // namespace ov
