// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "auto_func_test.hpp"
#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"

namespace ov {
namespace auto_plugin {
namespace tests {

using consistency_test_param = std::tuple<std::string,  // device name
                                          bool,         // get or set blob
                                          ov::AnyMap>;  // property

class Consistency_Test : public AutoFuncTests, public testing::WithParamInterface<consistency_test_param> {
    void SetUp() override {
        AutoFuncTests::SetUp();
        std::tie(target_device, use_get_tensor, property) = this->GetParam();
    };

public:
    static std::string getTestCaseName(const testing::TestParamInfo<consistency_test_param>& obj) {
        ov::AnyMap property;
        bool use_get_tensor;
        std::string target_device;
        std::tie(target_device, use_get_tensor, property) = obj.param;
        std::ostringstream result;
        result << "target_device=" << target_device << "_";
        result << std::string(use_get_tensor ? "_get_blob" : "_set_blob") << "_";
        if (!property.empty()) {
            for (auto& iter : property) {
                result << "priority=" << iter.first << "_" << iter.second.as<std::string>();
            }
        }
        return result.str();
    }

protected:
    bool use_get_tensor;
    ov::AnyMap property;
    std::string target_device;

    void run() {
        std::vector<ov::InferRequest> irs;
        std::vector<std::vector<ov::Tensor>> ref;
        std::map<std::shared_ptr<ov::Node>, ov::Tensor> input_data;

        auto compiled_model = core.compile_model(model_cannot_batch, target_device, property);
        auto inputs = compiled_model.inputs();
        auto outputs = compiled_model.outputs();
        auto num_requests = compiled_model.get_property(ov::optimal_number_of_infer_requests);
        for (size_t j = 0; j < num_requests; j++) {
            auto inf_req = compiled_model.create_infer_request();
            irs.push_back(inf_req);
            for (auto& iter : inputs) {
                auto tensor = ov::test::utils::create_and_fill_tensor(iter.get_element_type(), iter.get_shape());
                if (use_get_tensor)
                    memcpy(reinterpret_cast<uint8_t*>(inf_req.get_tensor(iter).data()),
                           reinterpret_cast<const uint8_t*>(tensor.data()),
                           tensor.get_byte_size());
                else
                    inf_req.set_tensor(iter, tensor);
                auto node_ptr = iter.get_node_shared_ptr();
                input_data.insert({std::const_pointer_cast<ov::Node>(node_ptr), tensor});
            }
            for (auto& iter : outputs) {
                if (!use_get_tensor) {
                    auto tensor = ov::Tensor(iter.get_element_type(), iter.get_shape());
                    inf_req.set_tensor(iter, tensor);
                }
            }
            auto refOutData = ov::test::utils::infer_on_template(model_cannot_batch, input_data);
            ref.push_back(refOutData);
        }
        for (size_t i = 0; i < 50; i++) {
            for (auto ir : irs) {
                ir.start_async();
            }

            for (auto ir : irs) {
                ir.wait();
            }
        }
        for (size_t i = 0; i < irs.size(); ++i) {
            for (auto& iter : outputs) {
                ov::test::utils::compare(irs[i].get_tensor(iter), ref[i][0]);
            }
        }
    }
};

TEST_P(Consistency_Test, infer_consistency_test) {
    run();
}

}  // namespace tests
}  // namespace auto_plugin
}  // namespace ov
