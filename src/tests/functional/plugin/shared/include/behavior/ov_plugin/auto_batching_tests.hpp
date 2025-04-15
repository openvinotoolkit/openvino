// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "base/ov_behavior_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/subgraph_builders/single_conv.hpp"
#include "common_test_utils/subgraph_builders/detection_output.hpp"
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"

namespace ov {
namespace test {
namespace behavior {
using AutoBatchTwoNetsParams = std::tuple<
        std::string,  // device name
        bool,         // get or set tensor
        size_t,       // number of streams
        size_t,       // number of requests
        size_t>;      // batch size>

class AutoBatching_Test : public OVPluginTestBase,
                          public testing::WithParamInterface<AutoBatchTwoNetsParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<AutoBatchTwoNetsParams> &obj) {
        size_t streams, requests, batch;
        bool use_get_tensor;
        std::string target_device;
        std::tie(target_device, use_get_tensor, streams, requests, batch) = obj.param;
        return target_device + std::string(use_get_tensor ? "_get_tensor" : "_set_blob") + "_batch_size_" +
               std::to_string(batch) +
               "_num_streams_" + std::to_string(streams) + "_num_req_" + std::to_string(requests);
    }

protected:
    bool use_get_tensor;
    size_t num_streams;
    size_t num_requests;
    size_t num_batch;
    std::vector<std::shared_ptr<ov::Model>> fn_ptrs;

    void SetUp() override {
        std::tie(target_device, use_get_tensor, num_streams, num_requests, num_batch) = this->GetParam();
        fn_ptrs = {ov::test::utils::make_single_conv(),
                   ov::test::utils::make_multi_single_conv()};
    };

    void TestAutoBatch() {
        auto core = ov::test::utils::PluginCache::get().core();

        ov::OutputVector outputs;
        std::vector<std::pair<std::shared_ptr<ov::Model>, ov::InferRequest>> irs;
        std::vector<ov::InferRequest> irs_ref;
        std::vector<size_t> outElementsCount;

        for (size_t i = 0; i < fn_ptrs.size(); ++i) {
            auto model = fn_ptrs[i];
            auto inputs = model->inputs();
            for (auto const & n : inputs) {
                n.get_node()->set_output_type(0, ov::element::f32, n.get_shape());
            }
            ov::AnyMap config;
            if (target_device.find("GPU") != std::string::npos) {
                config.insert(ov::num_streams(static_cast<int32_t>(num_streams)));
                config.insert(ov::hint::inference_precision(ov::element::f32));
            }

            if (target_device.find("CPU") != std::string::npos) {
                config.insert(ov::num_streams(static_cast<int32_t>(num_streams)));
                config.insert(ov::hint::inference_precision(ov::element::f32));
            }
            // minimize timeout to reduce test time
            config.insert(ov::auto_batch_timeout(1));

            auto compiled_model = core->compile_model(model, std::string(ov::test::utils::DEVICE_BATCH) + ":" +
                                                      target_device + "(" + std::to_string(num_batch) + ")",
                                                      config);

            auto network_outputs = model->outputs();
            ASSERT_EQ(network_outputs.size(), 1) << " Auto-Batching tests use networks with single output";
            auto const & output = network_outputs[0];
            for (size_t j = 0; j < num_requests; j++) {
                outputs.push_back(output);
                outElementsCount.push_back(
                        std::accumulate(begin(fn_ptrs[i]->get_output_shape(0)), end(fn_ptrs[i]->get_output_shape(0)), size_t(1),
                                        std::multiplies<size_t>()));

                auto inf_req = compiled_model.create_infer_request();
                irs.push_back({model, inf_req});

                auto compiled_model_ref = core->compile_model(model, ov::test::utils::DEVICE_TEMPLATE);
                auto inf_req_ref = compiled_model_ref.create_infer_request();
                irs_ref.push_back(inf_req_ref);

                std::vector<ov::Tensor> inData;
                for (auto const & input : inputs) {
                    auto tensor = ov::test::utils::create_and_fill_tensor(input.get_element_type(), input.get_shape());
                    if (use_get_tensor)
                        memcpy(inf_req.get_tensor(input).data(), tensor.data(), tensor.get_byte_size());
                    else
                        inf_req.set_tensor(input, tensor);

                    inf_req_ref.set_tensor(input, tensor);
                }

                if (!use_get_tensor) {
                    auto tensor = ov::test::utils::create_and_fill_tensor(output.get_element_type(), output.get_shape());
                    inf_req.set_tensor(output, tensor);
                }

                inf_req_ref.infer();
            }
        }

        {
            for (auto& ir : irs) {
                ir.second.start_async();
            }

            for (auto& ir : irs) {
                ir.second.wait();
            }
        }

        for (size_t i = 0; i < irs.size(); ++i) {
            auto output = irs[i].first->get_results().at(0);
            auto out = irs[i].second.get_tensor(output);
            auto out_ref = irs_ref[i].get_tensor(output);
            ov::test::utils::compare(out_ref, out);
        }
    }
};

class AutoBatching_Test_DetectionOutput : public AutoBatching_Test {
public:
    void SetUp() override {
        std::tie(target_device, use_get_tensor, num_streams, num_requests, num_batch) = GetParam();
        fn_ptrs = {ov::test::utils::make_detection_output(),
                   ov::test::utils::make_detection_output()};
    };

    static std::string getTestCaseName(const testing::TestParamInfo<AutoBatchTwoNetsParams> &obj) {
        return AutoBatching_Test::getTestCaseName(obj);
    }
};

TEST_P(AutoBatching_Test, compareAutoBatchingToSingleBatch) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    TestAutoBatch();
}

TEST_P(AutoBatching_Test_DetectionOutput, compareAutoBatchingToSingleBatch) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    TestAutoBatch();
}
}  // namespace behavior
}  // namespace test
}  // namespace ov
