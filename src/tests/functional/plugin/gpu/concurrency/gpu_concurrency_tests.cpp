// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "openvino/runtime/core.hpp"

#include <gpu/gpu_config.hpp>
#include <common_test_utils/test_common.hpp>
#include <functional_test_utils/plugin_cache.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "transformations/utils/utils.hpp"

using namespace ::testing;

using ConcurrencyTestParams = std::tuple<size_t,   // number of streams
                                         size_t>;  // number of requests

class OVConcurrencyTest : public CommonTestUtils::TestsCommon,
    public testing::WithParamInterface<ConcurrencyTestParams> {
    void SetUp() override {
        std::tie(num_streams, num_requests) = this->GetParam();
        fn_ptrs = {ngraph::builder::subgraph::makeSplitMultiConvConcat(),
                   ngraph::builder::subgraph::makeMultiSingleConv()};
    };
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcurrencyTestParams>& obj) {
        size_t streams, requests;
        std::tie(streams, requests) = obj.param;
        return "_num_streams_" + std::to_string(streams) + "_num_req_" +
            std::to_string(requests);
    }

protected:
    size_t num_streams;
    size_t num_requests;
    std::vector<std::shared_ptr<ngraph::Function>> fn_ptrs;
};

TEST_P(OVConcurrencyTest, canInferTwoExecNets) {
    auto ie = ov::Core();

    ov::ResultVector outputs;
    std::vector<ov::InferRequest> irs;
    std::vector<std::vector<uint8_t>> ref;
    std::vector<int> outElementsCount;

    for (size_t i = 0; i < fn_ptrs.size(); ++i) {
        auto fn = fn_ptrs[i];

        auto exec_net = ie.compile_model(fn_ptrs[i], CommonTestUtils::DEVICE_GPU,
                                         {{ov::ie::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS, std::to_string(num_streams)}});

        auto input = fn_ptrs[i]->get_parameters().at(0);
        auto output = fn_ptrs[i]->get_results().at(0);

        for (int j = 0; j < num_streams * num_requests; j++) {
            outputs.push_back(output);

            auto inf_req = exec_net.create_infer_request();
            irs.push_back(inf_req);

            auto tensor = FuncTestUtils::create_and_fill_tensor(input->get_element_type(), input->get_shape());
            inf_req.set_tensor(input, tensor);

            outElementsCount.push_back(ov::shape_size(fn_ptrs[i]->get_output_shape(0)));
            const auto in_tensor = inf_req.get_tensor(input);
            const auto tensorSize = in_tensor.get_byte_size();
            const auto inBlobBuf = static_cast<uint8_t*>(in_tensor.data());
            std::vector<uint8_t> inData(inBlobBuf, inBlobBuf + tensorSize);
            auto reOutData = ngraph::helpers::interpreterFunction(fn_ptrs[i], {inData}).front().second;
            ref.push_back(reOutData);
        }
    }

    const int niter = 10;
    for (int i = 0; i < niter; i++) {
        for (auto ir : irs) {
            ir.start_async();
        }

        for (auto ir : irs) {
            ir.wait();
        }
    }

    auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
    for (size_t i = 0; i < irs.size(); ++i) {
        const auto &refBuffer = ref[i].data();
        ASSERT_EQ(outElementsCount[i], irs[i].get_tensor(outputs[i]).get_size());
        FuncTestUtils::compareRawBuffers(irs[i].get_tensor(outputs[i]).data<float>(),
                                         reinterpret_cast<const float *>(refBuffer), outElementsCount[i],
                                         outElementsCount[i],
                                         thr);
    }
}

const std::vector<size_t> num_streams{ 1, 2 };
const std::vector<size_t> num_requests{ 1, 4 };

INSTANTIATE_TEST_SUITE_P(smoke_RemoteTensor, OVConcurrencyTest,
    ::testing::Combine(::testing::ValuesIn(num_streams),
        ::testing::ValuesIn(num_requests)),
    OVConcurrencyTest::getTestCaseName);
