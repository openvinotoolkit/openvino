// Copyright (C) 2018-2023 Intel Corporation
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
#include "ov_models/subgraph_builders.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "transformations/utils/utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/data_utils.hpp"

using namespace ::testing;

using ConcurrencyTestParams = std::tuple<size_t,   // number of streams
                                         size_t>;  // number of requests

class OVConcurrencyTest : public ov::test::TestsCommon,
    public testing::WithParamInterface<ConcurrencyTestParams> {
    void SetUp() override {
        std::tie(num_streams, num_requests) = this->GetParam();
        fn_ptrs = {ngraph::builder::subgraph::makeSplitMultiConvConcat(),
                   ngraph::builder::subgraph::makeMultiSingleConv(),
                   ngraph::builder::subgraph::makeTIwithLSTMcell()};
    };
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcurrencyTestParams>& obj) {
        size_t streams, requests;
        std::tie(streams, requests) = obj.param;
        return "_num_streams_" + std::to_string(streams) + "_num_req_" +
            std::to_string(requests);
    }

    void execute(bool is_caching_test = false) {
        auto ie = ov::Core();

        std::string cacheFolderName;
        if (is_caching_test) {
            std::stringstream ss;
            ss << "OVConcurrencyTest_nstreams_" << num_streams << "_nireq_" << num_requests;
            cacheFolderName = ss.str();
            ov::test::utils::removeFilesWithExt(cacheFolderName, "blob");
            ov::test::utils::removeFilesWithExt(cacheFolderName, "cl_cache");
            ov::test::utils::removeDir(cacheFolderName);
            ie.set_property(ov::cache_dir(cacheFolderName));
            ie.set_property(ov::intel_gpu::enable_loop_unrolling(false));
        }

        ov::ResultVector outputs;
        std::vector<ov::InferRequest> irs;
        std::vector<std::vector<uint8_t>> ref;
        std::vector<size_t> outElementsCount;

        for (size_t i = 0; i < fn_ptrs.size(); ++i) {
            auto fn = fn_ptrs[i];

            ov::CompiledModel exec_net;

            if (is_caching_test) {
                {
                    auto _dummy_exec_net = ie.compile_model(fn_ptrs[i], ov::test::utils::DEVICE_GPU,
                                                    ov::num_streams(ov::streams::Num(num_streams)), ov::hint::inference_precision(ov::element::f32));
                }
                {
                    exec_net = ie.compile_model(fn_ptrs[i], ov::test::utils::DEVICE_GPU,
                                                    ov::num_streams(ov::streams::Num(num_streams)), ov::hint::inference_precision(ov::element::f32));
                }
            } else {
                exec_net = ie.compile_model(fn_ptrs[i], ov::test::utils::DEVICE_GPU,
                                                ov::num_streams(ov::streams::Num(num_streams)), ov::hint::inference_precision(ov::element::f32));
            }

            auto output = fn_ptrs[i]->get_results().at(0);

            for (size_t j = 0; j < num_streams * num_requests; j++) {
                outputs.push_back(output);

                auto inf_req = exec_net.create_infer_request();
                irs.push_back(inf_req);

                std::vector<std::vector<uint8_t>> inputs;
                for (size_t param_idx = 0; param_idx < fn_ptrs[i]->get_parameters().size(); ++param_idx) {
                    auto input = fn_ptrs[i]->get_parameters().at(param_idx);
                    auto tensor = ov::test::utils::create_and_fill_tensor(input->get_element_type(), input->get_shape());
                    inf_req.set_tensor(input, tensor);

                    const auto in_tensor = inf_req.get_tensor(input);
                    const auto tensorSize = in_tensor.get_byte_size();
                    const auto inBlobBuf = static_cast<uint8_t*>(in_tensor.data());
                    std::vector<uint8_t> inData(inBlobBuf, inBlobBuf + tensorSize);
                    inputs.emplace_back(inData);
                }

                auto reOutData = ngraph::helpers::interpreterFunction(fn_ptrs[i], inputs).front().second;
                ref.push_back(reOutData);
                outElementsCount.push_back(ov::shape_size(fn_ptrs[i]->get_output_shape(0)));
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

        if (is_caching_test) {
            ov::test::utils::removeFilesWithExt(cacheFolderName, "blob");
            ov::test::utils::removeFilesWithExt(cacheFolderName, "cl_cache");
            ov::test::utils::removeDir(cacheFolderName);
        }
    }

protected:
    size_t num_streams;
    size_t num_requests;
    std::vector<std::shared_ptr<ngraph::Function>> fn_ptrs;
};

TEST_P(OVConcurrencyTest, canInferTwoExecNets) {
    this->execute(false);
}

TEST_P(OVConcurrencyTest, canInferTwoExecNets_cached) {
    this->execute(true);
}

const std::vector<size_t> num_streams{ 1, 2 };
const std::vector<size_t> num_requests{ 1, 4 };

INSTANTIATE_TEST_SUITE_P(smoke_RemoteTensor, OVConcurrencyTest,
    ::testing::Combine(::testing::ValuesIn(num_streams),
        ::testing::ValuesIn(num_requests)),
    OVConcurrencyTest::getTestCaseName);

TEST(canSwapTensorsBetweenInferRequests, inputs) {
    std::vector<std::vector<uint8_t>> ref;
    std::vector<ov::Tensor> input_tensors;
    auto fn = ngraph::builder::subgraph::makeSplitMultiConvConcat();

    auto ie = ov::Core();
    auto compiled_model = ie.compile_model(fn, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f32));

    const int infer_requests_num = 2;
    ov::InferRequest infer_request1 = compiled_model.create_infer_request();
    ov::InferRequest infer_request2 = compiled_model.create_infer_request();

    input_tensors.push_back(infer_request1.get_input_tensor());
    input_tensors.push_back(infer_request2.get_input_tensor());

    auto calc_ref_results = [&](const ov::Tensor& tensor){
        const auto tensor_size = tensor.get_byte_size();
        const auto in_blob_buf = static_cast<uint8_t*>(tensor.data());
        std::vector<uint8_t> inData(in_blob_buf, in_blob_buf + tensor_size);
        auto ref_out_data = ngraph::helpers::interpreterFunction(fn, {inData}).front().second;
        ref.push_back(ref_out_data);
    };

    auto compare_results = [&](ov::Tensor& result, const uint8_t* refResult) {
        auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
        ASSERT_EQ(ov::shape_size(fn->get_output_shape(0)), result.get_size());
        FuncTestUtils::compareRawBuffers(result.data<float>(),
                                        reinterpret_cast<const float *>(refResult), ov::shape_size(fn->get_output_shape(0)),
                                        ov::shape_size(fn->get_output_shape(0)),
                                        thr);
    };

    for (int32_t i = 0; i < infer_requests_num; i++) {
        ov::test::utils::fill_tensor_random(input_tensors[i], 10, -5, 1, i);
        calc_ref_results(input_tensors[i]);
    }

    // Swap tensors between IRs
    infer_request1.set_input_tensor(input_tensors[1]);
    infer_request2.set_input_tensor(input_tensors[0]);

    const int niter_limit = 10;
    int iter1 = 0, iter2 = 0;
    infer_request1.set_callback([&](std::exception_ptr exception_ptr) mutable {
        if (exception_ptr) {
            std::rethrow_exception(exception_ptr);
        } else {
            iter1++;
            ov::Tensor output_tensor = infer_request1.get_output_tensor();
            compare_results(output_tensor, ref[iter1 % 2].data());
            if (iter1 < niter_limit) {
                infer_request1.set_input_tensor(input_tensors[(iter1 + 1) % 2]);
                infer_request1.start_async();
            }
        }
    });

    infer_request2.set_callback([&](std::exception_ptr exception_ptr) mutable {
        if (exception_ptr) {
            std::rethrow_exception(exception_ptr);
        } else {
            iter2++;
            ov::Tensor output_tensor = infer_request2.get_output_tensor();
            compare_results(output_tensor, ref[(iter2 + 1) % 2].data());
            if (iter2 < niter_limit) {
                infer_request2.set_input_tensor(input_tensors[iter2 % 2]);
                infer_request2.start_async();
            }
        }
    });

    infer_request1.start_async();
    infer_request2.start_async();

    for (size_t i = 0; i < niter_limit; i++) {
        infer_request1.wait();
        infer_request2.wait();
    }
}

TEST(smoke_InferRequestDeviceMemoryAllocation, usmHostIsNotChanged) {
    auto fn = ngraph::builder::subgraph::makeDetectionOutput(ngraph::element::Type_t::f32);

    auto ie = ov::Core();
    auto compiled_model = ie.compile_model(fn, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f32));

    ov::InferRequest infer_request1 = compiled_model.create_infer_request();
    ov::InferRequest infer_request2 = compiled_model.create_infer_request();

    auto input_tensor1 = infer_request1.get_input_tensor();
    ov::test::utils::fill_tensor_random(input_tensor1, 20, 0, 1, 0);

    auto output_tensor1 = ov::test::utils::create_and_fill_tensor(compiled_model.output().get_element_type(), compiled_model.output().get_shape());
    auto output_tensor2 = infer_request2.get_output_tensor();

    // Use tensor from infer request #2 as an output for infer request #1
    infer_request1.set_output_tensor(output_tensor2);
    ASSERT_NO_THROW(infer_request1.infer());

    // Modify tensor somehow and save as a reference values
    ov::test::utils::fill_tensor_random(output_tensor2);

    std::vector<float> ref_values;
    ref_values.resize(output_tensor2.get_byte_size());
    std::memcpy(ref_values.data(), output_tensor2.data(), output_tensor2.get_byte_size());

    // Perform second infer() call with a system host memory tensor
    infer_request1.set_output_tensor(output_tensor1);
    ASSERT_NO_THROW(infer_request1.infer());

    // Expect that output_tensor2 will not change it's data after infer() call
    auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
    FuncTestUtils::compareRawBuffers(ref_values.data(),
                                     output_tensor2.data<float>(),
                                     ref_values.size(),
                                     ov::shape_size(output_tensor2.get_shape()),
                                     thr);
}

TEST(smoke_InferRequestDeviceMemoryAllocation, canSetSystemHostTensor) {
    auto fn = ngraph::builder::subgraph::makeDetectionOutput(ngraph::element::Type_t::f32);

    auto ie = ov::Core();
    auto compiled_model = ie.compile_model(fn, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f32));

    ov::InferRequest infer_request1 = compiled_model.create_infer_request();
    ov::InferRequest infer_request2 = compiled_model.create_infer_request();

    auto input_tensor1 = infer_request1.get_input_tensor();
    ov::test::utils::fill_tensor_random(input_tensor1, 20, 0, 1, 0);

    auto output_tensor1 = ov::test::utils::create_and_fill_tensor(compiled_model.output().get_element_type(), compiled_model.output().get_shape());
    auto output_tensor2 = infer_request2.get_output_tensor();

    infer_request1.set_output_tensor(output_tensor2);
    ASSERT_NO_THROW(infer_request1.infer());

    ov::test::utils::fill_tensor_random(input_tensor1, 10, 0, 1, 1);
    infer_request1.set_output_tensor(output_tensor1);
    ASSERT_NO_THROW(infer_request1.infer());
}

TEST(canSwapTensorsBetweenInferRequests, outputs) {
    std::vector<std::vector<uint8_t>> ref;
    std::vector<ov::Tensor> input_tensors;
    std::vector<ov::Tensor> output_tensors;
    auto fn = ngraph::builder::subgraph::makeSplitMultiConvConcat();

    auto ie = ov::Core();
    auto compiled_model = ie.compile_model(fn, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f32));

    const int infer_requests_num = 2;
    ov::InferRequest infer_request1 = compiled_model.create_infer_request();
    ov::InferRequest infer_request2 = compiled_model.create_infer_request();

    input_tensors.push_back(infer_request1.get_input_tensor());
    input_tensors.push_back(infer_request2.get_input_tensor());
    output_tensors.push_back(infer_request1.get_output_tensor());
    output_tensors.push_back(infer_request2.get_output_tensor());

    auto calc_ref_results = [&](const ov::Tensor& tensor){
        const auto tensor_size = tensor.get_byte_size();
        const auto in_blob_buf = static_cast<uint8_t*>(tensor.data());
        std::vector<uint8_t> inData(in_blob_buf, in_blob_buf + tensor_size);
        auto ref_out_data = ngraph::helpers::interpreterFunction(fn, {inData}).front().second;
        ref.push_back(ref_out_data);
    };

    auto compare_results = [&](ov::Tensor& result, const uint8_t* refResult) {
        auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
        ASSERT_EQ(ov::shape_size(fn->get_output_shape(0)), result.get_size());
        FuncTestUtils::compareRawBuffers(result.data<float>(),
                                        reinterpret_cast<const float *>(refResult), ov::shape_size(fn->get_output_shape(0)),
                                        ov::shape_size(fn->get_output_shape(0)),
                                        thr);
    };

    for (int32_t i = 0; i < infer_requests_num; i++) {
        ov::test::utils::fill_tensor_random(input_tensors[i], 10, -5, 1, i);
        calc_ref_results(input_tensors[i]);
    }

    // Swap tensors between IRs
    infer_request1.set_output_tensor(output_tensors[1]);
    infer_request2.set_output_tensor(output_tensors[0]);

    const int niter_limit = 10;
    int iter1 = 0, iter2 = 0;
    infer_request1.set_callback([&](std::exception_ptr exception_ptr) mutable {
        if (exception_ptr) {
            std::rethrow_exception(exception_ptr);
        } else {
            iter1++;
            ov::Tensor output_tensor = infer_request1.get_output_tensor();
            compare_results(output_tensor, ref[0].data());
            if (iter1 < niter_limit) {
                infer_request1.set_output_tensor(output_tensors[(iter1 + 1) % 2]);
                infer_request1.start_async();
            }
        }
    });

    infer_request2.set_callback([&](std::exception_ptr exception_ptr) mutable {
        if (exception_ptr) {
            std::rethrow_exception(exception_ptr);
        } else {
            iter2++;
            ov::Tensor output_tensor = infer_request2.get_output_tensor();
            compare_results(output_tensor, ref[1].data());
            if (iter2 < niter_limit) {
                infer_request2.set_output_tensor(output_tensors[iter2 % 2]);
                infer_request2.start_async();
            }
        }
    });

    infer_request1.start_async();
    infer_request2.start_async();

    for (size_t i = 0; i < niter_limit; i++) {
        infer_request1.wait();
        infer_request2.wait();
    }
}
