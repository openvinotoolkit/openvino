// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "common_test_utils/test_common.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "transformations/utils/utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/subgraph_builders/split_multi_conv_concat.hpp"
#include "common_test_utils/subgraph_builders/ti_with_lstm_cell.hpp"
#include "common_test_utils/subgraph_builders/detection_output.hpp"
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"

namespace {
using ConcurrencyTestParams = std::tuple<size_t,   // number of streams
                                         size_t>;  // number of requests

class OVConcurrencyTest : public ov::test::TestsCommon,
                          public testing::WithParamInterface<ConcurrencyTestParams> {
    void SetUp() override {
        std::tie(num_streams, num_requests) = this->GetParam();
        fn_ptrs = {ov::test::utils::make_split_multi_conv_concat(),
                   ov::test::utils::make_multi_single_conv(),
                   ov::test::utils::make_ti_with_lstm_cell()};
    };
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcurrencyTestParams>& obj) {
        size_t streams, requests;
        std::tie(streams, requests) = obj.param;
        return "_num_streams_" + std::to_string(streams) + "_num_req_" +
            std::to_string(requests);
    }

    void execute(bool is_caching_test = false) {
        auto core = ov::test::utils::PluginCache::get().core();

        std::string cacheFolderName;
        if (is_caching_test) {
            std::stringstream ss;
            ss << "OVConcurrencyTest_nstreams_" << num_streams << "_nireq_" << num_requests;
            cacheFolderName = ss.str();
            ov::test::utils::removeFilesWithExt(cacheFolderName, "blob");
            ov::test::utils::removeFilesWithExt(cacheFolderName, "cl_cache");
            ov::test::utils::removeDir(cacheFolderName);
            core->set_property(ov::cache_dir(cacheFolderName));
            core->set_property(ov::test::utils::DEVICE_GPU, ov::intel_gpu::enable_loop_unrolling(false));
        }

        std::vector<std::pair<std::shared_ptr<ov::Model>, ov::InferRequest>> irs;
        std::vector<ov::InferRequest> irs_ref;

        for (size_t i = 0; i < fn_ptrs.size(); ++i) {
            auto fn = fn_ptrs[i];
            ov::CompiledModel exec_net;
            if (is_caching_test) {
                {
                    auto _dummy_exec_net = core->compile_model(fn, ov::test::utils::DEVICE_GPU,
                                                    ov::num_streams(ov::streams::Num(num_streams)), ov::hint::inference_precision(ov::element::f32));
                }
                {
                    exec_net = core->compile_model(fn, ov::test::utils::DEVICE_GPU,
                                                    ov::num_streams(ov::streams::Num(num_streams)), ov::hint::inference_precision(ov::element::f32));
                }
            } else {
                exec_net = core->compile_model(fn, ov::test::utils::DEVICE_GPU,
                                                ov::num_streams(ov::streams::Num(num_streams)), ov::hint::inference_precision(ov::element::f32));
            }

            for (size_t j = 0; j < num_streams * num_requests; j++) {
                auto inf_req = exec_net.create_infer_request();
                irs.push_back({fn, inf_req});

                auto compiled_model_ref = core->compile_model(fn, ov::test::utils::DEVICE_TEMPLATE);
                auto inf_req_ref = compiled_model_ref.create_infer_request();
                irs_ref.push_back(inf_req_ref);

                std::vector<ov::Tensor> input_tensors;
                for (size_t param_idx = 0; param_idx < fn->get_parameters().size(); ++param_idx) {
                    auto input = fn->get_parameters().at(param_idx);
                    auto tensor = ov::test::utils::create_and_fill_tensor(input->get_element_type(), input->get_shape());
                    inf_req.set_tensor(input, tensor);
                    inf_req_ref.set_tensor(input, tensor);
                }
                inf_req_ref.infer();
            }
        }

        const int niter = 10;
        for (int i = 0; i < niter; i++) {
            for (auto ir : irs) {
                ir.second.start_async();
            }

            for (auto ir : irs) {
                ir.second.wait();
            }
        }

        for (size_t i = 0; i < irs.size(); ++i) {
            // TODO now it compares only 1st output
            // When CVS-126856 is fixed, update to compare all outputs
            auto output = irs[i].first->get_results().at(0);
            auto out = irs[i].second.get_tensor(output);
            auto out_ref = irs_ref[i].get_tensor(output);
            ov::test::utils::compare(out_ref, out);
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
    std::vector<std::shared_ptr<ov::Model>> fn_ptrs;
};

TEST_P(OVConcurrencyTest, canInferTwoExecNets) {
    execute(false);
}

TEST_P(OVConcurrencyTest, canInferTwoExecNets_cached) {
    execute(true);
}

const std::vector<size_t> num_streams{ 1, 2 };
const std::vector<size_t> num_requests{ 1, 4 };

INSTANTIATE_TEST_SUITE_P(smoke_RemoteTensor, OVConcurrencyTest,
    ::testing::Combine(::testing::ValuesIn(num_streams),
        ::testing::ValuesIn(num_requests)),
    OVConcurrencyTest::getTestCaseName);

TEST(canSwapTensorsBetweenInferRequests, inputs) {
    std::vector<ov::Tensor> ref;
    std::vector<ov::Tensor> input_tensors;
    auto fn = ov::test::utils::make_split_multi_conv_concat();

    auto core = ov::test::utils::PluginCache::get().core();
    auto compiled_model = core->compile_model(fn, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f32));

    const int infer_requests_num = 2;
    ov::InferRequest infer_request1 = compiled_model.create_infer_request();
    ov::InferRequest infer_request2 = compiled_model.create_infer_request();

    input_tensors.push_back(infer_request1.get_input_tensor());
    input_tensors.push_back(infer_request2.get_input_tensor());

    auto calc_ref_results = [&](const ov::Tensor& tensor){
        auto compiled_model_ref = core->compile_model(fn, ov::test::utils::DEVICE_TEMPLATE);
        auto inf_req_ref = compiled_model_ref.create_infer_request();

        auto input = fn->input(0);
        inf_req_ref.set_tensor(input, tensor);
        inf_req_ref.infer();

        auto output = fn->get_result();
        auto out_ref = inf_req_ref.get_tensor(output);
        ref.push_back(out_ref);
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
            ov::test::utils::compare(ref[iter1 % 2], output_tensor);
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
            ov::test::utils::compare(ref[(iter2 + 1) % 2], output_tensor);
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
    auto fn = ov::test::utils::make_detection_output(ov::element::f32);

    auto core = ov::test::utils::PluginCache::get().core();
    auto compiled_model = core->compile_model(fn, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f32));

    ov::InferRequest infer_request1 = compiled_model.create_infer_request();
    ov::InferRequest infer_request2 = compiled_model.create_infer_request();

    auto input_tensor1 = infer_request1.get_input_tensor();
    ov::test::utils::fill_tensor_random(input_tensor1, 20, 0, 1, 0);

    auto output_tensor1 = ov::test::utils::create_and_fill_tensor(compiled_model.output().get_element_type(), compiled_model.output().get_shape());
    auto output_tensor2 = infer_request2.get_output_tensor();

    // Use tensor from infer request #2 as an output for infer request #1
    infer_request1.set_output_tensor(output_tensor2);
    OV_ASSERT_NO_THROW(infer_request1.infer());

    // Modify tensor somehow and save as a reference values
    ov::test::utils::fill_tensor_random(output_tensor2);

    ov::Tensor ref_tensor(output_tensor2.get_element_type(), output_tensor2.get_shape());
    output_tensor2.copy_to(ref_tensor);

    // Perform second infer() call with a system host memory tensor
    infer_request1.set_output_tensor(output_tensor1);
    OV_ASSERT_NO_THROW(infer_request1.infer());

    // Expect that output_tensor2 will not change it's data after infer() call
    ov::test::utils::compare(ref_tensor, output_tensor2, 1e-4);
}

TEST(smoke_InferRequestDeviceMemoryAllocation, canSetSystemHostTensor) {
    auto fn = ov::test::utils::make_detection_output(ov::element::f32);

    auto core = ov::test::utils::PluginCache::get().core();
    auto compiled_model = core->compile_model(fn, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f32));

    ov::InferRequest infer_request1 = compiled_model.create_infer_request();
    ov::InferRequest infer_request2 = compiled_model.create_infer_request();

    auto input_tensor1 = infer_request1.get_input_tensor();
    ov::test::utils::fill_tensor_random(input_tensor1, 20, 0, 1, 0);

    auto output_tensor1 = ov::test::utils::create_and_fill_tensor(compiled_model.output().get_element_type(), compiled_model.output().get_shape());
    auto output_tensor2 = infer_request2.get_output_tensor();

    infer_request1.set_output_tensor(output_tensor2);
    OV_ASSERT_NO_THROW(infer_request1.infer());

    ov::test::utils::fill_tensor_random(input_tensor1, 10, 0, 1, 1);
    infer_request1.set_output_tensor(output_tensor1);
    OV_ASSERT_NO_THROW(infer_request1.infer());
}

TEST(canSwapTensorsBetweenInferRequests, outputs) {
    std::vector<ov::Tensor> ref;
    std::vector<ov::Tensor> input_tensors;
    std::vector<ov::Tensor> output_tensors;
    auto fn = ov::test::utils::make_split_multi_conv_concat();

    auto core = ov::test::utils::PluginCache::get().core();
    auto compiled_model = core->compile_model(fn, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f32));

    const int infer_requests_num = 2;
    ov::InferRequest infer_request1 = compiled_model.create_infer_request();
    ov::InferRequest infer_request2 = compiled_model.create_infer_request();

    input_tensors.push_back(infer_request1.get_input_tensor());
    input_tensors.push_back(infer_request2.get_input_tensor());
    output_tensors.push_back(infer_request1.get_output_tensor());
    output_tensors.push_back(infer_request2.get_output_tensor());

    auto calc_ref_results = [&](const ov::Tensor& tensor){
        auto compiled_model_ref = core->compile_model(fn, ov::test::utils::DEVICE_TEMPLATE);
        auto inf_req_ref = compiled_model_ref.create_infer_request();

        auto input = fn->input(0);
        inf_req_ref.set_tensor(input, tensor);
        inf_req_ref.infer();

        auto output = fn->get_result();
        auto out_ref = inf_req_ref.get_tensor(output);
        ref.push_back(out_ref);
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
            ov::test::utils::compare(ref[0], output_tensor);
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
            ov::test::utils::compare(ref[1], output_tensor);
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
} // namespace