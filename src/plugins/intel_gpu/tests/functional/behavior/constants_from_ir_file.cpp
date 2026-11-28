// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/core.hpp"

namespace {

// Constants of a model read from an IR file, with and without ENABLE_MMAP. With mmap the constant data stays in the
// mapped .bin file; on discrete GPUs it is uploaded to usm_device memory at compile time, and some drivers fail a
// direct copy from such memory (DG2 on Windows: clEnqueueMemcpyINTEL returns CL_OUT_OF_RESOURCES when the data does
// not start on a page boundary). The 2-byte constant of the Add is serialized first, so the weight starts at offset 2.
// The weight is stored as [n, k] with transpose_b, the layout FullyConnected uses, so the plugin uploads the mapped
// data itself; a [k, n] weight would be transposed into a new buffer first.
// The weight is larger than 64 MiB, the size of the buffer constants are uploaded through, so it takes two uploads.
class ConstantsFromIRFile : public ::testing::Test, public ::testing::WithParamInterface<bool> {
public:
    static std::string get_test_case_name(::testing::TestParamInfo<bool> obj) {
        return std::string("mmap=") + (obj.param ? "true" : "false");
    }

protected:
    static constexpr size_t k = 4096;
    static constexpr size_t n = 8448;  // f16 weight of 8448 x 4096 = 66 MiB
    static constexpr float shift = 0.5f;

    std::shared_ptr<ov::Core> core;
    std::shared_ptr<ov::Model> model;  // the model the IR file was serialized from
    std::string xml_path;
    std::string bin_path;

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        core = ov::test::utils::PluginCache::get().core();

        const auto prefix = ov::test::utils::generateTestFilePrefix();
        xml_path = prefix + ".xml";
        bin_path = prefix + ".bin";

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, k});
        auto shifted = std::make_shared<ov::op::v1::Add>(input, ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {shift}));
        const auto weight_data = ov::test::utils::create_and_fill_tensor(ov::element::f16, ov::Shape{n, k}, ov::test::utils::InputGenerateData(-1, 2, 1000, 1));
        auto weight = std::make_shared<ov::op::v0::Constant>(weight_data);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(shifted, weight, false, true);
        model = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(matmul)}, ov::ParameterVector{input});
        ov::pass::Serialize(xml_path, bin_path).run_on_model(model);
    }

    void TearDown() override {
        ov::test::utils::removeIRFiles(xml_path, bin_path);
    }
};

// The model read from the file must compile and compute (input + shift) x weight. The reference is computed here,
// independent of how the plugin uploads the weight: a piece uploaded to the wrong place changes outputs by about 10,
// far above the tolerance, which only covers f16 accumulation over k = 4096 products of values below 1.5.
TEST_P(ConstantsFromIRFile, InferMatchesReference) {
    const auto file_model = core->read_model(xml_path, std::string{}, ov::AnyMap{ov::enable_mmap(GetParam())});
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(file_model, ov::test::utils::DEVICE_GPU));

    auto request = compiled_model.create_infer_request();
    const auto input = ov::test::utils::create_and_fill_tensor(ov::element::f16, ov::Shape{1, k}, ov::test::utils::InputGenerateData(-1, 2, 100, 7));
    request.set_input_tensor(input);
    request.infer();

    const auto weight = ov::as_type_ptr<ov::op::v0::Constant>(model->get_results()[0]->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(1));
    const auto* w = weight->get_data_ptr<ov::float16>();
    const auto* x = input.data<ov::float16>();
    ov::Tensor expected(ov::element::f16, ov::Shape{1, n});
    auto* e = expected.data<ov::float16>();
    for (size_t j = 0; j < n; ++j) {
        float sum = 0.f;
        for (size_t i = 0; i < k; ++i) {
            const float xs = static_cast<float>(ov::float16(static_cast<float>(x[i]) + shift));  // the Add runs in f16
            sum += xs * static_cast<float>(w[j * k + i]);
        }
        e[j] = ov::float16(sum);
    }
    ov::test::utils::compare(expected, request.get_output_tensor(), ov::element::f16, 0.25, 0.01);
}

INSTANTIATE_TEST_SUITE_P(smoke_ConstantsFromIRFile, ConstantsFromIRFile, ::testing::Bool(), ConstantsFromIRFile::get_test_case_name);

}  // namespace
