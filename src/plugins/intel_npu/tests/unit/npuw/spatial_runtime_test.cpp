// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "just_sync_infer_request.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/opsets/opset13.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"

namespace {

class SpatialCPUInferRequest : public ov::ISyncInferRequest {
public:
    SpatialCPUInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled,
                           ov::InferRequest request,
                           std::shared_ptr<size_t> submissions)
        : ov::ISyncInferRequest(compiled),
          m_request(std::move(request)),
          m_submissions(std::move(submissions)) {
        for (size_t i = 0; i < compiled->inputs().size(); ++i) {
            set_tensor(compiled->inputs()[i], ov::get_tensor_impl(m_request.get_input_tensor(i)));
        }
        for (size_t i = 0; i < compiled->outputs().size(); ++i) {
            set_tensor(compiled->outputs()[i], ov::get_tensor_impl(m_request.get_output_tensor(i)));
        }
    }

    void infer() override {
        for (size_t i = 0; i < get_inputs().size(); ++i) {
            m_request.set_input_tensor(i, ov::make_tensor(get_tensor(get_inputs()[i])));
        }
        for (size_t i = 0; i < get_outputs().size(); ++i) {
            m_request.set_output_tensor(i, ov::make_tensor(get_tensor(get_outputs()[i])));
        }
        m_request.infer();
        ++*m_submissions;
    }

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }

    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return m_request.get_profiling_info();
    }

private:
    ov::InferRequest m_request;
    std::shared_ptr<size_t> m_submissions;
};

class SpatialCPUCompiledModel : public ov::ICompiledModel {
public:
    SpatialCPUCompiledModel(const std::shared_ptr<const ov::Model>& model,
                            const std::shared_ptr<const ov::IPlugin>& plugin,
                            ov::CompiledModel compiled)
        : ov::ICompiledModel(model, plugin),
          m_compiled(std::move(compiled)) {}

    void export_model(std::ostream& stream) const override {
        m_compiled.export_model(stream);
    }

    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_compiled.get_runtime_model();
    }

    void set_property(const ov::AnyMap& props) override {
        m_compiled.set_property(props);
    }

    ov::Any get_property(const std::string& name) const override {
        return m_compiled.get_property(name);
    }

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        return std::make_shared<SpatialCPUInferRequest>(shared_from_this(),
                                                        m_compiled.create_infer_request(),
                                                        submissions);
    }

    std::shared_ptr<size_t> submissions = std::make_shared<size_t>(0);

private:
    mutable ov::CompiledModel m_compiled;
};

class SpatialRuntimeTest : public testing::Test {
protected:
    static constexpr size_t width = 8;
    static constexpr size_t nway = 16;
    using Desc = ov::npuw::CompiledModelDescTestAccessor::SubmodelDesc;
    std::shared_ptr<ov::Core> cpu = std::make_shared<ov::Core>();
    std::shared_ptr<ov::test::npuw::NullPlugin> plugin = std::make_shared<ov::test::npuw::NullPlugin>();
    std::shared_ptr<testing::NiceMock<ov::MockICore>> core = std::make_shared<testing::NiceMock<ov::MockICore>>();
    std::shared_ptr<ov::npuw::JustInferRequest> spatial;
    ov::InferRequest reference;
    std::map<size_t, size_t> spatial_calls;

    static ov::AnyMap cpu_properties() {
        return {{ov::hint::inference_precision.name(), ov::element::f32},
                {ov::num_streams.name(), 1},
                {ov::inference_num_threads.name(), 1}};
    }

    void SetUp() override {
        using testing::_;
        ON_CALL(*core, get_property("NPU", ov::available_devices.name(), _))
            .WillByDefault(testing::Return(ov::Any(std::vector<std::string>{})));
        ON_CALL(*core, get_property("CPU", _, _))
            .WillByDefault([this](const std::string&, const std::string& name, const ov::AnyMap&) {
                return cpu->get_property("CPU", name);
            });
        ON_CALL(*core, get_supported_property(_, _, _))
            .WillByDefault([this](const std::string&, const ov::AnyMap& props, bool) {
                ov::AnyMap filtered;
                for (const auto& prop : cpu->get_property("CPU", ov::supported_properties)) {
                    if (prop.is_mutable() && props.count(prop)) {
                        filtered.emplace(prop, props.at(prop));
                    }
                }
                return filtered;
            });
        ON_CALL(
            *core,
            compile_model(testing::A<const std::shared_ptr<const ov::Model>&>(), testing::A<const std::string&>(), _))
            .WillByDefault([this](const std::shared_ptr<const ov::Model>& model,
                                  const std::string& device,
                                  const ov::AnyMap& props) {
                EXPECT_EQ(device, "CPU");
                auto config = props;
                for (const auto& prop : cpu_properties()) {
                    config[prop.first] = prop.second;
                }
                return ov::SoPtr<ov::ICompiledModel>{
                    std::make_shared<SpatialCPUCompiledModel>(model, plugin, cpu->compile_model(model, "CPU", config))};
            });
        plugin->set_core(core);
    }

    static std::shared_ptr<ov::Model> model(size_t range, bool dynamic_mask) {
        using namespace ov::opset13;
        auto input = std::make_shared<Parameter>(ov::element::f32, ov::Shape{1, range, width});
        input->set_friendly_name("input");
        input->output(0).set_names({"input"});
        ov::ParameterVector params{input};
        ov::Output<ov::Node> value = input;
        if (dynamic_mask) {
            auto mask = std::make_shared<Parameter>(ov::element::i64, ov::Shape{1, range});
            mask->set_friendly_name("attention_mask");
            mask->output(0).set_names({"attention_mask"});
            params.push_back(mask);
            auto converted = std::make_shared<Convert>(mask, ov::element::f32);
            auto expanded =
                std::make_shared<Unsqueeze>(converted, Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
            value = std::make_shared<Multiply>(value, expanded);
        }
        for (size_t block = 0; block < 4; ++block) {
            auto bias = Constant::create(ov::element::f32,
                                         ov::Shape{width},
                                         {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f});
            auto residual = std::make_shared<Add>(value, bias);
            value = ov::test::npuw::RMSNorm(width, ov::element::f32)(residual, "norm_" + std::to_string(block));
        }
        auto result = std::make_shared<Result>(value);
        return std::make_shared<ov::Model>(ov::ResultVector{result}, params, "spatial_runtime");
    }

    const Desc& desc(size_t idx) const {
        return *static_cast<const Desc*>(spatial->get_submodel_desc(idx));
    }

    void compile(size_t range, bool dynamic_mask = false) {
        auto graph = model(range, dynamic_mask);
        reference = cpu->compile_model(graph->clone(), "CPU", cpu_properties()).create_infer_request();
        ov::AnyMap props{{"NPUW_DEVICES", "CPU"},
                         {"NPUW_FOLD", "YES"},
                         {"NPUW_F16IC", "NO"},
                         {"NPUW_FUNCALL_ASYNC", "NO"},
                         {"NPUW_UNFOLD_IREQS", "NO"},
                         {"NPUW_ONLINE_PIPELINE", "SPATIAL"},
                         {"NPUW_ONLINE_KEEP_BLOCKS", "2"},
                         {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "1"},
                         {"NPUW_SPATIAL_NWAY", std::to_string(nway)},
                         {"NPUW_SPATIAL_DYN", dynamic_mask ? "YES" : "NO"}};
        auto compiled = std::make_shared<ov::npuw::CompiledModel>(graph->clone(), plugin, props);
        spatial = std::dynamic_pointer_cast<ov::npuw::JustInferRequest>(compiled->create_base_infer_request());
        ASSERT_NE(spatial, nullptr);
        for (size_t idx = 0; idx < spatial->total_subrequests(); ++idx) {
            const auto& call = desc(idx);
            if (call.replaced_by && desc(*call.replaced_by).spatial) {
                ++spatial_calls[*call.replaced_by];
            }
        }
        ASSERT_FALSE(spatial_calls.empty()) << "SPATIAL must produce a spatial function";
        size_t calls = 0;
        for (const auto& entry : spatial_calls) {
            const auto& body = desc(entry.first);
            ASSERT_TRUE(body.spatial.has_value());
            const auto& s = *body.spatial;
            EXPECT_EQ(s.range, range);
            EXPECT_EQ(s.nway, nway);
            ASSERT_GE(s.nway_iters, 2u);
            EXPECT_EQ(s.nway_iters, range / nway);
            if (range % nway) {
                ASSERT_GT(s.tail_size, 0u);
                EXPECT_EQ(s.tail_size, range % nway);
            } else {
                ASSERT_EQ(s.tail_size, 0u);
            }
            EXPECT_EQ(s.out_dim, 1u);
            ASSERT_FALSE(s.params.empty());
            ASSERT_NE(body.compiled_model, nullptr);
            EXPECT_EQ(body.compiled_model->outputs().front().get_shape()[1], nway);
            calls += entry.second;
        }
        ASSERT_GE(calls, 2u) << "The compute function must be repeated";
    }

    void run(size_t range, size_t valid_begin, size_t valid_end, bool dynamic_mask = false, size_t salt = 0) {
        ov::Tensor input(ov::element::f32, {1, range, width});
        for (size_t row = 0; row < range; ++row) {
            for (size_t col = 0; col < width; ++col) {
                input.data<float>()[row * width + col] =
                    (static_cast<int>((row * 17 + col * 13 + salt) % 101) - 50) / 37.0f + row / 64.0f;
            }
        }
        reference.set_tensor("input", input);
        auto base = std::static_pointer_cast<ov::npuw::IBaseInferRequest>(spatial);
        base->set_tensor(base->get_inputs()[0], ov::get_tensor_impl(input));
        if (dynamic_mask) {
            ov::Tensor mask(ov::element::i64, {1, range});
            for (size_t row = 0; row < range; ++row) {
                mask.data<int64_t>()[row] = row >= valid_begin && row < valid_end ? 1 : 0;
            }
            reference.set_tensor("attention_mask", mask);
            base->set_tensor(base->get_inputs()[1], ov::get_tensor_impl(mask));
        }
        for (const auto& entry : spatial_calls) {
            auto child = std::dynamic_pointer_cast<SpatialCPUCompiledModel>(desc(entry.first).compiled_model._ptr);
            ASSERT_NE(child, nullptr);
            *child->submissions = 0;
        }
        auto output = base->get_tensor(base->get_outputs()[0]);
        reference.infer();
        base->infer();
        const auto expected = reference.get_output_tensor();
        ASSERT_EQ(output->get_shape(), expected.get_shape());
        const auto* actual_data = output->data<const float>();
        const auto* expected_data = expected.data<const float>();
        for (size_t row = valid_begin; row < valid_end; ++row) {
            for (size_t col = 0; col < width; ++col) {
                const auto i = row * width + col;
                ASSERT_NEAR(actual_data[i], expected_data[i], 1e-5f + 1e-5f * std::abs(expected_data[i]))
                    << "row=" << row << " col=" << col;
            }
        }
        for (const auto& entry : spatial_calls) {
            const auto& body = desc(entry.first);
            const auto& s = *body.spatial;
            size_t slices = 0;
            for (size_t offset = 0; offset < range; offset += nway) {
                const auto length = std::min(nway, range - offset);
                slices += !dynamic_mask || !(offset + length < valid_begin || offset >= valid_end);
            }
            if (dynamic_mask) {
                EXPECT_LT(slices, s.nway_iters + (s.tail_size > 0));
                EXPECT_GT(slices, 0u);
            }
            auto child = std::dynamic_pointer_cast<SpatialCPUCompiledModel>(body.compiled_model._ptr);
            EXPECT_EQ(*child->submissions, entry.second * slices);
        }
    }
};

TEST_F(SpatialRuntimeTest, SpatialFullSlices) {
    ASSERT_NO_FATAL_FAILURE(compile(64));
    ASSERT_NO_FATAL_FAILURE(run(64, 0, 64));
}

TEST_F(SpatialRuntimeTest, SpatialWithTail) {
    ASSERT_NO_FATAL_FAILURE(compile(70));
    ASSERT_NO_FATAL_FAILURE(run(70, 0, 70));
}

TEST_F(SpatialRuntimeTest, SpatialDynAttentionMask) {
    ASSERT_NO_FATAL_FAILURE(compile(70, true));
    ASSERT_NO_FATAL_FAILURE(run(70, 33, 70, true));
    ASSERT_NO_FATAL_FAILURE(run(70, 1, 17, true, 29));
}

}  // namespace
