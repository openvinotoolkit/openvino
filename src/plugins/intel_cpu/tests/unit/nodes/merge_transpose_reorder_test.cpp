// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <common/blocked_desc_creator.h>
#include <cpu_types.h>
#include <edge.h>
#include <gtest/gtest.h>
#include <ie_common.h>
#include <memory_desc/cpu_memory_desc_utils.h>
#include <memory_desc/dnnl_memory_desc.h>
#include <node.h>
#include <nodes/reorder.h>
#include <graph.h>

#include <common/memory_desc_wrapper.hpp>
#include <dnnl.hpp>
#include <utility>

#include "common_test_utils/common_utils.hpp"
#include "cache/multi_cache.h"
#include "nodes/input.h"

#include "utils/rt_info/memory_formats_attribute.hpp"
#include "ov_models/builders.hpp"
#include "common_test_utils/test_assertions.hpp"
#include <exec_graph_info.hpp>
#include "transformations/utils/utils.hpp"
#include "openvino/runtime/system_conf.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

using namespace InferenceEngine;
using namespace ov::intel_cpu;

inline void fillData(const ov::intel_cpu::IMemory& inputMemory,
                     const ov::element::Type& prescision) {
    ov::intel_cpu::DnnlMemoryDescPtr dnnlMdInput = inputMemory.getDescWithType<DnnlMemoryDesc>();
    const dnnl::impl::memory_desc_wrapper mdInput{dnnlMdInput->getDnnlDesc().get()};
    auto elemNum = mdInput.nelems();
    auto inputData = inputMemory.getData();
    switch (prescision) {
    case ov::element::f32:
        for (int64_t i = 0; i < elemNum; ++i)
            *(static_cast<float*>(inputData) + mdInput.off_l(i, false)) = static_cast<float>(i);
        break;
    case ov::element::i8:
        for (int64_t i = 0; i < elemNum; ++i)
            *(static_cast<int8_t*>(inputData) + mdInput.off_l(i, false)) = static_cast<int8_t>(i);
        break;
    default:
        FAIL() << "Unsupported data precision in the test" << prescision.to_string();
    }
}

inline void checkResult(const ov::intel_cpu::IMemory& outputMemory,
                        const std::vector<float> reference,
                        const ov::element::Type& prescision) {
    auto dstData = outputMemory.getData();
    auto mdOutput = outputMemory.getDescWithType<DnnlMemoryDesc>()->getDnnlDesc();

    const dnnl::impl::memory_desc_wrapper mdwOutput(mdOutput.get());
    auto nelems = mdwOutput.nelems();

    ASSERT_EQ(reference.size(), nelems) << "mismatch data size";

    for (dnnl::impl::dim_t i = 0; i < nelems; ++i) {
        auto dstOffset = mdwOutput.off_l(i, false);
        switch (prescision) {
        case ov::element::f32: {
            auto s = reference[i];
            auto d = *(static_cast<float*>(dstData) + dstOffset);
            ASSERT_EQ(s, d) << "mismatch at position " << i;
            break;
        }
        case ov::element::i8: {
            auto s = static_cast<int8_t>(reference[i]);
            auto d = *(static_cast<int8_t*>(dstData) + dstOffset);
            ASSERT_EQ(s, d) << "mismatch at position " << i;
            break;
        }
        default:
            FAIL() << "Unsupported data precision in the test" << prescision.to_string();
        }
    }
}

inline const std::vector<float>& cal_refs() {
    static const std::vector<float> result_vals =
        {0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 3.0, 7.0, 11.0, 15.0, 19.0, 23.0};
    return result_vals;
}

using MergeTransposeReorderParams = std::tuple<
        ov::Shape,                       // Input shape
        ov::element::Type                // Input precision
>;

const std::vector<ov::Shape> input_shapes = {
    ov::Shape{1, 2, 3, 2, 2}
};
const std::vector<ov::element::Type> types = {
        ov::element::f32,
        ov::element::i8
};
const auto mergeTransposeAndReorderParams = ::testing::Combine(
        ::testing::ValuesIn(input_shapes),
        ::testing::ValuesIn(types)
);

/*
 * MergeTransposeReorderInplacedCPUTest to test the CPU plugin-in MergeTransposeReorder graph optimizer
 * under the circumstance that the upstream node or downstream node is inPlaced thereby the inserted Reorder
 * cannot be optimized.
 */
class MergeTransposeReorderInplacedCPUTest : public ::testing::Test,
                       public ::testing::WithParamInterface<MergeTransposeReorderParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MergeTransposeReorderParams>& obj) {
        std::ostringstream result;
        ov::Shape inShape;
        ov::element::Type inPrec;
        std::tie(inShape, inPrec) = obj.param;

        result << "IS=" << ov::test::utils::vec2str(inShape) << "_";
        result << "Precision=" << inPrec.to_string();

        return result.str();
    }

    void Run() {
        generateInput(inputShape, inPrec);
        infer();
        validate();
    }

    bool isSupportedTestCase() {
        if (ov::with_cpu_x86_avx512f())
            return true;
        return false;
    }

protected:
    void generateInput(const ov::Shape& input_shape,
            const ov::element::Type& input_prec) {
        auto input0 = m_graph->getInputNodeByName("param0");
        const auto& input0Mem = input0->getChildEdgesAtPort(0).front()->getMemory();
        fillData(input0Mem, input_prec);
    }

    void infer() {
        //
        m_graph->ResetInferCount();
        m_graph->Infer();
    }

    void validate(void) const {
        CheckTransposeCount(0);
        CheckReorderOptimized(std::string("_fake"), false);  // the fused node is of name "reshape_abcd_acdb_fake"

        auto getOutputEdgeByName = [&](const std::string &name) -> EdgePtr {
            const auto &outMap = m_graph->GetOutputNodesMap();
            for (const auto &out : m_function->get_results()) {
                if (out->get_friendly_name() != name) continue;

                const auto prev = out->input_value(0);
                const auto inputID = ov::op::util::create_ie_output_name(prev);
                auto outNode = outMap.find(inputID);
                if (outNode != outMap.end()) {
                    return outNode->second->getParentEdgeAt(0);
                }
            }
            return nullptr;
        };

        auto output0 = getOutputEdgeByName("result0");
        ASSERT_TRUE(output0) << "Unable to get output result0";
        const auto& output0Mem = output0->getMemory();
        checkResult(output0Mem, cal_refs(), inPrec);
    }

    void SetUp() override {
        std::tie(inputShape, inPrec) = this->GetParam();
        ASSERT_TRUE(inputShape.size() == 5) << "Only inputShape of size 5 is supported by this unit test.";

        if (!isSupportedTestCase()) {
            GTEST_SKIP() << "Skip test since such combination of parameters is not supported." << std::endl;
        }

        CreateGraph();
    }

    /*  graph typology
                ---------                               
                |Input  |
                ---------
                    |
                ---------
                |Reshape|
                ---------
                    |
             |---------------|
             |   ----------  |
             |   |Transpose| |
             |   ---------   |
             |       |       |
             |   ---------   |
             |   |Reorder |  |          <*NOTE: Reorder is inheristically inserted since Multiply is asking NSPC input.*>
             |   ---------   |
             |---------------|
                    |
                ----------
                |Multiply |
                ----------
                    |
                ---------
                |Output |
                ---------
    */
   const std::shared_ptr<const ov::Model>& build_model() {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inPrec, ov::Shape(inputShape))};
        params[0]->set_friendly_name("param0");

        // reshape: shape(a,b,c,d,e) -> shape(a,b,c,d*e)
        auto reshape_param = std::vector<int32_t>{0, 0, 0, -1};
        const auto constReshape = ngraph::builder::makeConstant<int>(ov::element::i32, {reshape_param.size()}, reshape_param);
        constReshape->set_friendly_name("constReshape");
        const auto reshape = std::make_shared<ov::op::v1::Reshape>(params[0], constReshape, true/*special_zero*/);
        reshape->set_friendly_name("reshape");

        // transpose
        auto order = std::vector<int32_t>{0, 3, 1, 2};
        auto constOrder = ngraph::builder::makeConstant(ngraph::element::i32, {order.size()}, order);
        constOrder->set_friendly_name("constOrder");
        auto transpose = std::make_shared<ngraph::opset5::Transpose>(reshape, constOrder);
        std::string memFmt1 = "nchw";
        transpose->get_rt_info()[InputMemoryFormats::get_type_info_static()] = ov::intel_cpu::InputMemoryFormats("cpu:" + memFmt1);
        transpose->get_rt_info()[OutputMemoryFormats::get_type_info_static()] = ov::intel_cpu::OutputMemoryFormats("cpu:" + memFmt1);
        transpose->get_rt_info()["enforceBF16evenForGraphTail"] = true;

        // multiply
        const auto constMulitply = ngraph::builder::makeConstant<float>(inPrec, {1}, {1.0f});
        const auto mulitply = std::make_shared<ngraph::opset1::Multiply>(transpose, constMulitply);
        mulitply->set_friendly_name("mulitply");
        std::string memFmt = "nhwc";
        mulitply->get_rt_info()[InputMemoryFormats::get_type_info_static()] = ov::intel_cpu::InputMemoryFormats("cpu:" + memFmt);
        mulitply->get_rt_info()[OutputMemoryFormats::get_type_info_static()] = ov::intel_cpu::OutputMemoryFormats("cpu:" + memFmt);
        mulitply->get_rt_info()["enforceBF16evenForGraphTail"] = true;

        ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(mulitply)};
        results[0]->set_friendly_name("result0");
        m_function = std::make_shared<const ngraph::Function>(results, params, "FuseTransposeReorder");

        return m_function;
    }

    void CreateGraph() {
        //
        Config conf;
        conf.rtCacheCapacity = 100;
        auto context = std::make_shared<GraphContext>(conf, nullptr, nullptr, false);
        const dnnl::engine cpuEngine = context->getEngine();

        m_graph = std::unique_ptr<Graph>(new Graph());

        const std::shared_ptr<const ov::Model>& function = build_model();
        m_graph->CreateGraph(function, context);
    }

    // helper to check if Transpose node is fused.
    void CheckTransposeCount(const size_t expectedTransposeCount) const {
        auto function = m_graph->dump();
        ASSERT_NE(nullptr, function);
        size_t actualTransposeCount = 0;
        for (const auto &node : function->get_ops()) {
            const auto & rtInfo = node->get_rt_info();
            auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
                auto it = rtInfo.find(paramName);
                IE_ASSERT(rtInfo.end() != it);
                return it->second.as<std::string>();
            };
            if (getExecValue(ExecGraphInfoSerialization::LAYER_TYPE) == "Transpose") {
                actualTransposeCount++;
            }
        }

        ASSERT_EQ(expectedTransposeCount, actualTransposeCount);
    }

    // helper to check isOptimized of Reorder node with a part of its name
    void CheckReorderOptimized(const std::string &patial_name, const bool expectedOptimized) const {
        const std::vector<NodePtr>& graph_nodes = m_graph->GetNodes();
        size_t actualCount = 0;
        for (auto &node : graph_nodes) {
            auto reorder_node = std::dynamic_pointer_cast<node::Reorder>(node);
            if (reorder_node && node->getName().find(patial_name) != std::string::npos) {
                ASSERT_EQ(expectedOptimized, reorder_node->getOptimized());
                actualCount++;
            }
        }

        ASSERT_EQ(1, actualCount);
    }

private:
    ov::Shape inputShape;
    ov::element::Type inPrec;

    std::unique_ptr<Graph> m_graph;
    std::shared_ptr<const ov::Model> m_function;
}; // class MergeTransposeReorderInplacedCPUTest

TEST_P(MergeTransposeReorderInplacedCPUTest, CompareResult) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_MergeTransposeReorderInplaced,
                         MergeTransposeReorderInplacedCPUTest,
                         mergeTransposeAndReorderParams,
                         MergeTransposeReorderInplacedCPUTest::getTestCaseName);
