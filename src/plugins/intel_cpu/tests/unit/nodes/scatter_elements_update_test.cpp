// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/shape.hpp>
#include <openvino/core/strides.hpp>
#include <openvino/core/type/element_type.hpp>
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "openvino/core/except.hpp"
#include "openvino/core/partial_shape.hpp"

#include "graph.h"

#include "nodes/input.h"
#include "nodes/scatter_update.h"

#include "openvino/op/result.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/scatter_elements_update.hpp"

#include "cpu_memory.h"
#include "cpu_tensor.h"
#include "openvino/runtime/itensor.hpp"

#include <common/memory_desc_wrapper.hpp>
#include <dnnl.hpp>

using namespace ov::intel_cpu;
using namespace ov::intel_cpu::node;
using namespace ov::op;

inline MemoryPtr create_memory(ov::element::Type prc, const ov::PartialShape& shape) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    CpuBlockedMemoryDescPtr desc;
    desc = std::make_shared<CpuBlockedMemoryDesc>(prc, Shape(shape));
    return std::make_shared<Memory>(eng, desc);
}

class SEUCPUTestNodeUtil {
public:
    void getScatterUpdateNode(const ov::element::Type& dataPrec,
                              const ov::element::Type& intPrec,
                              const ov::PartialShape& data_shape,
                              const ov::PartialShape& updates_shape,
                              const ov::PartialShape& indices_shape,
                              const int32_t axis) {
        using Reduction = ov::op::v12::ScatterElementsUpdate::Reduction;
        auto data_param = std::make_shared<v0::Parameter>(dataPrec, data_shape);
        auto updates_param = std::make_shared<v0::Parameter>(dataPrec, updates_shape);
        auto indices_param = std::make_shared<v0::Parameter>(intPrec, indices_shape);
        auto axis_param = std::make_shared<v0::Parameter>(ov::element::i32, ov::Shape{});
        auto scatterupdate_result = std::make_shared<v12::ScatterElementsUpdate>(data_param, indices_param, updates_param, axis_param, Reduction::SUM);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(scatterupdate_result)};
        const auto model = std::make_shared<const ov::Model>(results, ov::ParameterVector{data_param, indices_param, updates_param, axis_param}, "scatterupdates");

        Config conf;
        conf.rtCacheCapacity = 100;
        const auto context = std::make_shared<const GraphContext>(conf, nullptr, false);
        std::shared_ptr<Graph> graph = std::shared_ptr<Graph>(new Graph());
        graph->CreateGraph(model, context);

        NodePtr seu_node;
        for (auto &node : graph->GetNodes()) {
            if (node->getType() == Type::ScatterElementsUpdate) {
                seu_node = node;
            }
        }
        scatterupdateNode = std::dynamic_pointer_cast<ScatterUpdate>(seu_node);
    }

protected:
    void checkData(const ov::intel_cpu::IMemory& inputMemory,
                    const ov::intel_cpu::IMemory& outputMemory,
                    const ov::element::Type& prescision) {
        auto srcData = inputMemory.getData();
        auto dstData = outputMemory.getData();
        auto mdInput = inputMemory.getDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
        auto mdOutput = outputMemory.getDescWithType<DnnlMemoryDesc>()->getDnnlDesc();

        const dnnl::impl::memory_desc_wrapper mdwInput(mdInput.get());
        const dnnl::impl::memory_desc_wrapper mdwOutput(mdOutput.get());
        auto nelems = mdwInput.nelems();

        for (dnnl::impl::dim_t i = 0; i < nelems; ++i) {
            auto srcOffset = mdwInput.off_l(i, false);
            auto dstOffset = mdwOutput.off_l(i, false);
            switch (prescision) {
            case ov::element::f32: {
                auto s = *(static_cast<float*>(srcData) + srcOffset);
                auto d = *(static_cast<float*>(dstData) + dstOffset);
                ASSERT_EQ(s, d) << "mismatch at position " << i;
                break;
            }
            case ov::element::i32: {
                auto s = *(static_cast<int32_t*>(srcData) + srcOffset);
                auto d = *(static_cast<int32_t*>(dstData) + dstOffset);
                ASSERT_EQ(s, d) << "mismatch at position " << i;
                break;
            }
            default:
                FAIL() << "Unsupported data precision in the test" << prescision.get_type_name();
            }
        }
    }

    void fillData(const ov::intel_cpu::IMemory& inputMemory, const ov::element::Type& prec,
                  const int32_t max_value = std::numeric_limits<int32_t>::max(),
                  const int32_t divisor = 1) {
        ov::intel_cpu::DnnlMemoryDescPtr dnnlMdInput = inputMemory.getDescWithType<DnnlMemoryDesc>();
        const dnnl::impl::memory_desc_wrapper mdInput{dnnlMdInput->getDnnlDesc().get()};
        auto elemNum = mdInput.nelems();
        auto inputData = inputMemory.getData();
        switch (prec) {
        case ov::element::f32:
            for (int64_t i = 0; i < elemNum; ++i)
                *(static_cast<float*>(inputData) + mdInput.off_l(i, false)) = static_cast<float>(i % max_value);
            break;
        case ov::element::i32:
            for (int64_t i = 0; i < elemNum; ++i)
                *(static_cast<int32_t*>(inputData) + mdInput.off_l(i, false)) = static_cast<int32_t>((i / divisor) % max_value);
            break;
        default:
            FAIL() << "Unsupported data precision in the test" << prec.get_type_name();
        }
    }

    std::shared_ptr<ScatterUpdate> scatterupdateNode;
};

using SEU1DTestTestParamSet = std::tuple<int32_t,int32_t>;

class SEU1DTest : public ::testing::Test,
                  public ::testing::WithParamInterface<SEU1DTestTestParamSet>,
                  public SEUCPUTestNodeUtil {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SEU1DTestTestParamSet>& obj) {
        int32_t algo, axis;
        std::tie(algo, axis) = obj.param;
        std::ostringstream result;
        result << "(";
        result << "ALGO=" << algo;
        result << "_axis=" << axis;
        result << ")";
        return result.str();
    }

protected:
    void SetUp() override {
        std::tie(use_algo, use_axis) = this->GetParam();

        getScatterUpdateNode(dataPrec, intPrec, data_shape, updates_shape, indices_shape, use_axis);
        ASSERT_TRUE(scatterupdateNode);
    }

    void Run() {
        auto data_memptr = create_memory(dataPrec, data_shape);
        auto updates_memptr = create_memory(dataPrec, updates_shape);
        auto indices_memptr = create_memory(intPrec, indices_shape);
        auto indices1D_memptr = create_memory(intPrec, ov::PartialShape({indices_shape[use_axis]}));

        // generate inputs
        memset(data_memptr->getData(), 0, data_memptr->getSize());  // zeros
        fillData(*updates_memptr, dataPrec);
        const int32_t divisor = use_axis == 0 ?  data_shape[1].get_length(): 1;
        fillData(*indices_memptr, intPrec, data_shape[use_axis].get_length(), divisor);
        fillData(*indices1D_memptr, intPrec, data_shape[use_axis].get_length());

        // do
        if (use_algo == 0) {
            auto start = std::chrono::steady_clock::now();
            scatterupdateNode->scatterElementsUpdate(data_memptr, indices_memptr, updates_memptr, use_axis, ScatterUpdate::ReduceAdd<float>{});
            auto end = std::chrono::steady_clock::now();
            std::cout << "============================ " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
        } else if (use_algo == 1) {
            auto start = std::chrono::steady_clock::now();
            scatterupdateNode->scatterElementsUpdateAdvance(data_memptr, indices_memptr, updates_memptr, use_axis, ScatterUpdate::ReduceAdd<float>{});
            auto end = std::chrono::steady_clock::now();
            std::cout << "============================(Advance) " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
        } else {
            auto start = std::chrono::steady_clock::now();
            scatterupdateNode->scatterElementsUpdate1D(data_memptr, indices1D_memptr, updates_memptr, use_axis, ScatterUpdate::ReduceAdd<float>{});
            auto end = std::chrono::steady_clock::now();
            std::cout << "============================(1D) " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
        }

        // validate
        checkData(*updates_memptr, *data_memptr, dataPrec);
    }

private:
    const ov::element::Type dataPrec = ov::element::f32;
    const ov::element::Type intPrec = ov::element::i32;

    ov::PartialShape data_shape{1024, 4096};
    ov::PartialShape updates_shape{1024, 4096};
    ov::PartialShape indices_shape{1024, 4096};

    int32_t use_algo = 0;
    int32_t use_axis = 0;
};

TEST_P(SEU1DTest, performance) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_SEU1DTest,
                         SEU1DTest,
                         ::testing::Combine(::testing::Values(0, 1, 2),   // use_algo
                                            ::testing::Values(0, 1)),     // use_axis
                         SEU1DTest::getTestCaseName);
