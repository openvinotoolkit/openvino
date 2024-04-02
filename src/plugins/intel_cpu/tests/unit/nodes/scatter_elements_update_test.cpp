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

using namespace ov::intel_cpu;
using namespace ov::intel_cpu::node;
using namespace ov::op;

using SEU1DTest = ::testing::Test;

inline MemoryPtr create_memory(ov::element::Type prc, const ov::PartialShape& shape) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    CpuBlockedMemoryDescPtr desc;
    desc = std::make_shared<CpuBlockedMemoryDesc>(prc, Shape(shape));
    return std::make_shared<Memory>(eng, desc);
}

TEST_F(SEU1DTest, performance) {
    constexpr ov::element::Type dataPrec = ov::element::f32;
    constexpr ov::element::Type intPrec = ov::element::i32;
    constexpr int32_t axis = 0;
    ov::PartialShape data_shape{1024, 4096};
    ov::PartialShape updates_shape{1024, 4096};
    ov::PartialShape indices_shape(updates_shape);
    ov::PartialShape indices1D_shape{1024};

    auto getScatterUpdateNode = [&]() {
        using Reduction = ov::op::v12::ScatterElementsUpdate::Reduction;
        auto data_param = std::make_shared<v0::Parameter>(dataPrec, data_shape);
        auto updates_param = std::make_shared<v0::Parameter>(dataPrec, updates_shape);
        auto indices_param = std::make_shared<v0::Parameter>(intPrec, indices_shape);
        auto axis_const = std::make_shared<v0::Constant>(intPrec, ov::Shape{}, std::vector<int>{axis});
        auto scatterupdate_result = std::make_shared<v12::ScatterElementsUpdate>(data_param, indices_param, updates_param, axis_const, Reduction::SUM);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(scatterupdate_result)};
        const auto model = std::make_shared<const ov::Model>(results, ov::ParameterVector{data_param, indices_param, updates_param}, "scatterupdates");

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
        return std::dynamic_pointer_cast<ScatterUpdate>(seu_node);
    };
    const std::shared_ptr<ScatterUpdate> scatterupdateNode = getScatterUpdateNode();
    ASSERT_TRUE(scatterupdateNode);
    
    auto data_memptr = create_memory(dataPrec, data_shape);
    auto updates_memptr = create_memory(dataPrec, updates_shape);
    auto indices_memptr = create_memory(intPrec, indices_shape);

    const char* penv = getenv("USE_ALGO");
    const int32_t use_algo = penv ? std::atoi(penv) : 0;
    if (use_algo == 0) {
        auto start = std::chrono::steady_clock::now();
        scatterupdateNode->scatterElementsUpdate<float>(data_memptr, indices_memptr, updates_memptr, axis, scatter_elements_update::ReduceAdd{});
        auto end = std::chrono::steady_clock::now();
        std::cout << "============================ " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    } else if (use_algo == 1) {
        auto start = std::chrono::steady_clock::now();
        scatterupdateNode->scatterElementsUpdateAdvance<float>(data_memptr, indices_memptr, updates_memptr, axis, scatter_elements_update::ReduceAdd{});
        auto end = std::chrono::steady_clock::now();
        std::cout << "============================(Advance) " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    } else {
        auto indices1D_memptr = create_memory(intPrec, indices1D_shape);
        auto start = std::chrono::steady_clock::now();
        scatterupdateNode->scatterElementsUpdate1D<float>(data_memptr, indices1D_memptr, updates_memptr, axis, scatter_elements_update::ReduceAdd{});
        auto end = std::chrono::steady_clock::now();
        std::cout << "============================(1D) " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }
}
