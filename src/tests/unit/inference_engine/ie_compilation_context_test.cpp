// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <gtest/gtest.h>
#include <fstream>
#include <thread>
#include <chrono>

#include "compilation_context.hpp"
#include "ngraph/function.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/variant.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"
#include "cpp/ie_cnn_network.h"

#include "common_test_utils/test_constants.hpp"

using namespace InferenceEngine;
using namespace ngraph;
using namespace ::testing;
using namespace std::chrono;

static std::string generateTestFilePrefix() {
    // Generate unique file names based on test name, thread id and timestamp
    // This allows execution of tests in parallel (stress mode)
    auto testInfo = UnitTest::GetInstance()->current_test_info();
    std::string testName = testInfo->test_case_name();
    testName += testInfo->name();
    testName = std::to_string(std::hash<std::string>()(testName));
    std::stringstream ss;
    auto ts = duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch());
    ss << testName << "_" << std::this_thread::get_id() << "_" << ts.count();
    testName = ss.str();
    return testName;
}

class FileGuard {
    std::string m_fileName;
public:
    explicit FileGuard(std::string name): m_fileName(std::move(name)) {}
    ~FileGuard() { std::remove(m_fileName.c_str()); }
};

class NetworkContext_CalcFileInfoTests : public Test {
public:
    std::string m_fileName = "test.blob";

    static void createFile(const std::string& fileName, std::size_t size = 1) {
        std::ofstream str(fileName, std::ios::binary);
        if (!str.good()) {
            GTEST_SKIP();
        }
        for (std::size_t i = 0; i < size; i++)
            str.put('a');
    }

    // Sets up the test fixture.
    void SetUp() override {
        auto testName = generateTestFilePrefix();
        m_fileName = testName + m_fileName;
        createFile(m_fileName);
    }

    // Tears down the test fixture.
    void TearDown() override {
        std::remove(m_fileName.c_str());
    }
};

TEST_F(NetworkContext_CalcFileInfoTests, NoFile) {
    ASSERT_NE(NetworkCompilationContext::calculateFileInfo("notexisting.abc"),
              NetworkCompilationContext::calculateFileInfo("notexisting2.abc"));

    std::string fileName(100, 'a');
    std::string fileName2(fileName);
    ASSERT_EQ(NetworkCompilationContext::calculateFileInfo(fileName),
              NetworkCompilationContext::calculateFileInfo(fileName2));
}

TEST_F(NetworkContext_CalcFileInfoTests, ExistingFile) {
    ASSERT_EQ(NetworkCompilationContext::calculateFileInfo(m_fileName),
              NetworkCompilationContext::calculateFileInfo(m_fileName));
}

TEST_F(NetworkContext_CalcFileInfoTests, ExistingDiffFiles) {
    auto hash1 = NetworkCompilationContext::calculateFileInfo(m_fileName);
    std::string newName = m_fileName + "2";
    std::rename(m_fileName.c_str(), newName.c_str());
    m_fileName = std::move(newName);
    auto hash2 = NetworkCompilationContext::calculateFileInfo(m_fileName);
    ASSERT_NE(hash1, hash2);
}

TEST_F(NetworkContext_CalcFileInfoTests, ExistingFile_sameAbsPath) {
    std::string file1 = m_fileName;
    std::string file2 = std::string(".") + CommonTestUtils::FileSeparator + m_fileName;
    ASSERT_EQ(NetworkCompilationContext::calculateFileInfo(file1),
              NetworkCompilationContext::calculateFileInfo(file2)) <<
              "Hash of [" << file1 << "] is not equal to hash of [" << file2 << "]";
}

TEST_F(NetworkContext_CalcFileInfoTests, DateModified) {
    auto info1 = NetworkCompilationContext::calculateFileInfo(m_fileName);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    createFile(m_fileName);
    auto info2 = NetworkCompilationContext::calculateFileInfo(m_fileName);
    ASSERT_NE(info1, info2);
}

TEST_F(NetworkContext_CalcFileInfoTests, SizeModified) {
    createFile(m_fileName, 1);
    auto info1 = NetworkCompilationContext::calculateFileInfo(m_fileName);
    createFile(m_fileName, 2);
    auto info2 = NetworkCompilationContext::calculateFileInfo(m_fileName);
    ASSERT_NE(info1, info2);
}

////////////////////////////////////////////////////

static std::shared_ptr<ngraph::Function> create_simple_function() {
    // This example is taken from docs, shows how to create ngraph::Function
    //
    // Parameter--->Multiply--->Add--->Result
    //    Constant---'          /
    //              Constant---'

    // Create opset6::Parameter operation with static shape
    auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i8, ngraph::Shape{3, 1, 2});
    data->set_friendly_name("Parameter");
    data->get_output_tensor(0).set_names({"parameter"});

    auto mul_constant = ngraph::opset6::Constant::create(ngraph::element::i8, ngraph::Shape{1}, {3});
    mul_constant->set_friendly_name("mul_constant");
    mul_constant->get_output_tensor(0).set_names({"mul_constant"});
    auto mul = std::make_shared<ngraph::opset6::Multiply>(data, mul_constant);
    mul->set_friendly_name("mul");
    mul->get_output_tensor(0).set_names({"mul"});

    auto add_constant = ngraph::opset6::Constant::create(ngraph::element::i8, ngraph::Shape{1}, {2});
    add_constant->set_friendly_name("add_constant");
    add_constant->get_output_tensor(0).set_names({"add_constant"});
    auto add = std::make_shared<ngraph::opset6::Add>(mul, add_constant);
    add->set_friendly_name("add");
    add->get_output_tensor(0).set_names({"add"});

    // Create opset3::Result operation
    auto res = std::make_shared<ngraph::opset6::Result>(add);
    res->set_friendly_name("res");

    // Create nGraph function
    auto func = std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data});
    return func;
}

static CNNNetwork createNetwork() {
    CNNNetwork res(create_simple_function());
    return res;
}

static CNNNetwork createNetworkWithLayout(const ov::Layout& layout) {
    auto fun = create_simple_function();
    fun->get_parameters()[0]->set_layout(layout);
    fun->get_results()[0]->set_layout(layout);
    return CNNNetwork(fun);
}

static void checkCustomRt(const std::function<void(Node::RTMap&)>& emptyCb,
                          const std::function<void(Node::RTMap&, const std::string& name)>& nameCb) {
    auto net1 = createNetwork();
    auto net2 = createNetwork();
    auto & op1 = net1.getFunction()->get_ops().front()->get_rt_info();
    auto & op2 = net2.getFunction()->get_ops().front()->get_rt_info();

    emptyCb(op2);
    ASSERT_NE(NetworkCompilationContext::computeHash(net1, {}),
              NetworkCompilationContext::computeHash(net2, {}));

    emptyCb(op1);
    ASSERT_EQ(NetworkCompilationContext::computeHash(net1, {}),
              NetworkCompilationContext::computeHash(net2, {}));

    nameCb(op1, "test");
    ASSERT_NE(NetworkCompilationContext::computeHash(net1, {}),
              NetworkCompilationContext::computeHash(net2, {}));

    nameCb(op2, "test");
    ASSERT_EQ(NetworkCompilationContext::computeHash(net1, {}),
              NetworkCompilationContext::computeHash(net2, {}));

    nameCb(op1, "test2");
    ASSERT_NE(NetworkCompilationContext::computeHash(net1, {}),
              NetworkCompilationContext::computeHash(net2, {}));
}

TEST(NetworkContext_CNNNetwork, HashOfSame) {
    auto net1 = createNetwork();
    auto net2 = createNetwork();
    ASSERT_EQ(NetworkCompilationContext::computeHash(net1, {}),
              NetworkCompilationContext::computeHash(net2, {}));
}

TEST(NetworkContext_CNNNetwork, HashWithConfig) {
    auto net1 = createNetwork();
    auto net2 = createNetwork();
    ASSERT_NE(NetworkCompilationContext::computeHash(net1, {{"key", "value"}}),
              NetworkCompilationContext::computeHash(net2, {}));
    ASSERT_EQ(NetworkCompilationContext::computeHash(net1, {{"key", "value"}}),
              NetworkCompilationContext::computeHash(net2, {{"key", "value"}}));
}

TEST(NetworkContext_CNNNetwork, HashWithPrimitivesPriority) {
    auto net1 = createNetwork();
    auto net2 = createNetwork();
    auto net3 = createNetwork();
    auto & op2 = net2.getFunction()->get_ops().front()->get_rt_info();
    op2[ov::PrimitivesPriority::get_type_info_static()] = ov::PrimitivesPriority("testPriority");

    auto & op3 = net3.getFunction()->get_ops().front()->get_rt_info();
    op3["PrimitivesPriority"] = "testPriority";

    ASSERT_NE(NetworkCompilationContext::computeHash(net1, {}),
              NetworkCompilationContext::computeHash(net2, {}));

    ASSERT_EQ(NetworkCompilationContext::computeHash(net2, {}),
              NetworkCompilationContext::computeHash(net3, {}));
}

TEST(NetworkContext_CNNNetwork, HashWithFusedNames) {
    auto setFusedEmpty = [&](Node::RTMap& rtInfo) {
        rtInfo[ngraph::FusedNames::get_type_info_static()] = ngraph::FusedNames();
    };
    auto setFused = [&](Node::RTMap& rtInfo, const std::string& name) {
        rtInfo[ngraph::FusedNames::get_type_info_static()] = ngraph::FusedNames(name);
    };
    checkCustomRt(setFusedEmpty, setFused);
}

TEST(NetworkContext_CNNNetwork, HashWithPrimitivesPriorityType) {
    auto setPrimEmpty = [&](Node::RTMap& rtInfo) {
        rtInfo[ov::PrimitivesPriority::get_type_info_static()] = ov::PrimitivesPriority("");
    };
    auto setPrim = [&](Node::RTMap& rtInfo, const std::string& name) {
        rtInfo[ov::PrimitivesPriority::get_type_info_static()] = ov::PrimitivesPriority(name);
    };
    checkCustomRt(setPrimEmpty, setPrim);
}

TEST(NetworkContext_CNNNetwork, HashWithAffinity) {
    auto net1 = createNetwork();
    auto net2 = createNetwork();
    auto net3 = createNetwork();
    auto & op2 = net2.getFunction()->get_ops().front()->get_rt_info();
    op2["affinity"] = "testAffinity";

    auto & op3 = net3.getFunction()->get_ops().front()->get_rt_info();
    op3["affinity"] = "testAffinity";

    ASSERT_NE(NetworkCompilationContext::computeHash(net1, {}),
              NetworkCompilationContext::computeHash(net2, {}));

    ASSERT_EQ(NetworkCompilationContext::computeHash(net2, {}),
              NetworkCompilationContext::computeHash(net3, {}));
}

TEST(NetworkContext_CNNNetwork, HashWithFutureRt_string) {
    auto net1 = createNetwork();
    auto net2 = createNetwork();
    auto net3 = createNetwork();

    auto & op1 = net1.getFunction()->get_ops().front()->get_rt_info();
    op1["someFutureKey"] = "hello";

    auto & op2 = net2.getFunction()->get_ops().front()->get_rt_info();
    op2["someFutureKey"] = "hello";

    auto & op3 = net3.getFunction()->get_ops().front()->get_rt_info();
    op3["someFutureKey"] = "olleh";

    ASSERT_EQ(NetworkCompilationContext::computeHash(net1, {}),
              NetworkCompilationContext::computeHash(net2, {}));

    ASSERT_NE(NetworkCompilationContext::computeHash(net2, {}),
              NetworkCompilationContext::computeHash(net3, {}));
}

TEST(NetworkContext_CNNNetwork, HashWithFutureRt_int64) {
    auto net1 = createNetwork();
    auto net2 = createNetwork();
    auto net3 = createNetwork();

    auto & op1 = net1.getFunction()->get_ops().front()->get_rt_info();
    op1["someFutureKey"] = int64_t(42);

    auto & op2 = net2.getFunction()->get_ops().front()->get_rt_info();
    op2["someFutureKey"] = int64_t(42);

    auto & op3 = net3.getFunction()->get_ops().front()->get_rt_info();
    op3["someFutureKey"] = int64_t(43);

    ASSERT_EQ(NetworkCompilationContext::computeHash(net1, {}),
              NetworkCompilationContext::computeHash(net2, {}));

    ASSERT_NE(NetworkCompilationContext::computeHash(net2, {}),
              NetworkCompilationContext::computeHash(net3, {}));
}

TEST(NetworkContext_CNNNetwork, HashWithLayout) {
    auto net1 = createNetworkWithLayout("NCH");
    auto net2 = createNetworkWithLayout("nch");
    auto net3 = createNetworkWithLayout("?CH");
    auto net3_1 = createNetworkWithLayout("?C?");
    auto net4 = createNetworkWithLayout("");
    auto fun5 = create_simple_function();
    fun5->get_parameters()[0]->set_layout("NCH");
    fun5->get_parameters()[0]->set_layout("");
    fun5->get_results()[0]->set_layout("NHC");
    fun5->get_results()[0]->set_layout(ov::Layout());
    auto net5 = CNNNetwork(fun5);

    EXPECT_EQ(NetworkCompilationContext::computeHash(net1, {}),
              NetworkCompilationContext::computeHash(net2, {}));

    EXPECT_NE(NetworkCompilationContext::computeHash(net2, {}),
              NetworkCompilationContext::computeHash(net3, {}));

    EXPECT_NE(NetworkCompilationContext::computeHash(net3, {}),
              NetworkCompilationContext::computeHash(net3_1, {}));

    EXPECT_NE(NetworkCompilationContext::computeHash(net3, {}),
              NetworkCompilationContext::computeHash(net4, {}));

    EXPECT_EQ(NetworkCompilationContext::computeHash(net4, {}),
              NetworkCompilationContext::computeHash(net5, {}));
}

TEST(NetworkContext_CNNNetwork, HashWithTensorNames) {
    auto fun1 = create_simple_function();
    auto fun2 = create_simple_function();
    auto fun3 = create_simple_function();
    std::unordered_set<std::string> names1, names2;
    std::vector<std::string> testNames;
    testNames.reserve(100);
    for (int i = 0; i < 100; i++) {
        testNames.push_back("test" + std::to_string(i));
    }
    std::for_each(testNames.begin(), testNames.end(), [&names1](const std::string& name) {
        names1.insert(name);
    });
    std::for_each(testNames.rbegin(), testNames.rend(), [&names2](const std::string& name) {
        names2.insert(name);
    });

    fun1->input().set_names(names1);
    fun2->input().set_names(names2);

    auto net1 = CNNNetwork(fun1);
    auto net2 = CNNNetwork(fun2);
    auto net3 = CNNNetwork(fun3);

    ASSERT_EQ(NetworkCompilationContext::computeHash(net1, {}),
              NetworkCompilationContext::computeHash(net2, {}));

    ASSERT_NE(NetworkCompilationContext::computeHash(net2, {}),
              NetworkCompilationContext::computeHash(net3, {}));
}

TEST(NetworkContext_CNNNetwork, HashWithDifferentResults) {
    auto net1 = createNetwork();
    auto net2 = createNetwork();
    net2.getFunction()->remove_result(net2.getFunction()->get_results().front());
    auto net3 = createNetwork();
    net3.getFunction()->remove_result(net3.getFunction()->get_results().front());
    ASSERT_NE(NetworkCompilationContext::computeHash(net1, {}),
              NetworkCompilationContext::computeHash(net2, {}));
    ASSERT_EQ(NetworkCompilationContext::computeHash(net2, {}),
              NetworkCompilationContext::computeHash(net3, {}));
}

TEST(NetworkContext_CNNNetwork, HashWithDifferentMeanValues) {
    auto updatePreprocess = [&](CNNNetwork& cnnNet) {
        auto &preProcess = cnnNet.getInputsInfo().begin()->second->getPreProcess();
        preProcess.init(3);
        preProcess[0]->stdScale = 2;
        preProcess[1]->stdScale = 3;
        preProcess[2]->stdScale = 4;
        preProcess[0]->meanValue = 0;
        preProcess[1]->meanValue = 1;
        preProcess[2]->meanValue = 2;
        preProcess.setVariant(InferenceEngine::MEAN_VALUE);
    };
    auto net1 = createNetwork();
    auto net2 = createNetwork();
    updatePreprocess(net2);
    auto net3 = createNetwork();
    updatePreprocess(net3);
    ASSERT_NE(NetworkCompilationContext::computeHash(net1, {}),
              NetworkCompilationContext::computeHash(net2, {}));
    ASSERT_EQ(NetworkCompilationContext::computeHash(net2, {}),
              NetworkCompilationContext::computeHash(net3, {}));
}

// Verify all internal hash calculations are thread-safe (like ngraph::function serialization)
TEST(NetworkContext_CNNNetwork, HashOfSameMultiThreading) {
    auto net1 = createNetwork();
    auto net2 = createNetwork();
    std::atomic_bool fail{false};
    const auto TEST_DURATION_MS = 1000;
    auto start = high_resolution_clock::now();
    int t1Count = 0, t2Count = 0;
    auto threadFun = [&](int& count) {
        do {
            count++;
            auto hash1 = NetworkCompilationContext::computeHash(net1, {});
            auto hash2 = NetworkCompilationContext::computeHash(net2, {});
            if (hash1 != hash2) {
                fail = true;
                break;
            }
        } while (!fail && duration_cast<milliseconds>(high_resolution_clock::now() - start).count() < TEST_DURATION_MS);
    };
    std::thread t1(threadFun, std::ref(t1Count));
    std::thread t2(threadFun, std::ref(t2Count));
    t1.join();
    t2.join();
    std::cout << "Hash threading test finished. Total runs = " << t1Count + t2Count << std::endl;
    ASSERT_FALSE(fail);
}

////////////////////////////////////////////

TEST(NetworkContext_ModelName, HashOfSame) {
    ASSERT_EQ(NetworkCompilationContext::computeHash("model1", {}),
              NetworkCompilationContext::computeHash("model1", {}));

    ASSERT_NE(NetworkCompilationContext::computeHash("model1", {}),
              NetworkCompilationContext::computeHash("model2", {}));

    ASSERT_NE(NetworkCompilationContext::computeHash("model1", {{"key", "value"}}),
              NetworkCompilationContext::computeHash("model1", {}));

    ASSERT_EQ(NetworkCompilationContext::computeHash("model1", {{"key", "value"}}),
              NetworkCompilationContext::computeHash("model1", {{"key", "value"}}));
}

TEST(NetworkContext_ModelName, HashOfExistingFile) {
    auto file1 = generateTestFilePrefix() + ".xml";
    auto file2 = std::string(".") + CommonTestUtils::FileSeparator + file1;

    FileGuard guard(file1);
    {
        std::ofstream os(file1);
        os << "test";
    }
    ASSERT_EQ(NetworkCompilationContext::computeHash(file1, {}),
              NetworkCompilationContext::computeHash(file1, {}));

    ASSERT_EQ(NetworkCompilationContext::computeHash(file1, {}),
              NetworkCompilationContext::computeHash(file2, {}));

    ASSERT_NE(NetworkCompilationContext::computeHash(file1, {{"key", "value"}}),
              NetworkCompilationContext::computeHash(file2, {}));

    ASSERT_EQ(NetworkCompilationContext::computeHash(file1, {{"key", "value"}}),
              NetworkCompilationContext::computeHash(file2, {{"key", "value"}}));
}
