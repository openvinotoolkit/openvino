// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <atomic>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "ie_plugin_ptr.hpp"
#include "ngraph/function.hpp"
#include "details/ie_so_loader.h"
#include "ie_metric_helpers.hpp"
#include "ie_iexecutable_network.hpp"

#include "cpp_interfaces/impl/ie_executable_network_internal.hpp"
#include "cpp_interfaces/impl/ie_plugin_internal.hpp"

#include "common_test_utils/unicode_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_constants.hpp"

#include "functional_test_utils/test_model/test_model.hpp"
#include "functional_test_utils/network_utils.hpp"

#include "unit_test_utils/mocks/mock_iexecutable_network.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "ie_plugin_cpp.hpp"

using namespace InferenceEngine;
using namespace ::testing;
using namespace InferenceEngine::details;
using namespace std::placeholders;
using namespace std::chrono;

enum class TestLoadType {
    ECNN,
    EContext,
    EModelName
};
using TestParam = std::tuple<TestLoadType, std::string, bool>;

//  GCC4.8 limitation: have to specify type of each element in list
static const std::vector<TestParam> loadVariants = {
        TestParam { TestLoadType::ECNN, std::string("ByCNNNetwork"), false },
        TestParam { TestLoadType::EContext, std::string("ByRemoteContext"), true },
        TestParam { TestLoadType::EModelName, std::string("ByModelName"), false },
};

static const std::vector<std::string> cacheFolders {
        std::string("testCache"),
};

std::string getTestCaseName(const testing::TestParamInfo<std::tuple<TestParam, std::string>> &obj) {
    return std::get<1>(std::get<0>(obj.param)) + "_" + std::get<1>(obj.param);
}

class MockRemoteContext : public RemoteContext {
    std::string m_name;
public:
    MockRemoteContext(std::string name): m_name(std::move(name)) {}
    std::string getDeviceName() const noexcept { return m_name; }
    MOCK_METHOD2(CreateBlob, RemoteBlob::Ptr(const TensorDesc&, const ParamMap&));
    MOCK_QUALIFIED_METHOD0(getParams, const, ParamMap());
};

class MockCachingInferencePlugin : public InferenceEngine::InferencePluginInternal {
public:
    MockCachingInferencePlugin() = default;
    ~MockCachingInferencePlugin() = default;

    MOCK_METHOD2(LoadExeNetworkImpl, ExecutableNetworkInternal::Ptr(const CNNNetwork& network,
                                                      const std::map<std::string, std::string>& config));

    MOCK_METHOD3(LoadExeNetworkImpl, ExecutableNetworkInternal::Ptr(const CNNNetwork& network, RemoteContext::Ptr context,
                                                      const std::map<std::string, std::string>& config));

    MOCK_METHOD2(ImportNetworkImpl, ExecutableNetworkInternal::Ptr(std::istream& networkModel,
                                                                   const std::map<std::string, std::string>& config));

    MOCK_METHOD3(ImportNetworkImpl, ExecutableNetworkInternal::Ptr(std::istream& networkModel,
                                                                   const RemoteContext::Ptr& context,
                                                                   const std::map<std::string, std::string>& config));

    MOCK_CONST_METHOD2(QueryNetwork, QueryNetworkResult(const CNNNetwork& network,
                                    const std::map<std::string, std::string>& config));

    MOCK_CONST_METHOD2(GetMetric, Parameter(const std::string& name, const std::map<std::string, Parameter>& options));
    MOCK_METHOD1(GetDefaultContext, RemoteContext::Ptr(const ParamMap& params));
};

class MockExecutableNetwork : public ExecutableNetworkInternal {
public:
    MockExecutableNetwork() {}
    MOCK_METHOD1(ExportImpl, void(std::ostream& networkModel));
    MOCK_METHOD0(CreateInferRequest, IInferRequest::Ptr());
    MOCK_CONST_METHOD0(GetInputsInfo, ConstInputsDataMap());
    MOCK_CONST_METHOD0(GetOutputsInfo, ConstOutputsDataMap());
    MOCK_CONST_METHOD1(GetConfig, Parameter(const std::string& name));
    MOCK_CONST_METHOD1(GetMetric, Parameter(const std::string& name));
};

//------------------------------------------------------
class MkDirGuard {
    std::string m_dir;
public:
    MkDirGuard(const std::string &dir = std::string()): m_dir(dir) {
        if (!m_dir.empty()) {
            CommonTestUtils::createDirectory(m_dir);
        }
    }

    MkDirGuard(const MkDirGuard&) = delete;
    MkDirGuard& operator=(const MkDirGuard&) = delete;

    ~MkDirGuard() {
        if (!m_dir.empty()) {
            CommonTestUtils::removeFilesWithExt(m_dir, "blob");
            CommonTestUtils::removeDir(m_dir);
        }
    }
};

class CachingTest : public ::testing::TestWithParam<std::tuple<TestParam, std::string>> {
public:
    std::unique_ptr<SharedObjectLoader> sharedObjectLoader;
    std::function<void(IInferencePlugin*)> injectProxyEngine;
    std::string modelName = "Caching_test.xml";
    std::string weightsName = "Caching_test.bin";
    std::string deviceName = "mock";
    std::string deviceToLoad = "mock";
    std::shared_ptr<MockCachingInferencePlugin> mockPlugin;
    std::shared_ptr<MockExecutableNetwork> net;
    std::unique_ptr<MkDirGuard> m_dirCreator;
    TestLoadType                m_type;
    std::string                 m_cacheDir;
    using LoadFunction = std::function<void(Core&)>;
    using LoadFunctionWithCfg = std::function<void(Core&, const std::map<std::string, std::string> &)>;
    LoadFunction                m_testFunction;
    LoadFunctionWithCfg         m_testFunctionWithCfg;
    bool                        m_remoteContext = false;
    using CNNCallback = std::function<void(CNNNetwork&)>;
    CNNCallback                 m_cnnCallback = nullptr;

    std::string get_mock_engine_name() {
        std::string mockEngineName("mock_engine");
        return CommonTestUtils::pre + mockEngineName + IE_BUILD_POSTFIX + CommonTestUtils::ext;
    }

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

    void initParamTest() {
        m_type = std::get<0>(std::get<0>(GetParam()));
        m_cacheDir = std::get<1>(GetParam());
        m_testFunction = getLoadFunction(m_type);
        m_testFunctionWithCfg = getLoadFunctionWithCfg(m_type);
        m_remoteContext = std::get<2>(std::get<0>(GetParam()));
        auto testName = generateTestFilePrefix();
        modelName = testName + ".xml";
        weightsName = testName + ".bin";
        m_cacheDir = testName + m_cacheDir;
        m_dirCreator = std::unique_ptr<MkDirGuard>(new MkDirGuard(m_cacheDir));
    }

    void SetUp() override {
        initParamTest();
        mockPlugin = std::make_shared<MockCachingInferencePlugin>();
        net = std::make_shared<MockExecutableNetwork>();
        setupMock(*mockPlugin);
        std::string libraryName = get_mock_engine_name();
        sharedObjectLoader.reset(new SharedObjectLoader(libraryName.c_str()));
        injectProxyEngine = make_std_function<void(IInferencePlugin*)>("InjectProxyEngine");

        FuncTestUtils::TestModel::generateTestModel(modelName, weightsName);
    }

    void TearDown() override {
        CommonTestUtils::removeIRFiles(modelName, weightsName);
    }

    void testLoad(std::function<void(Core& ie)> func) {
        Core ie;
        injectProxyEngine(mockPlugin.get());
        ie.RegisterPlugin(std::string("mock_engine") + IE_BUILD_POSTFIX, deviceName);
        func(ie);
        ie.UnregisterPlugin(deviceName);
    }

    LoadFunction getLoadFunction(TestLoadType type) const {
        switch (type) {
            case TestLoadType::ECNN:
                return [&](Core& ie) { performReadAndLoad(ie); };
            case TestLoadType::EContext:
                return [&](Core& ie) { performReadAndLoadWithContext(ie); };
            case TestLoadType::EModelName:
                return [&](Core& ie) { performLoadByName(ie); };
        }
        return nullptr;
    }

    LoadFunctionWithCfg getLoadFunctionWithCfg(TestLoadType type) const {
        switch (type) {
            case TestLoadType::ECNN:
                return std::bind(&CachingTest::performReadAndLoad, this, _1, _2);
            case TestLoadType::EContext:
                return std::bind(&CachingTest::performReadAndLoadWithContext, this, _1, _2);
            case TestLoadType::EModelName:
                return std::bind(&CachingTest::performLoadByName, this, _1, _2);
        }
        return nullptr;
    }

    void performLoadByName(Core& ie, const std::map<std::string, std::string>& config = {}) const {
        ie.LoadNetwork(modelName, deviceToLoad, config);
    }

    void performReadAndLoad(Core& ie, const std::map<std::string, std::string>& config = {}) const {
        auto cnnNetwork = ie.ReadNetwork(modelName);
        if (m_cnnCallback) m_cnnCallback(cnnNetwork);
        ie.LoadNetwork(cnnNetwork, deviceToLoad, config);
    }

    void performReadAndLoadWithContext(Core& ie, const std::map<std::string, std::string>& config = {}) const {
        auto cnnNetwork = ie.ReadNetwork(modelName);
        EXPECT_CALL(*mockPlugin, GetDefaultContext(_)).Times(AnyNumber());
        auto context = ie.GetDefaultContext(deviceToLoad);
        if (m_cnnCallback) m_cnnCallback(cnnNetwork);
        ie.LoadNetwork(cnnNetwork, context, config);
    }

    std::shared_ptr<MockExecutableNetwork> createMockIExecutableNet() {
        auto mock = std::make_shared<MockExecutableNetwork>();
        EXPECT_CALL(*mock, GetInputsInfo()).Times(AnyNumber()).WillRepeatedly(Return(ConstInputsDataMap{}));
        EXPECT_CALL(*mock, GetOutputsInfo()).Times(AnyNumber()).WillRepeatedly(Return(ConstOutputsDataMap{}));
        EXPECT_CALL(*mock, GetConfig(PluginConfigParams::KEY_PERF_COUNT)).Times(AnyNumber()).WillRepeatedly(Return(Parameter{PluginConfigParams::NO}));
        EXPECT_CALL(*mock, GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))).Times(AnyNumber()).WillRepeatedly(Return(Parameter{1u}));
        auto ptr = std::make_shared<MockIInferRequest>();
        EXPECT_CALL(*ptr, SetCompletionCallback(_)).Times(AnyNumber()).WillRepeatedly(Return(OK));
        EXPECT_CALL(*ptr, SetUserData(_, _)).Times(AnyNumber()).WillRepeatedly(Return(OK));
        EXPECT_CALL(*mock, CreateInferRequest()).Times(AnyNumber()).WillRepeatedly(Return(ptr));
        return mock;
    }

private:
    template <class T>
    std::function<T> make_std_function(const std::string& functionName) {
        std::function <T> ptr(reinterpret_cast<T*>(sharedObjectLoader->get_symbol(functionName.c_str())));
        return ptr;
    }

    void setupMock(MockCachingInferencePlugin& plugin) {
        ON_CALL(plugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).
                WillByDefault(Invoke([&](const std::string &, const std::map<std::string, Parameter> &) {
            std::vector<std::string> res;
            res.push_back(METRIC_KEY(IMPORT_EXPORT_SUPPORT));
            res.push_back(METRIC_KEY(DEVICE_ARCHITECTURE));
            return res;
        }));
        ON_CALL(plugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).
                WillByDefault(Return(true));

        ON_CALL(plugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).
                WillByDefault(Invoke([&](const std::string &, const std::map<std::string, Parameter> &) {
            std::vector<std::string> res;
            res.push_back("SomeConfig");
            return res;
        }));

        ON_CALL(plugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).
                WillByDefault(Return("mock"));

        ON_CALL(plugin, ImportNetworkImpl(_, _, _)).
                WillByDefault(Invoke([&](std::istream &istr, RemoteContext::Ptr,
                                         const std::map<std::string, std::string> &) {
            return createMockIExecutableNet();
        }));

        ON_CALL(plugin, ImportNetworkImpl(_, _)).
                WillByDefault(Invoke([&](std::istream &istr, const std::map<std::string, std::string> &) {
            return createMockIExecutableNet();
        }));

        ON_CALL(plugin, LoadExeNetworkImpl(_, _, _)).
                WillByDefault(Invoke([&](const CNNNetwork &, RemoteContext::Ptr,
                                         const std::map<std::string, std::string> &) {
            return net;
        }));

        ON_CALL(plugin, LoadExeNetworkImpl(_, _)).
                WillByDefault(Invoke([&](const CNNNetwork &,
                                         const std::map<std::string, std::string> &) {
            return net;
        }));

        ON_CALL(plugin, GetDefaultContext(_)).
                WillByDefault(Invoke([&](const ParamMap &) {
            return std::make_shared<MockRemoteContext>(deviceToLoad);
        }));

        ON_CALL(plugin, QueryNetwork(_, _)).
                WillByDefault(Invoke([&](const CNNNetwork &network, const std::map<std::string, std::string>&) {
            QueryNetworkResult res;
            auto function = network.getFunction();
            EXPECT_TRUE(function);

            for (auto &&node : function->get_ops()) {
                res.supportedLayersMap.emplace(node->get_friendly_name(), deviceName);
            }
            return res;
        }));

        EXPECT_CALL(*net, GetInputsInfo()).Times(AnyNumber())
                .WillRepeatedly(Return(ConstInputsDataMap{}));
        EXPECT_CALL(*net, GetOutputsInfo()).Times(AnyNumber())
                .WillRepeatedly(Return(ConstOutputsDataMap{}));
        EXPECT_CALL(*net, GetConfig(PluginConfigParams::KEY_PERF_COUNT)).Times(AnyNumber())
                .WillRepeatedly(Return(PluginConfigParams::NO));
        EXPECT_CALL(*net, GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))).Times(AnyNumber())
                .WillRepeatedly(Return((unsigned int) 1));
        EXPECT_CALL(*net, GetMetric(METRIC_KEY(NETWORK_NAME))).Times(AnyNumber())
                .WillRepeatedly(Return("mock_net"));
        EXPECT_CALL(*net, GetMetric(METRIC_KEY(SUPPORTED_METRICS))).Times(AnyNumber())
                .WillRepeatedly(Invoke([&](const std::string &) {
            std::vector<std::string> res;
            res.push_back(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
            res.push_back(METRIC_KEY(NETWORK_NAME));
            return res;
        }));
        EXPECT_CALL(*net, CreateInferRequest()).Times(AnyNumber())
                .WillRepeatedly(Invoke([&]() {
            std::vector<std::string> res;
            auto inferReq = std::make_shared<MockIInferRequest>();
            EXPECT_CALL(*inferReq, SetCompletionCallback(_)).Times(AnyNumber()).WillRepeatedly(Return(OK));
            EXPECT_CALL(*inferReq, SetUserData(_, _)).Times(AnyNumber()).WillRepeatedly(Return(OK));
            return inferReq;
        }));
    }
};

TEST_P(CachingTest, TestLoad) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestLoadCustomImportExport) {
    const int customNumber = 1234;
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    ON_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).
            WillByDefault(Invoke([&](std::istream& s, RemoteContext::Ptr,
                                     const std::map<std::string, std::string> &) {
        int a;
        s >> a;
        EXPECT_EQ(customNumber, a);
        auto mock = std::make_shared<MockExecutableNetwork>();
        EXPECT_CALL(*mock, GetInputsInfo()).Times(AnyNumber()).WillRepeatedly(Return(ConstInputsDataMap{}));
        EXPECT_CALL(*mock, GetOutputsInfo()).Times(AnyNumber()).WillRepeatedly(Return(ConstOutputsDataMap{}));
        return mock;
    }));

    ON_CALL(*mockPlugin, ImportNetworkImpl(_, _)).
            WillByDefault(Invoke([&](std::istream &s, const std::map<std::string, std::string> &) {
        int a;
        s >> a;
        EXPECT_EQ(customNumber, a);
        auto mock = std::make_shared<MockExecutableNetwork>();
        EXPECT_CALL(*mock, GetInputsInfo()).Times(AnyNumber()).WillRepeatedly(Return(ConstInputsDataMap{}));
        EXPECT_CALL(*mock, GetOutputsInfo()).Times(AnyNumber()).WillRepeatedly(Return(ConstOutputsDataMap{}));
        return mock;
    }));

    ON_CALL(*net, ExportImpl(_)).WillByDefault(Invoke([&] (std::ostream& s) {
        s << customNumber;
    }));

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
}

// Brief: when LoadNetwork is called from different config - old cache shall not be used
TEST_P(CachingTest, TestChangeLoadConfig) {
    const std::string CUSTOM_KEY = "CUSTOM_KEY";
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    ON_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).
            WillByDefault(Invoke([&](const std::string &, const std::map<std::string, Parameter> &) {
        std::vector<std::string> res;
        res.push_back(CUSTOM_KEY);
        return res;
    }));
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunctionWithCfg(ie, {{CUSTOM_KEY, "0"}});
        });
    }

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunctionWithCfg(ie, {{CUSTOM_KEY, "1"}});
        });
    }
}

TEST_P(CachingTest, TestNoCacheEnabled) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(0);
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestNoCacheSupported) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _))
            .Times(AnyNumber()).WillRepeatedly(Return(false));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestNoCacheMetricSupported) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _))
            .Times(AnyNumber()).WillRepeatedly(Return(std::vector<std::string>{}));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(0);
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(0);
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestLoadChangeCacheDir) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }

    {
        std::string newCacheDir = m_cacheDir + "2";
        MkDirGuard dir(newCacheDir);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), newCacheDir}});
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestClearCacheDir) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(0);
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), ""}});
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestChangeOtherConfig) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            ie.SetConfig({{"someKey", "someValue"}});
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestChangeCacheDirFailure) {
    std::string longName(1000000, ' ');
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            EXPECT_ANY_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir + "/" + longName}}));
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestCacheDirCreateRecursive) {
    std::string newCacheDir1 = m_cacheDir + CommonTestUtils::FileSeparator + "a";
    std::string newCacheDir2 = newCacheDir1 + CommonTestUtils::FileSeparator + "b";
    std::string newCacheDir3 = newCacheDir2 + CommonTestUtils::FileSeparator + CommonTestUtils::FileSeparator;

    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            EXPECT_NO_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), newCacheDir3}}));
            EXPECT_NO_THROW(m_testFunction(ie));
        });
    }
    CommonTestUtils::removeFilesWithExt(newCacheDir2, "blob");
    std::remove(newCacheDir2.c_str());
    std::remove(newCacheDir1.c_str());
}

TEST_P(CachingTest, TestDeviceArchitecture) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber())
            .WillRepeatedly(Invoke([&](const std::string&, const std::map<std::string, Parameter>& options) {
                auto id = options.at("DEVICE_ID").as<std::string>();
                if (std::stoi(id) < 10) {
                    return "mock_first_architecture";
                } else {
                    return "mock_another_architecture";
                }
            }));
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            deviceToLoad = "mock.0";
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            deviceToLoad = "mock.1";
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            deviceToLoad = "mock.50";
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            deviceToLoad = "mock.51";
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestNoDeviceArchitecture) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber())
            .WillRepeatedly(Invoke([&] (const std::string&, const std::map<std::string, Parameter>&) {
                return std::vector<std::string>{METRIC_KEY(IMPORT_EXPORT_SUPPORT)};
            }));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(0);
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            deviceToLoad = "mock.0";
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            deviceToLoad = "mock.50";
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestThrowOnExport) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1).WillOnce(Throw(1));
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            EXPECT_ANY_THROW(m_testFunction(ie));
        });
    }
}

TEST_P(CachingTest, TestThrowOnImport) {
    ON_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).WillByDefault(Throw(1));
    ON_CALL(*mockPlugin, ImportNetworkImpl(_, _)).WillByDefault(Throw(1));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            EXPECT_ANY_THROW(m_testFunction(ie));
        });
    }
    { // Step 3: same load, cache should be deleted due to unsuccessful import on step 2
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            EXPECT_NO_THROW(m_testFunction(ie));
        });
    }
}

TEST_P(CachingTest, TestNetworkModified) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            EXPECT_NO_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}));
            EXPECT_NO_THROW(m_testFunction(ie));
        });
    }
    if (m_type == TestLoadType::EModelName) {
        // Modify model file
        std::fstream stream(modelName, std::fstream::out | std::fstream::app);
        stream << " ";
    } else {
        // Modify loaded CNN network
        m_cnnCallback = [&](CNNNetwork& network) {
            auto f = network.getFunction();
            auto res = f->get_results();
            f->remove_result(res.front());
        };
    }
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            EXPECT_NO_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}));
            EXPECT_NO_THROW(m_testFunction(ie));
        });
    }
    { // Step 3: same load, should be ok now
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            EXPECT_NO_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}));
            EXPECT_NO_THROW(m_testFunction(ie));
        });
    }
}

TEST_P(CachingTest, TestCacheFileCorrupted) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            EXPECT_NO_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}));
            EXPECT_NO_THROW(m_testFunction(ie));
        });
    }
    {
        auto blobs = CommonTestUtils::listFilesWithExt(m_cacheDir, "blob");
        for (const auto& fileName : blobs) {
            std::ofstream stream(fileName, std::ios_base::binary);
            stream << "SomeCorruptedText";
        }
    }
    { // Step 2. Cache is corrupted, will be silently removed
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            EXPECT_NO_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}));
            EXPECT_NO_THROW(m_testFunction(ie));
        });
    }
    { // Step 3: same load, should be ok now due to re-creation of cache
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            EXPECT_NO_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}));
            EXPECT_NO_THROW(m_testFunction(ie));
        });
    }
}

TEST_P(CachingTest, TestCacheFileOldVersion) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            EXPECT_NO_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}));
            EXPECT_NO_THROW(m_testFunction(ie));
        });
    }
    {
        auto blobs = CommonTestUtils::listFilesWithExt(m_cacheDir, "blob");
        for (const auto& fileName : blobs) {
            std::string content;
            {
                std::ifstream inp(fileName, std::ios_base::binary);
                std::ostringstream ostr;
                ostr << inp.rdbuf();
                content = ostr.str();
            }
            std::string buildNum = GetInferenceEngineVersion()->buildNumber;
            std::string zeroBuild(buildNum.size(), '0');
            auto index = content.find(buildNum);
            if (index != std::string::npos) {
                content.replace(index, buildNum.size(), zeroBuild);
            } else {
                SKIP();
            }
            std::ofstream out(fileName, std::ios_base::binary);
            out.write(content.c_str(), content.size());
        }
    }
    { // Step 2. Build number mismatch, cache will be silently removed
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            EXPECT_NO_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}));
            EXPECT_NO_THROW(m_testFunction(ie));
        });
    }
    { // Step 3: same load, should be ok now due to re-creation of cache
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            EXPECT_NO_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}));
            EXPECT_NO_THROW(m_testFunction(ie));
        });
    }
}

TEST_P(CachingTest, LoadHetero_NoCacheMetric) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _))
            .Times(AnyNumber()).WillRepeatedly(Return(std::vector<std::string>{}));
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _))
            .Times(AnyNumber()).WillRepeatedly(Return(std::vector<std::string>{}));
    // Hetero supports Import/Export, but mock plugin does not
    deviceToLoad = CommonTestUtils::DEVICE_HETERO + std::string(":mock.1,mock.2");
    if (m_remoteContext) {
        return; // skip the remote Context test for Hetero plugin
    }
    for (int i = 0; i < 2; i++) {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, LoadHetero_OneDevice) {
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    deviceToLoad = CommonTestUtils::DEVICE_HETERO + std::string(":mock");
    if (m_remoteContext) {
        return; // skip the remote Context test for Hetero plugin
    }
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
        // Ensure that only 1 blob (for Hetero) is created
        EXPECT_EQ(CommonTestUtils::listFilesWithExt(m_cacheDir, "blob").size(), 1);
    }

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, LoadHetero_TargetFallbackFromCore) {
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    deviceToLoad = CommonTestUtils::DEVICE_HETERO;
    if (m_remoteContext) {
        return; // skip the remote Context test for Hetero plugin
    }
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            ie.SetConfig({{"TARGET_FALLBACK", "mock"}}, CommonTestUtils::DEVICE_HETERO);
            m_testFunction(ie);
        });
        // Ensure that only 1 blob (for Hetero) is created
        EXPECT_EQ(CommonTestUtils::listFilesWithExt(m_cacheDir, "blob").size(), 1);
    }

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            ie.SetConfig({{"TARGET_FALLBACK", "mock"}}, CommonTestUtils::DEVICE_HETERO);
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, LoadHetero_MultiArchs) {
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    int customNumber = 1234;
    ON_CALL(*mockPlugin, ImportNetworkImpl(_, _)).
            WillByDefault(Invoke([&](std::istream &s, const std::map<std::string, std::string> &) {
        int a;
        s >> a;
        EXPECT_EQ(customNumber, a);
        auto mock = std::make_shared<MockExecutableNetwork>();
        EXPECT_CALL(*mock, GetInputsInfo()).Times(AnyNumber()).WillRepeatedly(Return(ConstInputsDataMap{}));
        EXPECT_CALL(*mock, GetOutputsInfo()).Times(AnyNumber()).WillRepeatedly(Return(ConstOutputsDataMap{}));
        return mock;
    }));

    ON_CALL(*net, ExportImpl(_)).WillByDefault(Invoke([&] (std::ostream& s) {
        s << customNumber;
    }));
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber()).WillRepeatedly(
            Invoke([&](const CNNNetwork &network, const std::map<std::string, std::string> &config) {
                QueryNetworkResult res;
                auto function = network.getFunction();
                EXPECT_TRUE(function);

                auto id = config.at("DEVICE_ID");
                bool supportsRelu = std::stoi(id) < 10;

                for (auto &&node : function->get_ops()) {
                    std::string nodeType = node->get_type_name();
                    if ((nodeType == "Relu" && supportsRelu) ||
                            (nodeType != "Relu" && !supportsRelu)) {
                        res.supportedLayersMap.emplace(node->get_friendly_name(), deviceName + "." + id);
                    }
                }
                return res;
            }));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber())
            .WillRepeatedly(Invoke([&](const std::string &, const std::map<std::string, Parameter> &options) {
                auto id = options.at("DEVICE_ID").as<std::string>();
                if (std::stoi(id) < 10) {
                    return "mock_first_architecture";
                } else {
                    return "mock_another_architecture";
                }
            }));
    deviceToLoad = CommonTestUtils::DEVICE_HETERO + std::string(":mock.1,mock.51");
    if (m_remoteContext) {
        return; // skip the remote Context test for Hetero plugin
    }
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(AtLeast(2)); // for .1 and for .51
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(AtLeast(2)); // for .1 and for .51
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
        // Ensure that only 1 blob (for Hetero) is created
        EXPECT_EQ(CommonTestUtils::listFilesWithExt(m_cacheDir, "blob").size(), 1);
    }

    deviceToLoad = CommonTestUtils::DEVICE_HETERO + std::string(":mock.2,mock.52");
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(AtLeast(2)); // for .2 and for .52
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
    deviceToLoad = CommonTestUtils::DEVICE_HETERO + std::string(":mock.53,mock.3");
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(AtLeast(1));
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(AtLeast(1));
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, LoadHetero_MultiArchs_TargetFallback_FromCore) {
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber())
            .WillRepeatedly(Invoke([&](const std::string &, const std::map<std::string, Parameter> &options) {
                auto id = options.at("DEVICE_ID").as<std::string>();
                if (std::stoi(id) < 10) {
                    return "mock_first_architecture";
                } else {
                    return "mock_another_architecture";
                }
            }));
    deviceToLoad = CommonTestUtils::DEVICE_HETERO;
    if (m_remoteContext) {
        return; // skip the remote Context test for Hetero plugin
    }
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            ie.SetConfig({{"TARGET_FALLBACK", "mock.1"}}, CommonTestUtils::DEVICE_HETERO);
            m_testFunction(ie);
        });
    }

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            ie.SetConfig({{"TARGET_FALLBACK", "mock.1"}}, CommonTestUtils::DEVICE_HETERO);
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            ie.SetConfig({{"TARGET_FALLBACK", "mock.51"}}, CommonTestUtils::DEVICE_HETERO);
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
}

// MULTI-DEVICE test
// Test that it is safe to load multiple devices sharing same cache
TEST_P(CachingTest, LoadMulti_race) {
    const auto TEST_DURATION_MS = 2000;
    const auto TEST_DEVICE_MAX_COUNT = 10;
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return; // skip the remote Context test for Multi plugin
    }
    int index = 0;
    auto start = high_resolution_clock::now();
    do {
        std::string cacheDir = m_cacheDir + std::to_string(index);
        MkDirGuard guard(cacheDir);
        int devCount = 1 + index % (TEST_DEVICE_MAX_COUNT - 1); // try dynamic number of devices from 1 to max
        deviceToLoad = CommonTestUtils::DEVICE_MULTI;
        deviceToLoad += ":mock.0";
        for (int i = 1; i < devCount; i++) {
            deviceToLoad += ",mock." + std::to_string(i);
        }

        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(devCount - 1);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), cacheDir}});
            ASSERT_NO_THROW(m_testFunction(ie));
        });
        index++;
    } while (duration_cast<milliseconds>(high_resolution_clock::now() - start).count() < TEST_DURATION_MS);
    std::cout << "Caching LoadMulti Test completed. Tried " << index << " times" << std::endl;
}

TEST_P(CachingTest, Load_threads) {
    const auto TEST_DURATION_MS = 2000;
    const auto THREADS_COUNT = 4;
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return; // skip the remote Context test for Multi plugin
    }
    auto start = high_resolution_clock::now();
    int index = 0;
    do {
        std::string cacheDir = m_cacheDir + std::to_string(index);
        MkDirGuard guard(cacheDir);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(THREADS_COUNT - 1);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), cacheDir}});
            std::vector<std::thread> threads;
            for (int i = 0; i < THREADS_COUNT; i++) {
                threads.emplace_back(([&]() { m_testFunction(ie); }));
            }
            for (int i = 0; i < THREADS_COUNT; i++) {
                threads[i].join();
            }
        });
        index++;
    } while (duration_cast<milliseconds>(high_resolution_clock::now() - start).count() < TEST_DURATION_MS);
    std::cout << "Caching Load multiple threads test completed. Tried " << index << " times" << std::endl;
}

// MULTI-DEVICE test
// Test that loading of device with one architecture doesn't block loading of device with another architecture
TEST_P(CachingTest, LoadMulti_Archs) {
    const auto IMPORT_DELAY_LONG_MS = 3000;
    const auto TEST_DEVICE_MAX_COUNT = 30; // Shall be >= 2
    const auto IMPORT_DELAY_SHORT_MS = 100;
    const auto EXP_MAX_EXEC_TIME_MS = 5500;
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber())
            .WillRepeatedly(Invoke([&](const std::string &, const std::map<std::string, Parameter> &options) {
                auto id = options.at("DEVICE_ID").as<std::string>();
                if (std::stoi(id) < 2) {
                    return "mock_first_architecture";
                } else {
                    return "mock_another_architecture";
                }
            }));
    if (m_remoteContext) {
        return; // skip the remote Context test for Multi plugin
    }

    deviceToLoad = CommonTestUtils::DEVICE_MULTI;
    deviceToLoad += ":mock.0";
    for (int i = 1; i < TEST_DEVICE_MAX_COUNT; i++) {
        deviceToLoad += ",mock." + std::to_string(i);
    }

    auto start = high_resolution_clock::now();
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(2);

        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(TEST_DEVICE_MAX_COUNT - 2)
                .WillRepeatedly(Invoke([&](std::istream &, const std::map<std::string, std::string> &opt) {
            auto id = opt.at("DEVICE_ID");
            if (std::stoi(id) < 2) {
                std::this_thread::sleep_for(milliseconds(IMPORT_DELAY_LONG_MS));
            } else {
                std::this_thread::sleep_for(milliseconds(IMPORT_DELAY_SHORT_MS));
            }
            return createMockIExecutableNet();
        }));
        EXPECT_CALL(*net, ExportImpl(_)).Times(2);
        testLoad([&](Core &ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            ASSERT_NO_THROW(m_testFunction(ie));
        });
    }
    ASSERT_LT(duration_cast<milliseconds>(high_resolution_clock::now() - start).count(), EXP_MAX_EXEC_TIME_MS);
}

INSTANTIATE_TEST_CASE_P(CachingTest, CachingTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(loadVariants),
                            ::testing::ValuesIn(cacheFolders)),
                        getTestCaseName);
