// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <atomic>
#include <string>
#include <vector>
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

using namespace InferenceEngine;
using namespace ::testing;
using namespace InferenceEngine::details;
using namespace std::placeholders;

enum class TestLoadType {
    ECNN,
    EContext,
    EModelName
};
using TestParam = std::tuple<TestLoadType, std::string, bool>;

static const std::vector<TestParam> loadVariants = {
        { TestLoadType::ECNN, "ByCNNNetwork", false },
        { TestLoadType::EContext, "ByRemoteContext", true },
        { TestLoadType::EModelName, "ByModelName", false },
};

static const std::vector<std::string> cacheFolders = {
        "testCache",
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

    MOCK_METHOD2(ImportNetworkImpl, ExecutableNetwork(std::istream& networkModel,
                                        const std::map<std::string, std::string>& config));

    MOCK_METHOD3(ImportNetworkImpl, ExecutableNetwork(std::istream& networkModel,
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
};

//------------------------------------------------------
class MkDirGuard {
    std::string m_dir;
public:
    MkDirGuard(const std::string &dir = std::string()): m_dir(dir) {
        if (!m_dir.empty()) {
            CommonTestUtils::makeDir(m_dir);
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
    std::function<void(Core&)>  m_testFunction;
    bool                        m_remoteContext = false;


    std::string get_mock_engine_name() {
        std::string mockEngineName("mock_engine");
        return CommonTestUtils::pre + mockEngineName + IE_BUILD_POSTFIX + CommonTestUtils::ext;
    }

    void initParamTest() {
        m_type = std::get<0>(std::get<0>(GetParam()));
        m_cacheDir = std::get<1>(GetParam());
        m_testFunction = getLoadFunction(m_type);
        m_remoteContext = std::get<2>(std::get<0>(GetParam()));
        m_dirCreator = std::unique_ptr<MkDirGuard>(new MkDirGuard(m_cacheDir));
    }

    void SetUp() override {
        mockPlugin = std::make_shared<MockCachingInferencePlugin>();
        net = std::make_shared<MockExecutableNetwork>();
        setupMock(*mockPlugin);
        initParamTest();
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

    std::function<void(Core&)> getLoadFunction(TestLoadType type) const {
        switch (type) {
            case TestLoadType::ECNN:
                return std::bind(&CachingTest::performReadAndLoad, this, _1);
            case TestLoadType::EContext:
                return std::bind(&CachingTest::performReadAndLoadWithContext, this, _1);
            case TestLoadType::EModelName:
                return std::bind(&CachingTest::performLoadByName, this, _1);
        }
        return nullptr;
    }
    void performLoadByName(Core &ie) const {
        auto exeNet = ie.LoadNetwork(modelName, deviceToLoad);
        (void)exeNet;
    }

    void performReadAndLoad(Core &ie) const {
        auto cnnNetwork = ie.ReadNetwork(modelName);
        auto exeNet = ie.LoadNetwork(cnnNetwork, deviceToLoad);
    }

    void performReadAndLoadWithContext(Core &ie) const {
        auto cnnNetwork = ie.ReadNetwork(modelName);
        EXPECT_CALL(*mockPlugin, GetDefaultContext(_)).Times(AnyNumber());
        auto context = ie.GetDefaultContext(deviceToLoad);
        ie.LoadNetwork(cnnNetwork, context);
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
                WillByDefault(Invoke([&](std::istream &, RemoteContext::Ptr,
                                         const std::map<std::string, std::string> &) {
            return ExecutableNetwork(std::make_shared<MockIExecutableNetwork>());
        }));

        ON_CALL(plugin, ImportNetworkImpl(_, _)).
                WillByDefault(Invoke([&](std::istream &, const std::map<std::string, std::string> &) {
            return ExecutableNetwork(std::make_shared<MockIExecutableNetwork>());
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
    }
};

TEST_P(CachingTest, TestLoadByName) {
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
            EXPECT_ANY_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), "?*. / *qwe/someFile.tmpCache"}}));
            m_testFunction(ie);
        });
    }
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

TEST_P(CachingTest, TestFileModified) {
    // Test is only for loadByName
    if (m_type != TestLoadType::EModelName) {
        return;
    }
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            EXPECT_NO_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), cacheFolders[0]}}));
            EXPECT_NO_THROW(getLoadFunction(TestLoadType::EModelName)(ie));
        });
    }
    {
        std::fstream stream(modelName, std::fstream::out | std::fstream::app);
        stream << " ";
    }
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*net, ExportImpl(_)).Times(1);
        testLoad([&](Core &ie) {
            EXPECT_NO_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), cacheFolders[0]}}));
            EXPECT_NO_THROW(getLoadFunction(TestLoadType::EModelName)(ie));
        });
    }
    { // Step 3: same load, should be ok now
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*net, ExportImpl(_)).Times(0);
        testLoad([&](Core &ie) {
            EXPECT_NO_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), cacheFolders[0]}}));
            EXPECT_NO_THROW(getLoadFunction(TestLoadType::EModelName)(ie));
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

TEST_P(CachingTest, LoadHeteroWithCorrectConfig) {
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    // TODO: test also HETERO with 1 plugin but different architectures, e.g. "HETERO:mock.1,mock.51"
    deviceToLoad = CommonTestUtils::DEVICE_HETERO + std::string(":mock.1,mock.2");
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

INSTANTIATE_TEST_CASE_P(CachingTest, CachingTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(loadVariants),
                            ::testing::ValuesIn(cacheFolders)),
                        getTestCaseName);
