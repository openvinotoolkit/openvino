// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "functional_test_utils/test_model/test_model.hpp"
#include "ie_core.hpp"
#include "ie_metric_helpers.hpp"
#include "ie_remote_context.hpp"
#include "ngraph/function.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/util/file_util.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "unit_test_utils/mocks/mock_iexecutable_network.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"

using namespace InferenceEngine;
using namespace ::testing;
using namespace InferenceEngine::details;
using namespace std::placeholders;
using namespace std::chrono;

enum class TestLoadType { ECNN, EContext, EModelName };

using TestParam = std::tuple<TestLoadType, std::string, bool>;

//  GCC4.8 limitation: have to specify type of each element in list
static const std::vector<TestParam> loadVariants = {
    TestParam{TestLoadType::ECNN, std::string("ByCNNNetwork"), false},
    TestParam{TestLoadType::EContext, std::string("ByRemoteContext"), true},
    TestParam{TestLoadType::EModelName, std::string("ByModelName"), false},
};

static const std::vector<std::string> cacheFolders{
    std::string("testCache"),
};

class MockRemoteContext : public RemoteContext {
    std::string m_name;

public:
    MockRemoteContext(std::string name) : m_name(std::move(name)) {}
    std::string getDeviceName() const noexcept override {
        return m_name;
    }
    MOCK_METHOD2(CreateBlob, RemoteBlob::Ptr(const TensorDesc&, const ParamMap&));
    MOCK_CONST_METHOD0(getParams, ParamMap());
};

class MockCachingInferencePluginBase : public InferenceEngine::IInferencePlugin {
public:
    MockCachingInferencePluginBase() = default;
    ~MockCachingInferencePluginBase() = default;

    ov::SoPtr<IExecutableNetworkInternal> LoadNetwork(const std::string& modelPath,
                                                      const std::map<std::string, std::string>& config) override {
        // In GTEST, it is not possible to call base implementation inside of mocked class
        // Thus, we define a proxy callback and will use
        // EXPECT_CALL(OnLoadNetworkFromFile) instead of EXPECT_CALL(LoadNetwork)
        OnLoadNetworkFromFile();
        return InferenceEngine::IInferencePlugin::LoadNetwork(modelPath, config);
    }

    virtual void OnLoadNetworkFromFile() const {}
};

class MockCachingInferencePlugin : public MockCachingInferencePluginBase {
public:
    MockCachingInferencePlugin() = default;
    ~MockCachingInferencePlugin() = default;

    MOCK_METHOD2(LoadExeNetworkImpl,
                 std::shared_ptr<IExecutableNetworkInternal>(const CNNNetwork& network,
                                                             const std::map<std::string, std::string>& config));

    MOCK_METHOD3(LoadExeNetworkImpl,
                 std::shared_ptr<IExecutableNetworkInternal>(const CNNNetwork& network,
                                                             const RemoteContext::Ptr& context,
                                                             const std::map<std::string, std::string>& config));

    MOCK_CONST_METHOD0(OnLoadNetworkFromFile, void(void));

    MOCK_METHOD2(ImportNetwork,
                 IExecutableNetworkInternal::Ptr(std::istream& networkModel,
                                                 const std::map<std::string, std::string>& config));

    MOCK_METHOD3(ImportNetwork,
                 IExecutableNetworkInternal::Ptr(std::istream& networkModel,
                                                 const RemoteContext::Ptr& context,
                                                 const std::map<std::string, std::string>& config));

    MOCK_CONST_METHOD2(QueryNetwork,
                       QueryNetworkResult(const CNNNetwork& network, const std::map<std::string, std::string>& config));

    MOCK_CONST_METHOD2(GetMetric, Parameter(const std::string& name, const std::map<std::string, Parameter>& options));
    MOCK_METHOD1(SetConfig, void(const std::map<std::string, std::string>& options));
    MOCK_METHOD1(GetDefaultContext, std::shared_ptr<RemoteContext>(const ParamMap& params));
};

class MockExecutableNetwork : public IExecutableNetworkInternal {
    std::mutex m_pluginMutex;
    std::shared_ptr<ov::Model> m_model = nullptr;

public:
    MockExecutableNetwork() {}

    MOCK_METHOD1(Export, void(std::ostream& networkModel));
    MOCK_METHOD0(CreateInferRequest, IInferRequestInternal::Ptr());
    MOCK_CONST_METHOD0(GetInputsInfo, ConstInputsDataMap());
    MOCK_CONST_METHOD0(GetOutputsInfo, ConstOutputsDataMap());
    MOCK_CONST_METHOD1(GetConfig, Parameter(const std::string& name));
    MOCK_CONST_METHOD1(GetMetric, Parameter(const std::string& name));
    MOCK_METHOD2(CreateInferRequestImpl, IInferRequestInternal::Ptr(InputsDataMap, OutputsDataMap));
    MOCK_METHOD1(setNetworkInputs, void(const InputsDataMap& networkInputs));
    MOCK_METHOD1(setNetworkOutputs, void(const OutputsDataMap& networkOutputs));
    MOCK_METHOD0(GetExecGraphInfo, std::shared_ptr<ov::Model>());

    // void Export(std::ostream& networkModel) override {
    //     std::lock_guard<std::mutex> guard(m_pluginMutex);
    //     IExecutableNetworkInternal::Export(networkModel);
    // }

    void set_model(const std::shared_ptr<const ov::Model>& model) {
        m_model = model->clone();
    }
    const std::shared_ptr<ov::Model>& get_model() const {
        return m_model;
    }

    void SetPointerToPlugin(const IInferencePlugin::Ptr& plugin) override {
        std::lock_guard<std::mutex> guard(m_pluginMutex);
        IExecutableNetworkInternal::SetPointerToPlugin(plugin);
    }
};

//------------------------------------------------------
class MkDirGuard {
    std::string m_dir;

public:
    explicit MkDirGuard(std::string dir = std::string()) : m_dir(std::move(dir)) {
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
    std::shared_ptr<void> sharedObjectLoader;
    std::function<void(IInferencePlugin*)> injectProxyEngine;
    std::string modelName = "Caching_test.xml";
    std::string weightsName = "Caching_test.bin";
    std::string deviceName = "mock";
    std::string deviceToLoad = "mock";
    std::shared_ptr<MockCachingInferencePlugin> mockPlugin;
    std::vector<std::shared_ptr<MockExecutableNetwork>> networks;
    std::mutex mock_creation_mutex;  // Internal gmock object registration is not thread-safe
    using ExeNetCallback = std::function<void(MockExecutableNetwork&)>;
    std::vector<ExeNetCallback> m_post_mock_net_callbacks = {};
    std::unique_ptr<MkDirGuard> m_dirCreator;
    TestLoadType m_type = TestLoadType::ECNN;
    std::string m_cacheDir;
    using LoadFunction = std::function<ExecutableNetwork(Core&)>;
    using LoadFunctionWithCfg = std::function<void(Core&, const std::map<std::string, std::string>&)>;
    LoadFunction m_testFunction;
    LoadFunctionWithCfg m_testFunctionWithCfg;
    bool m_remoteContext = false;
    using CNNCallback = std::function<void(CNNNetwork&)>;
    CNNCallback m_cnnCallback = nullptr;
    std::map<std::string, InputsDataMap> m_inputs_map;
    std::map<std::string, OutputsDataMap> m_outputs_map;
    using CheckConfigCb = std::function<void(const std::map<std::string, std::string>&)>;
    CheckConfigCb m_checkConfigCb = nullptr;

    static std::string get_mock_engine_path() {
        std::string mockEngineName("mock_engine");
        return ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(),
                                                  mockEngineName + IE_BUILD_POSTFIX);
    }

    void initParamTest() {
        m_type = std::get<0>(std::get<0>(GetParam()));
        m_cacheDir = std::get<1>(GetParam());
        m_testFunction = getLoadFunction(m_type);
        m_testFunctionWithCfg = getLoadFunctionWithCfg(m_type);
        m_remoteContext = std::get<2>(std::get<0>(GetParam()));
        auto testName = CommonTestUtils::generateTestFilePrefix();
        modelName = testName + ".xml";
        weightsName = testName + ".bin";
        m_cacheDir = testName + m_cacheDir;
        m_dirCreator = std::unique_ptr<MkDirGuard>(new MkDirGuard(m_cacheDir));
    }

    static std::shared_ptr<MockExecutableNetwork> createMockIExecutableNet(const std::string& name,
                                                                           const InputsDataMap& inputs_map,
                                                                           const OutputsDataMap& outputs_map) {
        auto mock = std::make_shared<MockExecutableNetwork>();
        ConstInputsDataMap inputMap;
        for (const auto& input_item : inputs_map) {
            inputMap.insert({input_item.first, input_item.second});
        }
        ConstOutputsDataMap outputMap;
        for (const auto& output_item : outputs_map) {
            outputMap.insert({output_item.first, output_item.second});
        }
        EXPECT_CALL(*mock, GetInputsInfo()).Times(AnyNumber()).WillRepeatedly(Return(inputMap));
        EXPECT_CALL(*mock, GetOutputsInfo()).Times(AnyNumber()).WillRepeatedly(Return(outputMap));
        EXPECT_CALL(*mock, GetConfig(ov::enable_profiling.name()))
            .Times(AnyNumber())
            .WillRepeatedly(Return(Parameter{PluginConfigParams::NO}));
        EXPECT_CALL(*mock, GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)))
            .Times(AnyNumber())
            .WillRepeatedly(Return(Parameter{1u}));
        EXPECT_CALL(*mock, GetExecGraphInfo()).Times(AnyNumber()).WillRepeatedly(Return([] {
            ngraph::ParameterVector parameters;
            parameters.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 8, 8}));
            auto notOp = std::make_shared<ov::op::v1::LogicalNot>(parameters.back());
            ngraph::ResultVector results;
            results.push_back(std::make_shared<ov::op::v0::Result>(notOp));
            return std::make_shared<ov::Model>(results, parameters, "empty_function");
        }()));
        auto ptr = std::make_shared<MockIInferRequestInternal>();
        EXPECT_CALL(*ptr, SetCallback(_)).Times(AnyNumber());
        EXPECT_CALL(*mock, CreateInferRequest()).Times(AnyNumber()).WillRepeatedly(Return(ptr));

        EXPECT_CALL(*mock, GetMetric(METRIC_KEY(NETWORK_NAME))).Times(AnyNumber()).WillRepeatedly(Return("mock_net"));
        EXPECT_CALL(*mock, GetMetric(METRIC_KEY(SUPPORTED_METRICS)))
            .Times(AnyNumber())
            .WillRepeatedly(Invoke([&](const std::string&) {
                std::vector<std::string> res;
                res.emplace_back(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
                res.emplace_back(METRIC_KEY(NETWORK_NAME));
                return res;
            }));
        EXPECT_CALL(*mock, GetMetric(ov::supported_properties.name()))
            .Times(AnyNumber())
            .WillRepeatedly(Return(std::vector<ov::PropertyName>{ov::supported_properties.name(),
                                                                 ov::optimal_number_of_infer_requests.name(),
                                                                 ov::model_name.name()}));
        EXPECT_CALL(*mock, setNetworkInputs(_)).Times(AnyNumber());
        EXPECT_CALL(*mock, setNetworkOutputs(_)).Times(AnyNumber());
        mock->setNetworkInputs(copyInfo(inputs_map));
        mock->setNetworkOutputs(copyInfo(outputs_map));
        ON_CALL(*mock, Export(_)).WillByDefault(Invoke([name](std::ostream& s) {
            s << name;
            s << ' ';
        }));
        return mock;
    }

    void SetUp() override {
        initParamTest();
        mockPlugin = std::make_shared<MockCachingInferencePlugin>();
        setupMock(*mockPlugin);
        std::string libraryPath = get_mock_engine_path();
        sharedObjectLoader = ov::util::load_shared_object(libraryPath.c_str());
        injectProxyEngine = make_std_function<void(IInferencePlugin*)>("InjectProxyEngine");

        FuncTestUtils::TestModel::generateTestModel(modelName, weightsName);
    }

    void TearDown() override {
        m_inputs_map = {};
        m_outputs_map = {};
        for (const auto& net : networks) {
            EXPECT_TRUE(Mock::VerifyAndClearExpectations(net.get()));
        }
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockPlugin.get()));
        CommonTestUtils::removeIRFiles(modelName, weightsName);
    }

    void testLoad(const std::function<void(Core& ie)>& func) {
        Core ie;
        injectProxyEngine(mockPlugin.get());
        ie.RegisterPlugin(ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(),
                                                             std::string("mock_engine") + IE_BUILD_POSTFIX),
                          deviceName);
        func(ie);
        ie.UnregisterPlugin(deviceName);
    }

    LoadFunction getLoadFunction(TestLoadType type) const {
        switch (type) {
        case TestLoadType::ECNN:
            return [&](Core& ie) {
                return performReadAndLoad(ie);
            };
        case TestLoadType::EContext:
            return [&](Core& ie) {
                return performReadAndLoadWithContext(ie);
            };
        case TestLoadType::EModelName:
            return [&](Core& ie) {
                return performLoadByName(ie);
            };
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

    ExecutableNetwork performLoadByName(Core& ie, const std::map<std::string, std::string>& config = {}) const {
        return ie.LoadNetwork(modelName, deviceToLoad, config);
    }

    ExecutableNetwork performReadAndLoad(Core& ie, const std::map<std::string, std::string>& config = {}) const {
        auto cnnNetwork = ie.ReadNetwork(modelName);
        if (m_cnnCallback)
            m_cnnCallback(cnnNetwork);
        return ie.LoadNetwork(cnnNetwork, deviceToLoad, config);
    }

    ExecutableNetwork performReadAndLoadWithContext(Core& ie,
                                                    const std::map<std::string, std::string>& config = {}) const {
        auto cnnNetwork = ie.ReadNetwork(modelName);
        EXPECT_CALL(*mockPlugin, GetDefaultContext(_)).Times(AnyNumber());
        auto context = ie.GetDefaultContext(deviceToLoad);
        if (m_cnnCallback)
            m_cnnCallback(cnnNetwork);
        return ie.LoadNetwork(cnnNetwork, context, config);
    }

private:
    template <class T>
    std::function<T> make_std_function(const std::string& functionName) {
        std::function<T> ptr(reinterpret_cast<T*>(ov::util::get_symbol(sharedObjectLoader, functionName.c_str())));
        return ptr;
    }

    void setupMock(MockCachingInferencePlugin& plugin) {
        ON_CALL(plugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _))
            .WillByDefault(Invoke([&](const std::string&, const std::map<std::string, Parameter>&) {
                std::vector<std::string> res;
                res.emplace_back(METRIC_KEY(IMPORT_EXPORT_SUPPORT));
                res.emplace_back(METRIC_KEY(DEVICE_ARCHITECTURE));
                return res;
            }));
        ON_CALL(plugin, GetMetric(ov::supported_properties.name(), _))
            .WillByDefault(Invoke([&](const std::string&, const std::map<std::string, Parameter>&) {
                return std::vector<ov::PropertyName>{ov::supported_properties.name(),
                                                     METRIC_KEY(IMPORT_EXPORT_SUPPORT),
                                                     ov::device::capabilities.name(),
                                                     ov::device::architecture.name()};
            }));

        ON_CALL(plugin, GetMetric(METRIC_KEY(OPTIMIZATION_CAPABILITIES), _))
            .WillByDefault(Return(std::vector<std::string>()));

        ON_CALL(plugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).WillByDefault(Return(true));

        ON_CALL(plugin, GetMetric(ov::device::capabilities.name(), _))
            .WillByDefault(Invoke([&](const std::string&, const std::map<std::string, Parameter>&) {
                return decltype(ov::device::capabilities)::value_type{ov::device::capability::EXPORT_IMPORT};
            }));

        ON_CALL(plugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _))
            .WillByDefault(Invoke([&](const std::string&, const std::map<std::string, Parameter>&) {
                std::vector<std::string> res;
                res.emplace_back("SomeConfig");
                return res;
            }));

        ON_CALL(plugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _))
            .WillByDefault(Invoke([&](const std::string&, const std::map<std::string, Parameter>&) {
                return "mock";
            }));

        ON_CALL(plugin, ImportNetwork(_, _, _))
            .WillByDefault(Invoke(
                [&](std::istream& istr, const RemoteContext::Ptr&, const std::map<std::string, std::string>& config) {
                    if (m_checkConfigCb) {
                        m_checkConfigCb(config);
                    }
                    std::string name;
                    istr >> name;
                    char space;
                    istr.read(&space, 1);
                    std::lock_guard<std::mutex> lock(mock_creation_mutex);
                    return createMockIExecutableNet({}, m_inputs_map[name], m_outputs_map[name]);
                }));

        ON_CALL(plugin, ImportNetwork(_, _))
            .WillByDefault(Invoke([&](std::istream& istr, const std::map<std::string, std::string>& config) {
                if (m_checkConfigCb) {
                    m_checkConfigCb(config);
                }
                std::string name;
                istr >> name;
                char space;
                istr.read(&space, 1);
                std::lock_guard<std::mutex> lock(mock_creation_mutex);
                return createMockIExecutableNet({}, m_inputs_map[name], m_outputs_map[name]);
            }));

        ON_CALL(plugin, LoadExeNetworkImpl(_, _, _))
            .WillByDefault(Invoke([&](const CNNNetwork& cnn,
                                      const RemoteContext::Ptr&,
                                      const std::map<std::string, std::string>& config) {
                if (m_checkConfigCb) {
                    m_checkConfigCb(config);
                }
                std::lock_guard<std::mutex> lock(mock_creation_mutex);
                std::string name = cnn.getFunction()->get_friendly_name();
                m_inputs_map[name] = cnn.getInputsInfo();
                m_outputs_map[name] = cnn.getOutputsInfo();
                auto exe_net = createMockIExecutableNet(cnn.getFunction()->get_friendly_name(),
                                                        m_inputs_map[name],
                                                        m_outputs_map[name]);
                exe_net->set_model(cnn.getFunction());
                for (const auto& cb : m_post_mock_net_callbacks) {
                    cb(*exe_net);
                }
                networks.push_back(exe_net);
                return exe_net;
            }));

        ON_CALL(plugin, LoadExeNetworkImpl(_, _))
            .WillByDefault(Invoke([&](const CNNNetwork& cnn, const std::map<std::string, std::string>& config) {
                if (m_checkConfigCb) {
                    m_checkConfigCb(config);
                }
                std::string name = cnn.getFunction()->get_friendly_name();
                std::lock_guard<std::mutex> lock(mock_creation_mutex);
                m_inputs_map[name] = cnn.getInputsInfo();
                m_outputs_map[name] = cnn.getOutputsInfo();
                auto exe_net = createMockIExecutableNet(cnn.getFunction()->get_friendly_name(),
                                                        m_inputs_map[name],
                                                        m_outputs_map[name]);
                exe_net->set_model(cnn.getFunction());
                for (const auto& cb : m_post_mock_net_callbacks) {
                    cb(*exe_net);
                }
                networks.push_back(exe_net);
                return exe_net;
            }));

        ON_CALL(plugin, GetDefaultContext(_)).WillByDefault(Invoke([&](const ParamMap&) {
            return std::make_shared<MockRemoteContext>(deviceToLoad);
        }));

        ON_CALL(plugin, QueryNetwork(_, _))
            .WillByDefault(Invoke([&](const CNNNetwork& network, const std::map<std::string, std::string>&) {
                QueryNetworkResult res;
                auto function = network.getFunction();
                EXPECT_TRUE(function);

                for (auto&& node : function->get_ops()) {
                    res.supportedLayersMap.emplace(node->get_friendly_name(), deviceName);
                }
                return res;
            }));

        EXPECT_CALL(plugin, SetConfig(_))
            .Times(AnyNumber())
            .WillRepeatedly(Invoke([](const std::map<std::string, std::string>&) {
                throw InferenceEngine::NotImplemented("Not implemented");
            }));
    }
};

TEST_P(CachingTest, TestLoad) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
        EXPECT_EQ(networks.size(), 1);
    }

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : networks) {
            EXPECT_CALL(*net, Export(_)).Times(0);  // No more 'Export' for existing networks
        }
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
        EXPECT_EQ(networks.size(), 1);
    }
}

/// \brief Verifies that ie.SetConfig({{"CACHE_DIR", <dir>}}, "deviceName"}}); enables caching for one device
TEST_P(CachingTest, TestLoad_by_device_name) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}, "mock");
            m_testFunction(ie);
        });
        EXPECT_EQ(networks.size(), 1);
    }

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : networks) {
            EXPECT_CALL(*net, Export(_)).Times(0);  // No more 'Export' for existing networks
        }
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}, "mock");
            m_testFunction(ie);
        });
        EXPECT_EQ(networks.size(), 1);
    }
}

TEST_P(CachingTest, TestLoadCustomImportExport) {
    const char customData[] = {1, 2, 3, 4, 5};
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    ON_CALL(*mockPlugin, ImportNetwork(_, _, _))
        .WillByDefault(
            Invoke([&](std::istream& s, const RemoteContext::Ptr&, const std::map<std::string, std::string>&) {
                char a[sizeof(customData)];
                s.read(a, sizeof(customData));
                EXPECT_EQ(memcmp(a, customData, sizeof(customData)), 0);
                std::string name;
                s >> name;
                std::lock_guard<std::mutex> lock(mock_creation_mutex);
                return createMockIExecutableNet({}, m_inputs_map[name], m_outputs_map[name]);
            }));

    ON_CALL(*mockPlugin, ImportNetwork(_, _))
        .WillByDefault(Invoke([&](std::istream& s, const std::map<std::string, std::string>&) {
            char a[sizeof(customData)];
            s.read(a, sizeof(customData));
            EXPECT_EQ(memcmp(a, customData, sizeof(customData)), 0);
            std::string name;
            s >> name;
            std::lock_guard<std::mutex> lock(mock_creation_mutex);
            return createMockIExecutableNet({}, m_inputs_map[name], m_outputs_map[name]);
        }));

    m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
        ON_CALL(net, Export(_)).WillByDefault(Invoke([&](std::ostream& s) {
            s.write(customData, sizeof(customData));
            s << net.get_model()->get_friendly_name();
        }));
    });

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : networks) {
            EXPECT_CALL(*net, Export(_)).Times(0);  // No 'Export' for existing networks
        }
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
}

// Brief: when LoadNetwork is called from different config - old cache shall not be used
TEST_P(CachingTest, TestChangeLoadConfig) {
    const std::string CUSTOM_KEY = "CUSTOM_KEY";
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::caching_properties.name(), _)).Times(AnyNumber());
    ON_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _))
        .WillByDefault(Invoke([&](const std::string&, const std::map<std::string, Parameter>&) {
            return std::vector<ov::PropertyName>{ov::supported_properties.name(),
                                                 METRIC_KEY(IMPORT_EXPORT_SUPPORT),
                                                 ov::device::capabilities.name(),
                                                 ov::device::architecture.name(),
                                                 ov::caching_properties.name()};
        }));
    ON_CALL(*mockPlugin, GetMetric(ov::caching_properties.name(), _))
        .WillByDefault(Invoke([&](const std::string&, const std::map<std::string, Parameter>&) {
            std::vector<ov::PropertyName> res;
            res.push_back(ov::PropertyName(CUSTOM_KEY, ov::PropertyMutability::RO));
            return decltype(ov::caching_properties)::value_type(res);
        }));
    ON_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _))
        .WillByDefault(Invoke([&](const std::string&, const std::map<std::string, Parameter>&) {
            std::vector<std::string> res;
            res.push_back(ov::caching_properties.name());
            return res;
        }));
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunctionWithCfg(ie, {{CUSTOM_KEY, "0"}});
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunctionWithCfg(ie, {{CUSTOM_KEY, "1"}});
        });
    }
}

/// \brief Verifies that ie.LoadNetwork(cnn, "deviceName", {{"CACHE_DIR", <dir>>}}) works
TEST_P(CachingTest, TestChangeLoadConfig_With_Cache_Dir_inline) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    ON_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _))
        .WillByDefault(Invoke([&](const std::string&, const std::map<std::string, Parameter>&) {
            return std::vector<std::string>{};
        }));
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            m_testFunctionWithCfg(ie, {{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : networks) {
            EXPECT_CALL(*net, Export(_)).Times(0);  // No more 'Export' for existing networks
        }
        testLoad([&](Core& ie) {
            m_testFunctionWithCfg(ie, {{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
        });
        EXPECT_EQ(networks.size(), 1);
    }
}

TEST_P(CachingTest, TestNoCacheEnabled) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(0);
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(0);
        });
        testLoad([&](Core& ie) {
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestNoCacheSupported) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(false));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::device::capabilities.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(decltype(ov::device::capabilities)::value_type{}));

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, OnLoadNetworkFromFile()).Times(m_type == TestLoadType::EModelName ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(0);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestNoCacheMetricSupported) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{}));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<std::string>{}));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(0);
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(0);
    EXPECT_CALL(*mockPlugin, GetMetric(ov::device::capabilities.name(), _)).Times(0);
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, OnLoadNetworkFromFile()).Times(m_type == TestLoadType::EModelName ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(0);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
}

/// \brief If device doesn't support 'cache_dir' or 'import_export' - setting cache_dir is ignored
TEST_P(CachingTest, TestNoCacheMetricSupported_by_device_name) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{}));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<std::string>{}));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(0);
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(0);
    EXPECT_CALL(*mockPlugin, GetMetric(ov::device::capabilities.name(), _)).Times(0);
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, OnLoadNetworkFromFile()).Times(m_type == TestLoadType::EModelName ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(0);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}, "mock");
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestNoCacheMetric_hasCacheDirConfig) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<std::string>{METRIC_KEY(SUPPORTED_CONFIG_KEYS)}));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<std::string>{CONFIG_KEY(CACHE_DIR)}));
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{ov::supported_properties.name(), ov::cache_dir.name()}));
    EXPECT_CALL(*mockPlugin, SetConfig(_))
        .Times(AtLeast(1))
        .WillRepeatedly(Invoke([](const std::map<std::string, std::string>& config) {
            ASSERT_GT(config.count(CONFIG_KEY(CACHE_DIR)), 0);
        }));

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, OnLoadNetworkFromFile()).Times(m_type == TestLoadType::EModelName ? 1 : 0);
        ASSERT_NO_THROW(testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        }));
    }
}

/// \brief If device supports 'cache_dir' or 'import_export' - setting cache_dir is passed to plugin on ie.LoadNetwork
TEST_P(CachingTest, TestNoCacheMetric_hasCacheDirConfig_inline) {
    m_checkConfigCb = [](const std::map<std::string, std::string>& config) {
        EXPECT_NE(config.count(CONFIG_KEY(CACHE_DIR)), 0);
    };
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<std::string>{METRIC_KEY(SUPPORTED_CONFIG_KEYS)}));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<std::string>{CONFIG_KEY(CACHE_DIR)}));
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{ov::supported_properties.name(), ov::cache_dir.name()}));

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, OnLoadNetworkFromFile()).Times(m_type == TestLoadType::EModelName ? 1 : 0);
        ASSERT_NO_THROW(testLoad([&](Core& ie) {
            m_testFunctionWithCfg(ie, {{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
        }));
    }
}

/// \brief ie.SetConfig(<cachedir>, "deviceName") is propagated to plugin's SetConfig if device supports CACHE_DIR
TEST_P(CachingTest, TestNoCacheMetric_hasCacheDirConfig_by_device_name) {
    m_checkConfigCb = [](const std::map<std::string, std::string>& config) {
        // Shall be '0' as appropriate 'cache_dir' is expected in SetConfig, not in Load/Import network
        EXPECT_EQ(config.count(CONFIG_KEY(CACHE_DIR)), 0);
    };
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<std::string>{METRIC_KEY(SUPPORTED_CONFIG_KEYS)}));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<std::string>{CONFIG_KEY(CACHE_DIR)}));
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{ov::supported_properties.name(), ov::cache_dir.name()}));
    EXPECT_CALL(*mockPlugin, SetConfig(_))
        .Times(AtLeast(1))
        .WillRepeatedly(Invoke([](const std::map<std::string, std::string>& config) {
            ASSERT_GT(config.count(CONFIG_KEY(CACHE_DIR)), 0);
        }));

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, OnLoadNetworkFromFile()).Times(m_type == TestLoadType::EModelName ? 1 : 0);
        ASSERT_NO_THROW(testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}, "mock");
            m_testFunction(ie);
        }));
    }
}

TEST_P(CachingTest, TestCacheEnabled_noConfig) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<std::string>{METRIC_KEY(SUPPORTED_CONFIG_KEYS)}));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<std::string>{}));
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{ov::supported_properties.name()}));
    EXPECT_CALL(*mockPlugin, SetConfig(_))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([](const std::map<std::string, std::string>& config) {
            ASSERT_EQ(config.count(CONFIG_KEY(CACHE_DIR)), 0);
        }));

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, OnLoadNetworkFromFile()).Times(m_type == TestLoadType::EModelName ? 1 : 0);
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestNoCacheMetric_configThrow) {
    m_checkConfigCb = [](const std::map<std::string, std::string>& config) {
        EXPECT_NE(config.count(CONFIG_KEY(CACHE_DIR)), 0);
    };
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<std::string>{METRIC_KEY(SUPPORTED_CONFIG_KEYS)}));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<std::string>{CONFIG_KEY(CACHE_DIR)}));
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{ov::supported_properties.name(), ov::cache_dir.name()}));
    EXPECT_CALL(*mockPlugin, SetConfig(_))
        .Times(AtLeast(1))
        .WillRepeatedly(Invoke([](const std::map<std::string, std::string>& config) {
            ASSERT_GT(config.count(CONFIG_KEY(CACHE_DIR)), 0);
            throw InferenceEngine::GeneralError("Error occurred");
        }));

    ASSERT_ANY_THROW(testLoad([&](Core& ie) {
        ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
        m_testFunction(ie);
    }));
}

TEST_P(CachingTest, TestNoCacheEnabled_cacheDirConfig) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<std::string>{METRIC_KEY(SUPPORTED_CONFIG_KEYS)}));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<std::string>{CONFIG_KEY(CACHE_DIR)}));
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{ov::supported_properties.name(), ov::cache_dir.name()}));
    EXPECT_CALL(*mockPlugin, SetConfig(_))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([](const std::map<std::string, std::string>& config) {
            ASSERT_EQ(config.count(CONFIG_KEY(CACHE_DIR)), 0);
        }));

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        testLoad([&](Core& ie) {
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestLoadChangeCacheDir) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {
        std::string newCacheDir = m_cacheDir + "2";
        MkDirGuard dir(newCacheDir);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), newCacheDir}});
            m_testFunction(ie);
        });
    }
}

/// \brief Change CACHE_DIR during working with same 'Core' object. Verifies that new dir is used for caching
TEST_P(CachingTest, TestLoadChangeCacheDirOneCore) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, SetConfig(_))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([](const std::map<std::string, std::string>& config) {
            ASSERT_EQ(config.count(CONFIG_KEY(CACHE_DIR)), 0);
        }));
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        testLoad([&](Core& ie) {
            m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
                EXPECT_CALL(net, Export(_)).Times(1);
            });
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
            std::string newCacheDir = m_cacheDir + "2";
            m_post_mock_net_callbacks.pop_back();
            m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
                EXPECT_CALL(net, Export(_)).Times(1);
            });
            MkDirGuard dir(newCacheDir);
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), newCacheDir}});
            m_testFunction(ie);
        });
    }
}

/// \brief Change CACHE_DIR during working with same 'Core' object
/// Initially set for 'device', then is overwritten with global 'cache_dir' for all devices
TEST_P(CachingTest, TestLoadChangeCacheDirOneCore_overwrite_device_dir) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, SetConfig(_))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([](const std::map<std::string, std::string>& config) {
            ASSERT_EQ(config.count(CONFIG_KEY(CACHE_DIR)), 0);
        }));
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        testLoad([&](Core& ie) {
            m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
                EXPECT_CALL(net, Export(_)).Times(1);
            });
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}, "mock");
            m_testFunction(ie);
            std::string newCacheDir = m_cacheDir + "2";
            m_post_mock_net_callbacks.pop_back();
            m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
                EXPECT_CALL(net, Export(_)).Times(1);
            });
            MkDirGuard dir(newCacheDir);
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), newCacheDir}});
            m_testFunction(ie);
        });
    }
}

/// \brief Change CACHE_DIR during working with same 'Core' object for device which supports 'CACHE_DIR' config, not
/// import_export Expectation is that SetConfig for plugin will be called 2 times - with appropriate cache_dir values
TEST_P(CachingTest, TestLoadChangeCacheDirOneCore_SupportsCacheDir_NoImportExport) {
    m_checkConfigCb = [](const std::map<std::string, std::string>& config) {
        EXPECT_EQ(config.count(CONFIG_KEY(CACHE_DIR)), 0);
    };
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<std::string>{CONFIG_KEY(CACHE_DIR)}));
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{ov::supported_properties.name(), ov::cache_dir.name()}));
    EXPECT_CALL(*mockPlugin, GetMetric(ov::device::capabilities.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<std::string>{METRIC_KEY(SUPPORTED_CONFIG_KEYS)}));
    EXPECT_CALL(*mockPlugin, GetMetric(ov::device::capabilities.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(decltype(ov::device::capabilities)::value_type{}));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    std::string set_cache_dir = {};
    EXPECT_CALL(*mockPlugin, SetConfig(_))
        .Times(AtLeast(2))
        .WillRepeatedly(Invoke([&](const std::map<std::string, std::string>& config) {
            ASSERT_NE(config.count(CONFIG_KEY(CACHE_DIR)), 0);
            set_cache_dir = config.at(CONFIG_KEY(CACHE_DIR));
        }));
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, OnLoadNetworkFromFile()).Times(m_type == TestLoadType::EModelName ? 2 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(0);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
            EXPECT_EQ(set_cache_dir, m_cacheDir);

            std::string new_cache_dir = m_cacheDir + "2";
            MkDirGuard dir(new_cache_dir);
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), new_cache_dir}});
            m_testFunction(ie);
            EXPECT_EQ(set_cache_dir, new_cache_dir);
        });
    }
}

/// \brief Change CACHE_DIR per device during working with same 'Core' object - expected that new cache dir is used
TEST_P(CachingTest, TestLoadChangeCacheDirOneCore_by_device_name) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, SetConfig(_))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([](const std::map<std::string, std::string>& config) {
            ASSERT_EQ(config.count(CONFIG_KEY(CACHE_DIR)), 0);
        }));
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        testLoad([&](Core& ie) {
            m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
                EXPECT_CALL(net, Export(_)).Times(1);
            });
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}, "mock");
            m_testFunction(ie);
            m_post_mock_net_callbacks.pop_back();
            m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
                EXPECT_CALL(net, Export(_)).Times(1);
            });
            std::string newCacheDir = m_cacheDir + "2";
            MkDirGuard dir(newCacheDir);
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), newCacheDir}}, "mock");
            m_testFunction(ie);
        });
    }
}

/// \brief Change CACHE_DIR per device during working with same 'Core' object - device supports CACHE_DIR
/// Verifies that no 'export' is called and cache_dir is propagated to set_config
TEST_P(CachingTest, TestLoadChangeCacheDirOneCore_by_device_name_supports_cache_dir) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<std::string>{CONFIG_KEY(CACHE_DIR)}));
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{ov::cache_dir.name()}));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(false));
    EXPECT_CALL(*mockPlugin, GetMetric(ov::device::capabilities.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(decltype(ov::device::capabilities)::value_type{}));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, SetConfig(_))
        .Times(AtLeast(2))
        .WillRepeatedly(Invoke([](const std::map<std::string, std::string>& config) {
            ASSERT_GT(config.count(CONFIG_KEY(CACHE_DIR)), 0);
        }));
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, OnLoadNetworkFromFile()).Times(m_type == TestLoadType::EModelName ? 2 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        testLoad([&](Core& ie) {
            m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
                EXPECT_CALL(net, Export(_)).Times(0);
            });
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}, "mock");
            m_testFunction(ie);
            m_post_mock_net_callbacks.pop_back();
            m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
                EXPECT_CALL(net, Export(_)).Times(0);
            });
            std::string newCacheDir = m_cacheDir + "2";
            MkDirGuard dir(newCacheDir);
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), newCacheDir}}, "mock");
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestClearCacheDir) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(0);
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        for (auto& net : networks) {
            EXPECT_CALL(*net, Export(_)).Times(0);
        }
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), ""}});
            m_testFunction(ie);
        });
        EXPECT_EQ(networks.size(), 1);
    }
}

TEST_P(CachingTest, TestChangeOtherConfig) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            ie.SetConfig({{"someKey", "someValue"}});
            m_testFunction(ie);
        });
        EXPECT_EQ(networks.size(), 1);
    }
}

TEST_P(CachingTest, TestChangeCacheDirFailure) {
    std::string longName(1000000, ' ');
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
        EXPECT_EQ(networks.size(), 1);
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(!m_remoteContext ? 1 : 0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
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

    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            EXPECT_NO_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), newCacheDir3}}));
            EXPECT_NO_THROW(m_testFunction(ie));
        });
    }
    CommonTestUtils::removeFilesWithExt(newCacheDir2, "blob");
    CommonTestUtils::removeDir(newCacheDir2);
    CommonTestUtils::removeDir(newCacheDir1);
}

TEST_P(CachingTest, TestDeviceArchitecture) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _))
        .Times(AnyNumber())
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
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            deviceToLoad = "mock.0";
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : networks) {
            EXPECT_CALL(*net, Export(_)).Times(0);
        }
        testLoad([&](Core& ie) {
            deviceToLoad = "mock.1";
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            deviceToLoad = "mock.50";
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : networks) {
            EXPECT_CALL(*net, Export(_)).Times(0);
        }
        testLoad([&](Core& ie) {
            deviceToLoad = "mock.51";
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestNoDeviceArchitecture) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([&](const std::string&, const std::map<std::string, Parameter>&) {
            return std::vector<ov::PropertyName>{ov::device::capabilities.name()};
        }));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([&](const std::string&, const std::map<std::string, Parameter>&) {
            return std::vector<std::string>{METRIC_KEY(IMPORT_EXPORT_SUPPORT)};
        }));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::device::capabilities.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(decltype(ov::device::capabilities)::value_type{ov::device::capability::EXPORT_IMPORT}));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(0);
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            deviceToLoad = "mock.0";
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : networks) {
            EXPECT_CALL(*net, Export(_)).Times(0);
        }
        testLoad([&](Core& ie) {
            deviceToLoad = "mock.50";
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestThrowOnExport) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1).WillOnce(Throw(1));
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            EXPECT_ANY_THROW(m_testFunction(ie));
        });
    }
}

// TODO: temporary behavior is to no re-throw exception on import error (see 54335)
// In future add separate 'no throw' test for 'blob_outdated' exception from plugin
TEST_P(CachingTest, TestThrowOnImport) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        if (m_remoteContext) {
            EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(1).WillOnce(Throw(1));
            EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        } else {
            EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
            EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(1).WillOnce(Throw(1));
        }
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            EXPECT_NO_THROW(m_testFunction(ie));
        });
    }
    {  // Step 3: same load, cache is re-created on export on step 2 and shall be successfully imported now
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : networks) {
            EXPECT_CALL(*net, Export(_)).Times(0);
        }
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            EXPECT_NO_THROW(m_testFunction(ie));
        });
    }
}

TEST_P(CachingTest, TestNetworkModified) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
    if (m_type == TestLoadType::EModelName) {
        // Modify model file
        std::fstream stream(modelName, std::fstream::out | std::fstream::app);
        stream << " ";
    } else {
        // Modify loaded CNN network
        m_cnnCallback = [&](CNNNetwork& network) {
            network.getInputsInfo()["Param_1"]->setLayout(Layout::NHWC);
        };
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {  // Step 3: same load, should be ok now
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : networks) {
            EXPECT_CALL(*net, Export(_)).Times(0);
        }
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
}

TEST_P(CachingTest, TestCacheFileCorrupted) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
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
    m_post_mock_net_callbacks.pop_back();
    {  // Step 2. Cache is corrupted, will be silently removed
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            EXPECT_NO_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}));
            EXPECT_NO_THROW(m_testFunction(ie));
        });
    }
    {  // Step 3: same load, should be ok now due to re-creation of cache
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : networks) {
            EXPECT_CALL(*net, Export(_)).Times(0);
        }
        testLoad([&](Core& ie) {
            EXPECT_NO_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}));
            EXPECT_NO_THROW(m_testFunction(ie));
        });
    }
}

TEST_P(CachingTest, TestCacheFileOldVersion) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
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
                return;  // skip test
            }
            std::ofstream out(fileName, std::ios_base::binary);
            out.write(content.c_str(), static_cast<std::streamsize>(content.size()));
        }
    }
    m_post_mock_net_callbacks.pop_back();
    {  // Step 2. Build number mismatch, cache will be silently removed
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            EXPECT_NO_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}));
            EXPECT_NO_THROW(m_testFunction(ie));
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {  // Step 3: same load, should be ok now due to re-creation of cache
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : networks) {
            EXPECT_CALL(*net, Export(_)).Times(0);
        }
        testLoad([&](Core& ie) {
            EXPECT_NO_THROW(ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}}));
            EXPECT_NO_THROW(m_testFunction(ie));
        });
    }
}

#if defined(ENABLE_HETERO)
TEST_P(CachingTest, LoadHetero_NoCacheMetric) {
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<std::string>{}));
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(SUPPORTED_METRICS), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<std::string>{}));
    EXPECT_CALL(*mockPlugin, GetMetric(ov::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{}));
    // Hetero supports Import/Export, but mock plugin does not
    deviceToLoad = CommonTestUtils::DEVICE_HETERO + std::string(":mock.1,mock.2");
    if (m_remoteContext) {
        return;  // skip the remote Context test for Hetero plugin
    }
    for (int i = 0; i < 2; i++) {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        for (auto& net : networks) {
            EXPECT_CALL(*net, Export(_)).Times(0);
        }
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
            networks.clear();
        });
    }
}

TEST_P(CachingTest, LoadHetero_OneDevice) {
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    deviceToLoad = CommonTestUtils::DEVICE_HETERO + std::string(":mock");
    if (m_remoteContext) {
        return;  // skip the remote Context test for Hetero plugin
    }
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
        // Ensure that only 1 blob (for Hetero) is created
        EXPECT_EQ(CommonTestUtils::listFilesWithExt(m_cacheDir, "blob").size(), 1);
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(1);
        for (auto& net : networks) {
            EXPECT_CALL(*net, Export(_)).Times(0);
        }
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
            networks.clear();
        });
    }
}

TEST_P(CachingTest, LoadHetero_TargetFallbackFromCore) {
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    deviceToLoad = CommonTestUtils::DEVICE_HETERO;
    if (m_remoteContext) {
        return;  // skip the remote Context test for Hetero plugin
    }
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            ie.SetConfig({{"TARGET_FALLBACK", "mock"}}, CommonTestUtils::DEVICE_HETERO);
            m_testFunction(ie);
        });
        // Ensure that only 1 blob (for Hetero) is created
        EXPECT_EQ(CommonTestUtils::listFilesWithExt(m_cacheDir, "blob").size(), 1);
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(1);
        for (auto& net : networks) {
            EXPECT_CALL(*net, Export(_)).Times(0);
        }
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            ie.SetConfig({{"TARGET_FALLBACK", "mock"}}, CommonTestUtils::DEVICE_HETERO);
            m_testFunction(ie);
            networks.clear();
        });
    }
}

TEST_P(CachingTest, LoadHetero_MultiArchs) {
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());

    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([&](const CNNNetwork& network, const std::map<std::string, std::string>& config) {
            QueryNetworkResult res;
            auto function = network.getFunction();
            EXPECT_TRUE(function);

            auto id = config.at("DEVICE_ID");
            bool supportsRelu = std::stoi(id) < 10;

            for (auto&& node : function->get_ops()) {
                std::string nodeType = node->get_type_name();
                if ((nodeType == "Relu" && supportsRelu) || (nodeType != "Relu" && !supportsRelu)) {
                    res.supportedLayersMap.emplace(node->get_friendly_name(), deviceName + "." + id);
                }
            }
            return res;
        }));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([&](const std::string&, const std::map<std::string, Parameter>& options) {
            auto id = options.at("DEVICE_ID").as<std::string>();
            if (std::stoi(id) < 10) {
                return "mock_first_architecture";
            } else {
                return "mock_another_architecture";
            }
        }));
    deviceToLoad = CommonTestUtils::DEVICE_HETERO + std::string(":mock.1,mock.51");
    if (m_remoteContext) {
        return;  // skip the remote Context test for Hetero plugin
    }
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(AtLeast(2));  // for .1 and for .51
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(AtLeast(1));
        });
        testLoad([&](Core& ie) {
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
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(AtLeast(2));  // for .2 and for .52
        for (auto& net : networks) {
            EXPECT_CALL(*net, Export(_)).Times(0);
        }
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
    deviceToLoad = CommonTestUtils::DEVICE_HETERO + std::string(":mock.53,mock.3");
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(AtLeast(1));
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(AtLeast(1));
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
            networks.clear();
        });
    }
}

TEST_P(CachingTest, LoadHetero_MultiArchs_TargetFallback_FromCore) {
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([&](const std::string&, const std::map<std::string, Parameter>& options) {
            auto id = options.at("DEVICE_ID").as<std::string>();
            if (std::stoi(id) < 10) {
                return "mock_first_architecture";
            } else {
                return "mock_another_architecture";
            }
        }));
    deviceToLoad = CommonTestUtils::DEVICE_HETERO;
    if (m_remoteContext) {
        return;  // skip the remote Context test for Hetero plugin
    }
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            ie.SetConfig({{"TARGET_FALLBACK", "mock.1"}}, CommonTestUtils::DEVICE_HETERO);
            m_testFunction(ie);
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(1);
        for (auto& net : networks) {
            EXPECT_CALL(*net, Export(_)).Times(0);
        }
        testLoad([&](Core& ie) {
            ie.SetConfig({{"TARGET_FALLBACK", "mock.1"}}, CommonTestUtils::DEVICE_HETERO);
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{"TARGET_FALLBACK", "mock.51"}}, CommonTestUtils::DEVICE_HETERO);
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
            networks.clear();
        });
    }
}
#endif  // define(ENABLE_HETERO)

#if defined(ENABLE_AUTO)
// AUTO-DEVICE test
// Single device
TEST_P(CachingTest, LoadAUTO_OneDevice) {
    const auto TEST_COUNT = 2;
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Auto plugin
    }
    m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
        EXPECT_CALL(net, Export(_)).Times(1);
    });
    std::string cacheDir = m_cacheDir;
    MkDirGuard guard(cacheDir);
    for (int index = 0; index < TEST_COUNT; index++) {
        deviceToLoad = CommonTestUtils::DEVICE_AUTO;
        deviceToLoad += ":mock.0";
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(TEST_COUNT - index - 1);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(index);
        ASSERT_NO_THROW(testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), cacheDir}});
            m_testFunction(ie);
        }));
    }
    std::cout << "Caching LoadAuto Test completed. Tried " << TEST_COUNT << " times" << std::endl;
}
// AUTO-DEVICE test
// load network with config
TEST_P(CachingTest, LoadAUTOWithConfig) {
    const auto TEST_COUNT = 2;
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Auto plugin
    }
    int index = 0;
    m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
        EXPECT_CALL(net, Export(_)).Times(1);
    });
    std::string cacheDir = m_cacheDir;
    MkDirGuard guard(cacheDir);
    for (; index < TEST_COUNT; index++) {
        deviceToLoad = CommonTestUtils::DEVICE_AUTO;
        deviceToLoad += ":mock.0";
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(TEST_COUNT - index - 1);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(index);
        ASSERT_NO_THROW(testLoad([&](Core& ie) {
            m_testFunctionWithCfg(ie, {{CONFIG_KEY(CACHE_DIR), cacheDir}});
        }));
    }
    std::cout << "Caching LoadAuto Test completed. Tried " << index << " times" << std::endl;
}
// Single device not support import/export
TEST_P(CachingTest, LoadAUTO_OneDeviceNoImportExport) {
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(false));
    EXPECT_CALL(*mockPlugin, GetMetric(ov::device::capabilities.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(decltype(ov::device::capabilities)::value_type{}));
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Auto plugin
    }
    EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(m_remoteContext ? 2 : 0);
    EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(!m_remoteContext ? 2 : 0);
    EXPECT_CALL(*mockPlugin, OnLoadNetworkFromFile()).Times(m_type == TestLoadType::EModelName ? 2 : 0);
    EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
    EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
    testLoad([&](Core& ie) {
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(0);
        });
        deviceToLoad = CommonTestUtils::DEVICE_AUTO;
        deviceToLoad += ":mock.0";
        ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
        m_testFunction(ie);
        m_post_mock_net_callbacks.pop_back();
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(0);
        });
        m_testFunction(ie);
    });
}
// MULTI-DEVICE test
// Test that it is safe to load multiple devices sharing same cache
// In case of sporadic failures - increase 'TEST_DURATION_MS' 100x times for better reproducibility
TEST_P(CachingTest, LoadMulti_race) {
    const auto TEST_DURATION_MS = 2000;
    const auto TEST_DEVICE_MAX_COUNT = 10;
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Multi plugin
    }
    int index = 0;
    auto start = high_resolution_clock::now();
    m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
        EXPECT_CALL(net, Export(_)).Times(1);
    });
    do {
        std::string cacheDir = m_cacheDir + std::to_string(index);
        MkDirGuard guard(cacheDir);
        int devCount = 1 + index % (TEST_DEVICE_MAX_COUNT - 1);  // try dynamic number of devices from 1 to max
        deviceToLoad = CommonTestUtils::DEVICE_MULTI;
        deviceToLoad += ":mock.0";
        for (int i = 1; i < devCount; i++) {
            deviceToLoad += ",mock." + std::to_string(i);
        }

        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(devCount - 1);
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), cacheDir}});
            ASSERT_NO_THROW(m_testFunction(ie));
        });
        index++;
    } while (duration_cast<milliseconds>(high_resolution_clock::now() - start).count() < TEST_DURATION_MS);
    std::cout << "Caching LoadMulti Test completed. Tried " << index << " times" << std::endl;
}

// MULTI-DEVICE test
// Test that it is safe to load multiple devices through loadNetwork
// In case of sporadic failures - increase 'TEST_DURATION_MS' 100x times for better reproducibility
TEST_P(CachingTest, LoadMultiWithConfig_race) {
    const auto TEST_DURATION_MS = 2000;
    const auto TEST_DEVICE_MAX_COUNT = 10;
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Multi plugin
    }
    int index = 0;
    auto start = high_resolution_clock::now();
    m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
        EXPECT_CALL(net, Export(_)).Times(1);
    });
    do {
        std::string cacheDir = m_cacheDir + std::to_string(index);
        MkDirGuard guard(cacheDir);
        int devCount = 1 + index % (TEST_DEVICE_MAX_COUNT - 1);  // try dynamic number of devices from 1 to max
        deviceToLoad = CommonTestUtils::DEVICE_MULTI;
        deviceToLoad += ":mock.0";
        for (int i = 1; i < devCount; i++) {
            deviceToLoad += ",mock." + std::to_string(i);
        }

        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(devCount - 1);
        testLoad([&](Core& ie) {
            ASSERT_NO_THROW(m_testFunctionWithCfg(ie, {{CONFIG_KEY(CACHE_DIR), cacheDir}}));
        });
        index++;
    } while (duration_cast<milliseconds>(high_resolution_clock::now() - start).count() < TEST_DURATION_MS);
    std::cout << "Caching LoadMulti Test completed. Tried " << index << " times" << std::endl;
}

// MULTI-DEVICE test
// Test loading of devices with different architectures
// In case of sporadic failures - increase 'TEST_DEVICE_MAX_COUNT' 100x times for better reproducibility
TEST_P(CachingTest, LoadMulti_Archs) {
    const auto TEST_DEVICE_MAX_COUNT = 30;  // Shall be >= 2
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([&](const std::string&, const std::map<std::string, Parameter>& options) {
            auto id = options.at("DEVICE_ID").as<std::string>();
            auto i = std::stoi(id) / 2;
            return "mock_architecture" + std::to_string(i);
        }));
    if (m_remoteContext) {
        return;  // skip the remote Context test for Multi plugin
    }

    deviceToLoad = CommonTestUtils::DEVICE_MULTI;
    deviceToLoad += ":mock.0";
    for (int i = 1; i < TEST_DEVICE_MAX_COUNT; i++) {
        deviceToLoad += ",mock." + std::to_string(i);
    }

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(TEST_DEVICE_MAX_COUNT / 2);
        // Load network from file shall not be called for plugins with caching supported
        EXPECT_CALL(*mockPlugin, OnLoadNetworkFromFile()).Times(0);

        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _))
            .Times(TEST_DEVICE_MAX_COUNT / 2)
            .WillRepeatedly(Invoke([&](std::istream& s, const std::map<std::string, std::string>&) {
                std::string name;
                s >> name;
                std::lock_guard<std::mutex> lock(mock_creation_mutex);
                return createMockIExecutableNet({}, m_inputs_map[name], m_outputs_map[name]);
            }));
        m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
            EXPECT_CALL(net, Export(_)).Times(1);  // each net will be exported once
        });
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            m_testFunction(ie);
        });
    }
}

// MULTI-DEVICE test
// Test loading of devices which don't support caching
// In case of sporadic failures - increase 'TEST_DEVICE_MAX_COUNT' 100x times for better reproducibility
TEST_P(CachingTest, LoadMulti_NoCachingOnDevice) {
    const auto TEST_DEVICE_MAX_COUNT = 100;  // Looks enough to catch potential race conditions
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(Parameter{false}));
    EXPECT_CALL(*mockPlugin, GetMetric(ov::device::capabilities.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(decltype(ov::device::capabilities)::value_type{}));
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());

    DataPtr inData = std::make_shared<Data>("Param_1", Precision::FP32);
    InputInfo inpInfo;
    inpInfo.setInputData(inData);
    InputInfo::CPtr cptr = std::make_shared<InputInfo>(inpInfo);
    ConstInputsDataMap inputMap{{"Param_1", cptr}};
    CDataPtr dataptr = std::make_shared<Data>("Reshape_2", Precision::FP32);
    ConstOutputsDataMap outputMap{{"Reshape_2", dataptr}};
    m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
        EXPECT_CALL(net, GetInputsInfo()).Times(AnyNumber()).WillRepeatedly(Return(inputMap));
        EXPECT_CALL(net, GetOutputsInfo()).Times(AnyNumber()).WillRepeatedly(Return(outputMap));
    });
    if (m_remoteContext) {
        return;  // skip the remote Context test for Multi plugin
    }

    deviceToLoad = CommonTestUtils::DEVICE_MULTI;
    deviceToLoad += ":mock.0";
    for (int i = 1; i < TEST_DEVICE_MAX_COUNT; i++) {
        deviceToLoad += ",mock." + std::to_string(i);
    }

    {
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(TEST_DEVICE_MAX_COUNT);
        // Load network from file shall not be called by Multi plugin for devices with caching supported
        EXPECT_CALL(*mockPlugin, OnLoadNetworkFromFile())
            .Times(m_type == TestLoadType::ECNN ? 0 : TEST_DEVICE_MAX_COUNT);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(0);
        for (auto& net : networks) {
            EXPECT_CALL(*net, Export(_)).Times(0);
        }
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), m_cacheDir}});
            ExecutableNetwork exeNet;
            exeNet = m_testFunction(ie);
            // Verify that inputs and outputs are set for Multi Executable Network
            ASSERT_EQ(exeNet.GetInputsInfo().size(), inputMap.size());
            ASSERT_EQ(exeNet.GetOutputsInfo().size(), outputMap.size());
            networks.clear();
        });
    }
}
#endif  // defined(ENABLE_AUTO)

#if defined(ENABLE_AUTO_BATCH)
// BATCH-DEVICE test
// load network with config
TEST_P(CachingTest, LoadBATCHWithConfig) {
    const auto TEST_COUNT = 2;
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Auto plugin
    }
    int index = 0;
    m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
        EXPECT_CALL(net, Export(_)).Times(1);
    });
    std::string cacheDir = m_cacheDir;
    MkDirGuard guard(cacheDir);
    for (; index < TEST_COUNT; index++) {
        deviceToLoad = CommonTestUtils::DEVICE_BATCH;
        deviceToLoad += ":mock.0";
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(TEST_COUNT - index - 1);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(index);
        ASSERT_NO_THROW(testLoad([&](Core& ie) {
            m_testFunctionWithCfg(ie, {{CONFIG_KEY(CACHE_DIR), cacheDir}});
        }));
    }
    std::cout << "Caching LoadAuto Test completed. Tried " << index << " times" << std::endl;
}
#endif  // defined(ENABLE_AUTO_BATCH)

// In case of sporadic failures - increase 'TEST_DURATION_MS' 100x times for better reproducibility
TEST_P(CachingTest, Load_threads) {
    const auto TEST_DURATION_MS = 2000;
    const auto THREADS_COUNT = 4;
    EXPECT_CALL(*mockPlugin, GetMetric(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, QueryNetwork(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, GetMetric(METRIC_KEY(DEVICE_ARCHITECTURE), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Multi plugin
    }
    auto start = high_resolution_clock::now();
    int index = 0;
    m_post_mock_net_callbacks.emplace_back([&](MockExecutableNetwork& net) {
        EXPECT_CALL(net, Export(_)).Times(1);
    });
    do {
        std::string cacheDir = m_cacheDir + std::to_string(index);
        MkDirGuard guard(cacheDir);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, LoadExeNetworkImpl(_, _)).Times(1);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, ImportNetwork(_, _)).Times(THREADS_COUNT - 1);
        testLoad([&](Core& ie) {
            ie.SetConfig({{CONFIG_KEY(CACHE_DIR), cacheDir}});
            std::vector<std::thread> threads;
            for (int i = 0; i < THREADS_COUNT; i++) {
                threads.emplace_back(([&]() {
                    m_testFunction(ie);
                }));
            }
            for (int i = 0; i < THREADS_COUNT; i++) {
                threads[i].join();
            }
        });
        index++;
    } while (duration_cast<milliseconds>(high_resolution_clock::now() - start).count() < TEST_DURATION_MS);
    std::cout << "Caching Load multiple threads test completed. Tried " << index << " times" << std::endl;
}

#if defined(ENABLE_OV_IR_FRONTEND)

static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<TestParam, std::string>>& obj) {
    return std::get<1>(std::get<0>(obj.param)) + "_" + std::get<1>(obj.param);
}

INSTANTIATE_TEST_SUITE_P(CachingTest,
                         CachingTest,
                         ::testing::Combine(::testing::ValuesIn(loadVariants), ::testing::ValuesIn(cacheFolders)),
                         getTestCaseName);
#endif  // defined(ENABLE_OV_IR_FRONTEND)
