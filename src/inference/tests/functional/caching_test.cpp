// Copyright (C) 2018-2025 Intel Corporation
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
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#ifdef PROXY_PLUGIN_ENABLED
#    include "openvino/proxy/properties.hpp"
#endif
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_iasync_infer_request.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icompiled_model.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_iplugin.hpp"

using namespace ::testing;
using namespace std::placeholders;
using namespace std::chrono;

enum class TestLoadType { EModel, EContext, EModelName };

using TestParam = std::tuple<TestLoadType, std::string, bool>;

//  GCC4.8 limitation: have to specify type of each element in list
static const std::vector<TestParam> loadVariants = {
    TestParam{TestLoadType::EModel, std::string("ByModel"), false},
    TestParam{TestLoadType::EContext, std::string("ByRemoteContext"), true},
    TestParam{TestLoadType::EModelName, std::string("ByModelName"), false},
};

static const std::vector<std::string> cacheFolders{
    std::string("testCache"),
};

class MockRemoteContext : public ov::IRemoteContext {
    std::string m_name;

public:
    MockRemoteContext(std::string name) : m_name(std::move(name)) {}
    const std::string& get_device_name() const override {
        return m_name;
    }
    MOCK_METHOD(ov::SoPtr<ov::IRemoteTensor>,
                create_tensor,
                (const ov::element::Type&, const ov::Shape&, const ov::AnyMap&));
    MOCK_METHOD(const ov::AnyMap&, get_property, (), (const));
};

class MockCachingIPluginBase : public ov::MockIPlugin {
public:
    MockCachingIPluginBase() = default;
    ~MockCachingIPluginBase() = default;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::string& model_path,
                                                      const ov::AnyMap& config) const override {
        // In GTEST, it is not possible to call base implementation inside of mocked class
        // Thus, we define a proxy callback and will use
        // EXPECT_CALL(OnCompileModelFromFile) instead of EXPECT_CALL(compile_model)
        OnCompileModelFromFile();
        return ov::IPlugin::compile_model(model_path, config);
    }

    virtual void OnCompileModelFromFile() const {}
};

class MockCachingIPlugin : public MockCachingIPluginBase {
public:
    MockCachingIPlugin() = default;
    ~MockCachingIPlugin() = default;

    MOCK_METHOD(void, OnCompileModelFromFile, (), (const));
};

class MockICompiledModelImpl : public ov::MockICompiledModel {
    std::mutex m_pluginMutex;
    std::shared_ptr<const ov::Model> m_model = nullptr;

public:
    MockICompiledModelImpl(const std::shared_ptr<const ov::Model>& model,
                           const std::shared_ptr<const ov::IPlugin>& plugin)
        : ov::MockICompiledModel(model, plugin) {
        m_model = model;
    }

    const std::shared_ptr<const ov::Model>& get_model() const {
        return m_model;
    }
};

//------------------------------------------------------
class MkDirGuard {
    std::string m_dir;

public:
    explicit MkDirGuard(std::string dir = std::string()) : m_dir(std::move(dir)) {
        if (!m_dir.empty()) {
            ov::test::utils::createDirectory(m_dir);
        }
    }

    MkDirGuard(const MkDirGuard&) = delete;
    MkDirGuard& operator=(const MkDirGuard&) = delete;

    ~MkDirGuard() {
        if (!m_dir.empty()) {
            ov::test::utils::removeFilesWithExt(m_dir, "blob");
            ov::test::utils::removeDir(m_dir);
        }
    }
};

class CachingTest : public ::testing::TestWithParam<std::tuple<TestParam, std::string>> {
public:
    std::shared_ptr<void> sharedObjectLoader;
    std::function<void(ov::IPlugin*)> injectPlugin;
    std::string modelName = "Caching_test.xml";
    std::string weightsName = "Caching_test.bin";
    std::string deviceName = "mock";
    std::string deviceToLoad = "mock";
    std::shared_ptr<MockCachingIPlugin> mockPlugin;
    std::vector<std::shared_ptr<MockICompiledModelImpl>> comp_models;
    std::mutex mock_creation_mutex;  // Internal gmock object registration is not thread-safe
    using ExeNetCallback = std::function<void(MockICompiledModelImpl&)>;
    std::vector<ExeNetCallback> m_post_mock_net_callbacks = {};
    std::unique_ptr<MkDirGuard> m_dirCreator;
    TestLoadType m_type = TestLoadType::EModel;
    std::string m_cacheDir;
    using LoadFunction = std::function<ov::CompiledModel(ov::Core&)>;
    using LoadFunctionWithCfg = std::function<void(ov::Core&, const ov::AnyMap&)>;
    LoadFunction m_testFunction;
    LoadFunctionWithCfg m_testFunctionWithCfg;
    using ModelCallbackFunc = std::function<void(std::shared_ptr<ov::Model>&)>;
    ModelCallbackFunc m_modelCallback;
    bool m_remoteContext = false;
    using CheckConfigCb = std::function<void(const ov::AnyMap&)>;
    CheckConfigCb m_checkConfigCb = nullptr;
    std::shared_ptr<ov::Model> m_model;
    std::map<std::string, std::shared_ptr<ov::Model>> m_models;

    static std::string get_mock_engine_path() {
        std::string mockEngineName("mock_engine");
        return ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                  mockEngineName + OV_BUILD_POSTFIX);
    }

    void initParamTest() {
        m_type = std::get<0>(std::get<0>(GetParam()));
        m_cacheDir = std::get<1>(GetParam());
        m_testFunction = getLoadFunction(m_type);
        m_testFunctionWithCfg = getLoadFunctionWithCfg(m_type);
        m_remoteContext = std::get<2>(std::get<0>(GetParam()));
        auto testName = ov::test::utils::generateTestFilePrefix();
        modelName = testName + ".xml";
        weightsName = testName + ".bin";
        m_cacheDir = testName + m_cacheDir;
        m_dirCreator = std::unique_ptr<MkDirGuard>(new MkDirGuard(m_cacheDir));
    }

    static std::shared_ptr<MockICompiledModelImpl> create_mock_compiled_model(
        const std::shared_ptr<const ov::Model>& model,
        const std::shared_ptr<ov::IPlugin>& plugin) {
        auto mock = std::make_shared<MockICompiledModelImpl>(model, plugin);
        EXPECT_CALL(*mock, inputs()).Times(AnyNumber()).WillRepeatedly(ReturnRefOfCopy(model->inputs()));
        EXPECT_CALL(*mock, outputs()).Times(AnyNumber()).WillRepeatedly(ReturnRefOfCopy(model->outputs()));
        EXPECT_CALL(*mock, get_runtime_model()).Times(AnyNumber()).WillRepeatedly(Return(model));
        auto ptr = std::make_shared<ov::MockIAsyncInferRequest>();
        EXPECT_CALL(*ptr, set_callback(_)).Times(AnyNumber());
        EXPECT_CALL(*mock, create_infer_request()).Times(AnyNumber()).WillRepeatedly(Return(ptr));

        EXPECT_CALL(*mock, get_property(ov::enable_profiling.name()))
            .Times(AnyNumber())
            .WillRepeatedly(Return(ov::Any{false}));
        EXPECT_CALL(*mock, get_property(ov::optimal_number_of_infer_requests.name()))
            .Times(AnyNumber())
            .WillRepeatedly(Return(ov::Any{1u}));
        EXPECT_CALL(*mock, get_property(ov::model_name.name())).Times(AnyNumber()).WillRepeatedly(Return("mock_net"));
        EXPECT_CALL(*mock, get_property(ov::supported_properties.name()))
            .Times(AnyNumber())
            .WillRepeatedly(Return(std::vector<ov::PropertyName>{ov::supported_properties.name(),
                                                                 ov::optimal_number_of_infer_requests.name(),
                                                                 ov::model_name.name()}));
        ON_CALL(*mock, export_model(_)).WillByDefault(Invoke([model](std::ostream& s) {
            s << model->get_friendly_name();
            s << ' ';
        }));
        return mock;
    }

    void SetUp() override {
        initParamTest();
        mockPlugin = std::make_shared<MockCachingIPlugin>();
        setupMock(*mockPlugin);
        std::string libraryPath = get_mock_engine_path();
        sharedObjectLoader = ov::util::load_shared_object(libraryPath.c_str());
        injectPlugin = make_std_function<void(ov::IPlugin*)>("InjectPlugin");

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::Serialize>(modelName, weightsName);
        manager.run_passes(ov::test::utils::make_conv_pool_relu({1, 3, 227, 227}, ov::element::Type_t::f32));
    }

    void TearDown() override {
        for (const auto& model : comp_models) {
            EXPECT_TRUE(Mock::VerifyAndClearExpectations(model.get()));
        }
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockPlugin.get()));
        ov::test::utils::removeIRFiles(modelName, weightsName);
    }

    void testLoad(const std::function<void(ov::Core& core)>& func) {
        ov::Core core;
        injectPlugin(mockPlugin.get());
        core.register_plugin(ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                                std::string("mock_engine") + OV_BUILD_POSTFIX),
                             deviceName);
        func(core);
        core.unload_plugin(deviceName);
    }

    LoadFunction getLoadFunction(TestLoadType type) const {
        switch (type) {
        case TestLoadType::EModel:
            return [&](ov::Core& core) {
                return performReadAndLoad(core);
            };
        case TestLoadType::EContext:
            return [&](ov::Core& core) {
                return performReadAndLoadWithContext(core);
            };
        case TestLoadType::EModelName:
            return [&](ov::Core& core) {
                return performLoadByName(core);
            };
        }
        return nullptr;
    }

    LoadFunctionWithCfg getLoadFunctionWithCfg(TestLoadType type) const {
        switch (type) {
        case TestLoadType::EModel:
            return std::bind(&CachingTest::performReadAndLoad, this, _1, _2);
        case TestLoadType::EContext:
            return std::bind(&CachingTest::performReadAndLoadWithContext, this, _1, _2);
        case TestLoadType::EModelName:
            return std::bind(&CachingTest::performLoadByName, this, _1, _2);
        }
        return nullptr;
    }

    ov::CompiledModel performLoadByName(ov::Core& core, const ov::AnyMap& config = {}) const {
        return core.compile_model(modelName, deviceToLoad, config);
    }

    ov::CompiledModel performReadAndLoad(ov::Core& core, const ov::AnyMap& config = {}) const {
        auto model = core.read_model(modelName, {}, config);
        if (m_modelCallback)
            m_modelCallback(model);
        return core.compile_model(model, deviceToLoad, config);
    }

    ov::CompiledModel performReadAndLoadWithContext(ov::Core& core, const ov::AnyMap& config = {}) const {
        auto model = core.read_model(modelName, {}, config);
        EXPECT_CALL(*mockPlugin, get_default_context(_)).Times(AnyNumber());
        auto context = core.get_default_context(deviceToLoad);
        if (m_modelCallback)
            m_modelCallback(model);
        return core.compile_model(model, context, config);
    }

private:
    template <class T>
    std::function<T> make_std_function(const std::string& functionName) {
        std::function<T> ptr(reinterpret_cast<T*>(ov::util::get_symbol(sharedObjectLoader, functionName.c_str())));
        return ptr;
    }

    void setupMock(MockCachingIPlugin& plugin) {
        ON_CALL(plugin, get_property(_, _))
            .WillByDefault(Invoke([&](const std::string& name, const ov::AnyMap&) -> ov::Any {
                OPENVINO_THROW("Unexpected ", name);
            }));
        ON_CALL(plugin, get_property(ov::supported_properties.name(), _))
            .WillByDefault(Invoke([&](const std::string&, const ov::AnyMap&) {
                return std::vector<ov::PropertyName>{ov::supported_properties.name(),
                                                     ov::device::capabilities.name(),
                                                     ov::device::architecture.name()};
            }));
        ON_CALL(plugin, get_property(ov::internal::supported_properties.name(), _))
            .WillByDefault(Invoke([&](const std::string&, const ov::AnyMap&) {
                return std::vector<ov::PropertyName>{ov::internal::caching_properties.name()};
            }));
        ON_CALL(plugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).WillByDefault(Return(true));

        ON_CALL(plugin, get_property(ov::device::capabilities.name(), _))
            .WillByDefault(Invoke([&](const std::string&, const ov::AnyMap&) {
                return decltype(ov::device::capabilities)::value_type{ov::device::capability::EXPORT_IMPORT};
            }));

        ON_CALL(plugin, get_property(ov::device::architecture.name(), _))
            .WillByDefault(Invoke([&](const std::string&, const ov::AnyMap&) {
                return "mock";
            }));

        ON_CALL(plugin, get_property(ov::internal::caching_properties.name(), _))
            .WillByDefault(Invoke([&](const std::string&, const ov::AnyMap&) {
                std::vector<ov::PropertyName> cachingProperties = {ov::device::architecture.name()};
                return decltype(ov::internal::caching_properties)::value_type(cachingProperties);
            }));

        ON_CALL(plugin, import_model(_, _, _))
            .WillByDefault(
                Invoke([&](std::istream& istr, const ov::SoPtr<ov::IRemoteContext>&, const ov::AnyMap& config) {
                    if (m_checkConfigCb) {
                        m_checkConfigCb(config);
                    }
                    std::string name;
                    istr >> name;
                    char space;
                    istr.read(&space, 1);
                    std::lock_guard<std::mutex> lock(mock_creation_mutex);
                    return create_mock_compiled_model(m_models[name], mockPlugin);
                }));

        ON_CALL(plugin, import_model(_, _)).WillByDefault(Invoke([&](std::istream& istr, const ov::AnyMap& config) {
            if (m_checkConfigCb) {
                m_checkConfigCb(config);
            }
            std::string name;
            istr >> name;
            char space;
            istr.read(&space, 1);
            std::lock_guard<std::mutex> lock(mock_creation_mutex);
            return create_mock_compiled_model(m_models[name], mockPlugin);
        }));

        ON_CALL(plugin, compile_model(_, _, _))
            .WillByDefault(Invoke([&](const std::shared_ptr<const ov::Model>& model,
                                      const ov::AnyMap& config,
                                      const ov::SoPtr<ov::IRemoteContext>&) {
                if (m_checkConfigCb) {
                    m_checkConfigCb(config);
                }
                std::lock_guard<std::mutex> lock(mock_creation_mutex);
                m_models[model->get_friendly_name()] = model->clone();
                auto comp_model = create_mock_compiled_model(m_models[model->get_friendly_name()], mockPlugin);
                for (const auto& cb : m_post_mock_net_callbacks) {
                    cb(*comp_model);
                }
                comp_models.push_back(comp_model);
                return comp_model;
            }));

        ON_CALL(plugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .WillByDefault(Invoke([&](const std::shared_ptr<const ov::Model>& model, const ov::AnyMap& config) {
                if (m_checkConfigCb) {
                    m_checkConfigCb(config);
                }
                std::lock_guard<std::mutex> lock(mock_creation_mutex);
                m_models[model->get_friendly_name()] = model->clone();
                auto comp_model = create_mock_compiled_model(m_models[model->get_friendly_name()], mockPlugin);
                for (const auto& cb : m_post_mock_net_callbacks) {
                    cb(*comp_model);
                }
                comp_models.push_back(comp_model);
                return comp_model;
            }));

        ON_CALL(plugin, get_default_context(_)).WillByDefault(Invoke([&](const ov::AnyMap&) {
            return std::make_shared<MockRemoteContext>(deviceToLoad);
        }));

        ON_CALL(plugin, query_model(_, _))
            .WillByDefault(Invoke([&](const std::shared_ptr<const ov::Model>& model, const ov::AnyMap&) {
                ov::SupportedOpsMap res;
                EXPECT_TRUE(model);

                for (auto&& node : model->get_ops()) {
                    res.emplace(node->get_friendly_name(), deviceName);
                }
                return res;
            }));

        EXPECT_CALL(plugin, set_property(_)).Times(AnyNumber()).WillRepeatedly(Invoke([](const ov::AnyMap&) {
            OPENVINO_NOT_IMPLEMENTED;
        }));
    }
};

TEST_P(CachingTest, TestLoad) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
        EXPECT_EQ(comp_models.size(), 1);
    }

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& model : comp_models) {
            EXPECT_CALL(*model, export_model(_)).Times(0);  // No more 'export_model' for existing model
        }
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
        EXPECT_EQ(comp_models.size(), 1);
    }
}

/// \brief Verifies that core.set_property({{"CACHE_DIR", <dir>}}, "deviceName"}}); enables caching for one device
TEST_P(CachingTest, TestLoad_by_device_name) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            core.set_property("mock", ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
        EXPECT_EQ(comp_models.size(), 1);
    }

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& model : comp_models) {
            EXPECT_CALL(*model, export_model(_)).Times(0);  // No more 'export_model' for existing models
        }
        testLoad([&](ov::Core& core) {
            core.set_property("mock", ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
        EXPECT_EQ(comp_models.size(), 1);
    }
}

TEST_P(CachingTest, TestLoadCustomImportExport) {
    const char customData[] = {1, 2, 3, 4, 5};
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());

    ON_CALL(*mockPlugin, import_model(_, _, _))
        .WillByDefault(Invoke([&](std::istream& s, const ov::SoPtr<ov::IRemoteContext>&, const ov::AnyMap&) {
            char a[sizeof(customData)];
            s.read(a, sizeof(customData));
            EXPECT_EQ(memcmp(a, customData, sizeof(customData)), 0);
            std::string name;
            s >> name;
            std::lock_guard<std::mutex> lock(mock_creation_mutex);
            return create_mock_compiled_model(m_models[name], mockPlugin);
        }));

    ON_CALL(*mockPlugin, import_model(_, _)).WillByDefault(Invoke([&](std::istream& s, const ov::AnyMap&) {
        char a[sizeof(customData)];
        s.read(a, sizeof(customData));
        EXPECT_EQ(memcmp(a, customData, sizeof(customData)), 0);
        std::string name;
        s >> name;
        std::lock_guard<std::mutex> lock(mock_creation_mutex);
        return create_mock_compiled_model(m_models[name], mockPlugin);
    }));

    m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
        ON_CALL(net, export_model(_)).WillByDefault(Invoke([&](std::ostream& s) {
            s.write(customData, sizeof(customData));
            s << net.get_model()->get_friendly_name();
        }));
    });

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& model : comp_models) {
            EXPECT_CALL(*model, export_model(_)).Times(0);  // No 'export_model' for existing models
        }
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
}

// Brief: when compile_model is called from different config - old cache shall not be used
TEST_P(CachingTest, TestChangeLoadConfig) {
    const std::string CUSTOM_KEY = "CUSTOM_KEY";
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());

    ON_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _))
        .WillByDefault(Invoke([&](const std::string&, const ov::AnyMap&) {
            return std::vector<ov::PropertyName>{ov::supported_properties.name(),
                                                 ov::device::capabilities.name(),
                                                 ov::device::architecture.name()};
        }));
    ON_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _))
        .WillByDefault(Invoke([&](const std::string&, const ov::AnyMap&) {
            return std::vector<ov::PropertyName>{ov::internal::caching_properties.name()};
        }));
    ON_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _))
        .WillByDefault(Invoke([&](const std::string&, const ov::AnyMap&) {
            std::vector<ov::PropertyName> res;
            res.push_back(ov::PropertyName(CUSTOM_KEY, ov::PropertyMutability::RO));
            return decltype(ov::internal::caching_properties)::value_type(res);
        }));
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunctionWithCfg(core, {{CUSTOM_KEY, "0"}});
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunctionWithCfg(core, {{CUSTOM_KEY, "1"}});
        });
    }
}

/// \brief Verifies that core.compile_model(model, "deviceName", {{"CACHE_DIR", <dir>>}}) works
TEST_P(CachingTest, TestChangeLoadConfig_With_Cache_Dir_inline) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            m_testFunctionWithCfg(core, ov::AnyMap{{ov::cache_dir.name(), m_cacheDir}});
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& model : comp_models) {
            EXPECT_CALL(*model, export_model(_)).Times(0);  // No more 'export_model' for existing models
        }
        testLoad([&](ov::Core& core) {
            m_testFunctionWithCfg(core, ov::AnyMap{{ov::cache_dir.name(), m_cacheDir}});
        });
        EXPECT_EQ(comp_models.size(), 1);
    }
}

TEST_P(CachingTest, TestNoCacheEnabled) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(0);
        });
        testLoad([&](ov::Core& core) {
            m_testFunction(core);
        });
    }
}

TEST_P(CachingTest, TestNoCacheSupported) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(false));
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(decltype(ov::device::capabilities)::value_type{}));

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, OnCompileModelFromFile()).Times(m_type == TestLoadType::EModelName ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(0);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
}

TEST_P(CachingTest, TestNoCacheMetricSupported) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{}));
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(0);
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(0);
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{}));
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(0);
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(0);

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, OnCompileModelFromFile()).Times(m_type == TestLoadType::EModelName ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(0);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
}

/// \brief If device doesn't support 'cache_dir' or 'import_export' - setting cache_dir is ignored
TEST_P(CachingTest, TestNoCacheMetricSupported_by_device_name) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{}));
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(0);
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(0);
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{}));
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(0);
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(0);

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, OnCompileModelFromFile()).Times(m_type == TestLoadType::EModelName ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(0);
        });
        testLoad([&](ov::Core& core) {
            core.set_property("mock", ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
}

TEST_P(CachingTest, TestNoCacheMetric_hasCacheDirConfig) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{ov::supported_properties.name(), ov::cache_dir.name()}));
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{}));
    EXPECT_CALL(*mockPlugin, set_property(_)).Times(AtLeast(1)).WillRepeatedly(Invoke([](const ov::AnyMap& config) {
        ASSERT_GT(config.count(ov::cache_dir.name()), 0);
    }));

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, OnCompileModelFromFile()).Times(m_type == TestLoadType::EModelName ? 1 : 0);
        OV_ASSERT_NO_THROW(testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        }));
    }
}

/// \brief If device supports 'cache_dir' or 'import_export' - setting cache_dir is passed to plugin on
/// core.compile_model
TEST_P(CachingTest, TestNoCacheMetric_hasCacheDirConfig_inline) {
    m_checkConfigCb = [](const ov::AnyMap& config) {
        EXPECT_NE(config.count(ov::cache_dir.name()), 0);
    };
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{ov::supported_properties.name(), ov::cache_dir.name()}));
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{}));
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, OnCompileModelFromFile()).Times(m_type == TestLoadType::EModelName ? 1 : 0);
        OV_ASSERT_NO_THROW(testLoad([&](ov::Core& core) {
            m_testFunctionWithCfg(core, {{ov::cache_dir.name(), m_cacheDir}});
        }));
    }
}

/// \brief core.set_property(<cachedir>, "deviceName") is propagated to plugin's set_property if device supports
/// CACHE_DIR
TEST_P(CachingTest, TestNoCacheMetric_hasCacheDirConfig_by_device_name) {
    m_checkConfigCb = [](const ov::AnyMap& config) {
        // Shall be '0' as appropriate 'cache_dir' is expected in set_property, not in Load/Import model
        EXPECT_EQ(config.count(ov::cache_dir.name()), 0);
    };
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{ov::supported_properties.name(), ov::cache_dir.name()}));
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{}));
    EXPECT_CALL(*mockPlugin, set_property(_)).Times(AtLeast(1)).WillRepeatedly(Invoke([](const ov::AnyMap& config) {
        ASSERT_GT(config.count(ov::cache_dir.name()), 0);
    }));

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, OnCompileModelFromFile()).Times(m_type == TestLoadType::EModelName ? 1 : 0);
        OV_ASSERT_NO_THROW(testLoad([&](ov::Core& core) {
            core.set_property("mock", ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        }));
    }
}

TEST_P(CachingTest, TestCacheEnabled_noConfig) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{ov::supported_properties.name()}));
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{}));
    EXPECT_CALL(*mockPlugin, set_property(_)).Times(AnyNumber()).WillRepeatedly(Invoke([](const ov::AnyMap& config) {
        ASSERT_EQ(config.count(ov::cache_dir.name()), 0);
    }));

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, OnCompileModelFromFile()).Times(m_type == TestLoadType::EModelName ? 1 : 0);
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
}

TEST_P(CachingTest, TestNoCacheMetric_configThrow) {
    m_checkConfigCb = [](const ov::AnyMap& config) {
        EXPECT_NE(config.count(ov::cache_dir.name()), 0);
    };
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{ov::supported_properties.name(), ov::cache_dir.name()}));
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{}));
    EXPECT_CALL(*mockPlugin, set_property(_)).Times(AtLeast(1)).WillRepeatedly(Invoke([](const ov::AnyMap& config) {
        ASSERT_GT(config.count(ov::cache_dir.name()), 0);
        OPENVINO_THROW("Error occurred");
    }));

    ASSERT_ANY_THROW(testLoad([&](ov::Core& core) {
        core.set_property(ov::cache_dir(m_cacheDir));
        m_testFunction(core);
    }));
}

TEST_P(CachingTest, TestNoCacheEnabled_cacheDirConfig) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{ov::supported_properties.name(), ov::cache_dir.name()}));
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{}));
    EXPECT_CALL(*mockPlugin, set_property(_)).Times(AnyNumber()).WillRepeatedly(Invoke([](const ov::AnyMap& config) {
        ASSERT_EQ(config.count(ov::cache_dir.name()), 0);
    }));

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        testLoad([&](ov::Core& core) {
            m_testFunction(core);
        });
    }
}

TEST_P(CachingTest, TestLoadChangeCacheDir) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {
        std::string newCacheDir = m_cacheDir + "2";
        MkDirGuard dir(newCacheDir);
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(newCacheDir));
            m_testFunction(core);
        });
    }
}

/// \brief Change CACHE_DIR during working with same 'Core' object. Verifies that new dir is used for caching
TEST_P(CachingTest, TestLoadChangeCacheDirOneCore) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, set_property(_)).Times(AnyNumber()).WillRepeatedly(Invoke([](const ov::AnyMap& config) {
        ASSERT_EQ(config.count(ov::cache_dir.name()), 0);
    }));
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        testLoad([&](ov::Core& core) {
            m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
                EXPECT_CALL(net, export_model(_)).Times(1);
            });
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
            std::string newCacheDir = m_cacheDir + "2";
            m_post_mock_net_callbacks.pop_back();
            m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
                EXPECT_CALL(net, export_model(_)).Times(1);
            });
            MkDirGuard dir(newCacheDir);
            core.set_property(ov::cache_dir(newCacheDir));
            m_testFunction(core);
        });
    }
}

/// \brief Change CACHE_DIR during working with same 'Core' object
/// Initially set for 'device', then is overwritten with global 'cache_dir' for all devices
TEST_P(CachingTest, TestLoadChangeCacheDirOneCore_overwrite_device_dir) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, set_property(_)).Times(AnyNumber()).WillRepeatedly(Invoke([](const ov::AnyMap& config) {
        ASSERT_EQ(config.count(ov::cache_dir.name()), 0);
    }));
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        testLoad([&](ov::Core& core) {
            m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
                EXPECT_CALL(net, export_model(_)).Times(1);
            });
            core.set_property("mock", ov::cache_dir(m_cacheDir));
            m_testFunction(core);
            std::string newCacheDir = m_cacheDir + "2";
            m_post_mock_net_callbacks.pop_back();
            m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
                EXPECT_CALL(net, export_model(_)).Times(1);
            });
            MkDirGuard dir(newCacheDir);
            core.set_property({ov::cache_dir(newCacheDir)});
            m_testFunction(core);
        });
    }
}

/// \brief Change CACHE_DIR during working with same 'Core' object for device which supports 'CACHE_DIR' config, not
/// import_export Expectation is that set_property for plugin will be called 2 times - with appropriate cache_dir values
TEST_P(CachingTest, TestLoadChangeCacheDirOneCore_SupportsCacheDir_NoImportExport) {
    m_checkConfigCb = [](const ov::AnyMap& config) {
        EXPECT_EQ(config.count(ov::cache_dir.name()), 0);
    };
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{ov::supported_properties.name(), ov::cache_dir.name()}));
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(decltype(ov::device::capabilities)::value_type{}));
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    std::string set_cache_dir = {};
    EXPECT_CALL(*mockPlugin, set_property(_)).Times(AtLeast(2)).WillRepeatedly(Invoke([&](const ov::AnyMap& config) {
        ASSERT_NE(config.count(ov::cache_dir.name()), 0);
        set_cache_dir = config.at(ov::cache_dir.name()).as<std::string>();
    }));
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, OnCompileModelFromFile()).Times(m_type == TestLoadType::EModelName ? 2 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(0);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
            EXPECT_EQ(set_cache_dir, m_cacheDir);

            std::string new_cache_dir = m_cacheDir + "2";
            MkDirGuard dir(new_cache_dir);
            core.set_property(ov::cache_dir(new_cache_dir));
            m_testFunction(core);
            EXPECT_EQ(set_cache_dir, new_cache_dir);
        });
    }
}

/// \brief Change CACHE_DIR per device during working with same 'Core' object - expected that new cache dir is used
TEST_P(CachingTest, TestLoadChangeCacheDirOneCore_by_device_name) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, set_property(_)).Times(AnyNumber()).WillRepeatedly(Invoke([](const ov::AnyMap& config) {
        ASSERT_EQ(config.count(ov::cache_dir.name()), 0);
    }));
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        testLoad([&](ov::Core& core) {
            m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
                EXPECT_CALL(net, export_model(_)).Times(1);
            });
            core.set_property("mock", ov::cache_dir(m_cacheDir));
            m_testFunction(core);
            m_post_mock_net_callbacks.pop_back();
            m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
                EXPECT_CALL(net, export_model(_)).Times(1);
            });
            std::string newCacheDir = m_cacheDir + "2";
            MkDirGuard dir(newCacheDir);
            core.set_property("mock", ov::cache_dir(newCacheDir));
            m_testFunction(core);
        });
    }
}

/// \brief Change CACHE_DIR per device during working with same 'Core' object - device supports CACHE_DIR
/// Verifies that no 'export' is called and cache_dir is propagated to set_config
TEST_P(CachingTest, TestLoadChangeCacheDirOneCore_by_device_name_supports_cache_dir) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{ov::cache_dir.name()}));
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(false));
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(decltype(ov::device::capabilities)::value_type{}));
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, set_property(_)).Times(AtLeast(2)).WillRepeatedly(Invoke([](const ov::AnyMap& config) {
        ASSERT_GT(config.count(ov::cache_dir.name()), 0);
    }));
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 2 : 0);
        EXPECT_CALL(*mockPlugin, OnCompileModelFromFile()).Times(m_type == TestLoadType::EModelName ? 2 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        testLoad([&](ov::Core& core) {
            m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
                EXPECT_CALL(net, export_model(_)).Times(0);
            });
            core.set_property("mock", ov::cache_dir(m_cacheDir));
            m_testFunction(core);
            m_post_mock_net_callbacks.pop_back();
            m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
                EXPECT_CALL(net, export_model(_)).Times(0);
            });
            std::string newCacheDir = m_cacheDir + "2";
            MkDirGuard dir(newCacheDir);
            core.set_property("mock", ov::cache_dir(newCacheDir));
            m_testFunction(core);
        });
    }
}

TEST_P(CachingTest, TestClearCacheDir) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        for (auto& model : comp_models) {
            EXPECT_CALL(*model, export_model(_)).Times(0);
        }
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            core.set_property(ov::cache_dir(""));
            m_testFunction(core);
        });
        EXPECT_EQ(comp_models.size(), 1);
    }
}

TEST_P(CachingTest, TestChangeOtherConfig) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            core.set_property({{"someKey", "someValue"}});
            m_testFunction(core);
        });
        EXPECT_EQ(comp_models.size(), 1);
    }
}

TEST_P(CachingTest, TestChangeCacheDirFailure) {
    std::string longName(1000000, ' ');
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
        EXPECT_EQ(comp_models.size(), 1);
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(!m_remoteContext ? 1 : 0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            EXPECT_ANY_THROW(core.set_property(ov::cache_dir(m_cacheDir + "/" + longName)));
            m_testFunction(core);
        });
    }
}

TEST_P(CachingTest, TestCacheDirCreateRecursive) {
    std::string newCacheDir1 = m_cacheDir + ov::test::utils::FileSeparator + "a";
    std::string newCacheDir2 = newCacheDir1 + ov::test::utils::FileSeparator + "b";
    std::string newCacheDir3 = newCacheDir2 + ov::test::utils::FileSeparator + ov::test::utils::FileSeparator;

    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            EXPECT_NO_THROW(core.set_property(ov::cache_dir(newCacheDir3)));
            EXPECT_NO_THROW(m_testFunction(core));
        });
    }
    ov::test::utils::removeFilesWithExt(newCacheDir2, "blob");
    ov::test::utils::removeDir(newCacheDir2);
    ov::test::utils::removeDir(newCacheDir1);
}

TEST_P(CachingTest, TestDeviceArchitecture) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([&](const std::string&, const ov::AnyMap& options) {
            auto id = options.at("DEVICE_ID").as<std::string>();
            if (std::stoi(id) < 10) {
                return "mock_first_architecture";
            } else {
                return "mock_another_architecture";
            }
        }));
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            deviceToLoad = "mock.0";
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : comp_models) {
            EXPECT_CALL(*net, export_model(_)).Times(0);
        }
        testLoad([&](ov::Core& core) {
            deviceToLoad = "mock.1";
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            deviceToLoad = "mock.50";
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : comp_models) {
            EXPECT_CALL(*net, export_model(_)).Times(0);
        }
        testLoad([&](ov::Core& core) {
            deviceToLoad = "mock.51";
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
}

TEST_P(CachingTest, TestNoDeviceArchitecture) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([&](const std::string&, const ov::AnyMap&) {
            return std::vector<ov::PropertyName>{ov::device::capabilities.name()};
        }));
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([&](const std::string&, const ov::AnyMap&) {
            return std::vector<ov::PropertyName>{ov::internal::caching_properties.name()};
        }));
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([&](const std::string&, const ov::AnyMap&) {
            return std::vector<ov::PropertyName>{ov::supported_properties};
        }));
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(decltype(ov::device::capabilities)::value_type{ov::device::capability::EXPORT_IMPORT}));
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(0);
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            deviceToLoad = "mock.0";
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : comp_models) {
            EXPECT_CALL(*net, export_model(_)).Times(0);
        }
        testLoad([&](ov::Core& core) {
            deviceToLoad = "mock.50";
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
}

TEST_P(CachingTest, TestNoCachingProperties) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([&](const std::string&, const ov::AnyMap&) {
            return std::vector<ov::PropertyName>{ov::device::capabilities.name()};
        }));
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([&](const std::string&, const ov::AnyMap&) {
            return std::vector<ov::PropertyName>{};
        }));
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(0);
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(decltype(ov::device::capabilities)::value_type{ov::device::capability::EXPORT_IMPORT}));
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(0);
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, OnCompileModelFromFile()).Times(m_type == TestLoadType::EModelName ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(0);
        });
        testLoad([&](ov::Core& core) {
            deviceToLoad = "mock.0";
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
}

TEST_P(CachingTest, TestThrowOnExport) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1).WillOnce(Throw(1));
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            EXPECT_ANY_THROW(m_testFunction(core));
        });
    }
}

// TODO: temporary behavior is to no re-throw exception on import error (see 54335)
// In future add separate 'no throw' test for 'blob_outdated' exception from plugin
TEST_P(CachingTest, TestThrowOnImport) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        if (m_remoteContext) {
            EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(1).WillOnce(Throw(1));
            EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        } else {
            EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
            EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(1).WillOnce(Throw(1));
        }
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            EXPECT_NO_THROW(m_testFunction(core));
        });
    }
    {  // Step 3: same load, cache is re-created on export on step 2 and shall be successfully imported now
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : comp_models) {
            EXPECT_CALL(*net, export_model(_)).Times(0);
        }
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            EXPECT_NO_THROW(m_testFunction(core));
        });
    }
}

TEST_P(CachingTest, TestModelModified) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
    if (m_type == TestLoadType::EModelName) {
        // Modify model file
        std::fstream stream(modelName, std::fstream::out | std::fstream::app);
        stream << " ";
    } else {
        // Modify loaded ov::Model
        m_modelCallback = [&](std::shared_ptr<ov::Model>& model) {
            model->get_parameters()[0]->set_layout(ov::Layout("NHWC"));
        };
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {  // Step 3: same load, should be ok now
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : comp_models) {
            EXPECT_CALL(*net, export_model(_)).Times(0);
        }
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
}

TEST_P(CachingTest, TestCacheFileCorrupted) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            EXPECT_NO_THROW(core.set_property(ov::cache_dir(m_cacheDir)));
            EXPECT_NO_THROW(m_testFunction(core));
        });
    }
    {
        auto blobs = ov::test::utils::listFilesWithExt(m_cacheDir, "blob");
        for (const auto& fileName : blobs) {
            std::ofstream stream(fileName, std::ios_base::binary);
            stream << "SomeCorruptedText";
        }
    }
    m_post_mock_net_callbacks.pop_back();
    {  // Step 2. Cache is corrupted, will be silently removed
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            EXPECT_NO_THROW(core.set_property(ov::cache_dir(m_cacheDir)));
            EXPECT_NO_THROW(m_testFunction(core));
        });
    }
    {  // Step 3: same load, should be ok now due to re-creation of cache
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : comp_models) {
            EXPECT_CALL(*net, export_model(_)).Times(0);
        }
        testLoad([&](ov::Core& core) {
            EXPECT_NO_THROW(core.set_property(ov::cache_dir(m_cacheDir)));
            EXPECT_NO_THROW(m_testFunction(core));
        });
    }
}

TEST_P(CachingTest, TestCacheFileOldVersion) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            EXPECT_NO_THROW(core.set_property(ov::cache_dir(m_cacheDir)));
            EXPECT_NO_THROW(m_testFunction(core));
        });
    }
    {
        auto blobs = ov::test::utils::listFilesWithExt(m_cacheDir, "blob");
        for (const auto& fileName : blobs) {
            std::string content;
            {
                std::ifstream inp(fileName, std::ios_base::binary);
                std::ostringstream ostr;
                ostr << inp.rdbuf();
                content = ostr.str();
            }
            std::string buildNum = ov::get_openvino_version().buildNumber;
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
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            EXPECT_NO_THROW(core.set_property(ov::cache_dir(m_cacheDir)));
            EXPECT_NO_THROW(m_testFunction(core));
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {  // Step 3: same load, should be ok now due to re-creation of cache
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : comp_models) {
            EXPECT_CALL(*net, export_model(_)).Times(0);
        }
        testLoad([&](ov::Core& core) {
            EXPECT_NO_THROW(core.set_property(ov::cache_dir(m_cacheDir)));
            EXPECT_NO_THROW(m_testFunction(core));
        });
    }
}

TEST_P(CachingTest, TestCacheFileWithCompiledModelRuntimeProperties) {
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([&](const std::string&, const ov::AnyMap&) {
            return std::vector<ov::PropertyName>{ov::internal::caching_properties.name(),
                                                 ov::internal::compiled_model_runtime_properties.name(),
                                                 ov::internal::compiled_model_runtime_properties_supported.name()};
        }));
    const std::string compiled_model_runtime_properties("Mock compiled model format segment.");
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::compiled_model_runtime_properties.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(compiled_model_runtime_properties));
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::compiled_model_runtime_properties_supported.name(), _))
        .Times(AtLeast(1))
        .WillRepeatedly(Invoke([&](const std::string&, const ov::AnyMap& options) {
            auto it = options.find(ov::internal::compiled_model_runtime_properties.name());
            ov::Any ret = true;
            if (it == options.end() || it->second.as<std::string>() != compiled_model_runtime_properties)
                ret = false;
            return ret;
        }));
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            EXPECT_NO_THROW(core.set_property(ov::cache_dir(m_cacheDir)));
            EXPECT_NO_THROW(m_testFunction(core));
        });
    }
    {
        auto blobs = ov::test::utils::listFilesWithExt(m_cacheDir, "blob");
        for (const auto& fileName : blobs) {
            std::string content;
            {
                std::ifstream inp(fileName, std::ios_base::binary);
                std::ostringstream ostr;
                ostr << inp.rdbuf();
                content = ostr.str();
            }
            auto index = content.find(compiled_model_runtime_properties.c_str());
            std::string new_compiled_model_runtime_properties(compiled_model_runtime_properties.size(), '0');
            if (index != std::string::npos) {
                content.replace(index, compiled_model_runtime_properties.size(), new_compiled_model_runtime_properties);
            } else {
                return;  // skip test
            }
            std::ofstream out(fileName, std::ios_base::binary);
            out.write(content.c_str(), static_cast<std::streamsize>(content.size()));
        }
    }
    m_post_mock_net_callbacks.pop_back();
    {  // Step 2. compiled_model_runtime_properties mismatch, cache will be silently removed
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            EXPECT_NO_THROW(core.set_property(ov::cache_dir(m_cacheDir)));
            EXPECT_NO_THROW(m_testFunction(core));
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {  // Step 3: same load, should be ok now due to re-creation of cache
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(!m_remoteContext ? 1 : 0);
        for (auto& net : comp_models) {
            EXPECT_CALL(*net, export_model(_)).Times(0);
        }
        testLoad([&](ov::Core& core) {
            EXPECT_NO_THROW(core.set_property(ov::cache_dir(m_cacheDir)));
            EXPECT_NO_THROW(m_testFunction(core));
        });
    }
}

TEST_P(CachingTest, LoadHetero_NoCacheMetric) {
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{}));
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(std::vector<ov::PropertyName>{}));
    // Hetero supports Import/Export, but mock plugin does not
    deviceToLoad = ov::test::utils::DEVICE_HETERO + std::string(":mock.1,mock.2");
    if (m_remoteContext) {
        return;  // skip the remote Context test for Hetero plugin
    }
    for (int i = 0; i < 2; i++) {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(0);
            EXPECT_CALL(net, get_runtime_model()).Times(0);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
            comp_models.clear();
        });
    }
}

TEST_P(CachingTest, LoadHetero_OneDevice) {
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    // deviceToLoad = "mock";
    deviceToLoad = ov::test::utils::DEVICE_HETERO + std::string(":mock");
    if (m_remoteContext) {
        return;  // skip the remote Context test for Hetero plugin
    }
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
        // Ensure that only 1 blob (for Hetero) is created
        EXPECT_EQ(ov::test::utils::listFilesWithExt(m_cacheDir, "blob").size(), 1);
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(1);
        for (auto& net : comp_models) {
            EXPECT_CALL(*net, export_model(_)).Times(0);
        }
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
            comp_models.clear();
        });
    }
}

TEST_P(CachingTest, LoadHetero_TargetFallbackFromCore) {
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    deviceToLoad = ov::test::utils::DEVICE_HETERO;
    if (m_remoteContext) {
        return;  // skip the remote Context test for Hetero plugin
    }
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            core.set_property(ov::test::utils::DEVICE_HETERO, {{ov::device::priorities.name(), "mock"}});
            m_testFunction(core);
        });
        // Ensure that only 1 blob (for Hetero) is created
        EXPECT_EQ(ov::test::utils::listFilesWithExt(m_cacheDir, "blob").size(), 1);
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(1);
        for (auto& net : comp_models) {
            EXPECT_CALL(*net, export_model(_)).Times(0);
        }
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            core.set_property(ov::test::utils::DEVICE_HETERO, {{ov::device::priorities.name(), "mock"}});
            m_testFunction(core);
            comp_models.clear();
        });
    }
}

TEST_P(CachingTest, LoadHetero_MultiArchs) {
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());

    EXPECT_CALL(*mockPlugin, query_model(_, _))
        .Times(AnyNumber())
        .WillRepeatedly(
            Invoke([&](const std::shared_ptr<const ov::Model>& model, const ov::AnyMap& config) -> ov::SupportedOpsMap {
                ov::SupportedOpsMap res;
                EXPECT_TRUE(model);

                auto id = config.at("DEVICE_ID").as<std::string>();
                bool supportsRelu = std::stoi(id) < 10;

                for (auto&& node : model->get_ops()) {
                    std::string nodeType = node->get_type_name();
                    if ((nodeType == "Relu" && supportsRelu) || (nodeType != "Relu" && !supportsRelu)) {
                        res.emplace(node->get_friendly_name(), deviceName + "." + id);
                    }
                }
                return res;
            }));
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([&](const std::string&, const ov::AnyMap& options) {
            auto id = options.at("DEVICE_ID").as<std::string>();
            if (std::stoi(id) < 10) {
                return "mock_first_architecture";
            } else {
                return "mock_another_architecture";
            }
        }));
    deviceToLoad = ov::test::utils::DEVICE_HETERO + std::string(":mock.1,mock.51");
    if (m_remoteContext) {
        return;  // skip the remote Context test for Hetero plugin
    }
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(AtLeast(2));  // for .1 and for .51
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(AtLeast(1));
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
        // Ensure that only 1 blob (for Hetero) is created
        EXPECT_EQ(ov::test::utils::listFilesWithExt(m_cacheDir, "blob").size(), 1);
    }

    deviceToLoad = ov::test::utils::DEVICE_HETERO + std::string(":mock.2,mock.52");
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(AtLeast(2));  // for .2 and for .52
        for (auto& net : comp_models) {
            EXPECT_CALL(*net, export_model(_)).Times(0);
        }
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
    deviceToLoad = ov::test::utils::DEVICE_HETERO + std::string(":mock.53,mock.3");
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(AtLeast(1));
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(AtLeast(1));
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
            comp_models.clear();
        });
    }
}

TEST_P(CachingTest, LoadHetero_MultiArchs_TargetFallback_FromCore) {
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([&](const std::string&, const ov::AnyMap& options) {
            auto id = options.at("DEVICE_ID").as<std::string>();
            if (std::stoi(id) < 10) {
                return "mock_first_architecture";
            } else {
                return "mock_another_architecture";
            }
        }));
    deviceToLoad = ov::test::utils::DEVICE_HETERO;
    if (m_remoteContext) {
        return;  // skip the remote Context test for Hetero plugin
    }
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            core.set_property(ov::test::utils::DEVICE_HETERO, {{ov::device::priorities.name(), "mock.1"}});
            m_testFunction(core);
        });
    }
    m_post_mock_net_callbacks.pop_back();
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(1);
        for (auto& net : comp_models) {
            EXPECT_CALL(*net, export_model(_)).Times(0);
        }
        testLoad([&](ov::Core& core) {
            core.set_property(ov::test::utils::DEVICE_HETERO, {{ov::device::priorities.name(), "mock.1"}});
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::test::utils::DEVICE_HETERO, {{ov::device::priorities.name(), "mock.51"}});
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
            comp_models.clear();
        });
    }
}

#if defined(ENABLE_AUTO)
// AUTO-DEVICE test
// Single device
TEST_P(CachingTest, LoadAUTO_OneDevice) {
    const auto TEST_COUNT = 2;
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_default_context(_)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Auto plugin
    }
    m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
        EXPECT_CALL(net, export_model(_)).Times(1);
    });
    std::string cacheDir = m_cacheDir;
    MkDirGuard guard(cacheDir);
    for (int index = 0; index < TEST_COUNT; index++) {
        deviceToLoad = ov::test::utils::DEVICE_AUTO;
        deviceToLoad += ":mock.0";
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(TEST_COUNT - index - 1);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(index);
        OV_ASSERT_NO_THROW(testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(cacheDir));
            m_testFunction(core);
        }));
    }
    std::cout << "Caching LoadAuto Test completed. Tried " << TEST_COUNT << " times" << std::endl;
}
// AUTO-DEVICE test
// load model with config
TEST_P(CachingTest, LoadAUTOWithConfig) {
    const auto TEST_COUNT = 2;
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_default_context(_)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Auto plugin
    }
    int index = 0;
    m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
        EXPECT_CALL(net, export_model(_)).Times(1);
    });
    std::string cacheDir = m_cacheDir;
    MkDirGuard guard(cacheDir);
    for (; index < TEST_COUNT; index++) {
        deviceToLoad = ov::test::utils::DEVICE_AUTO;
        deviceToLoad += ":mock.0";
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(TEST_COUNT - index - 1);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(index);
        OV_ASSERT_NO_THROW(testLoad([&](ov::Core& core) {
            m_testFunctionWithCfg(core, {{ov::cache_dir.name(), cacheDir}});
        }));
    }
    std::cout << "Caching LoadAuto Test completed. Tried " << index << " times" << std::endl;
}
// Single device not support import/export
TEST_P(CachingTest, LoadAUTO_OneDeviceNoImportExport) {
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_default_context(_)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(false));
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(decltype(ov::device::capabilities)::value_type{}));
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Auto plugin
    }
    EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 2 : 0);
    EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
        .Times(!m_remoteContext ? 2 : 0);
    EXPECT_CALL(*mockPlugin, OnCompileModelFromFile()).Times(m_type == TestLoadType::EModelName ? 2 : 0);
    EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
    EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
    testLoad([&](ov::Core& core) {
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(0);
        });
        deviceToLoad = ov::test::utils::DEVICE_AUTO;
        deviceToLoad += ":mock.0";
        core.set_property(ov::cache_dir(m_cacheDir));
        m_testFunction(core);
        m_post_mock_net_callbacks.pop_back();
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(0);
        });
        m_testFunction(core);
    });
}
// MULTI-DEVICE test
// Test that it is safe to load multiple devices sharing same cache
// In case of sporadic failures - increase 'TEST_DURATION_MS' 100x times for better reproducibility
TEST_P(CachingTest, LoadMulti_race) {
    const auto TEST_DURATION_MS = 2000;
    const auto TEST_DEVICE_MAX_COUNT = 10;
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_default_context(_)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Multi plugin
    }
    int index = 0;
    auto start = high_resolution_clock::now();
    m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
        EXPECT_CALL(net, export_model(_)).Times(1);
    });
    do {
        std::string cacheDir = m_cacheDir + std::to_string(index);
        MkDirGuard guard(cacheDir);
        int devCount = 1 + index % (TEST_DEVICE_MAX_COUNT - 1);  // try dynamic number of devices from 1 to max
        deviceToLoad = ov::test::utils::DEVICE_MULTI;
        deviceToLoad += ":mock.0";
        for (int i = 1; i < devCount; i++) {
            deviceToLoad += ",mock." + std::to_string(i);
        }

        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(devCount - 1);
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(cacheDir));
            OV_ASSERT_NO_THROW(m_testFunction(core));
        });
        index++;
    } while (duration_cast<milliseconds>(high_resolution_clock::now() - start).count() < TEST_DURATION_MS);
    std::cout << "Caching LoadMulti Test completed. Tried " << index << " times" << std::endl;
}

// MULTI-DEVICE test
// Test that it is safe to load multiple devices through compile_model
// In case of sporadic failures - increase 'TEST_DURATION_MS' 100x times for better reproducibility
TEST_P(CachingTest, LoadMultiWithConfig_race) {
    const auto TEST_DURATION_MS = 2000;
    const auto TEST_DEVICE_MAX_COUNT = 10;
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_default_context(_)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Multi plugin
    }
    int index = 0;
    auto start = high_resolution_clock::now();
    m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
        EXPECT_CALL(net, export_model(_)).Times(1);
    });
    do {
        std::string cacheDir = m_cacheDir + std::to_string(index);
        MkDirGuard guard(cacheDir);
        int devCount = 1 + index % (TEST_DEVICE_MAX_COUNT - 1);  // try dynamic number of devices from 1 to max
        deviceToLoad = ov::test::utils::DEVICE_MULTI;
        deviceToLoad += ":mock.0";
        for (int i = 1; i < devCount; i++) {
            deviceToLoad += ",mock." + std::to_string(i);
        }

        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(devCount - 1);
        testLoad([&](ov::Core& core) {
            OV_ASSERT_NO_THROW(m_testFunctionWithCfg(core, {{ov::cache_dir.name(), cacheDir}}));
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
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_default_context(_)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Invoke([&](const std::string&, const ov::AnyMap& options) {
            auto id = options.at("DEVICE_ID").as<std::string>();
            auto i = std::stoi(id) / 2;
            return "mock_architecture" + std::to_string(i);
        }));
    if (m_remoteContext) {
        return;  // skip the remote Context test for Multi plugin
    }

    deviceToLoad = ov::test::utils::DEVICE_MULTI;
    deviceToLoad += ":mock.0";
    for (int i = 1; i < TEST_DEVICE_MAX_COUNT; i++) {
        deviceToLoad += ",mock." + std::to_string(i);
    }

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(TEST_DEVICE_MAX_COUNT / 2);
        // Load model from file shall not be called for plugins with caching supported
        EXPECT_CALL(*mockPlugin, OnCompileModelFromFile()).Times(0);

        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _))
            .Times(TEST_DEVICE_MAX_COUNT / 2)
            .WillRepeatedly(Invoke([&](std::istream& s, const ov::AnyMap&) {
                std::string name;
                s >> name;
                std::lock_guard<std::mutex> lock(mock_creation_mutex);
                return create_mock_compiled_model(m_models[name], mockPlugin);
            }));
        m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
            EXPECT_CALL(net, export_model(_)).Times(1);  // each net will be exported once
        });
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
}

// MULTI-DEVICE test
// Test loading of devices which don't support caching
// In case of sporadic failures - increase 'TEST_DEVICE_MAX_COUNT' 100x times for better reproducibility
TEST_P(CachingTest, LoadMulti_NoCachingOnDevice) {
    const auto TEST_DEVICE_MAX_COUNT = 100;  // Looks enough to catch potential race conditions
    EXPECT_CALL(*mockPlugin, get_default_context(_)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capability::EXPORT_IMPORT, _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(false));
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(decltype(ov::device::capabilities)::value_type{}));
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());

    std::vector<ov::Output<const ov::Node>> ins;
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
    ins.emplace_back(param->output(0));
    m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& model) {
        EXPECT_CALL(model, inputs()).Times(AnyNumber()).WillRepeatedly(ReturnRefOfCopy(ins));
        EXPECT_CALL(model, outputs()).Times(AnyNumber()).WillRepeatedly(ReturnRefOfCopy(ins));
    });
    if (m_remoteContext) {
        return;  // skip the remote Context test for Multi plugin
    }

    deviceToLoad = ov::test::utils::DEVICE_MULTI;
    deviceToLoad += ":mock.0";
    for (int i = 1; i < TEST_DEVICE_MAX_COUNT; i++) {
        deviceToLoad += ",mock." + std::to_string(i);
    }

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(TEST_DEVICE_MAX_COUNT);
        // Load model from file shall not be called by Multi plugin for devices with caching supported
        EXPECT_CALL(*mockPlugin, OnCompileModelFromFile())
            .Times(m_type == TestLoadType::EModel ? 0 : TEST_DEVICE_MAX_COUNT);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(0);
        for (auto& net : comp_models) {
            EXPECT_CALL(*net, export_model(_)).Times(0);
        }
        testLoad([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            ov::CompiledModel model;
            model = m_testFunction(core);
            // Verify that inputs and outputs are set for Multi Compiled Model
            ASSERT_EQ(model.inputs().size(), ins.size());
            ASSERT_EQ(model.outputs().size(), ins.size());
            comp_models.clear();
        });
    }
}
#endif  // defined(ENABLE_AUTO)

#if defined(ENABLE_AUTO_BATCH)
// BATCH-DEVICE test
// load model with config
TEST_P(CachingTest, LoadBATCHWithConfig) {
    const auto TEST_COUNT = 2;
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_default_context(_)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::hint::performance_mode.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(ov::hint::PerformanceMode::THROUGHPUT));
    if (m_remoteContext) {
        return;  // skip the remote Context test for Auto plugin
    }
    int index = 0;
    m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
        EXPECT_CALL(net, export_model(_)).Times(1);
    });
    std::string cacheDir = m_cacheDir;
    MkDirGuard guard(cacheDir);
    for (; index < TEST_COUNT; index++) {
        deviceToLoad = ov::test::utils::DEVICE_BATCH;
        deviceToLoad += ":mock.0";
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(TEST_COUNT - index - 1);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(index);
        testLoad([&](ov::Core& core) {
            m_testFunctionWithCfg(core, {{ov::cache_dir.name(), cacheDir}});
        });
    }
    std::cout << "Caching LoadAuto Test completed. Tried " << index << " times" << std::endl;
}
#endif  // defined(ENABLE_AUTO_BATCH)

// In case of sporadic failures - increase 'TEST_DURATION_MS' 100x times for better reproducibility
TEST_P(CachingTest, Load_threads) {
    const auto TEST_DURATION_MS = 2000;
    const auto THREADS_COUNT = 4;
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Multi plugin
    }
    auto start = high_resolution_clock::now();
    int index = 0;
    m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
        EXPECT_CALL(net, export_model(_)).Times(1);
    });
    do {
        std::string cacheDir = m_cacheDir + std::to_string(index);
        MkDirGuard guard(cacheDir);
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
        EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
        EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(THREADS_COUNT - 1);
        testLoad([&](ov::Core& core) {
            core.set_property({{ov::cache_dir.name(), cacheDir}});
            std::vector<std::thread> threads;
            for (int i = 0; i < THREADS_COUNT; i++) {
                threads.emplace_back(([&]() {
                    m_testFunction(core);
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

TEST_P(CachingTest, Load_mmap) {
    ON_CALL(*mockPlugin, import_model(_, _)).WillByDefault(Invoke([&](std::istream& istr, const ov::AnyMap& config) {
        if (m_checkConfigCb) {
            m_checkConfigCb(config);
        }
        ov::Tensor compiled_blob;
        if (config.count(ov::hint::compiled_blob.name()))
            compiled_blob = config.at(ov::hint::compiled_blob.name()).as<ov::Tensor>();

        EXPECT_TRUE(static_cast<bool>(compiled_blob));

        std::string name;
        istr >> name;
        char space;
        istr.read(&space, 1);
        std::lock_guard<std::mutex> lock(mock_creation_mutex);
        return create_mock_compiled_model(m_models[name], mockPlugin);
    }));

    ON_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _))
        .WillByDefault(Invoke([&](const std::string&, const ov::AnyMap&) {
            return std::vector<ov::PropertyName>{ov::internal::caching_properties.name(),
                                                 ov::internal::caching_with_mmap.name()};
        }));
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Multi plugin
    }
    int index = 0;
    m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
        EXPECT_CALL(net, export_model(_)).Times(1);
    });
    MkDirGuard guard(m_cacheDir);
    EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
    EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
    EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
    EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(1);
    testLoad([&](ov::Core& core) {
        core.set_property({{ov::cache_dir.name(), m_cacheDir}});
        m_testFunction(core);
        m_testFunction(core);
    });
    std::cout << "Caching Load multiple threads test completed. Tried " << index << " times" << std::endl;
}

TEST_P(CachingTest, Load_mmap_is_disabled) {
    ON_CALL(*mockPlugin, import_model(_, _)).WillByDefault(Invoke([&](std::istream& istr, const ov::AnyMap& config) {
        if (m_checkConfigCb) {
            m_checkConfigCb(config);
        }

        EXPECT_GT(config.count(ov::hint::compiled_blob.name()), 0);

        std::string name;
        istr >> name;
        char space;
        istr.read(&space, 1);
        std::lock_guard<std::mutex> lock(mock_creation_mutex);
        return create_mock_compiled_model(m_models[name], mockPlugin);
    }));
    ON_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _))
        .WillByDefault(Invoke([&](const std::string&, const ov::AnyMap&) {
            return std::vector<ov::PropertyName>{ov::internal::caching_properties.name(),
                                                 ov::internal::caching_with_mmap.name()};
        }));
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Multi plugin
    }
    int index = 0;
    m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
        EXPECT_CALL(net, export_model(_)).Times(1);
    });
    MkDirGuard guard(m_cacheDir);
    EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
    EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
    EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
    EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(1);
    testLoad([&](ov::Core& core) {
        core.set_property({{ov::cache_dir.name(), m_cacheDir}});
        core.set_property({ov::enable_mmap(false)});
        m_testFunction(core);
        m_testFunction(core);
    });
    std::cout << "Caching Load multiple threads test completed. Tried " << index << " times" << std::endl;
}

TEST_P(CachingTest, Load_mmap_is_not_supported_by_plugin) {
    ON_CALL(*mockPlugin, import_model(_, _)).WillByDefault(Invoke([&](std::istream& istr, const ov::AnyMap& config) {
        if (m_checkConfigCb) {
            m_checkConfigCb(config);
        }

        EXPECT_GT(config.count(ov::hint::compiled_blob.name()), 0);

        std::string name;
        istr >> name;
        char space;
        istr.read(&space, 1);
        std::lock_guard<std::mutex> lock(mock_creation_mutex);
        return create_mock_compiled_model(m_models[name], mockPlugin);
    }));
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Multi plugin
    }
    int index = 0;
    m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
        EXPECT_CALL(net, export_model(_)).Times(1);
    });
    MkDirGuard guard(m_cacheDir);
    EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
    EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
    EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
    EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(1);
    testLoad([&](ov::Core& core) {
        core.set_property({{ov::cache_dir.name(), m_cacheDir}});
        core.set_property({ov::enable_mmap(true)});
        m_testFunction(core);
        m_testFunction(core);
    });
    std::cout << "Caching Load multiple threads test completed. Tried " << index << " times" << std::endl;
}

TEST_P(CachingTest, Load_mmap_is_disabled_local_cfg) {
    ON_CALL(*mockPlugin, import_model(_, _)).WillByDefault(Invoke([&](std::istream& istr, const ov::AnyMap& config) {
        if (m_checkConfigCb) {
            m_checkConfigCb(config);
        }

        EXPECT_GT(config.count(ov::hint::compiled_blob.name()), 0);

        std::string name;
        istr >> name;
        char space;
        istr.read(&space, 1);
        std::lock_guard<std::mutex> lock(mock_creation_mutex);
        return create_mock_compiled_model(m_models[name], mockPlugin);
    }));
    ON_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _))
        .WillByDefault(Invoke([&](const std::string&, const ov::AnyMap&) {
            return std::vector<ov::PropertyName>{ov::internal::caching_properties.name(),
                                                 ov::internal::caching_with_mmap.name()};
        }));
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Multi plugin
    }
    int index = 0;
    m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
        EXPECT_CALL(net, export_model(_)).Times(1);
    });
    MkDirGuard guard(m_cacheDir);
    EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
    EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
    EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
    EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(1);
    testLoad([&](ov::Core& core) {
        const auto config = ov::AnyMap{{ov::cache_dir(m_cacheDir)}, {ov::enable_mmap(false)}};
        m_testFunctionWithCfg(core, config);
        m_testFunctionWithCfg(core, config);
    });
    std::cout << "Caching Load multiple threads test completed. Tried " << index << " times" << std::endl;
}

TEST_P(CachingTest, Load_mmap_is_not_supported_by_plugin_local_cfg) {
    ON_CALL(*mockPlugin, import_model(_, _)).WillByDefault(Invoke([&](std::istream& istr, const ov::AnyMap& config) {
        if (m_checkConfigCb) {
            m_checkConfigCb(config);
        }
        EXPECT_GT(config.count(ov::hint::compiled_blob.name()), 0);

        std::string name;
        istr >> name;
        char space;
        istr.read(&space, 1);
        std::lock_guard<std::mutex> lock(mock_creation_mutex);
        return create_mock_compiled_model(m_models[name], mockPlugin);
    }));
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Multi plugin
    }
    int index = 0;
    m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
        EXPECT_CALL(net, export_model(_)).Times(1);
    });
    MkDirGuard guard(m_cacheDir);
    EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
    EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
    EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
    EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(1);
    testLoad([&](ov::Core& core) {
        const auto config = ov::AnyMap{{ov::cache_dir(m_cacheDir)}, {ov::enable_mmap(false)}};
        m_testFunctionWithCfg(core, config);
        m_testFunctionWithCfg(core, config);
    });
    std::cout << "Caching Load multiple threads test completed. Tried " << index << " times" << std::endl;
}

TEST_P(CachingTest, import_from_cache_model_and_weights_path_properties_not_supported) {
    ON_CALL(*mockPlugin, import_model(_, _)).WillByDefault(Invoke([&](std::istream& istr, const ov::AnyMap& config) {
        if (m_checkConfigCb) {
            m_checkConfigCb(config);
        }
        EXPECT_EQ(config.count(ov::hint::compiled_blob.name()), 1);
        EXPECT_EQ(config.count(ov::hint::model.name()), 0);
        EXPECT_EQ(config.count(ov::weights_path.name()), 0);

        std::string name;
        istr >> name;
        char space;
        istr.read(&space, 1);
        std::lock_guard<std::mutex> lock(mock_creation_mutex);
        return create_mock_compiled_model(m_models[name], mockPlugin);
    }));
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Multi plugin
    }
    m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
        EXPECT_CALL(net, export_model(_)).Times(1);
    });
    MkDirGuard guard(m_cacheDir);
    EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
    EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
    EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
    EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(1);
    testLoad([&](ov::Core& core) {
        const auto config = ov::AnyMap{{ov::cache_dir(m_cacheDir)}};
        m_testFunctionWithCfg(core, config);
        // load from cache
        m_testFunctionWithCfg(core, config);
    });
}

TEST_P(CachingTest, import_from_cache_model_and_weights_path_properties_are_supported) {
    ON_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _))
        .WillByDefault(Invoke([&](const std::string&, const ov::AnyMap&) {
            return std::vector<ov::PropertyName>{
                ov::supported_properties.name(),
                ov::device::capabilities.name(),
                ov::device::architecture.name(),
                ov::hint::model.name(),
                ov::weights_path.name(),
            };
        }));
    ON_CALL(*mockPlugin, import_model(_, _)).WillByDefault(Invoke([&](std::istream& istr, const ov::AnyMap& config) {
        if (m_checkConfigCb) {
            m_checkConfigCb(config);
        }
        EXPECT_EQ(config.count(ov::hint::compiled_blob.name()), 1);
        EXPECT_EQ(config.count(ov::hint::model.name()), m_type != TestLoadType::EModelName ? 1 : 0);
        EXPECT_EQ(config.count(ov::weights_path.name()), m_type == TestLoadType::EModelName ? 1 : 0);

        std::string name;
        istr >> name;
        char space;
        istr.read(&space, 1);
        std::lock_guard<std::mutex> lock(mock_creation_mutex);
        return create_mock_compiled_model(m_models[name], mockPlugin);
    }));
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Multi plugin
    }
    m_post_mock_net_callbacks.emplace_back([&](MockICompiledModelImpl& net) {
        EXPECT_CALL(net, export_model(_)).Times(1);
    });
    MkDirGuard guard(m_cacheDir);
    EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
    EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
    EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
    EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(1);
    testLoad([&](ov::Core& core) {
        const auto config = ov::AnyMap{{ov::cache_dir(m_cacheDir)}};
        m_testFunctionWithCfg(core, config);
        // load from cache
        m_testFunctionWithCfg(core, config);
    });
}

TEST_P(CachingTest, import_from_compiled_blob_weights_path_property_is_supported) {
    ON_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _))
        .WillByDefault(Invoke([&](const std::string&, const ov::AnyMap&) {
            return std::vector<ov::PropertyName>{ov::supported_properties.name(),
                                                 ov::device::capabilities.name(),
                                                 ov::device::architecture.name(),
                                                 ov::weights_path.name(),
                                                 ov::hint::model.name()};
        }));
    ON_CALL(*mockPlugin, import_model(_, _)).WillByDefault(Invoke([&](std::istream& istr, const ov::AnyMap& config) {
        if (m_checkConfigCb) {
            m_checkConfigCb(config);
        }
        EXPECT_EQ(config.count(ov::hint::compiled_blob.name()), 1);
        EXPECT_EQ(config.count(ov::hint::model.name()), m_type != TestLoadType::EModelName ? 1 : 0);
        EXPECT_EQ(config.count(ov::weights_path.name()), m_type == TestLoadType::EModelName ? 1 : 0);
        return nullptr;
    }));
    EXPECT_CALL(*mockPlugin, get_property(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, query_model(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    if (m_remoteContext) {
        return;  // skip the remote Context test for Multi plugin
    }
    m_modelCallback = nullptr;
    MkDirGuard guard(m_cacheDir);
    EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(0);
    EXPECT_CALL(*mockPlugin, import_model(_, _, _)).Times(0);
    EXPECT_CALL(*mockPlugin, import_model(_, _)).Times(1);

    EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
    if (m_type == TestLoadType::EModelName) {
        EXPECT_CALL(*mockPlugin, OnCompileModelFromFile()).Times(1);
    }
    testLoad([&](ov::Core& core) {
        auto compiled_blob = ov::Tensor(ov::element::u8, ov::Shape{100});
        const auto config = ov::AnyMap{{ov::hint::compiled_blob(compiled_blob)}};
        m_testFunctionWithCfg(core, config);
    });
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

class CacheTestWithProxyEnabled : public CachingTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<TestParam, std::string>>& obj) {
        return std::get<1>(std::get<0>(obj.param)) + "_" + std::get<1>(obj.param);
    }

protected:
    void testLoadProxy(const std::function<void(ov::Core& core)>& func) {
        ov::Core core;
        injectPlugin(mockPlugin.get());
        core.register_plugin(ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                                std::string("mock_engine") + OV_BUILD_POSTFIX),
                             deviceName,
                             {{ov::proxy::configuration::alias.name(), "mock"},
                              {ov::proxy::configuration::internal_name.name(), "internal_mock"}});
        ON_CALL(*mockPlugin, get_default_context(_)).WillByDefault(Invoke([&](const ov::AnyMap&) {
            return std::make_shared<MockRemoteContext>("internal_mock");
        }));
        func(core);
        core.unload_plugin(deviceName);
    }
};

#ifdef PROXY_PLUGIN_ENABLED
TEST_P(CacheTestWithProxyEnabled, TestLoad) {
    ON_CALL(*mockPlugin, get_property(ov::available_devices.name(), _))
        .WillByDefault(Invoke([&](const std::string&, const ov::AnyMap&) {
            std::vector<std::string> available_devices = {};
            available_devices.push_back("mock");
            return decltype(ov::available_devices)::value_type(available_devices);
        }));
    EXPECT_CALL(*mockPlugin, get_default_context(_)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::architecture.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::supported_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::internal::caching_properties.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::available_devices.name(), _)).Times(AnyNumber());
    EXPECT_CALL(*mockPlugin, get_property(ov::device::capabilities.name(), _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(decltype(ov::device::capabilities)::value_type{}));
    // proxy should direct the compile from file to hardware plugin
    EXPECT_CALL(*mockPlugin, OnCompileModelFromFile()).Times(m_type == TestLoadType::EModelName ? 1 : 0);

    {
        EXPECT_CALL(*mockPlugin, compile_model(_, _, _)).Times(m_remoteContext ? 1 : 0);
        EXPECT_CALL(*mockPlugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(!m_remoteContext ? 1 : 0);
        testLoadProxy([&](ov::Core& core) {
            core.set_property(ov::cache_dir(m_cacheDir));
            m_testFunction(core);
        });
    }
}

INSTANTIATE_TEST_SUITE_P(CacheTestWithProxyEnabled,
                         CacheTestWithProxyEnabled,
                         ::testing::Combine(::testing::ValuesIn(loadVariants), ::testing::ValuesIn(cacheFolders)),
                         CacheTestWithProxyEnabled::getTestCaseName);
#endif
