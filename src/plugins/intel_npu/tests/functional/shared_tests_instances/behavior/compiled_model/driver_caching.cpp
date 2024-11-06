#include "overload/driver_caching.hpp"
#include "common/npu_test_env_cfg.hpp"
//#include "iostream"

namespace {

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> emptyConfig = {{}};
const std::vector<ov::AnyMap> ovCacheConfig = {{ov::cache_dir("testCacheDir")}};
//ov::AnyMap{{ov::cache_dir.name(), m_cacheDir}};
//ov::AnyMap({ov::cache_dir("test"), ov::force_tbb_terminate(false)}
const std::vector<ov::AnyMap> byPassConfig = {{ov::intel_npu::bypass_umd_caching(true)}};

// INSTANTIATE_TEST_SUITE_P(smoke_CompilationCacheFlag,
//                          CompileAndDriverCaching,
//                          ::testing::Combine(::testing::Values(getConstantGraph()),
//                                             ::testing::Values(ov::test::utils::DEVICE_NPU),
//                                             ::testing::ValuesIn(emptyConfig)),
//                          ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);

// INSTANTIATE_TEST_SUITE_P(smoke_CompilationCacheFlag,
//                          CompileAndDriverCaching,
//                          ::testing::Combine(::testing::Values(getConstantGraph()),
//                                             ::testing::Values(ov::test::utils::DEVICE_NPU),
//                                             ::testing::ValuesIn(ovCacheConfig)),
//                          ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);

// INSTANTIATE_TEST_SUITE_P(smoke_CompilationCacheFlag,
//                          CompileAndDriverCaching,
//                          ::testing::Combine(::testing::Values(getConstantGraph()),
//                                             ::testing::Values(ov::test::utils::DEVICE_NPU),
//                                             ::testing::ValuesIn(byPassConfig)),
//                          ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);

const std::vector<ov::AnyMap> Config = {{}, {ov::cache_dir("testCacheDir")}, {ov::intel_npu::bypass_umd_caching(true)}};

INSTANTIATE_TEST_SUITE_P(smoke_CompilationCacheFlag,
                         CompileAndDriverCaching,
                         ::testing::Combine(::testing::Values(getConstantGraph()),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(Config)),
                         ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);

// #ifdef WIN32
// INSTANTIATE_TEST_SUITE_P(smoke_CompilationTwiceOnWindwos,
//                          CompileAndDriverCaching,
//                          ::testing::Combine(::testing::Values(getConstantGraph()),
//                                             ::testing::Values(ov::test::utils::DEVICE_NPU),
//                                             ::testing::ValuesIn(emptyConfig)),
//                          ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);

// INSTANTIATE_TEST_SUITE_P(smoke_CompilationTwiceOnWindwos,
//                          CompileAndDriverCaching,
//                          ::testing::Combine(::testing::Values(getConstantGraph()),
//                                             ::testing::Values(ov::test::utils::DEVICE_NPU),
//                                             ::testing::ValuesIn(ovCacheConfig)),
//                          ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);

// INSTANTIATE_TEST_SUITE_P(smoke_CompilationTwiceOnWindwos,
//                          CompileAndDriverCaching,
//                          ::testing::Combine(::testing::Values(getConstantGraph()),
//                                             ::testing::Values(ov::test::utils::DEVICE_NPU),
//                                             ::testing::ValuesIn(byPassConfig)),
//                          ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);

// #else

// INSTANTIATE_TEST_SUITE_P(smoke_CompilationTwiceOnLinux,
//                          CompileAndDriverCaching,
//                          ::testing::Combine(::testing::Values(getConstantGraph()),
//                                             ::testing::Values(ov::test::utils::DEVICE_NPU),
//                                             ::testing::ValuesIn(emptyConfig)),
//                          ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);

// INSTANTIATE_TEST_SUITE_P(smoke_CompilationTwiceOnLinux,
//                          CompileAndDriverCaching,
//                          ::testing::Combine(::testing::Values(getConstantGraph()),
//                                             ::testing::Values(ovCacheConfig)),
//                          ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);

// INSTANTIATE_TEST_SUITE_P(smoke_CompilationTwiceOnLinux,
//                          CompileAndDriverCaching,
//                          ::testing::Combine(::testing::Values(getConstantGraph()),
//                                             ::testing::Values(byPassConfig)),
//                          ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);

// #endif

}  // namespace
