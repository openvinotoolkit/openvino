// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//
#if defined(ENABLE_UNICODE_PATH_SUPPORT) || defined(_WIN32)

#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "ie_core.hpp"
#include "ie_precision.hpp"
#include "file_utils.h"

#include "common_test_utils/unicode_utils.hpp"
#include "ngraph_functions/builders.hpp"

#include <gtest/gtest.h>
#include <stdlib.h>
#ifndef _WIN32
#include <dirent.h>
#endif

using activationParams = std::tuple<std::wstring, InferenceEngine::SizeVector>;

namespace {

const wchar_t pathSeparator =
#ifdef _WIN32
    L'\\';
#else
    L'/';
#endif

std::vector<std::wstring> listFiles(const std::wstring& path, const std::wstring& extension) {
    std::vector<std::wstring> result;
#ifdef _WIN32
    HANDLE hFind = INVALID_HANDLE_VALUE;
    WIN32_FIND_DATAW ffd;
    hFind = FindFirstFileW((path + L"\\*." + extension).c_str(), &ffd);

    if (INVALID_HANDLE_VALUE == hFind)
        return result;

    do {
        result.push_back(ffd.cFileName);
    } while (FindNextFileW(hFind, &ffd) != 0);
#else
    DIR* dir = nullptr;
    dirent* ent = nullptr;

    if ((dir = opendir(FileUtils::wStringtoMBCSstringChar(path).c_str())) != nullptr) {
        while ((ent = readdir(dir)) != nullptr) {
            const auto wname = FileUtils::multiByteCharToWString(ent->d_name);
            if (wname.rfind(extension) == wname.size() - extension.size()) // not the efficient one, but ok for tests
                result.push_back(wname);
        }
        closedir(dir);
    }
#endif
    return result;
}

std::vector<std::wstring> getAllFirmwareNames(const std::wstring& path) {
    auto result = listFiles(path, L"mvcmd");
#ifdef _WIN32
    const auto elfFiles = listFiles(path, L"elf");
    result.insert(result.end(), elfFiles.begin(), elfFiles.end());
#endif
    return result;
}

bool unicodeSetEnv(const std::wstring& key, const std::wstring& value) {
#ifdef _WIN32
    return SetEnvironmentVariableW(key.c_str(), value.c_str());
#else
    return 0 == setenv(FileUtils::wStringtoMBCSstringChar(key).c_str(), FileUtils::wStringtoMBCSstringChar(value).c_str(), 1);
#endif
}

} // namespace

class TemporaryFirmwareDir {
    std::wstring m_tempDirPath;
    std::vector<std::wstring> m_filenames;

public:
    TemporaryFirmwareDir(const std::wstring& tempDirPath,
                         const std::wstring& sourceDir,
                         const std::vector<std::wstring>& filenames)
        : m_tempDirPath(tempDirPath), m_filenames(filenames) {
        #ifdef _WIN32
            _wmkdir(m_tempDirPath.c_str());
        #else
            mkdir(FileUtils::wStringtoMBCSstringChar(m_tempDirPath).c_str(), 0755);
        #endif

        for (const auto& filename : m_filenames) {
            CommonTestUtils::copyFile(sourceDir + pathSeparator + filename, m_tempDirPath + pathSeparator + filename);
        }
    }

    ~TemporaryFirmwareDir() {
#ifdef _WIN32
        for (const auto& filename : m_filenames) {
            _wunlink((m_tempDirPath + pathSeparator + filename).c_str());
        }
        _wrmdir(m_tempDirPath.c_str());
#else
        for (const auto& filename : m_filenames) {
            unlink(FileUtils::wStringtoMBCSstringChar(m_tempDirPath + pathSeparator + filename).c_str());
        }
        rmdir(FileUtils::wStringtoMBCSstringChar(m_tempDirPath).c_str());
#endif
    }
};

class UnicodeLocationTest :
            public testing::WithParamInterface<activationParams>,
            public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<activationParams> &obj);

protected:
    InferenceEngine::SizeVector inputShapes;
    std::unique_ptr<TemporaryFirmwareDir> tempDir;

    virtual void envSetup() {
        const std::wstring testSubdir = std::get<0>(GetParam());
        const std::wstring libPath = InferenceEngine::getIELibraryPathW();
        const std::wstring tempPath = libPath + pathSeparator + testSubdir;
        tempDir.reset(new TemporaryFirmwareDir(tempPath, libPath, getAllFirmwareNames(libPath)));

        unicodeSetEnv(L"IE_VPU_FIRMWARE_DIR", tempPath);
    }

    void SetUp() override {
        envSetup();

        InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP32;
        InferenceEngine::Precision netPrecision = InferenceEngine::Precision::FP32;
        ngraph::helpers::ActivationTypes activationType = ngraph::helpers::ActivationTypes::Tanh;
        inputShapes = std::get<1>(GetParam());
        targetDevice = CommonTestUtils::DEVICE_MYRIAD;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShapes});
        auto activation = ngraph::builder::makeActivation(params[0], ngPrc, activationType);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{activation}, params);
    }

    void Run() override {
        LayerTestsUtils::LayerTestsCommon::Run();
        UnregisterCertainPlugin("MYRIAD");
    }
};

class UnicodeLocationTestUnexistent : public UnicodeLocationTest {
    void envSetup() override {
        const std::wstring testSubdir = std::get<0>(GetParam());
        const std::wstring tempPath = std::wstring(L"unexistent_path") + pathSeparator + testSubdir;
        unicodeSetEnv(L"IE_VPU_FIRMWARE_DIR", tempPath);
    }
};

TEST_P(UnicodeLocationTest, CompareWithRefs) {
    Run();
}

TEST_P(UnicodeLocationTestUnexistent, CompareWithRefs) {
    ASSERT_THROW(Run(), InferenceEngine::details::InferenceEngineException);
}

const auto basicCases = ::testing::Combine(
    ::testing::ValuesIn(CommonTestUtils::test_unicode_postfix_vector),
    ::testing::Values(std::vector<size_t>({1, 50})));

INSTANTIATE_TEST_CASE_P(Environment, UnicodeLocationTest, basicCases);
INSTANTIATE_TEST_CASE_P(Environment, UnicodeLocationTestUnexistent, basicCases);

#endif
