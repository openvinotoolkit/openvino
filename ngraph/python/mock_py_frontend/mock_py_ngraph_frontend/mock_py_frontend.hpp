// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "frontend_manager/frontend_manager.hpp"
#include "frontend_manager/frontend_manager_defs.hpp"
#include "ngraph/visibility.hpp"

// Defined if we are building the plugin DLL (instead of using it)
#ifdef mock_py_ngraph_frontend_EXPORTS
#define MOCK_API NGRAPH_HELPER_DLL_EXPORT
#else
#define MOCK_API NGRAPH_HELPER_DLL_IMPORT
#endif // mock1_ngraph_frontend_EXPORTS

// OK to have 'using' in mock header

using namespace ngraph;
using namespace ngraph::frontend;

class MOCK_API InputModelMockPy : public InputModel
{
};

struct MOCK_API FeCallStat
{
    FrontEndCapFlags m_loadFlags;
    std::vector<std::string> m_loadPaths;
    int m_convertModelCount = 0;
    int m_convertFuncCount = 0;
    int m_convertPartCount = 0;
    int m_decodeCount = 0;
    int m_normalizeCount = 0;
    // Getters
    FrontEndCapFlags get_loadFlags() const { return m_loadFlags; }
    std::vector<std::string> get_loadPaths() const { return m_loadPaths; }
    int get_convertModelCount() const { return m_convertModelCount; }
    int get_convertFuncCount() const { return m_convertFuncCount; }
    int get_convertPartCount() const { return m_convertPartCount; }
    int get_decodeCount() const { return m_decodeCount; }
    int get_normalizeCount() const { return m_normalizeCount; }
};

class MOCK_API FrontEndMockPy : public FrontEnd
{
    mutable FeCallStat m_stat;

public:
    FrontEndMockPy(FrontEndCapFlags flags) { m_stat.m_loadFlags = flags; }

    InputModel::Ptr load_from_file(const std::string& path) const override
    {
        m_stat.m_loadPaths.push_back(path);
        return std::make_shared<InputModelMockPy>();
    }

    std::shared_ptr<ngraph::Function> convert(InputModel::Ptr model) const override
    {
        m_stat.m_convertModelCount++;
        return std::make_shared<ngraph::Function>(NodeVector{}, ParameterVector{});
    }

    std::shared_ptr<ngraph::Function> convert(std::shared_ptr<ngraph::Function> func) const override
    {
        m_stat.m_convertFuncCount++;
        return func;
    }

    std::shared_ptr<ngraph::Function> convert_partially(InputModel::Ptr model) const override
    {
        m_stat.m_convertPartCount++;
        return std::make_shared<ngraph::Function>(NodeVector{}, ParameterVector{});
    }

    std::shared_ptr<ngraph::Function> decode(InputModel::Ptr model) const override
    {
        m_stat.m_decodeCount++;
        return std::make_shared<ngraph::Function>(NodeVector{}, ParameterVector{});
    }

    void normalize(std::shared_ptr<ngraph::Function> function) const override
    {
        m_stat.m_normalizeCount++;
    }

    FeCallStat get_stat() const { return m_stat; }

    void reset_stat() { m_stat = {}; }
};
