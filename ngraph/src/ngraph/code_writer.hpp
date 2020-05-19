//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <sstream>
#include <string>

namespace ngraph
{
    class CodeWriter;
}

class ngraph::CodeWriter
{
public:
    CodeWriter()
        : indent(0)
        , m_pending_indent(true)
        , m_temporary_name_count(0)
    {
    }
    std::string get_code() const { return m_ss.str(); }
    void operator+=(const std::string& s) { *this << s; }
    template <typename T>
    friend CodeWriter& operator<<(CodeWriter& out, const T& obj)
    {
        std::stringstream ss;
        ss << obj;

        for (char c : ss.str())
        {
            if (c == '\n')
            {
                out.m_pending_indent = true;
            }
            else
            {
                if (out.m_pending_indent)
                {
                    out.m_pending_indent = false;
                    for (size_t i = 0; i < out.indent; i++)
                    {
                        out.m_ss << "    ";
                    }
                }
            }
            out.m_ss << c;
        }

        return out;
    }

    std::string generate_temporary_name(const std::string& prefix = "tempvar")
    {
        std::stringstream ss;

        ss << prefix << m_temporary_name_count;
        m_temporary_name_count++;

        return ss.str();
    }

    void block_begin()
    {
        *this << "{\n";
        indent++;
    }

    void block_end()
    {
        indent--;
        *this << "}\n";
    }

    size_t indent;

private:
    std::stringstream m_ss;
    bool m_pending_indent;
    size_t m_temporary_name_count;
};
