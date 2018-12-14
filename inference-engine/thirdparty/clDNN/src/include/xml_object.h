/*
// Copyright (c) 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/
#pragma once
#include <string>
#include <type_traits>
#include <unordered_map>
#include <ostream>
#include <memory>

namespace cldnn
{
    class xml_base;
    using xml_key = std::string;
    using xml_base_ptr = std::shared_ptr<xml_base>;
    using xml_map = std::unordered_map<xml_key, xml_base_ptr>;

    class xml_base
    {
    public:
        virtual void dump(std::ostream& out, int offset) = 0;
    };

    template<class Type>
    class xml_leaf : public xml_base
    {
    private:
        Type value;
    public:
        xml_leaf(const Type& val) : value(val) {}
        xml_leaf(Type&& val) : value(std::move(val)) {}
        void dump(std::ostream& out, int) override
        {
            out << value;
        }
    };

    template<class Type>
    class xml_basic_array : public xml_base
    {
    private:
        std::vector<Type> values;
    public:
        xml_basic_array(const std::vector<Type>& arr) : values(arr) {}
        xml_basic_array(std::vector<Type>&& arr) : values(std::move(arr)) {}
        void dump(std::ostream& out, int) override
        {
            const char* delim = "";
            for (size_t i = 0; i < values.size(); i++)
            {
                out << delim << values[i];
                delim = ",";
            }
        }
    };

    class xml_composite : public xml_base
    {
    private:
        xml_map children;
    public:
        void dump(std::ostream& out, int offset = -1) override
        {
            offset++;
            bool first = true;
            static int offset_temp;
            std::string spaces(offset * 4, ' ');
            if (offset!=0) out << "\n";
            for (const auto& it : children)
            {
                if (first)
                {
                    out << spaces << "<" << it.first << ">";
                    first = false;
                }
                else
                    out << "\n" << spaces << "<" << it.first << ">";

                offset_temp = offset;
                it.second->dump(out, offset);

                std::string spaces_behind(0, ' ');
                if (offset_temp != offset)
                    spaces_behind = spaces;
                out << spaces_behind << "</" << it.first << ">";
                if (offset == 1)
                {
                    out << spaces << "\n";
                }
            };

            if (offset > 0)
            {
                out << spaces << "\n";
                offset--;
            }
        }

        template<class Type>
        void add(xml_key key, Type value)
        {
            children[key] = std::make_shared<xml_leaf<Type>>(value);
        }
        void add(xml_key key, xml_composite comp)
        {
            children[key] = std::make_shared<xml_composite>(comp);
        }
        template<class Type>
        void add(xml_key key, std::vector<Type> array)
        {
            children[key] = std::make_shared<xml_basic_array<Type>>(array);
        }
    };


}

