//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

namespace ngraph
{
    namespace test
    {
        /// These templates should be specialized for each test engine and they should contain
        /// a "static constexpr const bool value" member set to true or false.
        /// These traits are used in engine_factory.hpp

        /// Indicates that a given Engine can be constructed for different devices (IE engines)
        template <typename Engine>
        struct supports_devices;

        /// Indicates that a given Engine supports dynamic shapes
        template <typename Engine>
        struct supports_dynamic;

        /// Example:
        ///
        // template <>
        // struct supports_dynamic<EngineName> {
        //     static constexpr const bool value = true;
        // };
    }
}
