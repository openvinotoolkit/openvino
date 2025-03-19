// Copyright (C) 2024 Intel Corporationov::npuw::
// SPDX-License-Identifier: Apache-2.0
//

#include "repeated.hpp"

#include <sstream>
#include <tuple>

#include "../../logging.hpp"

using ov::npuw::online::Interconnect;
using ov::npuw::online::MetaInterconnect;
using ov::npuw::online::Repeated;

bool Repeated::Archetype::operator==(const Repeated::Archetype& other) const {
    if (metadesc != other.metadesc) {
        return false;
    }

    if (reptrack != other.reptrack) {
        return false;
    }

    return true;
}

void Repeated::exclude() {
    m_excluded = true;
}

void Repeated::resetExclude() {
    m_excluded = false;
}

bool Repeated::openForMerge() const {
    return !m_excluded;
}

bool Interconnect::operator==(const Interconnect& other) const {
    return input_port == other.input_port && output_port == other.output_port && input_node == other.input_node &&
           output_node == other.output_node;
}

bool MetaInterconnect::operator==(const MetaInterconnect& other) const {
    return other.input_meta == input_meta && other.output_meta == output_meta && other.input_port == input_port &&
           other.output_port == output_port && other.input_reptrack == input_reptrack &&
           other.output_reptrack == output_reptrack;
}

bool MetaInterconnect::operator<(const MetaInterconnect& other) const {
    return std::make_tuple(input_meta, input_port, input_reptrack, output_port, output_meta, output_reptrack) <
           std::make_tuple(other.input_meta,
                           other.input_port,
                           other.input_reptrack,
                           other.output_port,
                           other.output_meta,
                           other.output_reptrack);
}
