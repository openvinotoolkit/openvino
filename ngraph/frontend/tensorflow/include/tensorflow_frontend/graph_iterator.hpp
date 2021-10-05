// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tensorflow_frontend/decoder.hpp>

namespace ngraph {
namespace frontend {
/// Abstract representation for an input model graph that gives nodes in topologically sorted order
class GraphIterator {
public:
    typedef std::shared_ptr<GraphIterator> Ptr;

    /// \brief Get a number of operation nodes in the graph
    virtual size_t size() const = 0;

    /// \brief Set iterator to the start position
    virtual void reset() = 0;

    /// \brief Move to the next node in the graph
    virtual void next() = 0;

    /// \brief Returns true if iterator goes out of the range of available nodes
    virtual bool is_end() const = 0;

    /// \brief Return a pointer to a decoder of the current node
    virtual std::shared_ptr<::ngraph::frontend::DecoderBase> get_decoder() const = 0;
};

}  // namespace frontend
}  // namespace ngraph
