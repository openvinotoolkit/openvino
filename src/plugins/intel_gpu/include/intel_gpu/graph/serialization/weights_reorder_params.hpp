#pragma once

#include "intel_gpu/primitives/input_layout.hpp"
#include <intel_gpu/runtime/memory.hpp>

namespace cldnn {
    struct WeightsReorderParams {
        WeightsReorderParams() {}

        WeightsReorderParams(const layout& in_layout, const layout& out_layout, bool transposed = false, bool grouped = false)
            : _in_layout(in_layout),
              _out_layout(out_layout),
              _transposed(transposed),
              _grouped(grouped) {}

        size_t hash() const {
            size_t seed = hash_combine(_in_layout.hash(), _out_layout.hash());
            seed = hash_combine(seed, _transposed);
            seed = hash_combine(seed, _grouped);
            return seed;
        }

        bool operator==(const WeightsReorderParams& rhs) const {
            if (typeid(*this) != typeid(rhs))
                return false;

            return _in_layout == rhs._in_layout &&
                   _out_layout == rhs._out_layout &&
                   _transposed == rhs._transposed &&
                   _grouped == rhs._grouped;
        }

        layout get_input_layout() const { return _in_layout; }
        layout get_output_layout() const { return _out_layout; }
        bool should_be_transposed() const { return _transposed; }
        bool get_grouped() const { return _grouped; }

        void set_input_layout(const layout& layout) { _in_layout = layout; }
        void set_output_layout(const layout& layout) { _out_layout = layout; }

        virtual void save(cldnn::BinaryOutputBuffer& ob) const {
            ob << _in_layout;
            ob << _out_layout;
            ob << _transposed;
            ob << _grouped;
        }
        virtual void load(cldnn::BinaryInputBuffer& ib) {
            ib >> _in_layout;
            ib >> _out_layout;
            ib >> _transposed;
            ib >> _grouped;
        }
        virtual ~WeightsReorderParams() = default;

    protected:
        layout _in_layout;
        layout _out_layout;
        bool _transposed;
        bool _grouped;
    };

#ifdef ENABLE_ONEDNN_FOR_GPU
    namespace onednn {
        struct WeightsReorderParamsOneDNN : public cldnn::WeightsReorderParams {
            WeightsReorderParamsOneDNN() : cldnn::WeightsReorderParams() {}
            WeightsReorderParamsOneDNN(const layout& in_layout,
                                       const layout& out_layout,
                                       const dnnl::memory::desc& in_desc,
                                       const dnnl::memory::desc& out_desc,
                                       bool transposed,
                                       bool grouped = false)
                : WeightsReorderParams(in_layout, out_layout, transposed, grouped)
                , _in_desc(std::make_shared<dnnl::memory::desc>(in_desc))
                , _out_desc(std::make_shared<dnnl::memory::desc>(out_desc)) {}

            std::shared_ptr<dnnl::memory::desc> _in_desc = std::make_shared<dnnl::memory::desc>();
            std::shared_ptr<dnnl::memory::desc> _out_desc = std::make_shared<dnnl::memory::desc>();

            void save(cldnn::BinaryOutputBuffer& ob) const override {
                cldnn::WeightsReorderParams::save(ob);
                dnnl::memory::desc inTemp(*_in_desc);
                std::vector<uint8_t> in = inTemp.get_blob();
                ob << in;
                dnnl::memory::desc outTemp(*_out_desc);
                std::vector<uint8_t> out = outTemp.get_blob();
                ob << out;
            }
            void load(cldnn::BinaryInputBuffer& ib) override {
                cldnn::WeightsReorderParams::load(ib);
                std::vector<uint8_t> in;
                ib >> in;
                _in_desc = std::make_shared<dnnl::memory::desc>(in);
                std::vector<uint8_t> out;
                ib >> out;
                _out_desc = std::make_shared<dnnl::memory::desc>(out);
            }
        };
    }
#endif
}
