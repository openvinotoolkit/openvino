// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "backend/gna_types.h"
#include "nnet_base_matcher.hpp"
#include "frontend/quantization.h"

using TranspozedData = std::tuple<std::vector<uint16_t>*, int, int>;

class TranspozeIterator {
    std::pair<int, int> dims;
    int _offset = 0;
    int _row = 0;
    int _col = 0;
    int _outputRow = 0;
 public :
    TranspozeIterator(const std::pair<int, int> & dims) : dims(std::move(dims)) {
    }
    TranspozeIterator(const TranspozedData & data) : TranspozeIterator({std::get<1>(data), std::get<2>(data)}) {
    }

    TranspozeIterator operator ++ (int) {
        TranspozeIterator c(*this);
        this->operator++();
        return c;
   }

   void reset() {
       _offset = 0;
       _row = 0;
       _col = 0;
       _outputRow = 0;
    }

   //  prefix form
   TranspozeIterator& operator ++ () {
        if (dims.first == 0 || dims.second == 0) {
            _offset ++;
        } else {
            // step over whole row length
            _row++;
            // once number of rows hit max value
            if (_row == dims.second) {
                // increment offset within row
                _col++;
                // restart from first row
                _row = 0;
                // restart from next output channel
                if (_col == dims.first) {
                    _outputRow++;
                    _col = 0;
                }
            }
            _offset = _col + _row * dims.first  +  _outputRow * dims.first * dims.second;
        }
        return *this;
    }
    // getting index
    operator int() {
         return _offset;
    }
    int row() const noexcept {
        return _row;
    }
    int col() const noexcept {
        return _col;
    }
    int outputRow() const noexcept{
        return _outputRow;
    }
};

class WeightsMatcher : public ::testing::MatcherInterface<const gna_nnet_type_t*> {
    enum HowMatch{
        eNone,
        eEq,
        eTranspozed
    } eMatchKind;
    TranspozedData  transpozedData;

    mutable std::stringstream error;
    mutable TranspozeIterator iterator;
    mutable int actual;
 public:
    explicit WeightsMatcher(const TranspozedData & data) :
        eMatchKind(eTranspozed),
        transpozedData(data),
        iterator(data) {
        if (0 == std::get<1>(transpozedData) || 0 == std::get<2>(transpozedData)) {
            eMatchKind = eEq;
        }
    }
    bool MatchAndExplain(const gna_nnet_type_t *foo, ::testing::MatchResultListener *listener) const override {
        if (foo == nullptr)
            return false;
        iterator.reset();

        for(int i = 0; i < foo->nLayers; i++) {
            if (foo->pLayers[i].nLayerKind != INTEL_AFFINE &&
                foo->pLayers[i].nLayerKind != INTEL_AFFINE_DIAGONAL) continue;

            auto affine = (gna_affine_func_t*)foo->pLayers[i].pLayerStruct;

            auto affineWeightsSize = foo->pLayers[i].nOutputRows *
                (foo->pLayers[i].nLayerKind == INTEL_AFFINE_DIAGONAL ? 1 : foo->pLayers[i].nInputRows);

            if (affineWeightsSize != std::get<0>(transpozedData)->size()) {
                error << "gna-xnn layer(" << i << ") weights size mismatch: expected "
                      << std::get<0>(transpozedData)->size() << ", but was: " << affineWeightsSize;
                break;
            }

            auto pWeights = reinterpret_cast<uint16_t *>(affine->pWeights);

            for (int i = 0; i != affineWeightsSize; i++, iterator++) {
                auto savedVal = (&std::get<0>(transpozedData)->front())[iterator];
                if (pWeights[i] != savedVal) {
                    actual = pWeights[i];
                    return false;
                }
            }
            return true;
        }
        return false;
    };
    void DescribeTo(::std::ostream *os) const override {
        *os << error.str() << std::endl;
        if (eMatchKind == eEq) {
            *os << "weights of affine layers are not equal, error at: ";
        } else {
            *os << "weights of affine layers are not transpozed, error at: ";
        }
        *os << (int)iterator << ", actual=" << actual<<", expected=" << (&std::get<0>(transpozedData)->front())[iterator];
    }
};

class WeightsSizeMatcher : public ::testing::MatcherInterface<const gna_nnet_type_t*> {
    enum HowMatch{
        eNone,
        eEqAffine,
    } eMatchKind;

    mutable std::stringstream error;
    mutable int actual;
    size_t expected_weights_size;
 public:
    explicit WeightsSizeMatcher(const size_t data_len) :
        eMatchKind(eEqAffine),
        expected_weights_size(data_len){
    }
    bool MatchAndExplain(const gna_nnet_type_t *foo, ::testing::MatchResultListener *listener) const override {
        if (foo == nullptr)
            return false;

        size_t sizeTotal = 0;
        std::stringstream ss;
        for(int i = 0; i < foo->nLayers; i++) {
            if (foo->pLayers[i].nLayerKind != INTEL_AFFINE && eMatchKind == eEqAffine) continue;

            auto affineWeightsSize = foo->pLayers[i].nOutputRows *
                (foo->pLayers[i].nLayerKind == INTEL_AFFINE_DIAGONAL ? 1 : foo->pLayers[i].nInputRows);

            sizeTotal += affineWeightsSize;
            ss << "[" << i << "]: " << affineWeightsSize << ", ";

        }

        if (eMatchKind == eEqAffine &&  sizeTotal != expected_weights_size) {
            error << "gna-affine layers " << ss.str() << " have diff total weights size : " << sizeTotal
                  << ", while expected to have: " << expected_weights_size << "\n";
            return false;
        }
        return true;
    };
    void DescribeTo(::std::ostream *os) const override {
        *os << error.str() << std::endl;
    }
};


class WeightsSaver: public ::testing::MatcherInterface<const gna_nnet_type_t*> {
    mutable TranspozeIterator iterator;
    std::vector<uint16_t>* weights;
 public:
    explicit WeightsSaver(TranspozedData data) :
        weights(std::get<0>(data)), iterator(data) {
    }
    bool MatchAndExplain(const gna_nnet_type_t *foo, ::testing::MatchResultListener *listener) const override {
        if (foo == nullptr)
            return false;
        for(int i = 0; i < foo->nLayers; i++) {
            if (foo->pLayers[i].nLayerKind != INTEL_AFFINE) continue;

            auto affine = (gna_affine_func_t*)foo->pLayers[i].pLayerStruct;

            auto affineWeightsSize = foo->pLayers[i].nOutputRows * foo->pLayers[i].nInputRows;
            auto pWeights = reinterpret_cast<uint16_t *>(affine->pWeights);
            weights->resize(affineWeightsSize);

            for (int i=0; i != affineWeightsSize; i++, ++iterator) {
                (*weights)[i] = pWeights[iterator];
            }

            return true;
        }
        return false;
    };
    void DescribeTo(::std::ostream *os) const override {
        *os << "affine layer not found";
    }
};


void HasWeightsTranspozed(std::unique_ptr<NNetComponentMatcher>& components,  std::vector<uint16_t>* data, std::pair<int, int> dims) {
    components->add(new WeightsMatcher(make_tuple(data, dims.first, dims.second)));
}

void HasWeightsEq(std::unique_ptr<NNetComponentMatcher>& components,  std::vector<uint16_t>* data) {
    components->add(new WeightsMatcher(make_tuple(data, 0, 0)));
}

void SaveWeights(std::unique_ptr<NNetComponentMatcher>& components,  std::vector<uint16_t>* data, std::pair<int, int> dims) {
    components->add(new WeightsSaver(make_tuple(data, dims.first, dims.second)));
}

void HasWeightsSizeEq(std::unique_ptr<NNetComponentMatcher>& components,  size_t weights_size) {
    components->add(new WeightsSizeMatcher(weights_size));
}

