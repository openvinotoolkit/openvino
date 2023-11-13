// -*- coding: utf-8 -*-
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const ov = require('../build/Release/ov_node_addon.node');
const assert = require('assert');
const { test, describe, it } = require('node:test');
const getRandomBigInt = require('random-bigint');

const shape = [1, 3, 224, 224];
const elemNum = 1 * 3 * 224 * 224;
const data = Float32Array.from({ length: elemNum }, () => Math.random() );
const params = [
  [ov.element.i8, 'i8', Int8Array.from({ length: elemNum }, () => Math.random() )],
  [ov.element.u8, 'u8', Uint8Array.from({ length: elemNum }, () => Math.random() )],
  [ov.element.i16, 'i16', Int16Array.from({ length: elemNum }, () => Math.random() )],
  [ov.element.u16, 'u16', Uint16Array.from({ length: elemNum }, () => Math.random() )],
  [ov.element.i32, 'i32', Int32Array.from({ length: elemNum }, () => Math.random() )],
  [ov.element.u32, 'u32', Uint32Array.from({ length: elemNum }, () => Math.random() )],
  [ov.element.f32, 'f32', Float32Array.from({ length: elemNum }, () => Math.random() )],
  [ov.element.f64, 'f64', Float64Array.from({ length: elemNum }, () => Math.random() )],
  [ov.element.i64, 'i64', BigInt64Array.from({ length: elemNum }, () => getRandomBigInt(10) )],
  [ov.element.u64, 'u64', BigUint64Array.from({ length: elemNum }, () => getRandomBigInt(10) )],
];

test('Test for number of arguments in tensor', () => {
  assert.throws( () => new ov.Tensor(ov.element.f32),
    {message: 'Invalid number of arguments for Tensor constructor.'});
});

describe('Tensor without data parameters', () => {
  it('Tensor should have array with zeros and numbers of elements according to the shape', () => {
    const tensor = new ov.Tensor(ov.element.f32, shape);
    assert.strictEqual(tensor.data.length, elemNum);
  });
});

describe('Tensor data', () => {

  params.forEach(([type, stringType, data]) => {
    it(`Set tensor data with ${stringType} element type`, () => {
      const tensor = new ov.Tensor(type, shape, data);
      assert.deepStrictEqual(tensor.data, data);
    });
  });

  it('Test tensor getData()', () => {
    const tensor = new ov.Tensor(ov.element.f32, shape, data);
    assert.deepStrictEqual(tensor.getData(), data);
  });

  it('Set tensor data with Float32Array created from ArrayBuffer', () => {
    const size = elemNum * 4;
    const buffer = new ArrayBuffer(size);
    const view = new Float32Array(buffer);
    view.set(data);
    const tensor = new ov.Tensor(ov.element.f32, shape, view);
    assert.deepStrictEqual(tensor.data, data);
  });

  it('Set tensor data with too big Float32Array', () => {
    const size = elemNum * 8;
    const buffer = new ArrayBuffer(size);
    const view = new Float32Array(buffer);
    view.set(data);
    assert.throws( () => new ov.Tensor(ov.element.f32, shape, view),
      {message: /Memory allocated using shape and element::type mismatch/});
  });

  it('Third argument of a tensor cannot be an ArrayBuffer', () => {
    assert.throws(
      () => new ov.Tensor(ov.element.f32, shape, new ArrayBuffer(1234)),
      {message: 'Third argument of a tensor must be of type TypedArray.'});
  });

  it('Third argument of a tensor cannot be an array object', () => {
    assert.throws(
      () => new ov.Tensor(ov.element.f32, shape, [1, 2, 3, 4]),
      {message: 'Third argument of a tensor must be of type TypedArray.'});
  });
});

describe('Tensor shape', () => {

  it('ov::Shape from an array object', () => {
    const tensor = new ov.Tensor(ov.element.f32, [1, 3, 224, 224], data);
    assert.deepStrictEqual(tensor.getShape(), [1, 3, 224, 224]);
  });

  it('ov::Shape from an array object with floating point numbers', () => {
    const tensor =
    new ov.Tensor(ov.element.f32, [1, 3.0, 224.8, 224.4], data);
    assert.deepStrictEqual(tensor.getShape(), [1, 3, 224, 224]);
  });

  it('Array argument to create ov::Shape can only contain numbers', () => {
    assert.throws(
      () => new ov.Tensor(ov.element.f32, ['1', 3, 224, 224], data),
      {message: 'Invalid tensor argument. '
    + 'Passed array must contain only numbers.'});
  });

  it('ov::Shape from TypedArray -> Int32Array', () => {
    const shp = Int32Array.from([1, 224, 224, 3]);
    const tensor = new ov.Tensor(ov.element.f32, shp, data);
    assert.deepStrictEqual(tensor.getShape(), [1, 224, 224, 3]);
  });

  it('Cannot create ov::Shape from Float32Array', () => {
    const shape = Float32Array.from([1, 224, 224, 3]);
    assert.throws(
      () => new ov.Tensor(ov.element.f32, shape, data),
      /Invalid tensor argument./
    );
  });

  it('Cannot create ov::Shape from ArrayBuffer', () => {
    const shape = Int32Array.from([1, 224, 224, 3]);
    assert.throws(
      () => new ov.Tensor(ov.element.f32, shape.buffer, data),
      /Invalid tensor argument./
    );
  });
});

describe('Tensor element type', () => {
  params.forEach(([elemType, val]) => {
    it(`Comparison of ov.element.${elemType} to string ${val}`, () => {
      assert.strictEqual(elemType, val);
    });
  });

  params.forEach(([elemType, , data]) => {
    it(`Comparison of ov.element ${elemType} got from Tensor object`, () => {
      const tensor = new ov.Tensor(elemType, shape, data);
      assert.strictEqual(tensor.getElementType(), elemType);
    });
  });
});
