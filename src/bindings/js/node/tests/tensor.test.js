// -*- coding: utf-8 -*-
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const { addon: ov } = require('..');
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
  assert.throws( () => new ov.Tensor(ov.element.f32, shape, data, params),
    {message: /Invalid number of arguments for Tensor constructor./});
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

  it('Create string tensor', () => {
    const str_arr = ['text', 'more text', 'even more text'];
    const tensor = new ov.Tensor(str_arr);
    assert.deepStrictEqual(tensor.data, str_arr);
  });

  it('Create string tensor', () => {
    const str_arr = ['text', 'more text', 'even more text'];
    const tensor = new ov.Tensor(str_arr);
    assert.deepStrictEqual(tensor.data, str_arr);
  });

  it('String tensor - passed array does not contain string elements', () => {
    const str_arr = ['text', true];
    assert.throws(() => { new ov.Tensor(str_arr);},
      /The array passed to create string tensor must contain only strings./
    );
  });

  it('Set string tensor data', () => {
    const str_arr = ['H', 'e', 'l', 'l', 'o'];
    const tensor = new ov.Tensor(ov.element.string, [1, 1, 1, 5]);
    tensor.data = str_arr;
    assert.deepStrictEqual(tensor.data, str_arr);
  });

  it('Test tensor getData()', () => {
    const tensor = new ov.Tensor(ov.element.f32, shape, data);
    assert.deepStrictEqual(tensor.getData(), data);
  });

  it('Test tensor.data setter - different element type throws', () => {
    const float64_data = Float64Array.from([1, 2, 3] );
    const tensor = new ov.Tensor(ov.element.f32, [1, 3]);
    assert.throws(() => {
      tensor.data = float64_data;},
    /Passed array must have the same size as the Tensor!/
    );
  });

  it('Test tensor.data setter - different element length throws', () => {
    const float64_data = Float64Array.from([1, 2, 3] );
    const tensor = new ov.Tensor(ov.element.f64, [1, 2]);
    assert.throws(() => {
      tensor.data = float64_data;},
    /Passed array must have the same size as the Tensor!/
    );
  });

  it('Test tensor.data setter', () => {
    const testString = 'test';
    const tensor = new ov.Tensor(ov.element.f64, [1, 2]);
    assert.throws(() => {
      tensor.data = testString;},
    /Passed argument must be TypedArray, or Array if the tensor type is string./
    );
  });

  it('Test tensor.data setter', () => {
    const tensor = new ov.Tensor(ov.element.f32, shape);
    tensor.data = data;
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
      {message: /Third argument of a tensor must be TypedArray./});
  });

  it('Third argument of a tensor cannot be an array object', () => {
    assert.throws(
      () => new ov.Tensor(ov.element.f32, shape, [1, 2, 3, 4]),
      {message: /Third argument of a tensor must be TypedArray./});
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
      {message: /Passed array must contain only numbers/});
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
      /Passed argument must be an Int32Array or a Uint32Array./
    );
  });

  it('Cannot create ov::Shape from ArrayBuffer', () => {
    const shape = Int32Array.from([1, 224, 224, 3]);
    assert.throws(
      () => new ov.Tensor(ov.element.f32, shape.buffer, data),
      /Passed argument must be of type Array or TypedArray./
    );
  });

  it('getShape() method does not accept parameters', () => {
    const tensor = new ov.Tensor(ov.element.f32, [1, 3, 224, 224], data);
    assert.throws(
      () => tensor.getShape(1, 2, 3),
      { message: 'No parameters are allowed for the getShape() method.'}
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

describe('Tensor getSize', () => {

  it('getSize returns the correct total number of elements', () => {
    const tensor = new ov.Tensor(ov.element.f32, shape, data);
    const expectedSize = shape.reduce((acc, dim) => acc * dim, 1);
    assert.strictEqual(tensor.getSize(), expectedSize);
  });

  it('getSize should throw an error if arguments are provided', () => {
    const tensor = new ov.Tensor(ov.element.f32, shape, data);
    assert.throws(
      () => tensor.getSize(1),
      { message: 'getSize() does not accept any arguments.' }
    );
  });
});

describe('Tensor getSize for various shapes', () => {

  it('calculates size correctly for a common image data shape [3, 224, 224]', () => {
    const shape = [3, 224, 224];
    const expectedSize = 3*224*224;
    const tensorData = new Float32Array(expectedSize).fill(0);
    const tensor = new ov.Tensor(ov.element.f32, shape, tensorData);
    assert.strictEqual(tensor.getSize(), expectedSize);
  });

  it('calculates size correctly for a scalar wrapped in a tensor [1]', () => {
    const shape = [1];
    const expectedSize = 1;
    const tensorData = new Float32Array(expectedSize).fill(0);
    const tensor = new ov.Tensor(ov.element.f32, shape, tensorData);
    assert.strictEqual(tensor.getSize(), expectedSize);
  });

  it('calculates size correctly for a vector [10]', () => {
    const shape = [10];
    const expectedSize = 10;
    const tensorData = new Float32Array(expectedSize).fill(0);
    const tensor = new ov.Tensor(ov.element.f32, shape, tensorData);
    assert.strictEqual(tensor.getSize(), expectedSize);
  });
});
