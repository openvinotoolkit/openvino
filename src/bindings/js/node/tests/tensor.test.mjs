import {test, describe, it} from 'node:test';
import assert from 'node:assert';

import { addon } from '../dist/index.js';

const shape = [1, 3, 224, 224];
const elemNum = 1 * 3 * 224 * 224;
const data = Float32Array.from({length: elemNum}, () => Math.random() );

test('Test for number of arguments in tensor', () => {
  assert.throws( () => new addon.Tensor(addon.element.f32, shape),
    {message: 'Invalid number of arguments for Tensor constructor.'});
});

describe('Tensor data', () => {

  it('Set tensor data with Float32Array', () => {
    const tensor = new addon.Tensor('f32', shape, data);
    assert.deepStrictEqual(tensor.data, data);
  });

  it('Set tensor data with Float32Array created from ArrayBuffer', () => {
    const size = elemNum * 4;
    const buffer = new ArrayBuffer(size);
    const view = new Float32Array(buffer);
    view.set(data);
    const tensor = new addon.Tensor(addon.element.f32, shape, view);
    assert.deepStrictEqual(tensor.data, data);
  });

  it('Set tensor data with too big Float32Array', () => {
    const size = elemNum * 8;
    const buffer = new ArrayBuffer(size);
    const view = new Float32Array(buffer);
    view.set(data);
    assert.throws( () => new addon.Tensor(addon.element.f32, shape, view),
      {message: 'Invalid tensor argument. Memory allocated using shape '
          + 'and element::type mismatch passed data\'s size'});
  });

  it('Third argument of a tensor cannot be an ArrayBuffer', () => {
    assert.throws(
      () => new addon.Tensor(addon.element.f32, shape, new ArrayBuffer(1234)),
      {message: 'Third argument of a tensor must be of type TypedArray.'});
  });

  it('Third argument of a tensor cannot be an array object', () => {
    assert.throws(
      () => new addon.Tensor(addon.element.f32, shape, [1, 2, 3, 4]),
      {message: 'Third argument of a tensor must be of type TypedArray.'});
  });
});

describe('Tensor shape', () => {

  it('ov::Shape from an array object', () => {
    const tensor = new addon.Tensor(addon.element.f32, [1, 3, 224, 224], data);
    assert.deepStrictEqual(tensor.getShape().data, [1, 3, 224, 224]);
  });

  it('ov::Shape from an array object with floating point numbers', () => {
    const tensor =
    new addon.Tensor(addon.element.f32, [1, 3.0, 224.8, 224.4], data);
    assert.deepStrictEqual(tensor.getShape().data, [1, 3, 224, 224]);
  });

  it('Array argument to create ov::Shape can only contain numbers', () => {
    assert.throws(
      () => new addon.Tensor(addon.element.f32, ['1', 3, 224, 224], data),
      {message: 'Invalid tensor argument. '
    + 'Passed array must contain only numbers.'});
  });

  it('ov::Shape from TypedArray -> Int32Array', () => {
    const shp = Int32Array.from([1, 224, 224, 3]);
    const tensor = new addon.Tensor(addon.element.f32, shp, data);
    assert.deepStrictEqual(tensor.getShape().data, [1, 224, 224, 3]);
  });

  it('Cannot create ov::Shape from Float32Array', () => {
    const shape = Float32Array.from([1, 224, 224, 3]);
    assert.throws(
      () => new addon.Tensor(addon.element.f32, shape, data),
      /Invalid tensor argument./
    );
  });

  it('Cannot create ov::Shape from ArrayBuffer', () => {
    const shape = Int32Array.from([1, 224, 224, 3]);
    assert.throws(
      () => new addon.Tensor(addon.element.f32, shape.buffer, data),
      /Invalid tensor argument./
    );
  });
});

describe('Tensor element type', () => {
  const ets = [ [addon.element.f32, 'f32'],
    [addon.element.i32, 'i32'],
    [addon.element.u32, 'u32'] ];

  ets.forEach(([elemType, val]) => {
    it(`Comparison of ov.element.${elemType} to string ${val}`, () => {
      assert.strictEqual(elemType, val);
    });
  });

  ets.forEach(([elemType, ..._]) => {
    it(`Comparison of ov.element ${elemType} got from Tensor object`, () => {
      const tensor = new addon.Tensor(elemType, shape, data);
      assert.strictEqual(tensor.getPrecision(), elemType);
    });
  });
});
