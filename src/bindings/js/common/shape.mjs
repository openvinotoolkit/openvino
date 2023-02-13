import { jsTypeByPrecisionMap, heapLabelByTypeMap } from './types.mjs';

const defaultPrecision = 'uint16';
const TYPE = jsTypeByPrecisionMap[defaultPrecision];
const PRECISION = heapLabelByTypeMap[TYPE.name];

export default class Shape {
  #dimensions;

  static PRECISION = PRECISION;
  static TYPE = TYPE;

  constructor(...dimensions) {
    if (Array.isArray(dimensions[0])) dimensions = dimensions[0];

    this.#dimensions = new Shape.TYPE(dimensions.length);

    dimensions.map((d, i) => this.#dimensions[i] = d);
  }

  get dim() {
    return this.#dimensions.length;
  }

  get data() {
    return this.#dimensions;
  }

  static parse(ov, originalShape) {
    const originalDim = originalShape.getDim();
    const originalDataPointer = originalShape.getData();

    const dimensions = new Shape.TYPE(originalDim);

    for (let i = 0; i < originalDim; i++) {
      const dimension = ov[Shape.PRECISION][originalDataPointer/Shape.TYPE.BYTES_PER_ELEMENT + i];
      
      dimensions[i] = dimension;
    }

    return new Shape(...dimensions);
  }

  static convert(ov, shape) {
    const originalDimensions = new Shape.TYPE(shape.data);
    const elementSizeInBytes = originalDimensions.BYTES_PER_ELEMENT;
    const heapSpace = ov._malloc(originalDimensions.length*elementSizeInBytes);
    const offset = Math.sqrt(elementSizeInBytes);
    ov[Shape.PRECISION].set(originalDimensions, heapSpace>>offset); 
    
    return { obj: new ov.Shape(heapSpace, shape.dim), free: () => ov._free(heapSpace) };
  }

  convert(ov) {
    return Shape.convert(ov, this);
  }
};

/*

// const s = new Shape([1, 224, 224, 3]);
const shapeExample = new Shape(1, 224, 224, 3);

shapeExample.dim; // int of dimensions count
shapeExample.data; // typed array

const arrayData = [1, 2, 3, 4];
// const tensorExample = new Tensor(new UInt8Arrray(arrayData), new Shape(2, 2));
// const tensorExample = new Tensor('uint8', arrayData, new Shape(2, 2));
const tensorExample = new Tensor(arrayData, new Shape(2, 2)); // by default array type will = float16

const model;

const output = model.infer(tensorExample); // output is Tensor type

output.shape; // Shape type
output.data; // typed array

*/
