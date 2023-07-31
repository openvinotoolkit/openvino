declare module 'openvino' {
  export class Core {
    constructor();
    compileModel(): CompiledModel;
    readModelAsync(): Promise<CompiledModel>;
    readModel(): Model;
  }

  export class Model {
    outputs: Output[];
    inputs: Output[];

    // Constructor isn't available from JS side
    // constructor();
    output(nameOrId?: string | number): Output;
    getName(): string;
  }

  export class CompiledModel {
    outputs: Output[];
    inputs: Output[];
    // Constructor isn't available from JS side
    // constructor();
    output(nameOrId?: string | number);
    input(nameOrId?: string | number);
    createInferRequest(): InferRequest;
  }

  export class Tensor {
    // FIXME: now its only Float32Array
    data: number[];
    // FIXME: does empty constructor use?
    constructor();
    // FIXME: now tensorData can be only Float32Array
    constructor(type: element, shape: number[], tensorData: number[]);
    getPrecision(): element;
    getShape(): Shape;
    // FIXME: now it returns Float32Array
    getData(): number[];
  }

  export class Shape {
    data: number[];
    // FIXME: does empty constructor use?
    constructor();
    constructor(dimensions: number, data: number[]);
    getData(): number[];
    shapeSize(): number;
    getDim(): number;
  }

  export class InferRequest {
    // Constructor isn't available from JS side
    // constructor();
    // FIXME: are we going to add index parameter for this method?
    getOutputTensor(): Tensor;
    getOutputTensors(): Tensor[];
    getTensor(output: Output): Tensor;
    infer(inputData?: { [inputName: string]: Tensor });
    setInputTensor(tensor: Tensor);
  }

  export class Input {
    shape: Shape;
    // Constructor isn't available from JS side
    // constructor();
    getShape(): Shape;
  }

  export class Output {
    anyName: string;
    shape: number[];
    // Constructor isn't available from JS side
    // constructor();
    toString(): string;
    getAnyName(): string;
    getShape(): Shape;
    // FIXME: These methods are not available from JS side for some reason
    // (const ov::Node & ov::Node)
    setNames(names: string[]);
    getNames(): string[];
  }

  export class PrePostProcessor {
    constructor();
    // FIXME: should we return this after build() call?
    build(): PrePostProcessor;
    setInputElementType(idx: number, type: element);
    setInputModelLayout(layout: string[]);
    // FIXME: are we going to add index parameter for this method?
    setInputTensorLayout(layout: string[]);
    preprocessResizeAlgorithm(resizeAlgorithm: resizeAlgorithms);
    setInputTensorShape(shape: number[]): PrePostProcessor;
  }

  export enum element {
    u8,
    u32,
    u16,
    i8,
    i64,
    i32,
    i16,
    f64,
    f32,
  }

  export enum resizeAlgorithms {
    RESIZE_NEAREST,
    RESIZE_CUBIC,
    RESIZE_LINEAR,
  }
}
