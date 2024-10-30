# How to test the OpenVINO™ JavaScript API

## Build the OpenVINO™ JavaScript API
For detailed build instructions, refer to the [OpenVINO™ JavaScript API documentation](./README.md).


## Run OpenVINO™ JavaScript API tests
*For simplicity, begin by navigating to the [main JavaScript API folder](./../node):*
```shell
cd <openvino_repo>/src/bindings/js/node
```

Use this command to run OpenVINO JavaScript API tests:
```shell
npm run test
```

To run specific test files, you can pass one or more glob patterns:
```shell
node --test "tests/unit/core.test.js" "tests/unit/*model.test.js"
```

Before executing individual test files, a one-time setup is required. If you have not previously executed `npm run test`, initiate the setup by running the following command:

```shell
npm run test_setup
```

More information on running tests from the command line can be found in the [Node.js documentation]( https://nodejs.org/docs/latest/api/test.html#running-tests-from-the-command-line).


## Check the code style of the JavaScript API
*ESLint* is a tool to enforce a consistent coding style and to identify and fix potential issues in JavaScript code.

To check the code style of the JavaScript API, run the following commands:
```shell
npm run lint
```
*ESLint* can automatically fix many of the issues it detects. Use following command to run *ESLint* with the fix option on a single file:
```shell
npx eslint --fix "tests/unit/core.test.js"
```

It is recommended to run the code style check each time new tests are added.


## Writing OpenVINO™ JavaScript API tests

### Before start
Follow and complete [Examples of OpenVINO™ JavaScript API code](./code_examples.md).


### Adding new test-case in the correct place
Each new test should verify the correct behavior of the new functionality (e.g. class, method).

Unit test files are located in the `<openvino_repo>/src/bindings/js/node/tests/unit/` directory and their names correspond to the class/module to be tested.

Always add tests to the correct locations and create new files only when necessary. *Remember to include the license on top of each new file*.


### Test writing guidelines
Each test file starts with a `describe` block to group all tests related to a specific class or module. The name of the `describe` block should match the name of the class or module being tested, for example *ov.Core tests*.

Within the `describe` block, individual tests are defined using `test` or `it` blocks, with the name of the test reflecting what is being tested. If multiple tests relate to the same method, they can be grouped within a nested `describe` block.

 ```js
 const { describe, it, beforeEach } = require('node:test');
describe('ov.Tensor tests', () => {
  test('MyTensor.getElementType()', () => {
    const tensor = new ov.MyTensor(ov.element.f32, [1, 3]);
    assert.strictEqual(tensor.getElementType(), ov.element.f32);
  });

});
 ```
When writing tests, keep the following best practices in mind:

 * Focus on testing general usage and edge cases, but avoid excessive testing that leads to redundant test cases and can slow down validation pipelines.
  * Avoid initializing shared variables outside of `test` or `beforeEach` blocks to prevent tests from affecting each other.
 * Do not test built-in language features; instead, focus on the functionality provided by your code.
 * Use hardcoded expected results to verify the correctness of your code. Alternatively, generate reference values at runtime using reliable libraries or methods.
 * Extract and reuse common code snippets, such as helper functions or setup code, to improve test readability and maintainability.


## See also
 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO™ bindings README](../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
