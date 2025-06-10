/**
 * Unit tests for the action's main functionality, src/restoreImpl.js
 */
const core = require("@actions/core");
const path = require("path");
const os = require("os");
const fs = require("fs");
const findWheelImpl = require("../src/index.js");

const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "test-find-wheel"));
const wheelsPath = path.join(tempDir, "wheels");

const wheels = [
  "package_x-1.0.0-cp310-cp310-linux_x86_64.whl",
  "package_x-1.0.0-cp311-cp311-linux_x86_64.whl",
  "package_y-1.0.0-cp311-cp311-linux_x86_64.whl",
  "package_y-2.0.0-cp311-cp311-linux_x86_64.whl",
];

// Mock the GitHub Actions core library
const getInputMock = jest.spyOn(core, "getInput").mockImplementation();
const setOutputMock = jest.spyOn(core, "setOutput").mockImplementation();
const setFailedMock = jest.spyOn(core, "setFailed").mockImplementation();

// Clean up mock file system after each test
afterEach(() => {
  fs.rmSync(tempDir, { recursive: true });
});

// Mock the action's main function
const runMock = jest.spyOn(findWheelImpl, "run");

describe("run", () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // Set up mock file system before each test
    fs.mkdirSync(wheelsPath, { recursive: true });

    // Create test wheels
    for (const wheel of wheels) {
      const wheelPath = path.join(wheelsPath, wheel);
      fs.writeFileSync(wheelPath, "Fake wheel content");
    }
  });

  it("Find_one_wheel", async () => {
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation((name) => {
      switch (name) {
        case "wheels_dir":
          return wheelsPath;
        case "package_name":
          return "package_x";
        default:
          return "";
      }
    });

    await findWheelImpl.run();
    expect(runMock).toHaveReturned();

    // Verify that all of the core library functions were called correctly
    expect(setOutputMock).toHaveBeenNthCalledWith(
      1,
      "wheel_path",
      path.join(wheelsPath, wheels[1]),
    );
  });

  it("Fails_when_no_wheels_found", async () => {
    getInputMock.mockImplementation((name) => {
      switch (name) {
        case "wheels_dir":
          return wheelsPath;
        case "package_name":
          return "non_existent_package";
        default:
          return "";
      }
    });

    await findWheelImpl.run();

    expect(setFailedMock).toHaveBeenCalledWith(
      'No files found matching "non_existent_package"',
    );
  });

  it("Fails_when_multiple_wheels_found", async () => {
    getInputMock.mockImplementation((name) => {
      switch (name) {
        case "wheels_dir":
          return wheelsPath;
        case "package_name":
          return "package_y";
        default:
          return "";
      }
    });

    await findWheelImpl.run();

    const expectedWheels = [
      path.join(wheelsPath, wheels[3]),
      path.join(wheelsPath, wheels[2]),
    ];
    expect(setFailedMock).toHaveBeenCalledWith(
      `Multiple files found matching "package_y": ${JSON.stringify(expectedWheels)}`,
    );
  });
});
