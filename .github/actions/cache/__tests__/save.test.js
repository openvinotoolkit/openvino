/**
 * Unit tests for the action's main functionality, src/saveImpl.js
 */
const core = require('@actions/core');
const path = require('path');
const os = require('os');
const fs = require('fs');
const saveImpl = require('../src/saveImpl');

// Mock the GitHub Actions core library
const getInputMock = jest.spyOn(core, 'getInput').mockImplementation();
const setOutputMock = jest.spyOn(core, 'setOutput').mockImplementation();
const setFailedMock = jest.spyOn(core, 'setFailed').mockImplementation();

const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'test-'));
const cacheLocalPath = path.join(tempDir, 'cache_local');
const cacheRemotePath = path.join(tempDir, 'cache_remote');

// Clean up mock file system after each test
afterEach(() => {
  fs.rmSync(tempDir, { recursive: true });
});

// Mock the action's main function
const runMock = jest.spyOn(saveImpl, 'save');

describe('save', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // Set up mock file system before each test

    // Create cache pathes
    fs.mkdirSync(cacheLocalPath, { recursive: true });
    fs.mkdirSync(cacheRemotePath, { recursive: true });

    // Define test files
    const file1Path = path.join(cacheLocalPath, 'file1.txt');
    const file2Path = path.join(cacheLocalPath, 'file2.txt');

    // Write content to the test files
    fs.writeFileSync(file1Path, 'File 1 contents');
    fs.writeFileSync(file2Path, 'File 2 contents');
  });

  it('Cache files pass', async () => {
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'cache';
        case 'path':
          return cacheLocalPath;
        default:
          return '';
      }
    });

    await saveImpl.save();

    expect(runMock).toHaveReturned();

    expect(setOutputMock).toHaveBeenNthCalledWith(
      1,
      'cache-file',
      'cache.cache'
    );
    expect(setOutputMock).toHaveBeenNthCalledWith(2, 'cache-hit', true);

    expect(fs.existsSync(`${cacheRemotePath}/cache.cache`)).toBe(true); // Check if the tar file was created
  });

  it('Cache files: absent local cache dir', async () => {
    // Set the action's inputs as return values from core.getInput()
    const cachePathAbsent = path.join(tempDir, 'cache_local_absent');
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'cache';
        case 'path':
          return cachePathAbsent;
        default:
          return '';
      }
    });

    await saveImpl.save();
    expect(runMock).toHaveReturned();
    expect(fs.existsSync(`${cacheRemotePath}/cache.cache`)).toBe(false); // Check that the tar file was not created
  });

  it('Cache files: absent remote cache dir', async () => {
    // Set the action's inputs as return values from core.getInput()
    const cacheRemotePathAbsent = path.join(tempDir, 'cache_remote_absent');
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePathAbsent;
        case 'key':
          return 'cache';
        case 'path':
          return cacheLocalPath;
        default:
          return '';
      }
    });

    await saveImpl.save();

    expect(runMock).toHaveReturned();
    expect(setOutputMock).toHaveBeenNthCalledWith(
      1,
      'cache-file',
      'cache.cache'
    );
    expect(setOutputMock).toHaveBeenNthCalledWith(2, 'cache-hit', true);
    expect(fs.existsSync(`${cacheRemotePathAbsent}/cache.cache`)).toBe(true); // Check if the tar file was created
  });

  it('Cache files: key is not set', async () => {
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'path':
          return cacheLocalPath;
        default:
          return '';
      }
    });

    await saveImpl.save();

    expect(runMock).toHaveReturned();
    expect(setOutputMock).not.toHaveBeenCalled();
    expect(setFailedMock).not.toHaveBeenCalled();
  });

  it('Cache files: file already exists', async () => {
    const cacheFakePath = path.join(cacheRemotePath, 'cache-fake.cache');
    fs.writeFileSync(cacheFakePath, 'Fake content');
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'cache-fake';
        case 'path':
          return cacheLocalPath;
        default:
          return '';
      }
    });

    await saveImpl.save();

    expect(runMock).toHaveReturned();
    expect(setOutputMock).not.toHaveBeenCalled();
    expect(setFailedMock).not.toHaveBeenCalled();
  });

  it('Test unexpected behaviour', async () => {
    // Set folder permissions to read-only
    fs.chmodSync(cacheRemotePath, '555');
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'cache';
        case 'path':
          return cacheLocalPath;
        default:
          return '';
      }
    });

    await saveImpl.save();
    expect(runMock).toHaveReturned();
    expect(setOutputMock).not.toHaveBeenCalled();
    expect(setFailedMock).toHaveBeenCalled();
  });
});
