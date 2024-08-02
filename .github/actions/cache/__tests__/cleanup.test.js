/**
 * Unit tests for the action's main functionality, src/cleanupImpl.js
 */
const core = require('@actions/core');
const path = require('path');
const os = require('os');
const fs = require('fs');
const cleanupImpl = require('../src/cleanupImpl');

// Mock the GitHub Actions core library
const getInputMock = jest.spyOn(core, 'getInput').mockImplementation();
const setFailedMock = jest.spyOn(core, 'setFailed').mockImplementation();

const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'test-cleanup-'));
const cacheRemotePath = path.join(tempDir, 'cache_remote');

const cacheFiles = ['cache_1.cache', 'cache_2.cache', 'cache_3.cache'];
const minAccessTime = 7 * 24 * 60 * 60 * 1000; // 1 week

// Mock the action's main function
const runMock = jest.spyOn(cleanupImpl, 'cleanUp');

describe('cleanup', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // Set up file system before each test
    fs.mkdirSync(cacheRemotePath, { recursive: true });

    const fileSizeInBytes = 1024 * 1024 * 1024; // 1 GB
    const buffer = Buffer.alloc(fileSizeInBytes);
    let id = 1;
    for (const cache of cacheFiles) {
      const cachePath = path.join(cacheRemotePath, cache);
      fs.writeFileSync(cachePath, buffer);
      fs.utimesSync(
        cachePath,
        new Date(Date.now() - minAccessTime * id++),
        new Date()
      );
    }
  });

  // Clean up mock file system after each test
  afterEach(() => {
    fs.rmSync(tempDir, { recursive: true });
  });

  it('Cleanup old files using restore key', async () => {
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'key';
        case 'restore-keys':
          return 'cache';
        case 'cache-size':
          return 1;
        default:
          return '';
      }
    });

    await cleanupImpl.cleanUp();

    expect(runMock).toHaveReturned();
    // cache2 and cache3 should be removed
    for (const cache of cacheFiles.slice(1, 2)) {
      expect(fs.existsSync(path.join(cacheRemotePath, cache))).toBe(false);
    }
    // check that file1 exists
    expect(fs.existsSync(path.join(cacheRemotePath, cacheFiles[0]))).toBe(true);
  });

  it('Cleanup old files using key', async () => {
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'cache';
        case 'cache-size':
          return 1;
        default:
          return '';
      }
    });

    await cleanupImpl.cleanUp();

    expect(runMock).toHaveReturned();
    // cache2 and cache3 should be removed
    for (const cache of cacheFiles.slice(1, 2)) {
      expect(fs.existsSync(path.join(cacheRemotePath, cache))).toBe(false);
    }
    // check that file1 exists
    expect(fs.existsSync(path.join(cacheRemotePath, cacheFiles[0]))).toBe(true);
  });

  it('Skip cleanup old files by size', async () => {
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'key';
        case 'restore-keys':
          return 'cache';
        case 'cache-size':
          return 5;
        default:
          return '';
      }
    });

    await cleanupImpl.cleanUp();

    expect(runMock).toHaveReturned();

    for (const cache of cacheFiles) {
      expect(fs.existsSync(path.join(cacheRemotePath, cache))).toBe(true);
    }
  });

  it('Skip cleanup old files by atime', async () => {
    for (const cache of cacheFiles) {
      const cachePath = path.join(cacheRemotePath, cache);
      fs.utimesSync(
        cachePath,
        new Date(Date.now() - minAccessTime / 2),
        new Date()
      );
    }

    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'key';
        case 'restore-keys':
          return 'cache';
        case 'cache-size':
          return 1;
        case 'max-cache-size':
          return 5;
        default:
          return '';
      }
    });

    await cleanupImpl.cleanUp();

    expect(runMock).toHaveReturned();

    for (const cache of cacheFiles) {
      expect(fs.existsSync(path.join(cacheRemotePath, cache))).toBe(true);
    }
  });

  it('Cleanup recently used files by max cache limit', async () => {
    for (const cache of cacheFiles) {
      const cachePath = path.join(cacheRemotePath, cache);
      fs.utimesSync(
        cachePath,
        new Date(Date.now() - minAccessTime / 2),
        new Date()
      );
    }

    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'key';
        case 'restore-keys':
          return 'cache';
        case 'cache-size':
          return 1;
        case 'max-cache-size':
          return 2;
        default:
          return '';
      }
    });

    await cleanupImpl.cleanUp();

    expect(runMock).toHaveReturned();

    // cache2 and cache3 should be removed
    for (const cache of cacheFiles.slice(1, 2)) {
      expect(fs.existsSync(path.join(cacheRemotePath, cache))).toBe(false);
    }
    // check that file1 exists
    expect(fs.existsSync(path.join(cacheRemotePath, cacheFiles[0]))).toBe(true);
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
        case 'restore-keys':
          return 'cache';
        case 'cache-size':
          return 2;
        default:
          return '';
      }
    });

    await cleanupImpl.cleanUp();

    expect(runMock).toHaveReturned();
    expect(setFailedMock).not.toHaveBeenCalled();

    fs.chmodSync(cacheRemotePath, '777');
  });

  it('Cleanup absent directory', async () => {
    const cachePathAbsent = path.join(tempDir, 'cache_remote_absent');
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cachePathAbsent;
        case 'key':
          return 'cache';
        case 'restore-keys':
          return 'cache';
        case 'cache-size':
          return 2;
        default:
          return '';
      }
    });

    await cleanupImpl.cleanUp();

    expect(runMock).toHaveReturned();
    expect(setFailedMock).not.toHaveBeenCalled();
  });

  it('Cleanup directory with subdirectory', async () => {
    const cacheSubPath = path.join(cacheRemotePath, 'cache_subdir.cache');
    fs.mkdirSync(cacheSubPath, { recursive: true });
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'restore-keys':
          return 'cache';
        case 'cache-size':
          return 1;
        default:
          return '';
      }
    });

    await cleanupImpl.cleanUp();

    expect(runMock).toHaveReturned();
    // cache2 and cache3 should be removed
    for (const cache of cacheFiles.slice(1, 2)) {
      expect(fs.existsSync(path.join(cacheRemotePath, cache))).toBe(false);
    }
    // check that file1 exists
    expect(fs.existsSync(path.join(cacheRemotePath, cacheFiles[0]))).toBe(true);
    // check that sub directory exists
    expect(fs.existsSync(cacheSubPath)).toBe(true);
  });
});
