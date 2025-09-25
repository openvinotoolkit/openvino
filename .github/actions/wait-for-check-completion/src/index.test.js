const { waitForCheck } = require('../src/index');
const core = require('@actions/core');

// Mock the core module
jest.mock('@actions/core');

describe('Wait for Check Completion Action', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    test('should wait for check to complete successfully', async () => {
        const mockOctokit = {
            rest: {
                checks: {
                    listForRef: jest.fn().mockResolvedValue({
                        data: {
                            check_runs: [{
                                name: 'test-check',
                                status: 'completed',
                                conclusion: 'success',
                                started_at: new Date().toISOString()
                            }]
                        }
                    })
                }
            }
        };

        const result = await waitForCheck(
            mockOctokit,
            'owner',
            'repo',
            'sha123',
            'test-check',
            1, // 1 second interval
            60 // 60 second timeout
        );

        expect(result.status).toBe('completed');
        expect(result.conclusion).toBe('success');
        expect(mockOctokit.rest.checks.listForRef).toHaveBeenCalledWith({
            owner: 'owner',
            repo: 'repo',
            ref: 'sha123',
            check_name: 'test-check',
            per_page: 100
        });
    });

    test('should handle check that is still in progress', async () => {
        let callCount = 0;
        const mockOctokit = {
            rest: {
                checks: {
                    listForRef: jest.fn().mockImplementation(() => {
                        callCount++;
                        if (callCount === 1) {
                            return Promise.resolve({
                                data: {
                                    check_runs: [{
                                        name: 'test-check',
                                        status: 'in_progress',
                                        conclusion: null,
                                        started_at: new Date().toISOString()
                                    }]
                                }
                            });
                        } else {
                            return Promise.resolve({
                                data: {
                                    check_runs: [{
                                        name: 'test-check',
                                        status: 'completed',
                                        conclusion: 'success',
                                        started_at: new Date().toISOString()
                                    }]
                                }
                            });
                        }
                    })
                }
            }
        };

        const result = await waitForCheck(
            mockOctokit,
            'owner',
            'repo',
            'sha123',
            'test-check',
            0.1, // 0.1 second interval for fast test
            60 // 60 second timeout
        );

        expect(result.status).toBe('completed');
        expect(result.conclusion).toBe('success');
        expect(mockOctokit.rest.checks.listForRef).toHaveBeenCalledTimes(2);
    });

    test('should timeout when check does not complete', async () => {
        const mockOctokit = {
            rest: {
                checks: {
                    listForRef: jest.fn().mockResolvedValue({
                        data: {
                            check_runs: [{
                                name: 'test-check',
                                status: 'in_progress',
                                conclusion: null,
                                started_at: new Date().toISOString()
                            }]
                        }
                    })
                }
            }
        };

        await expect(
            waitForCheck(
                mockOctokit,
                'owner',
                'repo',
                'sha123',
                'test-check',
                0.1, // 0.1 second interval
                0.2 // 0.2 second timeout
            )
        ).rejects.toThrow('Timeout: Check "test-check" did not complete within 0.2 seconds');
    });

    test('should handle no check runs found', async () => {
        let callCount = 0;
        const mockOctokit = {
            rest: {
                checks: {
                    listForRef: jest.fn().mockImplementation(() => {
                        callCount++;
                        if (callCount === 1) {
                            return Promise.resolve({
                                data: { check_runs: [] }
                            });
                        } else {
                            return Promise.resolve({
                                data: {
                                    check_runs: [{
                                        name: 'test-check',
                                        status: 'completed',
                                        conclusion: 'success',
                                        started_at: new Date().toISOString()
                                    }]
                                }
                            });
                        }
                    })
                }
            }
        };

        const result = await waitForCheck(
            mockOctokit,
            'owner',
            'repo',
            'sha123',
            'test-check',
            0.1, // 0.1 second interval
            60 // 60 second timeout
        );

        expect(result.status).toBe('completed');
        expect(result.conclusion).toBe('success');
    });
});