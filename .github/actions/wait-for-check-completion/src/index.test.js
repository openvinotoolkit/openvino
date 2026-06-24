import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { waitForChecks } from '../src/index.js';

describe('Wait for Check Completion Action', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    describe('waitForChecks', () => {
        test('should wait for single check in array', async () => {
            const mockOctokit = {
                paginate: jest.fn().mockResolvedValue([{
                    name: 'test-check',
                    status: 'completed',
                    conclusion: 'success',
                    started_at: new Date().toISOString()
                }]),
                rest: {
                    checks: {
                        listForRef: jest.fn()
                    }
                }
            };

            const results = await waitForChecks(
                mockOctokit,
                'owner',
                'repo',
                'sha123',
                ['test-check'],
                1, // 1 second interval
                60 // 60 second timeout
            );

            expect(results['test-check'].status).toBe('completed');
            expect(results['test-check'].conclusion).toBe('success');
            expect(mockOctokit.paginate).toHaveBeenCalledWith(
                mockOctokit.rest.checks.listForRef,
                {
                    owner: 'owner',
                    repo: 'repo',
                    ref: 'sha123',
                    per_page: 100
                }
            );
        });

        test('should wait for multiple checks to complete successfully', async () => {
            const mockOctokit = {
                paginate: jest.fn().mockResolvedValue([
                    {
                        name: 'test-check-1',
                        status: 'completed',
                        conclusion: 'success',
                        started_at: new Date().toISOString()
                    },
                    {
                        name: 'test-check-2',
                        status: 'completed',
                        conclusion: 'success',
                        started_at: new Date().toISOString()
                    }
                ]),
                rest: {
                    checks: {
                        listForRef: jest.fn()
                    }
                }
            };

            const results = await waitForChecks(
                mockOctokit,
                'owner',
                'repo',
                'sha123',
                ['test-check-1', 'test-check-2'],
                1, // 1 second interval
                60 // 60 second timeout
            );

            expect(results['test-check-1'].status).toBe('completed');
            expect(results['test-check-1'].conclusion).toBe('success');
            expect(results['test-check-2'].status).toBe('completed');
            expect(results['test-check-2'].conclusion).toBe('success');
            expect(mockOctokit.paginate).toHaveBeenCalledWith(
                mockOctokit.rest.checks.listForRef,
                {
                    owner: 'owner',
                    repo: 'repo',
                    ref: 'sha123',
                    per_page: 100
                }
            );
        });

        test('should handle multiple checks with mixed progress states', async () => {
            let callCount = 0;
            const mockOctokit = {
                paginate: jest.fn().mockImplementation(() => {
                    callCount++;
                    if (callCount === 1) {
                        return Promise.resolve([
                            {
                                name: 'test-check-1',
                                status: 'completed',
                                conclusion: 'success',
                                started_at: new Date().toISOString()
                            },
                            {
                                name: 'test-check-2',
                                status: 'in_progress',
                                conclusion: null,
                                started_at: new Date().toISOString()
                            }
                        ]);
                    }
                    return Promise.resolve([
                        {
                            name: 'test-check-1',
                            status: 'completed',
                            conclusion: 'success',
                            started_at: new Date().toISOString()
                        },
                        {
                            name: 'test-check-2',
                            status: 'completed',
                            conclusion: 'success',
                            started_at: new Date().toISOString()
                        }
                    ]);
                }),
                rest: {
                    checks: {
                        listForRef: jest.fn()
                    }
                }
            };

            const results = await waitForChecks(
                mockOctokit,
                'owner',
                'repo',
                'sha123',
                ['test-check-1', 'test-check-2'],
                0.1, // 0.1 second interval for fast test
                60 // 60 second timeout
            );

            expect(results['test-check-1'].status).toBe('completed');
            expect(results['test-check-1'].conclusion).toBe('success');
            expect(results['test-check-2'].status).toBe('completed');
            expect(results['test-check-2'].conclusion).toBe('success');
            expect(mockOctokit.paginate).toHaveBeenCalledTimes(2);
        });

        test('should timeout when some checks do not complete', async () => {
            const mockOctokit = {
                paginate: jest.fn().mockResolvedValue([
                    {
                        name: 'test-check-1',
                        status: 'completed',
                        conclusion: 'success',
                        started_at: new Date().toISOString()
                    },
                    {
                        name: 'test-check-2',
                        status: 'in_progress',
                        conclusion: null,
                        started_at: new Date().toISOString()
                    }
                ]),
                rest: {
                    checks: {
                        listForRef: jest.fn()
                    }
                }
            };

            await expect(
                waitForChecks(
                    mockOctokit,
                    'owner',
                    'repo',
                    'sha123',
                    ['test-check-1', 'test-check-2'],
                    0.1, // 0.1 second interval
                    0.2 // 0.2 second timeout
                )
            ).rejects.toThrow('Timeout: Checks [test-check-2] did not complete within 0.2 seconds');
        });
    });
});