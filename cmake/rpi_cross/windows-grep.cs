// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

using System;
using System.Text.RegularExpressions;

public static class Grep
{
    public static int Main(string[] args)
    {
        string pattern = null;
        foreach (var arg in args)
        {
            if (arg == "-E")
            {
                continue;
            }
            pattern = arg;
        }

        if (string.IsNullOrEmpty(pattern))
        {
            return 1;
        }

        var regex = new Regex(pattern);
        string line;
        while ((line = Console.ReadLine()) != null)
        {
            if (regex.IsMatch(line))
            {
                Console.WriteLine(line);
            }
        }

        return 0;
    }
}
