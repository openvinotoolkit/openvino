#!/usr/bin/env python3
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--format', '-f', choices=['no', 'csv', 'md'], default='no', required=False, help="print data using format")
    parser.add_argument('--group_by', '-g', choices=['layerName', 'layerType', 'execType'], default=['layerType'], required=False, help="group data by column", nargs='+')
    parser.add_argument('benchmark_average_counters_file', type=str, action='store')
    return parser.parse_args();

def get_dataframe(path):
    with open(path, 'r', newline='') as csvfile:
        return pd.read_csv(path, delimiter=';', header=0, index_col=0)

def aggregate(df, group_by):
    # remove existing total
    df = df.drop(index='Total')

    aggregated = df.groupby(group_by, as_index=False)['cpuTime (ms)'].agg(['count','sum'])
    # sort by sum
    result = aggregated.sort_values(by=['sum'], ascending=False)
    # add percentage
    result['%'] = (result['sum'] / result['sum'].sum()) * 100
    # round percentage
    result = result.round({'%': 2})
    # add total
    result.loc['Total'] = result.sum(numeric_only=True)
    for group in group_by:
        result.at['Total', group] = 'Total'

    # ensure count as int (no trailing .0)
    result['count'] = result['count'].astype('int')
    # rename columns
    result = result.rename(columns={"count": "Count", "sum":"Sum (ms)"})
    return result

def print_df(df, format):
    if format == 'csv':
        print(df.to_csv(index=False))
    elif format == 'md':
        print(df.to_markdown(index=False))
    else:
        print(df.to_string(index=False))

if __name__ == "__main__":
    args = parse_args();

    df = get_dataframe(args.benchmark_average_counters_file);
    df = aggregate(df, args.group_by)
    print_df(df, args.format)
