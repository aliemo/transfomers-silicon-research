#!/usr/bin/env python

import yaml

import os
import argparse


# for k, v in data.items():
#     data[k]['url'] = " "
#     title = data[k]['title']
#     data[k]['title'] = f'"{title}"'


# with open('papers2.yaml', 'w') as f:
#     yaml.safe_dump(data,f,indent=2, sort_keys=False)

# data = [(v['year'],v) for k, v in data.items()]

# data.sort(key=lambda x:x[0])
# print(data)

# years = [2018, 2019, 2020, 2021, 2022, 2023]

# for d in data:
#     paper=d[1]
#     print(f'**{paper["title"]}**')
#     print()
#     print(f'![](https://img.shields.io/badge/{paper["publisher"]}-{paper["year"]}-skyblue?colorstyle=flat-square)')
#     print(f'[![DOI-Link](https://img.shields.io/badge/DOI-{paper["doi"]}-sandybrown?style=flat-square)]({paper["url"]})')
    
#     print()




def write_md(data, outf):
    if outf == 'stdout':
        for k, paper in data.items():
            print(f'**{paper["title"]}**')
            print()

            print(f'![](https://img.shields.io/badge/{paper["publisher"]}-{paper["year"]}-skyblue?colorstyle=flat-square)')
            print(f'[![DOI-Link](https://img.shields.io/badge/DOI-{paper["doi"]}-sandybrown?style=flat-square)]({paper["url"]})')
            print()
    else:
        with open(outf, 'w') as f:
            for k, paper in data.items():
                f.write(f'**{paper["title"]}**')
                f.write('\n')
                f.write('\n')
                f.write(f'![](https://img.shields.io/badge/{paper["publisher"]}-{paper["year"]}-skyblue?colorstyle=flat-square)')
                f.write('\n')
                f.write(f'[![DOI-Link](https://img.shields.io/badge/DOI-{paper["doi"]}-sandybrown?style=flat-square)]({paper["url"]})')
                f.write('\n')
                f.write('\n')

    
def read_yaml(inpf):
    data = {}
    with open(inpf) as f:
        content = yaml.safe_load(f)


    for k, p in content.items():
        if p['year'] is not None:
            data[k]=p

    return data

def main():

    
    parser = argparse.ArgumentParser(description='Generate README.md')
    parser.add_argument('-i', '--input', type=str, default='README.md', help='input yaml file')
    parser.add_argument('-o', '--output', type=str, default='stdout', help='output README file')
    args = parser.parse_args()

    inpf = args.input
    outf = args.output
    
    data = read_yaml(inpf)
    write_md(data, outf)    

if __name__ == '__main__':
    main()