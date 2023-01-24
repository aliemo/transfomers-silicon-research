#!/usr/bin/env python

import yaml

import os
import argparse




def write_md(data, outf, signle=True, with_header=True):

    if with_header:
        header = ''
        with open('data/header.txt') as f:
            lines = f.readlines()
            header += ''.join(lines)

        with open('data/basic.txt') as f:
            lines = f.readlines()
            header += ''.join(lines)
        
        with open('data/important.txt') as f:
            lines = f.readlines()
            header += ''.join(lines)

    if signle:
        if outf == 'stdout':
            print(header)
            print()
            for k, paper in data.items():
                print(f'**{paper["title"]}**')
                print()

                print(f'![](https://img.shields.io/badge/{paper["publisher"]}-{paper["year"]}-skyblue?colorstyle=flat-square)')
                print(f'[![DOI-Link](https://img.shields.io/badge/DOI-{paper["doi"]}-sandybrown?style=flat-square)]({paper["url"]})')
                
                if paper['pdf']:
                    print(f'[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)]({paper["pdf"]})')

                print()
        else:
            with open(outf, 'w') as f:
                f.write(header)
                f.write('\n')
                for k, paper in data.items():
                    f.write(f'**{paper["title"]}**')
                    f.write('\n')
                    f.write('\n')
                    f.write(f'![](https://img.shields.io/badge/{paper["publisher"]}-{paper["year"]}-skyblue?colorstyle=flat-square)')
                    f.write('\n')
                    f.write(f'[![DOI-Link](https://img.shields.io/badge/DOI-{paper["doi"]}-sandybrown?style=flat-square)]({paper["url"]})')
                    f.write('\n')
                    if paper['pdf']:
                        f.write(f'[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)]({paper["pdf"]})')
                    f.write('\n')
                    f.write('\n')

    else:
        if outf == 'stdout':
            print(header)
            print()
            for year, papers in data.items():
                print(f'### {year}')
                print()

                for k, paper in papers.items():
                    print(f'**{paper["title"]}**')
                    print()

                    print(f'![](https://img.shields.io/badge/{paper["publisher"]}-{paper["year"]}-skyblue?colorstyle=flat-square)')
                    print(f'[![DOI-Link](https://img.shields.io/badge/DOI-{paper["doi"]}-sandybrown?style=flat-square)]({paper["url"]})')
                    if paper['pdf']:
                        print(f'[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)]({paper["pdf"]})')
                    print()
                    print()
                
                print("---")
                print()
        else:
            with open(outf, 'w') as f:
                f.write(header)
                f.write('\n')
                for year, papers in data.items():
                    f.write(f'### {year}')
                    f.write('\n')

                    for k, paper in papers.items():
                        f.write(f'**{paper["title"]}**')
                        f.write('\n')
                        f.write('\n')
                        f.write(f'![](https://img.shields.io/badge/{paper["publisher"]}-{paper["year"]}-skyblue?colorstyle=flat-square)')
                        f.write('\n')
                        f.write(f'[![DOI-Link](https://img.shields.io/badge/DOI-{paper["doi"]}-sandybrown?style=flat-square)]({paper["url"]})')
                        f.write('\n')
                        if paper['pdf']:
                            f.write(f'[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)]({paper["pdf"]})')
                        f.write('\n')
                        f.write('\n')

                    f.write('---')
                    f.write('\n')    


def read_yaml(inpf):
    data = {}
    with open(inpf) as f:
        content = yaml.safe_load(f)


    for k, p in content.items():
        if p['year'] is not None:
            # print(k)
            if p['ignore'] == False:
                data[k]=p
    
    return data

def sort_by_year(xdata, signle=True):
    
    if signle:
        return {k: v for k, v in sorted(xdata.items(), key=lambda x: x[1]['year'])}
    
    
    years = list(set([p['year'] for _, p in xdata.items()]))

    data = {k:{} for k in years}
    for year in years:
        d = {k: p for k, p in xdata.items() if p['year'] == year}
        data[year] = d

    return data

def main():

    
    parser = argparse.ArgumentParser(description='Generate README.md')
    parser.add_argument('-i', '--input', type=str, default='README.md', help='input yaml file')
    parser.add_argument('-o', '--output', type=str, default='stdout', help='output README file')
    args = parser.parse_args()

    inpf = args.input
    outf = args.output
    
    data = read_yaml(inpf)
    data = sort_by_year(data, signle=False)
    write_md(data, outf, signle=False)    

if __name__ == '__main__':
    main()