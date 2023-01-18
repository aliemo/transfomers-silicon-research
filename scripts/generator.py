#!/usr/bin/env python

import yaml

data = {}
with open('papers.yaml') as f:
    content = yaml.safe_load(f)


for k, p in content.items():
    if p['year'] is not None:
        data[k]=p

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






for k, paper in data.items():

    
    print(f'**{paper["title"]}**')
    print()

    print(f'![](https://img.shields.io/badge/{paper["publisher"]}-{paper["year"]}-skyblue?colorstyle=flat-square)')
    print(f'[![DOI-Link](https://img.shields.io/badge/DOI-{paper["doi"]}-sandybrown?style=flat-square)]({paper["url"]})')
    print()
    
