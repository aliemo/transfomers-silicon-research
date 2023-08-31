import yaml
import argparse

def read_yaml(inpf):
    data = {}
    with open(inpf) as f:
        content = yaml.safe_load(f)


    for k, p in content.items():
            data[k]=p

    return data

def write_yaml(src:dict, outdir):
    title = ''
    year = ''
    doi = ''
    url = ''
    pdf = ''
    m_model = ''
    method = ''
    platform = ''
    publisher = ''
    pub = ''
    pubtype = ''
    pubshort = ''
    ignore = ''
    silicon = ''
    with open(outdir, 'w') as sf:
        for inx, (key, value) in enumerate(src.items()):

            title = value['title']
            year = value['year']
            doi = value['doi']
            url = value['url']
            pdf = value['pdf']
            model = value['model']
            method = value['method']
            platform = value['platform']
            publisher = value['publisher']
            pubname = value['pubname']
            pubtype = value['type']
            pubkey = value['pubkey']
            ignore = value['ignore']
            silicon = value['silicon']
            print(inx, key, inx+1 == key)

            sf.write(f'{inx+1}:')
            sf.write('\n')
            sf.write(f'  title: "{title}"')
            sf.write('\n')
            sf.write(f'  year: {year}')
            sf.write('\n')
            sf.write(f'  type: {pubtype}')
            sf.write('\n')
            sf.write(f'  doi: {doi}')
            sf.write('\n')
            sf.write(f'  url: {url}')
            sf.write('\n')
            sf.write(f'  pdf: {pdf}')
            sf.write('\n')
            sf.write(f'  ignore: {ignore}')
            sf.write('\n')
            sf.write(f'  silicon: {silicon}')
            sf.write('\n')
            sf.write(f'  platform: {platform}')
            sf.write('\n')
            sf.write(f'  model: {model}')
            sf.write('\n')
            sf.write(f'  method: {method}')
            sf.write('\n')
            sf.write(f'  publisher: {publisher}')
            sf.write('\n')
            sf.write(f'  pubkey: "{pubshort}"')
            sf.write('\n')
            sf.write(f'  pubname: "{pubname}"')
            sf.write('\n')
            sf.write(f'  reserved: DEADBEEF')
            sf.write('\n')
            sf.write('\n')


def main():
    parser = argparse.ArgumentParser(description='Reformat Yaml File')
    parser.add_argument('-i', '--input', type=str, default='papers.yaml', help='input yaml file')
    parser.add_argument('-o', '--output', type=str, default='stdout', help='output README file')
    args = parser.parse_args()

    inpf = args.input
    outf = args.output

    data = read_yaml(inpf)


    write_yaml(data, outf)
    print(len(data))

if __name__ == '__main__':
    main()
