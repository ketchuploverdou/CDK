import pandas as pd
import os


def convert2dict(path, type_='entity', save_dir='./data/OpenBG500'):
    df = pd.read_csv(path, sep='\t', header=None)
    tmp = list(df[0])
    p_file = os.path.join(save_dir, type_ + '.dict')
    with open(p_file, 'w') as f:
        f.write(f"[pred]\t{0}\n")
        for idx, i in enumerate(tmp):
            content = f"{i}\t{idx + 1}\n"
            f.write(content)
    print(f"file saved at {p_file}")


def convert2txt(path, mode='train', save_dir='./data/OpenBG500'):
    df = pd.read_csv(path, sep='\t', header=None)
    cols = df.columns
    size = len(df[0])
    p_file = os.path.join(save_dir, mode + '.txt')
    with open(p_file, 'w') as f:
        for i in range(size):
            content = ""
            for c in cols:
                content += f"{df[c][i]}\t"
            content += '[pred]\n' if mode == 'test' else '\n'
            f.write(content)
    print(f"file saved at {p_file}")


def main():
    p1 = "./data/OpenBG500/OpenBG500_entity2text.tsv"
    p2 = "./data/OpenBG500/OpenBG500_relation2text.tsv"
    convert2dict(p1, type_='entities')
    convert2dict(p2, type_='relations')
    p3 = './data/OpenBG500/OpenBG500_train.tsv'
    p4 = './data/OpenBG500/OpenBG500_test.tsv'
    p5 = './data/OpenBG500/OpenBG500_dev.tsv'
    convert2txt(p3, 'train')
    convert2txt(p4, 'test')
    convert2txt(p5, 'valid')


main()
