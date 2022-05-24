

# 来源
http://jmcauley.ucsd.edu/data/amazon/


# 下载
http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video_5.json.gz

# 预处理
1、解压
2、转为 txt 文件

``` python
import json
from tqdm.auto import tqdm

instants = []
u_set = dict()
i_set = dict()

with open("Amazon_Instant_Video_5.json", 'r', encoding='utf-8') as fp:
    for line in tqdm(fp, total=37126):
        rate = json.loads(line)
        
        uid = rate['reviewerID']
        iid = rate['asin']
        
        if uid not in u_set:
            u_set[uid] = len(u_set)
        
        if iid not in i_set:
            i_set[iid] = len(i_set)
        
        instants.append((u_set[uid], i_set[iid], rate['overall']))

with open('Amazon_Instant_Video_5.txt', 'w') as fp:
    for inst in tqdm(instants):
        fp.write(f"{inst[0]}\t{inst[1]}\t{inst[2]}\n")

```

