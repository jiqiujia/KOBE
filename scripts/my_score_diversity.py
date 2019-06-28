import io
import re

if __name__ == '__main__':
    stsSet = set()
    totalCnt = 0
    titleSet = set()
    with io.open('E:/dlprojects/data/kobe/item_desc_dataset.txt', encoding='utf-8') as fin:
        for line in fin.readlines():
            line = line.split("\t")
            titleSet.add(line[0])
            stses = re.split(r'[，。！；]', line[1])
            totalCnt += len(stses)
            for sts in stses:
                if sts not in stsSet:
                    stsSet.add(sts)
    print("total title num {}, sts num {}, distinct sts num {}".format(len(titleSet), totalCnt, len(stsSet)))

    totalCnt = 0
    inCnt = 0
    descStsSet = set()
    descTitleSet = set()
    outStsSet = set()
    with io.open('../aiProductTest.txt', 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            if not line.startswith("\t"):
                if len(line)>0 and not line[0]=='#':
                    descTitleSet.add(line.strip())
                continue
            line = line.strip()
            stses = re.split(r"[，。！；]", line)
            totalCnt += len(stses)
            for sts in stses:
                if sts in stsSet:
                    inCnt += 1
                else:
                    outStsSet.add(sts)
                if sts not in descStsSet:
                    descStsSet.add(sts)
    print("total sts num {}, in total num {}, distinct sts num {}".format(totalCnt, inCnt, len(descStsSet)))
    print("title intersect num {}".format(len(descTitleSet.intersection(titleSet))))
    print("outSts num {}".format(len(outStsSet)))
    for sts in outStsSet:
        print(sts)