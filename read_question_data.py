
def read_question_data(filename):
    f = open(filename, "r")
    raw_data = f.readlines()
    f.close()
    q = list()
    pos = list()
    neg = list()
    for td in raw_data:
        qi, posi, negi = [], [], []
        if (len(td.split('\t')) == 3):
            qi, posi, negi = td.split('\t')
            posi_list = posi.split()
            for pi in posi_list:
                q.append(qi)
                pos.append(pi)
                neg.append(negi)
        elif (len(td.split('\t')) == 4):
            qi, posi, alli, _ = td.split('\t')
            q.append(qi)
            pos.append(posi)
            posi_list = posi.split()
            alli_list = alli.split()
            negi_list = [x for x in alli_list if x not in posi_list]
            negi = " ".join(negi_list)
            # print("Negative example list: ")
            # print(negi, len(negi.split()))
            neg.append(negi)
    return (q, pos, neg)

