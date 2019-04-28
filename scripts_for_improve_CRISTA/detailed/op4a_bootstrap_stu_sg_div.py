from sklearn.externals import joblib
import numpy as np

np.random.seed(1337)


def gen_data(pos, neg, niter=100):
    n = pos.shape[0]
    nn = neg.shape[0]
    pos_lst = []
    neg_lst = []
    for i in range(niter):
        idx_pos = np.random.choice(range(n), size=n * 2, replace=True)
        idx_neg = np.random.choice(range(nn), size=n * 2, replace=True)
        pos_lst.append(pos[idx_pos])
        neg_lst.append(neg[idx_neg])
    return list(zip(pos_lst, neg_lst))


data_dt = joblib.load('data/stu_sg_div.pkl')
stulst, studt = joblib.load('stuids.pkl')
dt = dict.fromkeys(data_dt.keys())
# sgid = sgids[0]
for stuid in stulst:
    print(stuid)
    ot, neg = data_dt[stuid]
    lst = gen_data(ot, neg)
    dt[stuid] = lst

joblib.dump(dt, 'data/stu_sg_bs_div.pkl')

