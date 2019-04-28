import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import product

xlst = ['Leave-sgRNA-Out', 'Leave-Study-Out: Common', 'Leave-Study-Out: Unique']
hlst = ['RF with balanced sampling', 'CRISTA', 'RF without balanced sampling','CFD','OptCD','CCTop']

zlst = list(product(xlst, hlst))
flst = ['sg_div_ns_sp_res.tab', 'sg_div_ori_res.tab', 'sg_div_ns_res.tab','benchmark_cfd_sg_res.tab',
        'benchmark_mit_sg_res.tab', 'benchmark_cctop_sg_res.tab',
        'stu_div_common_ns_sp.tab', 'stu_div_common_ori.tab', 'stu_div_common_ns.tab', 'benchmark_cfd_stu_common.tab',
        'benchmark_mit_stu_common.tab', 'benchmark_cctop_stu_common.tab',
        'stu_div_unique_ns_sp.tab', 'stu_div_unique_ori.tab', 'stu_div_unique_ns.tab', 'benchmark_cfd_stu_unique.tab',
        'benchmark_mit_stu_unique.tab', 'benchmark_cctop_stu_unique.tab']
dt = {k:v for k, v in zip(zlst, flst)}

df = pd.read_csv('result/all_res.csv', index_col=None)
row_gen = ((row[1]['x'], row[1]['h']) for row in df.iterrows())

