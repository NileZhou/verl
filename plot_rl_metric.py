import re
import matplotlib.pyplot as plt
from collections import defaultdict

# log_file = "/njfs/train-aitech/projects/zhouyi9/projects/coderl/Archer2.0/scripts/train/s_3b_rl/s_run_3b_rl_code16k_0.125-0.875_offpolicy_no-uly_lcbpre.log"
log_file = "/njfs/train-aitech/projects/zy9/verl/recipe/my_dapo/run_my_dapo_offpolicy_mini_c3_latest.log"
log_file = "/njfs/train-aitech/projects/zy9/verl/recipe/dapo/run_archer_3b_code16k_baseline.log"
log_file = "/njfs/train-aitech/projects/zy9/verl/recipe/my_dapo/run_my_dapo_offpolicy_mini_c6_latest.log"
pattern = re.compile(r'step:(\d+) - (.+)')
metrics = defaultdict(dict)

with open(log_file) as f:
    for line in f:
        # 去掉 ANSI 颜色码
        clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
        m = pattern.search(clean)
        if m:
            step = int(m.group(1))
            pairs = m.group(2).split(' - ')
            for pair in pairs:
                if ':' in pair:
                    k, v = pair.split(':', 1)
                    try:
                        metrics[k][step] = float(v)
                    except ValueError:
                        pass

# 要画的指标
keys_to_plot = [
    'critic/score/mean',
    'actor/entropy',
    'actor/pg_loss',
    'actor/grad_norm',
    'response_length/mean',
    'val-core/livecodebench/acc/mean@4',
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for ax, key in zip(axes, keys_to_plot):
    if key in metrics:
        steps = sorted(metrics[key].keys())
        vals = [metrics[key][s] for s in steps]
        ax.plot(steps, vals)
        ax.set_title(key)
        ax.set_xlabel('step')
        ax.grid(True)
plt.tight_layout()
plt.savefig('/data0/users/zhouyi9/projects/obuild/training_metrics.png', dpi=150)