import os, json

results_dir = r'D:\federated_phishing\results'
files = sorted([f for f in os.listdir(results_dir) if f.startswith('E') and f.endswith('_final.json')])

print(f"{'Exp':<6} {'Model':<12} {'Aggregation':<10} {'Non-IID Type':<25} {'F1':>8} {'Train Size':>12}")
print('-'*78)

for f in files:
    with open(os.path.join(results_dir, f)) as fp:
        r = json.load(fp)
    train_size = r.get('train_size', 'unknown')
    print(
        f"{r['experiment_id']:<6} "
        f"{r['model']:<12} "
        f"{r['aggregation']:<10} "
        f"{r['non_iid_type']:<25} "
        f"{float(r['f1']):>8.4f} "
        f"{str(train_size):>12}"
    )

print(f"\nTotal experiments: {len(files)}")
