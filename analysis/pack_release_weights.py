# -*- coding: utf-8 -*-
"""收集论文复现必需的 best_model.h5（3 场景 × 4 模型 × 3 种子），按场景打包供 GitHub Release。

只保留 best_model.h5（ModelCheckpoint save_best_only=True，论文报告的数字来自它），
末轮 model.h5 与消融/敏感性/冒烟/容量实验的旧输出不归档。
"""
import csv, hashlib, os, tarfile, time

ROOT = r"C:\Users\12130\Desktop\CLA小论文撰写\experiments_v2"
OUT = r"C:\Users\12130\Desktop\CLA小论文撰写\拒稿重投\github_release"
REPO = r"C:\Users\12130\Desktop\CLA小论文撰写\拒稿重投\github_repo"

FROZEN = os.path.join(ROOT, "outputs_frozen_manuscript")
# 论文报告的全部数字与这三批逐位一致（见 verify_checkpoint_provenance.py），
# 因此 Release 归档必须取这三批，而不是 9/16 的冻结参数复验批次。
PRIMARY = {
    1: os.path.join(ROOT, "outputs", "optputs_linux_s1"),
    2: os.path.join(ROOT, "outputs", "optputs_linux_s2"),
    3: os.path.join(ROOT, "outputs", "outputs_linux_robustness"),
}

MODELS = ["CLA", "CNN_LSTM", "LSTM_Attention", "2D_CNN_Attention"]
SEEDS = [42, 43, 44]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locate(scenario, model, seed):
    """返回 (源文件路径, 来源标签)。

    首选与论文报告数字同源的批次（manuscript_reported）；只有那里缺失时才退回
    9/16 的冻结参数复验批次，并在清单里如实标注 frozen_revalidation。
    """
    rel = os.path.join("scenario_%d" % scenario, "iid", "order_original", model, "seed_%d" % seed, "best_model.h5")
    p = os.path.join(PRIMARY[scenario], rel)
    if os.path.isfile(p):
        return p, "manuscript_reported"
    p = os.path.join(FROZEN, rel)
    if os.path.isfile(p):
        return p, "frozen_revalidation"
    return None, None


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for scenario in (1, 2, 3):
        members, total = [], 0
        for model in MODELS:
            for seed in SEEDS:
                src, tag = locate(scenario, model, seed)
                if src is None:
                    print("MISSING: scenario_%d/%s/seed_%d" % (scenario, model, seed))
                    continue
                arcname = "scenario_%d/iid/order_original/%s/seed_%d/best_model.h5" % (scenario, model, seed)
                members.append((src, arcname, tag))
                total += os.path.getsize(src)
        tar_path = os.path.join(OUT, "scenario_%d_weights.tar" % scenario)
        t0 = time.time()
        with tarfile.open(tar_path, "w") as tar:
            for src, arcname, tag in members:
                tar.add(src, arcname=arcname, recursive=False)
        dt = time.time() - t0
        for src, arcname, tag in members:
            size = os.path.getsize(src)
            manifest.append({
                "scenario": scenario,
                "model": arcname.split("/")[3],
                "seed": arcname.split("/")[4].replace("seed_", ""),
                "file": arcname,
                "size_mb": round(size / 1048576, 1),
                "md5": md5(src),
                "source": tag,
                "tar": os.path.basename(tar_path),
            })
        print("scenario_%d: %d files, %.1f MB -> %s (%.1fs)"
              % (scenario, len(members), total / 1048576, os.path.basename(tar_path), dt))

    man_path = os.path.join(REPO, "data", "checkpoint_manifest.csv")
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "model", "seed", "file", "size_mb", "md5", "source", "tar"])
        w.writeheader()
        w.writerows(manifest)
    print("\nmanifest ->", man_path, "rows:", len(manifest))
    for t in sorted(os.listdir(OUT)):
        print("  %s  %.1f MB" % (t, os.path.getsize(os.path.join(OUT, t)) / 1048576))


if __name__ == "__main__":
    main()
