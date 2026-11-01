# 推送到 GitHub 的操作步骤

本地仓库已经初始化并提交完成（commit `a9eb173`，约 814 KB）。剩下只需三步。

---

## 第 1 步：在 GitHub 网页上创建一个空仓库

1. 打开 https://github.com/new
2. Repository name 填：`cla-reliability-aware-inversion`
3. Description 可填：
   `Reliability-aware source-term inversion under degraded atmospheric radiation monitoring: code, audits and figure source data`
4. 选 **Public**（JER 要求数据可公开获取）
5. **不要**勾选 "Add a README file"、"Add .gitignore"、"Choose a license"
   —— 这三个本地已经有了，勾选会导致推送冲突。
6. 点 Create repository。

创建后页面会显示一个地址，形如：
```
https://github.com/<你的用户名>/cla-reliability-aware-inversion.git
```

## 第 2 步：在本地关联并推送

在 Git Bash 里执行（把 `<你的用户名>` 换成实际用户名）：

```bash
cd "/c/Users/12130/Desktop/CLA小论文撰写/拒稿重投/github_repo"

git remote add origin https://github.com/<你的用户名>/cla-reliability-aware-inversion.git
git branch -M main
git push -u origin main
```

### 认证方式

GitHub 已不支持账号密码推送，三选一：

- **推荐：Personal Access Token**
  GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
  → Generate new token → 勾选 `repo` → 生成后复制。
  推送时 Username 填你的 GitHub 用户名，Password 填这个 token。
- **SSH**：`ssh-keygen -t ed25519` 生成密钥，把公钥加到 GitHub → Settings → SSH and GPG keys，
  然后把 remote 改成 `git@github.com:<用户名>/cla-reliability-aware-inversion.git`。
- **GitHub Desktop**：图形界面，自动处理认证，最省事。

### 如果国内访问 GitHub 不稳定

先测试连通性：
```bash
ssh -T git@github.com          # 走 SSH
git ls-remote https://github.com/<用户名>/cla-reliability-aware-inversion.git
```
若超时，可为 git 配置代理（按你实际代理端口改）：
```bash
git config http.proxy http://127.0.0.1:7890
git config https.proxy http://127.0.0.1:7890
```
或改用 Gitee 建镜像仓库，正文里同时给出 GitHub 与 Gitee 两个链接。

## 第 3 步：把 1.0 GB 权重作为 Release 附件上传（数据已在仓库里）

### 为什么权重不直接进仓库

HDF5 权重**几乎压不动**（实测 100.8 MB → 93.4 MB，只减 7%），原因是 Keras 存了 Adam 优化器状态。
而 GitHub 仓库内单文件硬上限 100 MB，单个 144 MB / 100.8 MB 的 `.h5` 直接 push 会被拒绝。
Release 附件单文件上限 2 GB，且**不计入仓库体积**，所以仓库仍是 60 MB，clone 只要几秒。

### 打包结果（已做好，在 `拒稿重投/github_release/`）

| 附件 | 内容 | 大小 | MD5 |
|---|---|---|---|
| `scenario_1_weights.tar` | 场景1：4 模型 × 3 种子 | 259.9 MB | `df960e9e8952c600c51b508437b1df7f` |
| `scenario_2_weights.tar` | 场景2：4 模型 × 3 种子 | 259.9 MB | `e003747e6e78476c85fad0505d24dca0` |
| `scenario_3_weights.tar` | 场景3：4 模型 × 3 种子 | 267.8 MB | `b136fe77a7dc452a3be3699383308bdd` |

合计 788 MB。内部路径 `scenario_X/iid/order_original/<MODEL>/seed_<SEED>/best_model.h5`，
只保留 `best_model.h5`（验证损失最小的那一轮），末轮 `model.h5` 不归档。
逐文件校验和见 `data/checkpoint_manifest.csv`（36 行，来源均为与稿件数字同源的那批运行）。

自己重做：
```bash
python3 analysis/verify_checkpoint_provenance.py   # 先核对数字来自哪一批
python3 analysis/pack_release_weights.py           # 再按场景打包
```

### 上传方式（三选一）

**A. 网页（最直观）**
1. 打开仓库页面 → 右侧 **Releases** → **Create a new release**
2. Tag version 填 `v1.0.0`，标题例如 `Model checkpoints for the JER submission`
3. 把三个 `.tar` 拖进 "Attach binaries" 框（单附件上限 2 GB，都合规）
4. 建议**逐个上传**：先建 release、只传第一个附件并保存，再编辑 release 补传后两个，
   避免一次性 1 GB 传到最后失败要重来
5. Publish release

**B. gh 命令行（可后台跑，断线可重试）**
```bash
gh auth login
gh release create v1.0.0 \
  "/c/Users/12130/Desktop/CLA小论文撰写/拒稿重投/github_release/scenario_1_weights.tar" \
  "/c/Users/12130/Desktop/CLA小论文撰写/拒稿重投/github_release/scenario_2_weights.tar" \
  "/c/Users/12130/Desktop/CLA小论文撰写/拒稿重投/github_release/scenario_3_weights.tar" \
  --repo <你的用户名>/cla-reliability-aware-inversion \
  --title "Model checkpoints for the JER submission" \
  --notes "36 best_model.h5 checkpoints (4 models x 3 seeds x 3 scenarios); see data/checkpoint_manifest.csv for per-file checksums."
```

**C. 一定要留在仓库目录里的话**
把每个 >100 MB 的 `.h5` 切成 ≤95 MB 分片再入仓：
```bash
split -b 95m best_model.h5 best_model.h5.part-
# 恢复：cat best_model.h5.part-* > best_model.h5
```
代价是仓库变成约 1 GB、clone 要几分钟，且 GitHub 会提示仓库偏大，所以不推荐。

### 正文 Data availability 建议改为

> Code, the full CALPUFF simulated datasets (gzip-compressed), the trained model checkpoints,
> the figure source data, the exact receptor coordinates and the CALPUFF input-parameter audit
> tables are deposited in a public GitHub repository (URL to be inserted at proof stage); the
> checkpoints are attached to the v1.0.0 release of that repository.

   把两处 "to be inserted at proof stage" 换成实际 URL 与 DOI 即可；
   GitHub 仓库 URL 直接用第 2 步的 `https://github.com/<用户名>/cla-reliability-aware-inversion`。

### 气象场：三个场景共用一份（已解决，已归档）

之前的疑问（场景 2 / 3 的气象场不在本机）已经澄清：**三个场景的扩散数据由同一份 CALMET 输出
生成**。三个 INP 里的 METDAT 指向 E 盘三个不同目录，但那份气象文件本身是同一份。

| 项目 | 值 |
|---|---|
| 归档文件 | `data/calpuff_inputs/CALMET01_shared.DAT`（2.4 MB） |
| 来源 | `CAL\GBHG气象、地形\` 根目录下标注"三个场景的扩散数据都是用这个气象文件生成的"那份 |
| 气象场网格 | 20×20，0.5 km，原点 341.580 / 6174.610，UTM 33，NZ=3 |
| 时段 | 1978-10-19，IBHR=0 → IEHR=**23**（0–23 时，20 min 步长） |
| 覆盖情况 | 论文模拟 12–16 时，完整落在气象场时段内，无需外推或复用最后一小时 |

三个场景的 CALPUFF 受体网格不同（s1 为 20×20 原点 341.58；s2/s3 为 25×25 原点 339.08），
由各自的 `CALPUFF-sX.INP` 定义，CALPUFF 在共享气象场上插值取值——这正是当时的实际配置。

同时归档的还有：`CALMET_shared.INP`（从 DAT 内嵌控制文件块逐条记录提取，与实际生成该气象场
时使用的一致）、`CALMET_shared.LST`、`surf.dat`、`up2.dat`、`geo.dat`、`lu.dat`、
`ctgproc.inp`、`GBHG_terrain.tif`。

未归档：CTG 原始地形栅格 `euogel20.tif`（169 MB，过大），详见 `docs/meteorology_inputs.md`。

注意：`CALPUFF模拟福岛核事故` 那套是**另一个项目**（日本 UTM 54、160 km 域、CS-137），
与本论文无关，不要混进归档。

---

## 推送前建议再检查一遍

```bash
# 确认没有超过 100 MB 的文件（GitHub 硬上限）
git ls-files | xargs -I{} du -m {} | sort -rn | head -5

# 确认 .gitignore 生效（不应出现 outputs/、*.npy、未压缩的 data s*.csv）
git ls-files | grep -E "outputs|\.npy|data s[123]" || echo "OK: 大文件已排除"
```

仓库现在约 63 MB：代码 12 个文件、分析脚本 10 个、逐图源数据 10 个 CSV、审计结果 11 个 CSV，
外加三份压缩后的模拟数据（60 MB）。推送 60 MB 数据在国内网络下大概几分钟到十几分钟，
若中途断线可以重复执行 `git push`（git 是可续传的，比网页上传稳）。

> 如果之后觉得 60 MB 还是太慢，可以用 Git LFS 托管这三个 .gz（GitHub 免费额度 1 GB 存储/月），
> 但普通 push 已经够用，不建议再增加复杂度。

---

## 可选：给仓库打一个与投稿版本对应的 tag

```bash
git tag -a v1.0.0 -m "Version submitted to Journal of Environmental Radioactivity"
git push origin v1.0.0
```

打 tag 后，如果之后在 Zenodo 上开启 GitHub 联动，每次发版会自动归档一份带 DOI 的快照，
方便引用"与稿件完全一致"的那一版代码。
