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

## 第 3 步：把大文件上传到 Zenodo，并在正文填 DOI

GitHub 单文件上限 100 MB、仓库建议不超过 1 GB，所以 ~900 MB 的模拟数据和模型权重**不要**放进仓库。

1. 打开 https://zenodo.org → Log in with GitHub → Upload → New upload
2. 分两个 deposit（便于分别引用）：
   - **Dataset**：`data s1.csv`、`data s2.csv`、`data s3.csv` + 三个 `CALPUFF-s*.INP` + CALMET 输出
   - **Models**：6 个训练好的 checkpoint（Scenario 3 的 CLA / CNN-LSTM / LSTM-Attention / 2D-CNN-Attention）
3. License 选 `Creative Commons Attribution 4.0`
4. Resource type 分别选 Dataset / Software
5. 发布后拿到两个 DOI，填回正文的 Data availability 段。
   对应稿件中那句话现在写的是：
   > Code, figure source data, the exact receptor coordinates and the CALPUFF input-parameter audit
   > tables are deposited in a public GitHub repository (URL to be inserted at proof stage); the
   > ~900 MB CALPUFF simulated datasets and the six trained model checkpoints are deposited in Zenodo
   > (DOI to be inserted at proof stage).

   把两处 "to be inserted at proof stage" 换成实际 URL 与 DOI 即可。

---

## 推送前建议再检查一遍

```bash
# 确认没有大文件被误加入
git ls-files | xargs -I{} du -h {} | sort -rh | head -5

# 确认 .gitignore 生效（不应出现 outputs/、*.npy、data s*.csv）
git ls-files | grep -E "outputs|\.npy|data s[123]" || echo "OK: 大文件已排除"
```

仓库里目前只有：代码 12 个文件、分析脚本 10 个、逐图源数据 10 个 CSV、审计结果 11 个 CSV，
总计 814 KB，可直接推送。

---

## 可选：给仓库打一个与投稿版本对应的 tag

```bash
git tag -a v1.0.0 -m "Version submitted to Journal of Environmental Radioactivity"
git push origin v1.0.0
```

打 tag 后，如果之后在 Zenodo 上开启 GitHub 联动，每次发版会自动归档一份带 DOI 的快照，
方便引用"与稿件完全一致"的那一版代码。
