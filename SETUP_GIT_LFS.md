# 📦 Setting Up Git LFS for Large Model Files

## What is Git LFS?

Git Large File Storage (LFS) is a Git extension that allows you to store large files (>100MB) on GitHub by replacing them with text pointers in your repository, while storing the actual file content on a remote server.

---

## 🚀 Quick Setup Guide

### Step 1: Install Git LFS

#### On Ubuntu/Debian:
```bash
sudo apt-get install git-lfs
```

#### On macOS:
```bash
brew install git-lfs
```

#### On Windows:
Download from: https://git-lfs.github.com/

### Step 2: Initialize Git LFS in Your Repository

```bash
# Navigate to your repository
cd ~/CCTV/cctv-anomaly-detection/cctv-anomaly-detection/cctv-anomaly-detection-1

# Initialize Git LFS
git lfs install
```

### Step 3: Track Large Files

```bash
# Track the ReID model file
git lfs track "models/person_reid_model.pth"

# Track all .pth files over 100MB (optional)
git lfs track "models/*.pth"

# This creates/updates .gitattributes file
git add .gitattributes
```

### Step 4: Add and Commit the Large File

```bash
# Add the ReID model
git add models/person_reid_model.pth

# Commit
git commit -m "Add Person ReID model via Git LFS (111MB)"

# Push to GitHub
git push origin main
```

---

## 📋 Complete Setup Commands

Run these commands in order:

```bash
# 1. Install Git LFS (if not already installed)
git lfs install

# 2. Track the large model file
git lfs track "models/person_reid_model.pth"

# 3. Add .gitattributes
git add .gitattributes

# 4. Remove the file from .gitignore
# Edit .gitignore and remove the line: models/person_reid_model.pth

# 5. Add the model file
git add models/person_reid_model.pth

# 6. Commit
git commit -m "🤖 Add Person ReID model via Git LFS (111MB)

- Added person_reid_model.pth using Git LFS
- Model size: 111MB
- Enables global person tracking across frames
- Required for full stealing detection functionality"

# 7. Push to GitHub
git push origin main
```

---

## ✅ Verification

After pushing, verify Git LFS is working:

```bash
# Check LFS status
git lfs ls-files

# Should show:
# person_reid_model.pth

# Check file size on GitHub
# The file should appear as ~111MB but stored via LFS
```

---

## 🔍 How Users Will Download It

When others clone your repository:

```bash
# Clone with LFS files
git clone https://github.com/aayishasameer/cctv-anomaly-detection.git
cd cctv-anomaly-detection

# LFS files are automatically downloaded
# If not, manually pull them:
git lfs pull
```

---

## 💰 GitHub LFS Limits

### Free Account:
- **Storage**: 1 GB free
- **Bandwidth**: 1 GB/month free
- Your 111MB model fits easily!

### If You Need More:
- **Data Packs**: $5/month for 50GB storage + 50GB bandwidth
- **GitHub Pro**: Includes more LFS storage

---

## 🎯 Alternative: GitHub Releases

If you don't want to use Git LFS, you can upload the model as a release asset:

### Step 1: Create a Release on GitHub

1. Go to: https://github.com/aayishasameer/cctv-anomaly-detection/releases
2. Click "Create a new release"
3. Tag: `v1.0.0`
4. Title: "CCTV Anomaly Detection v1.0 - Complete System"
5. Upload `person_reid_model.pth` as an asset
6. Publish release

### Step 2: Update Documentation

Add download instructions to README:

```markdown
## Download Large Model

The Person ReID model (111MB) is available in the releases:

1. Go to [Releases](https://github.com/aayishasameer/cctv-anomaly-detection/releases)
2. Download `person_reid_model.pth`
3. Place it in the `models/` directory
```

---

## 🔄 Comparison of Methods

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Git LFS** | ✅ Integrated with Git<br>✅ Automatic download<br>✅ Version control | ⚠️ Requires LFS setup<br>⚠️ Bandwidth limits | Regular updates |
| **GitHub Releases** | ✅ No LFS needed<br>✅ Unlimited size<br>✅ Easy to download | ❌ Manual download<br>❌ Not version controlled | One-time release |
| **External Storage** | ✅ No GitHub limits<br>✅ Fast downloads | ❌ Separate service<br>❌ Link management | Very large files |

---

## 🎯 Recommended Approach

**For your project, I recommend Git LFS because:**

1. ✅ Your model (111MB) is within free limits
2. ✅ Users get it automatically when cloning
3. ✅ Integrated with your Git workflow
4. ✅ Professional and standard approach
5. ✅ Easy to update the model later

---

## 📝 Step-by-Step Implementation

Let me help you set it up right now:

```bash
# Run these commands:

# 1. Install Git LFS
git lfs install

# 2. Track the model
git lfs track "models/person_reid_model.pth"

# 3. Update .gitignore (remove the exclusion)
# Edit .gitignore and remove: models/person_reid_model.pth

# 4. Add files
git add .gitattributes
git add models/person_reid_model.pth

# 5. Commit
git commit -m "🤖 Add Person ReID model via Git LFS (111MB)"

# 6. Push
git push origin main
```

---

## 🆘 Troubleshooting

### Issue: "This exceeds GitHub's file size limit"
**Solution**: Make sure Git LFS is installed and tracking is set up:
```bash
git lfs install
git lfs track "models/person_reid_model.pth"
```

### Issue: "Git LFS not found"
**Solution**: Install Git LFS first:
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs
```

### Issue: File still too large
**Solution**: Verify LFS is tracking the file:
```bash
git lfs ls-files
# Should show your model file
```

---

## 📚 Additional Resources

- Git LFS Documentation: https://git-lfs.github.com/
- GitHub LFS Guide: https://docs.github.com/en/repositories/working-with-files/managing-large-files
- Git LFS Tutorial: https://www.atlassian.com/git/tutorials/git-lfs

---

## ✅ After Setup

Once Git LFS is set up, your repository will:

1. ✅ Include the full ReID model
2. ✅ Allow automatic downloads for users
3. ✅ Support model updates via Git
4. ✅ Stay within GitHub's free tier
5. ✅ Provide professional model distribution

---

**Ready to set it up? Let me know and I'll help you through each step!**
