# LumiGrasp

## 1. Set Up

### conda environment

```
conda create -n mubl_sim python=3.11 -y
conda activate mubl_sim

python -m pip install -U pip
python -m pip install bpy

pip install openexr imath h5py # for blender Z buffer reading
pip install matplotlib # for data visualization

pip install torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124 # for training
```

### related repos

```
cd ../Opensource_Forks
git clone https://github.com/michaal94/MuBlE
git clone https://github.com/ARISE-Initiative/robomimic
ln -s MuBlE /path/to/LumiGrasp/third_party
ln -s robomimic /path/to/LumiGrasp/thrid_party
cd ../LumiGrasp
```



## 2. Bridge Demo

In our bridge setup:

- robosuite / MuJoCo handle physics simulation
- Blender handles rendering

We use a bridge to synchronize the two environments without modifying the source code of these site-packages.

Run `conda list` and check：

- environment name `mubl_sim`
- Python 3.11
- MuJoCo 3.8.1
- robosuite 1.5.2
- bpy 5.0.1
- torch 2.6.0 + torchvision 0.21.0 + cu124

More details, refer to `docs/readme_bridge.md`

## 3. Progress

Refer to `docs/progress.md`
