# LumiGrasp
<<<<<<< Updated upstream
=======

## 1. Set Up

### conda environment

初步调研，使用较新的稳定版本 `python=3.11` `bpy=5.0.1` 
```
conda create -n mubl_sim python=3.11 -y
conda activate mubl_sim

python -m pip install -U pip
python -m pip install bpy

pip install openexr imath h5py # for blender Z buffer 读取
```

### related repos

```
cd ../Opensource_Forks
git clone https://github.com/michaal94/MuBlE
git clone https://github.com/ARISE-Initiative/robomimic
ln -s MuBlE /path/to/LumiGrasp/thrid_party
ln -s robomimic /path/to/LumiGrasp/thrid_party
cd ../LumiGrasp
```



## 2. Bridge Demo

当前第一版桥接目标是：robosuite / MuJoCo 负责物理，Blender 负责渲染，项目代码只做状态读取和位姿同步，不修改三方库。

环境确认：

- -name `mubl_sim`
- Python 3.11
- MuJoCo 3.8.1
- robosuite 1.5.2
- bpy 5.0.1

详见文档 `docs/readme_bridge.md`

## 3. Progress

详见文档 `docs/progress.md`
>>>>>>> Stashed changes
