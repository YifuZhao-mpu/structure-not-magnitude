"""
给已建好的场景索引补上每帧的 3D 标注框。

用途: 累积多帧 LiDAR 作为遮挡证据时, 必须先剔除【动态物体】的点 ——
否则移动车辆/行人会在累积点云里拖出一条轨迹, 制造出相机曝光时刻根本不存在的
"遮挡物", 把 occluded 率从 ~1% 抬到 30%+ (实测)。这是 Codex 所说的
"Accumulation creates temporal geometry", 也是可见性证书最容易被污染的地方。

局限(必须写进论文): 标注框只覆盖被标注的类别。未标注的动态物体(飘动枝叶、
塑料袋等)无法剔除 —— 这正是 unknown 态存在的理由之一。
"""
import json, os, argparse
import numpy as np

NUSC = os.environ["NUSCENES_ROOT"]
V = f"{NUSC}/v1.0-trainval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/nusc_index")
    a = ap.parse_args()

    scenes = sorted(f[:-4] for f in os.listdir(a.index) if f.endswith(".npz"))
    print(f"加载 sample.json / sample_annotation.json ...", flush=True)
    sample = json.load(open(f"{V}/sample.json"))
    ann = json.load(open(f"{V}/sample_annotation.json"))
    scene_tok = {s["token"]: s for s in json.load(open(f"{V}/scene.json"))}
    name2tok = {v["name"]: k for k, v in scene_tok.items()}

    by_sample = {}
    for r in ann:
        by_sample.setdefault(r["sample_token"], []).append(r)
    samples_by_scene = {}
    for s in sample:
        samples_by_scene.setdefault(s["scene_token"], []).append(s)
    for v in samples_by_scene.values():
        v.sort(key=lambda x: x["timestamp"])

    for sn in scenes:
        tok = name2tok.get(sn)
        if tok is None:
            continue
        ss = samples_by_scene[tok]
        centers, sizes, rots, nper = [], [], [], []
        for s in ss:
            bs = by_sample.get(s["token"], [])
            nper.append(len(bs))
            for b in bs:
                centers.append(b["translation"]); sizes.append(b["size"]); rots.append(b["rotation"])
        d = dict(np.load(f"{a.index}/{sn}.npz", allow_pickle=True))
        d["box_centers"] = np.array(centers, np.float32) if centers else np.zeros((0,3), np.float32)
        d["box_sizes"]   = np.array(sizes, np.float32) if sizes else np.zeros((0,3), np.float32)
        d["box_rots"]    = np.array(rots, np.float32) if rots else np.zeros((0,4), np.float32)
        d["box_nper"]    = np.array(nper, np.int32)
        np.savez_compressed(f"{a.index}/{sn}.npz", **d)
    print(f"完成: {len(scenes)} 个场景已补标注框")


if __name__ == "__main__":
    main()
