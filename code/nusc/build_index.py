"""
GATE C 第一步: 把 nuScenes 的 1.9GB JSON metadata 压成每场景一个紧凑 npz。

后续所有步骤(可见性估计/训练/评估)只读这些 npz, 不再碰原始 JSON。

场景选择遵循 EXPERIMENT_PLAN 的实验单元定义: 场景是配对单元, seeds 嵌套其中,
所以需要 >=32 个独立场景才能测到 0.5-sd 的效应(实测场景间 CV=0.28)。
"""
import json, os, sys, argparse
import numpy as np

NUSC = os.environ["NUSCENES_ROOT"]
V = f"{NUSC}/v1.0-trainval"
CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])


def se3(rot, t):
    T = np.eye(4); T[:3, :3] = quat_to_R(rot); T[:3, 3] = t
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=40)
    ap.add_argument("--out", default="data/nusc_index")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("加载 metadata ...", flush=True)
    L = lambda n: json.load(open(f"{V}/{n}"))
    scene = L("scene.json"); sample = L("sample.json")
    calib = {r["token"]: r for r in L("calibrated_sensor.json")}
    sensor = {r["token"]: r for r in L("sensor.json")}
    print("  sample_data.json (1.3GB) ...", flush=True)
    sd_all = L("sample_data.json")
    print("  ego_pose.json (616MB) ...", flush=True)
    ego = {r["token"]: r for r in L("ego_pose.json")}

    # 只保留 keyframe 的 sample_data, 按 (sample, channel) 索引
    bysam = {}
    for r in sd_all:
        if not r["is_key_frame"]:
            continue
        ch = sensor[calib[r["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        bysam[(r["sample_token"], ch)] = r
    del sd_all
    samples_by_scene = {}
    for s in sample:
        samples_by_scene.setdefault(s["scene_token"], []).append(s)
    for v in samples_by_scene.values():
        v.sort(key=lambda x: x["timestamp"])

    kw = ["rain", "night", "construction"]
    cand = [s for s in scene
            if not any(k in s["description"].lower() for k in kw) and s["nbr_samples"] >= 39]
    cand.sort(key=lambda s: s["name"])
    sel = cand[:a.n_scenes]
    print(f"候选 {len(cand)} -> 选取 {len(sel)}", flush=True)

    for si, sc in enumerate(sel):
        ss = samples_by_scene[sc["token"]]
        rec = {"scene": sc["name"], "desc": sc["description"]}
        img_paths, Ks, c2w, cam_ids = [], [], [], []
        lid_paths, l2w = [], []
        for si2, s in enumerate(ss):
            sdl = bysam.get((s["token"], "LIDAR_TOP"))
            if sdl is None:
                continue
            cl = calib[sdl["calibrated_sensor_token"]]; el = ego[sdl["ego_pose_token"]]
            T = se3(el["rotation"], el["translation"]) @ se3(cl["rotation"], cl["translation"])
            lid_paths.append(sdl["filename"]); l2w.append(T)
            for ci, ch in enumerate(CAMS):
                sdc = bysam.get((s["token"], ch))
                if sdc is None:
                    continue
                cc = calib[sdc["calibrated_sensor_token"]]; ec = ego[sdc["ego_pose_token"]]
                Tc = se3(ec["rotation"], ec["translation"]) @ se3(cc["rotation"], cc["translation"])
                img_paths.append(sdc["filename"]); Ks.append(np.array(cc["camera_intrinsic"]))
                c2w.append(Tc); cam_ids.append(ci)
        np.savez_compressed(
            f"{a.out}/{sc['name']}.npz",
            img_paths=np.array(img_paths), Ks=np.array(Ks), c2w=np.array(c2w),
            cam_ids=np.array(cam_ids), lid_paths=np.array(lid_paths), l2w=np.array(l2w),
            scene=sc["name"], desc=sc["description"])
        if si % 10 == 0 or si == len(sel)-1:
            print(f"  [{si+1}/{len(sel)}] {sc['name']}: {len(img_paths)} imgs, {len(lid_paths)} sweeps", flush=True)
    print(f"完成 -> {a.out}")


if __name__ == "__main__":
    main()
