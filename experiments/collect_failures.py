"""Collect failures for the policy selected in config.py: python -m experiments.collect_failures."""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
import torch

from sim.simulation import CozmoSim
from experiments.failure_plots import state_comparison_report
from config import TASK, POLICY_TYPE
from train.utils.environment import CozmoEnv
from utils import HZ


STATE_FIELDS = ["pose_x_mm", "pose_y_mm", "pose_sin", "pose_cos",
                "lwheel_rad_s", "rwheel_rad_s", "lift_rad", "head_rad",
                "cube_accel_x_g", "cube_accel_y_g", "cube_accel_z_g"]

METRIC_FIELDS = ["initial_value", "cumulative_reward", "discounted_return"]


def trajectory_metrics(model, initial_obs, rewards):
    """Evaluate the critic on exact policy inputs; rewards exclude bootstrapping."""
    model.policy.set_training_mode(False)
    with torch.inference_mode():
        tensor_obs, _ = model.policy.obs_to_tensor(initial_obs)
        initial_value = float(model.policy.predict_values(tensor_obs).item())
    rewards = np.asarray(rewards, dtype=np.float64)
    return {
        "initial_value": initial_value,
        "cumulative_reward": float(rewards.sum()),
        "discounted_return": float(np.dot(model.gamma ** np.arange(len(rewards)), rewards)),
    }


def metric_metadata(model):
    return {
        "critic_gamma": float(model.gamma),
        "metric_definitions": {
            "initial_value": "PPO critic V(o_0), using saved state and noisy vision",
            "cumulative_reward": "sum(r_t), t=0..T-1; same as return",
            "discounted_return": "sum(gamma**t * r_t), t=0..T-1; no bootstrap",
        },
        "value_note": "Critic predicts discounted future reward. Timeout is truncation, so "
                      "discounted_return omits a possible gamma**T * V(o_T) continuation. "
                      "These reward metrics are not the paper's reach-avoid value.",
    }


def update_metrics(output, model, model_path):
    """Recompute reports from saved endpoint observations, without simulation."""
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != summary["model_sha256"]:
        raise SystemExit("Model hash differs from the collected policy; metrics were not updated.")
    records = []
    for path in sorted(output.glob("sample_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
            initial_obs = {key: arrays[f"initial_{key}"] for key in ("state", "vision")}
            if len(arrays["rewards"]) != record["steps"]:
                raise ValueError(f"{path}: reward count differs from recorded steps")
            metrics = trajectory_metrics(model, initial_obs, arrays["rewards"])
            if not np.isclose(metrics["cumulative_reward"], record["return"]):
                raise ValueError(f"{path}: saved rewards disagree with recorded return")
        record.update(metrics)
        record.pop("final_value", None)
        records.append(record)
    if len(records) != summary["failures_collected"]:
        raise ValueError("Saved sample count differs from summary; metrics were not updated")
    episodes_path = output / "episodes.json"
    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    by_seed = {record["seed"]: record for record in records}
    for episode in episodes:
        episode.pop("final_value", None)
        if not episode["is_success"] and episode["seed"] in by_seed:
            episode.update({key: by_seed[episode["seed"]][key] for key in METRIC_FIELDS})
    summary.update(metric_metadata(model))
    summary.update(episodes_evaluated=len(episodes),
                   successes_observed=sum(bool(ep["is_success"]) for ep in episodes))
    for record in records:
        (output / f'{record["sample"]}.json').write_text(json.dumps(record, indent=2), encoding="utf-8")
    episodes_path.write_text(json.dumps(episodes, indent=2), encoding="utf-8")
    write_report(output, records, summary)
    print(f"Total episodes evaluated: {len(episodes)} | "
          f"successes: {summary['successes_observed']} | failures collected: {len(records)}")
    print(f"Updated reproducible trajectory metrics for {len(records)} samples in {output}")


def update_plots(output, model, model_path):
    """Backfill legacy state endpoints with verified replays, then rebuild plots."""
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if (hashlib.sha256(model_path.read_bytes()).hexdigest() != summary["model_sha256"]
            or summary["task"] != str(TASK)):
        raise SystemExit("Plot replay requires the original model and task.")
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted(output.glob("sample_*.json"))]
    episodes_path = output / "episodes.json"
    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    endpoints = ("initial_state", "state", "initial_raw_state", "raw_state")
    by_seed = {record["seed"]: record for record in records}
    for ep in episodes:
        if not ep["is_success"] and ep["seed"] in by_seed:
            ep.update({key: by_seed[ep["seed"]][key] for key in endpoints})

    def save_progress():
        temporary = episodes_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(episodes, indent=2), encoding="utf-8")
        temporary.replace(episodes_path)
        write_report(output, records, summary)

    save_progress()
    missing = [ep for ep in episodes if any(key not in ep for key in endpoints)]
    if missing:
        env = CozmoEnv(CozmoSim(), TASK)
        try:
            for index, ep in enumerate(missing, 1):
                obs, _ = env.reset(seed=ep["seed"])
                initial_state = obs["state"].tolist()
                initial_raw = {key: float(value) for key, value in env.sim.get_raw_state().items()}
                rewards = []
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    rewards.append(float(reward))
                    if terminated or truncated:
                        break
                if (bool(info["is_success"]) != ep["is_success"] or len(rewards) != ep["steps"]
                        or not np.isclose(sum(rewards), ep["return"], rtol=1e-7, atol=1e-7)):
                    raise ValueError(f"Seed {ep['seed']}: replay differs from saved episode; "
                                     "its states were not saved. Previously verified states are retained.")
                ep.update(initial_state=initial_state, state=obs["state"].tolist(),
                          initial_raw_state=initial_raw,
                          raw_state={key: float(value) for key, value in env.sim.get_raw_state().items()})
                # Checkpoint every episode so interruption never loses verified replays.
                temporary = episodes_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(episodes, indent=2), encoding="utf-8")
                temporary.replace(episodes_path)
                print(f"Verified state replay {index}/{len(missing)} | seed {ep['seed']}", flush=True)
                if index % 10 == 0:
                    write_report(output, records, summary)
        finally:
            if env.sim.video_renderer is not None:
                env.sim.video_renderer.close()
            env.sim.renderer.close()
            env.close()
    save_progress()
    print(f"Updated state comparisons for all {len(episodes)} episodes in {output}")


def snapshot(sim, obs, renderer):
    """Read the current state without stepping or drawing additional sensor noise."""
    renderer.update_scene(sim.data, camera="cozmo_cam", scene_option=sim.cam_option)
    head = renderer.render().copy()
    robot = sim.data.body("cozmo").xpos.copy()
    cube = sim.data.joint(sim.target_joint).qpos[:3].copy()
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (robot + cube) / 2
    camera.distance = max(0.65, float(np.linalg.norm(robot - cube)) * 1.8)
    camera.azimuth = 135
    camera.elevation = -55
    renderer.update_scene(sim.data, camera=camera)
    world = renderer.render().copy()
    return {
        "head": head, "world": world,
        "state": obs["state"].copy(), "vision": obs["vision"].copy(),
        "raw": {key: float(value) for key, value in sim.get_raw_state().items()},
        "qpos": sim.data.qpos.copy(), "qvel": sim.data.qvel.copy(),
    }


def save_pair(path, shot, title):
    image = Image.new("RGB", (640, 310), "#17212b")
    draw = ImageDraw.Draw(image)
    draw.text((10, 8), title, fill="white")
    draw.text((10, 32), "Head camera (RGB, before sensor noise)", fill="white")
    draw.text((330, 32), "World view", fill="white")
    image.paste(Image.fromarray(shot["head"]), (0, 60))
    image.paste(Image.fromarray(shot["world"]), (320, 60))
    image.save(path)


def write_report(output, records, summary, render_plots=True):
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "samples.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["sample", "episode", "seed", "steps",
                                                "return", *METRIC_FIELDS, "distance_mm", *STATE_FIELDS])
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in writer.fieldnames})
    cards = []
    for record in records:
        name = record["sample"]
        rows = "".join(f"<tr><td>{field}</td><td>{a:.5f}</td><td>{b:.5f}</td></tr>"
                       for field, a, b in zip(STATE_FIELDS, record["initial_state"], record["state"]))
        cards.append(f'<article><h2>{name} | seed {record["seed"]}</h2>'
                     f'<p>Episode ended at {record["steps"]} steps; '
                     f'final cube distance {record["distance_mm"]:.1f} mm</p>'
                     f'<p>Critic V(initial): {record["initial_value"]:.6f}; '
                     f'actual cumulative reward: {record["cumulative_reward"]:.6f}; '
                     f'discounted return: {record["discounted_return"]:.6f}</p>'
                     f'<img src="{name}_initial.png" alt="Initial head and world views">'
                     f'<img src="{name}_failure.png" alt="Timeout head and world views">'
                     '<table><tr><th>Policy state component</th><th>Initial</th><th>Timeout</th></tr>'
                     f'{rows}</table><p><a href="{name}.json">Full state and metadata</a> | '
                     f'<a href="{name}.npz">Observation arrays and trajectory</a></p></article>')
    if render_plots:
        episodes_path = output / "episodes.json"
        episodes = json.loads(episodes_path.read_text(encoding="utf-8")) if episodes_path.exists() else []
        comparisons = state_comparison_report(output, episodes)
    else:
        comparisons = '<p>State comparison plots will be generated when collection finishes.</p>'
    page = ('<!doctype html><meta charset="utf-8"><title>Policy failures</title>'
            '<style>body{font:16px system-ui;background:#101820;color:#eee;margin:24px}'
            'article{background:#1c2935;padding:18px;margin:24px 0;border-radius:8px}'
            'img{width:640px;max-width:100%;margin:4px}table{border-collapse:collapse}'
            'td,th{padding:4px 18px;text-align:left;border-bottom:1px solid #456}'
            'a{color:#8cd2ff}pre{white-space:pre-wrap}'
            'img.comparison{width:100%;max-width:1430px}figure{margin:20px 0}'
            '</style><h1>Policy failure samples</h1>'
            f'<p><strong>Total episodes evaluated: {summary["episodes_evaluated"]}</strong> '
            f'(successes: {summary["successes_observed"]}; '
            f'failures collected: {summary["failures_collected"]})</p>'
            '<p>Failure = episode ended with is_success=false. '
            'One sample per failed episode. These are observed timeout states, not proof of where '
            'a failure first became inevitable. State vectors below are normalized policy inputs; '
            'physical state and privileged cube position are in each JSON. Images show initial '
            'and terminal states; NPZ includes the exact noisy 3-frame policy observations.</p>'
            + comparisons + f'<details><summary>Collection metadata</summary>'
            f'<pre>{html.escape(json.dumps(summary, indent=2))}</pre></details>' + ''.join(cards))
    (output / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--max-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted collection")
    parser.add_argument("--update-plots", action="store_true",
                        help="Build state plots, replaying legacy episodes missing saved states")
    parser.add_argument("--update-metrics", action="store_true",
                        help="Recompute metrics in existing samples using saved observations and rewards")
    args = parser.parse_args()
    if args.samples < 1 or args.max_episodes < args.samples or args.seed < 0:
        parser.error("samples must be positive, max-episodes >= samples, and seed >= 0")
    output = Path("experiments") / "failure_samples" / str(TASK)
    model_path = Path("models") / POLICY_TYPE / f"{TASK}.zip"
    torch.set_num_threads(1)
    if args.update_plots:
        update_plots(output, PPO.load(model_path, device="cpu"), model_path)
        return
    if args.update_metrics:
        update_metrics(output, PPO.load(model_path, device="cpu"), model_path)
        return
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not args.resume:
        raise SystemExit(f"{output} already contains samples. Move or clear them before collecting again.")
    model = PPO.load(model_path, device="cpu")
    records, episodes = [], []
    if args.resume:
        saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if (saved["model_sha256"] != hashlib.sha256(model_path.read_bytes()).hexdigest()
                or saved["base_seed"] != args.seed
                or saved["requested_samples"] != args.samples
                or "collection_seeds" in saved):
            raise SystemExit("Resume requires the original model, seed, sample target, and consecutive seeds.")
        episodes = json.loads((output / "episodes.json").read_text(encoding="utf-8"))
        records = [json.loads(path.read_text(encoding="utf-8"))
                   for path in sorted(output.glob("sample_*.json"))]
        if len(records) != saved["failures_collected"] or any(
                ep["seed"] != args.seed + i for i, ep in enumerate(episodes)):
            raise SystemExit("Saved collection is inconsistent; cannot resume.")
        if len(records) >= args.samples:
            print("Collection is already complete.")
            return
    env = CozmoEnv(CozmoSim(), TASK)
    renderer = mujoco.Renderer(env.sim.model, 240, 320)
    env.sim.add_context(renderer._gl_context, renderer._mjr_context)
    summary = {
        "model": str(model_path.resolve()),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "task": str(TASK), "policy_type": POLICY_TYPE, "deterministic_policy": True,
        "failure_definition": "episode ended with info['is_success'] false",
        "requested_samples": args.samples, "base_seed": args.seed,
        "state_fields": STATE_FIELDS, "state_units": "normalized [-1, 1]",
        "raw_state_units": "mm, rad, rad/s, g; pose and cube xy in robot spawn frame",
        **metric_metadata(model),
    }
    try:
        for episode in range(len(episodes), args.max_episodes):
            seed = args.seed + episode
            obs, _ = env.reset(seed=seed)
            initial = snapshot(env.sim, obs, renderer)
            initial_obs = {key: obs[key].copy() for key in ("state", "vision")}
            states, actions, rewards = [obs["state"].copy()], [], []
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                states.append(obs["state"].copy())
                actions.append(action.copy())
                rewards.append(float(reward))
                if terminated or truncated:
                    break
            failed = not bool(info["is_success"])
            metrics = trajectory_metrics(model, initial_obs, rewards)
            episodes.append({"episode": episode, "seed": seed, "is_success": bool(info["is_success"]),
                             "steps": env.sim.step_count, "return": sum(rewards), **metrics,
                             "initial_state": initial["state"].tolist(), "state": obs["state"].tolist(),
                             "initial_raw_state": initial["raw"],
                             "raw_state": {key: float(value) for key, value in env.sim.get_raw_state().items()}})
            if failed:
                shot = snapshot(env.sim, obs, renderer)
                name = f"sample_{len(records) + 1:03d}"
                save_pair(output / f"{name}_initial.png", initial, f"{name} | initial | seed {seed}")
                save_pair(output / f"{name}_failure.png", shot, f"{name} | failure | step {env.sim.step_count}")
                raw = shot["raw"]
                record = {
                    "sample": name, "episode": episode, "seed": seed,
                    "steps": env.sim.step_count, "seconds": env.sim.step_count / HZ,
                    "return": sum(rewards), "is_success": False, "terminated": bool(terminated),
                    **metrics,
                    "truncated": bool(truncated), "failure_reason": "timeout_without_success" if truncated else "terminated_without_success",
                    "distance_mm": float(np.hypot(raw["cube_x"] - raw["pose_x"], raw["cube_y"] - raw["pose_y"])),
                    "state_fields": STATE_FIELDS, "state": shot["state"].tolist(),
                    "initial_state": initial["state"].tolist(),
                    "raw_state": raw, "initial_raw_state": initial["raw"],
                    **dict(zip(STATE_FIELDS, shot["state"].tolist())),
                }
                (output / f"{name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                np.savez_compressed(output / f"{name}.npz",
                                    initial_state=initial["state"], state=shot["state"],
                                    initial_vision=initial["vision"], vision=shot["vision"],
                                    initial_qpos=initial["qpos"], initial_qvel=initial["qvel"],
                                    qpos=shot["qpos"], qvel=shot["qvel"],
                                    states=np.asarray(states), actions=np.asarray(actions),
                                    rewards=np.asarray(rewards))
                records.append(record)
            summary.update(episodes_evaluated=len(episodes), failures_collected=len(records),
                           successes_observed=sum(bool(ep["is_success"]) for ep in episodes),
                           success_rate=sum(ep["is_success"] for ep in episodes) / len(episodes),
                           complete=len(records) >= args.samples)
            (output / "episodes.json").write_text(json.dumps(episodes, indent=2), encoding="utf-8")
            write_report(output, records, summary, render_plots=False)
            print(f"Episode {episode + 1}: {'FAIL' if failed else 'success'} | "
                  f"total episodes evaluated: {len(episodes)} | "
                  f"{len(records)}/{args.samples} failures collected | {output}", flush=True)
            if len(records) >= args.samples:
                break
    finally:
        renderer.close()
        if env.sim.video_renderer is not None:
            env.sim.video_renderer.close()
        env.sim.renderer.close()
        env.close()
    write_report(output, records, summary)
    print(f"Total episodes evaluated: {len(episodes)} | "
          f"successes: {summary['successes_observed']} | failures collected: {len(records)}")
    if len(records) < args.samples:
        raise SystemExit(f"Episode limit reached: collected {len(records)}/{args.samples}. See {output}")


if __name__ == "__main__":
    main()
