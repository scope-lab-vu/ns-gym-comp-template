from time import perf_counter
from datetime import datetime
from AAMAS_Comp import base_agent
import os
import json
import zipfile
from pathlib import Path
from tqdm import tqdm
import numpy as np


def _default_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def run_single_episode(env, agent, seed):
    """Runs single environment episode.

    Args:
        env (NSFrozenLakeWrapper): The gymnasium environment. Must be wrapped with NSFrozenLakeWrapper (or a subclass) to expose the necessary non-stationary interfaces like `get_planning_env()`.
        agent (Union[AAMAS_Comp.base_agent.ModelBasedAgent, AAMAS_Comp.base_agent.ModelFreeAgent]): Evaluation Agent.
        seed (int): Random number generator seed.

    Returns:
        dict: A dictionary containing lists of rewards, observations, actions, and decision times.
    """

    obs, reward = env.reset(seed=seed)

    done = False
    truncated = False

    episode_metrics = {
        "step_number": [],
        "rewards": [],
        "observations": [],
        "notification": [],
        "actions": [],
        "decision_time": [],
        "info": [],
        "env_change": []
    }

    is_model_based_agent = isinstance(agent, base_agent.ModelBasedAgent)
    count = 0

    while not (done or truncated):

        if is_model_based_agent:
            planning_env = env.get_planning_env()
            action, decision_time = agent.validate_and_get_action(
                obs, planning_env, reward=reward, done=done
            )

        else:
            action, decision_time = agent.validate_and_get_action(
                obs, env.action_space, reward=reward, done=done
            )

        obs, reward, done, truncated, info = env.step(action)



        episode_metrics["step_number"].append(count)
        episode_metrics["observations"].append(obs["state"])
        episode_metrics["notification"].append(obs["env_change"])
        episode_metrics["rewards"].append(reward)
        episode_metrics["actions"].append(action)
        episode_metrics["decision_time"].append(decision_time)
        episode_metrics["info"].append(info)
        episode_metrics["env_change"].append(max(info['Ground Truth Env Change'].values()))

    return episode_metrics


def run_complete_evaluation(env, agent, start_seed, num_episodes, name_prefix, save_dir="results/"):
    """Runs multiple episodes with deterministic sequential seeding. Saves results as Compressed JSON file.
    Args:
        env: The gymnasium environment wrapped with NSFrozenLakeWrapper.
        agent: Evaluation Agent (ModelBasedAgent or ModelFreeAgent).
        start_seed (int): Starting seed. Each episode uses start_seed + i.
        num_episodes (int): Number of episodes to run.
        name_prefix (str): Experiment name prefix.
        save_dir (Path): path to save directory. Defaults to results.


    Returns:
        dict: Maps seed (str) to that episode's metrics dict.
    """
    results_table = {}

    start_time = perf_counter()

    for i in tqdm(range(num_episodes), desc=name_prefix):
        seed = start_seed + i
        episode_metrics = run_single_episode(env, agent, seed)
        results_table[str(seed)] = episode_metrics

    total_time = perf_counter() - start_time

    if not isinstance(save_dir, Path):
        save_dir = Path(save_dir)

    experiment_dir = save_dir / name_prefix
    os.makedirs(experiment_dir, exist_ok=True)

    metadata = {
        "name_prefix": name_prefix,
        "start_seed": start_seed,
        "end_seed": start_seed + num_episodes - 1,
        "num_episodes": num_episodes,
        "change_notification": env.change_notification, 
        "delta_change_notification": env.delta_change_notification,
        "total_time_seconds": total_time,
        "timestamp": datetime.now().isoformat(),
        
    }

    with open(experiment_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    json_path = experiment_dir / f"{name_prefix}.json"
    zip_path = experiment_dir / f"{name_prefix}.zip"

    with open(json_path, "w") as f:
        json.dump(results_table, f, default=_default_serializer)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, json_path.name)

    json_path.unlink()

    # Per-episode summary
    summary = {}
    for seed_key, ep in results_table.items():
        rewards = ep["rewards"]
        decision_times = ep["decision_time"]
        summary[seed_key] = {
            "total_reward": sum(rewards),
            "num_steps": len(rewards),
            "mean_decision_time": np.mean(decision_times),
            "std_decision_time": np.std(decision_times),
            "num_transition_fn_changes": sum(ep["env_change"]),
        }

    # Aggregate across all episodes
    all_rewards = [s["total_reward"] for s in summary.values()]
    all_steps = [s["num_steps"] for s in summary.values()]
    all_mean_dt = [s["mean_decision_time"] for s in summary.values()]
    all_tf_changes = [s["num_transition_fn_changes"] for s in summary.values()]

    aggregate = {
        "mean_total_reward": float(np.mean(all_rewards)),
        "std_total_reward": float(np.std(all_rewards)),
        "mean_episode_steps": float(np.mean(all_steps)),
        "mean_decision_time": float(np.mean(all_mean_dt)),
        "std_decision_time": float(np.std(all_mean_dt)),
        "mean_transition_fn_changes": float(np.mean(all_tf_changes)),
        "std_transition_fn_changes": float(np.std(all_tf_changes)),
    }

    summary_data = {
        "aggregate": aggregate,
        "per_episode": summary,
    }

    with open(experiment_dir / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2, default=_default_serializer)

    print(f"\n{'='*50}")
    print(f"Evaluation Summary: {name_prefix}")
    print(f"{'='*50}")
    print(f"Episodes:            {num_episodes}")
    print(f"Seed Range:          {start_seed} - {start_seed + num_episodes - 1}")
    print(f"Total Time:          {total_time:.2f}s")
    print(f"Mean Total Reward:   {aggregate['mean_total_reward']:.4f} +/- {aggregate['std_total_reward']:.4f}")
    print(f"Mean Episode Steps:  {aggregate['mean_episode_steps']:.1f}")
    print(f"Mean Decision Time:  {aggregate['mean_decision_time']:.6f}s +/- {aggregate['std_decision_time']:.6f}s")
    print(f"Mean Number of T(s,a) Changes: {aggregate['mean_transition_fn_changes']:.2f} +/- {aggregate['std_transition_fn_changes']:.2f}")
    print(f"{'='*50}")
    print(f"Results saved to:    {experiment_dir}")

    return results_table
