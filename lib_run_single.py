import datetime
import json
import logging
import os
import time
from io import BytesIO
from wrapt_timeout_decorator import *
from PIL import Image, ImageChops, ImageStat
from lib_results_logger import log_task_completion
from mm_agents.evocua.region_focus import RegionFocusConfig, RegionFocusRefiner, parse_click_action

logger = logging.getLogger("desktopenv.experiment")


def setup_logger(example, example_result_dir):
    runtime_logger = logging.getLogger(f"desktopenv.example.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(logging.FileHandler(os.path.join(example_result_dir, "runtime.log")))
    return runtime_logger


def _build_region_focus(args):
    """Create a RegionFocus refiner when enabled by args or environment.

    CLI flags are optional; without parser support you can still enable it with:
        export EVO_REGION_FOCUS=1
    """
    config = RegionFocusConfig.from_args(args)
    if not config.enabled:
        return None
    logger.info(
        "RegionFocus enabled: trigger_policy=%s crop_ratios=%s min_crop_size=%s max_calls=%s confidence_threshold=%s nochange_threshold=%s save_debug=%s",
        config.trigger_policy,
        config.crop_ratios,
        config.min_crop_size,
        config.max_calls,
        config.confidence_threshold,
        config.nochange_threshold,
        config.save_debug,
    )
    return RegionFocusRefiner(config)


def _screenshot_delta(before_bytes, after_bytes):
    """Return a cheap mean absolute pixel delta in [0, 255].

    Small values mean the screen barely changed. This is used as a lightweight
    NoChange trigger for RegionFocus retries. It deliberately downsamples the
    screenshots to avoid expensive per-pixel comparisons.
    """
    try:
        before = Image.open(BytesIO(before_bytes)).convert("L").resize((64, 64))
        after = Image.open(BytesIO(after_bytes)).convert("L").resize((64, 64))
        diff = ImageChops.difference(before, after)
        return float(ImageStat.Stat(diff).mean[0])
    except Exception as exc:
        logger.debug("Failed to compute screenshot delta: %s", exc)
        return 255.0


def _should_try_region_focus_retry(region_focus, raw_action, before_obs, after_obs, done):
    if region_focus is None or done:
        return False, None
    if parse_click_action(raw_action) is None:
        return False, None
    delta = _screenshot_delta(before_obs.get("screenshot"), after_obs.get("screenshot"))
    return delta <= region_focus.config.nochange_threshold, delta


def run_single_example_evocua(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    """
    Unified run function for EvoCUAAgent (supporting both S1 and S2 modes).
    """
    runtime_logger = setup_logger(example, example_result_dir)
    
    # Reset Environment
    env.reset(task_config=example)
    
    # Reset Agent
    # Handle agent reset signature differences if any
    try:
        agent.reset(runtime_logger, vm_ip=env.vm_ip)
    except Exception:
        try:
            agent.reset(runtime_logger)
        except Exception:
            agent.reset()

    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
    done = False
    step_idx = 0
    region_focus = _build_region_focus(args)

    env.controller.start_recording()
    while not done and step_idx < max_steps:
        # EvoCUAAgent.predict unified signature: returns (response, actions)
        # It handles both modes internally.
        predict_res = agent.predict(instruction, obs)
        
        # Check return signature logic
        if len(predict_res) == 3:
            # Compatibility with S1 original signature if agent was updated to match
            response, actions, info_dict = predict_res
        else:
            response, actions = predict_res
            info_dict = {}

        logger.info(f"Step {step_idx + 1} Actions: {actions}")
        
        # Break if no actions (fail-safe)
        if not actions or (len(actions) == 1 and (actions[0] == "" or "error" in actions[0].lower())):
             # Allow "FAIL" or "DONE" to process through execution loop if agent outputs them as actions
             if not (actions and actions[0] in ["FAIL", "DONE"]):
                 logger.warning("No valid actions returned. Breaking loop.")
                 break

        for action in actions:
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            raw_action = action
            executed_action = action
            before_obs = obs
            region_focus_info = None
            rf_retry_info = None

            # Optional pre-click policy. This reproduces the first RegionFocus-lite
            # behavior, but the default policy is now nochange to avoid refining
            # every click and slowing down / perturbing successful actions.
            if region_focus is not None and region_focus.config.trigger_policy in {"pre_click", "both"}:
                try:
                    executed_action, region_focus_info = region_focus.refine_action(
                        agent=agent,
                        obs=before_obs,
                        instruction=instruction,
                        model_response=response,
                        action=raw_action,
                        step_idx=step_idx + 1,
                        action_timestamp=action_timestamp,
                        debug_dir=example_result_dir,
                    )
                except Exception as rf_e:
                    logger.warning("RegionFocus pre-click refinement failed; falling back to raw action: %s", rf_e)
                    region_focus_info = {
                        "enabled": True,
                        "trigger_policy": getattr(region_focus.config, "trigger_policy", "unknown"),
                        "changed": False,
                        "original_action": raw_action,
                        "refined_action": raw_action,
                        "reason": f"exception: {rf_e}",
                    }
                    executed_action = raw_action

            logger.info("Executing action: %s", executed_action)
            if executed_action != raw_action:
                logger.info("Raw action before RegionFocus: %s", raw_action)
            
            # Execute the selected action.
            obs, reward, done, info = env.step(executed_action, args.sleep_after_execution)

            # Default RegionFocus policy: only retry when a click-like action produced
            # almost no visual change. This is closer to the intended use case:
            # RegionFocus is a fallback for suspicious grounding, not a mandatory
            # step on every action.
            if region_focus is not None and region_focus.config.trigger_policy in {"nochange", "both"}:
                should_retry, delta = _should_try_region_focus_retry(region_focus, raw_action, before_obs, obs, done)
                if should_retry:
                    retry_timestamp = f"{action_timestamp}_retry"
                    logger.info(
                        "RegionFocus no-change trigger fired: delta=%.4f <= %.4f",
                        delta,
                        region_focus.config.nochange_threshold,
                    )
                    try:
                        refined_retry_action, rf_retry_info = region_focus.refine_action(
                            agent=agent,
                            obs=before_obs,
                            instruction=instruction,
                            model_response=response,
                            action=raw_action,
                            step_idx=step_idx + 1,
                            action_timestamp=retry_timestamp,
                            debug_dir=example_result_dir,
                        )
                        if refined_retry_action != raw_action:
                            logger.info("Executing RegionFocus retry action: %s", refined_retry_action)
                            obs, reward, done, info = env.step(refined_retry_action, args.sleep_after_execution)
                            executed_action = refined_retry_action
                            if isinstance(rf_retry_info, dict):
                                rf_retry_info["retry_executed"] = True
                        else:
                            logger.info("RegionFocus retry did not change the action; skipping duplicate execution.")
                            if isinstance(rf_retry_info, dict):
                                rf_retry_info["retry_executed"] = False
                                rf_retry_info["skip_reason"] = "same_as_raw_action"
                    except Exception as rf_e:
                        logger.warning("RegionFocus no-change retry failed: %s", rf_e)
                        rf_retry_info = {
                            "enabled": True,
                            "trigger_policy": getattr(region_focus.config, "trigger_policy", "unknown"),
                            "changed": False,
                            "original_action": raw_action,
                            "refined_action": raw_action,
                            "reason": f"retry_exception: {rf_e}",
                            "nochange_delta": delta,
                            "retry_executed": False,
                        }
                elif delta is not None:
                    rf_retry_info = {
                        "enabled": True,
                        "trigger_policy": getattr(region_focus.config, "trigger_policy", "unknown"),
                        "changed": False,
                        "reason": "screen_changed_no_retry",
                        "nochange_delta": delta,
                        "threshold": region_focus.config.nochange_threshold,
                        "retry_executed": False,
                    }
            
            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            
            # Save screenshot
            screenshot_file = f"step_{step_idx + 1}_{action_timestamp}.png"
            with open(os.path.join(example_result_dir, screenshot_file), "wb") as _f:
                _f.write(obs['screenshot'])
            
            # Log Trajectory
            log_entry = {
                "step_num": step_idx + 1,
                "action_timestamp": action_timestamp,
                "raw_action": raw_action,
                "action": executed_action,
                "response": response,
                "reward": reward,
                "done": done,
                "info": info,
                "screenshot_file": screenshot_file
            }
            if region_focus_info is not None:
                log_entry["region_focus"] = region_focus_info
            if rf_retry_info is not None:
                log_entry["region_focus_retry"] = rf_retry_info
            # Add natural language info if available (S1 style)
            if info_dict:
                log_entry["natural_language_action"] = info_dict.get("action")
            
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False))
                f.write("\n")
                
            if done:
                logger.info("The episode is done.")
                break
        
        step_idx += 1
        
    time.sleep(20) # Wait for environment to settle
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
    
    log_task_completion(example, result, example_result_dir, args)

    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
