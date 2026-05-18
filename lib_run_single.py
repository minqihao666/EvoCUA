import datetime
import json
import logging
import os
import time
from wrapt_timeout_decorator import *
from lib_results_logger import log_task_completion
from mm_agents.evocua.region_focus import RegionFocusConfig, RegionFocusRefiner

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
        "RegionFocus enabled: crop_ratios=%s min_crop_size=%s max_calls=%s confidence_threshold=%s save_debug=%s",
        config.crop_ratios,
        config.min_crop_size,
        config.max_calls,
        config.confidence_threshold,
        config.save_debug,
    )
    return RegionFocusRefiner(config)


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
            region_focus_info = None

            if region_focus is not None:
                try:
                    executed_action, region_focus_info = region_focus.refine_action(
                        agent=agent,
                        obs=obs,
                        instruction=instruction,
                        model_response=response,
                        action=raw_action,
                        step_idx=step_idx + 1,
                        action_timestamp=action_timestamp,
                        debug_dir=example_result_dir,
                    )
                except Exception as rf_e:
                    # RegionFocus should never break the rollout. Fall back to the original action.
                    logger.warning("RegionFocus refinement failed; falling back to raw action: %s", rf_e)
                    region_focus_info = {
                        "enabled": True,
                        "changed": False,
                        "original_action": raw_action,
                        "refined_action": raw_action,
                        "reason": f"exception: {rf_e}",
                    }
                    executed_action = raw_action

            logger.info("Executing action: %s", executed_action)
            if executed_action != raw_action:
                logger.info("Raw action before RegionFocus: %s", raw_action)
            
            # Execute
            obs, reward, done, info = env.step(executed_action, args.sleep_after_execution)
            
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
