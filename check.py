import os
import json

# 修改这里即可
RESULT_DIR = "./results/evocua32b_full"
ACTION_SPACE = "pyautogui"
OBSERVATION_TYPE = "screenshot"
MODEL = "EvoCUA-S2"
TEST_ALL_META_PATH = "evaluation_examples/test_nogdrive.json"

# 改成 "all" 看全部；改成 "chrome" / "gimp" / "vs_code" / "multi_apps" 等看单个 domain
DOMAIN = "chrome"

target_dir = os.path.join(RESULT_DIR, ACTION_SPACE, OBSERVATION_TYPE, MODEL)

with open(TEST_ALL_META_PATH, "r", encoding="utf-8") as f:
    test_all_meta = json.load(f)

if DOMAIN != "all":
    test_all_meta = {DOMAIN: test_all_meta[DOMAIN]}

finished = {}
scores = []

if os.path.exists(target_dir):
    domains_to_scan = [DOMAIN] if DOMAIN != "all" else os.listdir(target_dir)

    for domain in domains_to_scan:
        domain_path = os.path.join(target_dir, domain)
        if not os.path.isdir(domain_path):
            continue

        finished[domain] = []
        for example_id in os.listdir(domain_path):
            if example_id == "onboard":
                continue

            example_path = os.path.join(domain_path, example_id)
            if not os.path.isdir(example_path):
                continue

            result_file = os.path.join(example_path, "result.txt")
            if os.path.exists(result_file):
                finished[domain].append(example_id)
                try:
                    with open(result_file, "r", encoding="utf-8") as rf:
                        scores.append(float(rf.read().strip()))
                except Exception:
                    scores.append(0.0)

left_total = 0
print("Left tasks:")
for domain, examples in test_all_meta.items():
    done = set(finished.get(domain, []))
    left = [x for x in examples if x not in done]
    print(f"{domain}: {len(left)}")
    left_total += len(left)

print(f"\nTotal left: {left_total}")

if scores:
    avg = sum(scores) / len(scores)
    print(f"Finished tasks: {len(scores)}")
    print(f"Current Success Rate ({DOMAIN}): {avg * 100:.6f} %")
else:
    print("Finished tasks: 0")
    print(f"Current Success Rate ({DOMAIN}): 0.000000 %")