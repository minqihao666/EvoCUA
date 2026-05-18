import os
import json
import argparse


def read_score(result_file: str) -> float:
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return 0.0
        return float(text.split()[0])
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Check EvoCUA/OSWorld evaluation progress and score by domain."
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        default="./results/evocua32b_full",
        help="Result directory. Default: ./results/evocua32b_full",
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="pyautogui",
        help="Action space. Default: pyautogui",
    )
    parser.add_argument(
        "--observation_type",
        type=str,
        default="screenshot",
        help="Observation type. Default: screenshot",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EvoCUA-S2",
        help="Model name in result path. Default: EvoCUA-S2",
    )
    parser.add_argument(
        "--test_all_meta_path",
        type=str,
        default="evaluation_examples/test_nogdrive.json",
        help="Path to test meta json. Default: evaluation_examples/test_nogdrive.json",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="all",
        help="Domain to check, e.g. chrome/gimp/vs_code/all. Default: all",
    )
    parser.add_argument(
        "--show_left",
        action="store_true",
        help="Show unfinished task ids.",
    )
    parser.add_argument(
        "--show_zero",
        action="store_true",
        help="Show finished task ids whose score is 0.",
    )

    args = parser.parse_args()

    target_dir = os.path.join(
        args.result_dir,
        args.action_space,
        args.observation_type,
        args.model,
    )

    if not os.path.exists(args.test_all_meta_path):
        raise FileNotFoundError(f"test_all_meta_path not found: {args.test_all_meta_path}")

    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    if args.domain != "all":
        if args.domain not in test_all_meta:
            print(f"Domain not found: {args.domain}")
            print("Available domains:")
            for d in test_all_meta.keys():
                print(f"  - {d}")
            return
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    rows = []
    all_left_ids = {}
    all_zero_ids = {}

    grand_total = 0
    grand_finished = 0
    grand_score_sum = 0.0

    for domain, example_ids in test_all_meta.items():
        total = len(example_ids)
        finished = 0
        score_sum = 0.0
        left_ids = []
        zero_ids = []

        for example_id in example_ids:
            result_file = os.path.join(target_dir, domain, example_id, "result.txt")

            if os.path.exists(result_file):
                finished += 1
                score = read_score(result_file)
                score_sum += score
                if score == 0:
                    zero_ids.append(example_id)
            else:
                left_ids.append(example_id)

        left = total - finished
        score_rate = (score_sum / total * 100) if total else 0.0
        finished_avg = (score_sum / finished * 100) if finished else 0.0

        rows.append({
            "domain": domain,
            "total": total,
            "finished": finished,
            "left": left,
            "score_sum": score_sum,
            "score_rate": score_rate,
            "finished_avg": finished_avg,
        })

        all_left_ids[domain] = left_ids
        all_zero_ids[domain] = zero_ids

        grand_total += total
        grand_finished += finished
        grand_score_sum += score_sum

    print(f"Result path: {target_dir}")
    print(f"Domain: {args.domain}")
    print()

    header = f"{'domain':<24} {'total':>7} {'finished':>9} {'left':>7} {'score_sum':>12} {'score_rate':>12} {'finished_avg':>14}"
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['domain']:<24} "
            f"{r['total']:>7} "
            f"{r['finished']:>9} "
            f"{r['left']:>7} "
            f"{r['score_sum']:>12.4f} "
            f"{r['score_rate']:>11.4f}% "
            f"{r['finished_avg']:>13.4f}%"
        )

    print("-" * len(header))

    grand_left = grand_total - grand_finished
    grand_score_rate = (grand_score_sum / grand_total * 100) if grand_total else 0.0
    grand_finished_avg = (grand_score_sum / grand_finished * 100) if grand_finished else 0.0

    print(
        f"{'TOTAL':<24} "
        f"{grand_total:>7} "
        f"{grand_finished:>9} "
        f"{grand_left:>7} "
        f"{grand_score_sum:>12.4f} "
        f"{grand_score_rate:>11.4f}% "
        f"{grand_finished_avg:>13.4f}%"
    )

    if args.show_left:
        print("\nUnfinished task ids:")
        for domain, ids in all_left_ids.items():
            if ids:
                print(f"\n[{domain}] {len(ids)} left")
                for x in ids:
                    print(x)

    if args.show_zero:
        print("\nZero-score finished task ids:")
        for domain, ids in all_zero_ids.items():
            if ids:
                print(f"\n[{domain}] {len(ids)} zero-score")
                for x in ids:
                    print(x)


if __name__ == "__main__":
    main()
