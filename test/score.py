"""
score.py — Compare houses.csv (prédictions de logreg_predict.py)
           avec dataset_train_validate.csv (vérité terrain).

Usage:
    python score.py
    python score.py --truth datasets/dataset_train_validate.csv --pred houses.csv
    python score.py --show-errors    # affiche seulement les erreurs
    python score.py --no-detail      # stats uniquement, pas de tableau
"""

import argparse
import pandas as pd
from collections import defaultdict

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
COLORS = {
    "Gryffindor": "\033[91m",
    "Hufflepuff":  "\033[93m",
    "Ravenclaw":   "\033[94m",
    "Slytherin":   "\033[92m",
}
RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"

def ch(house):
    return f"{COLORS.get(house, '')}{house}{RESET}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", default="datasets/dataset_train_validate.csv")
    parser.add_argument("--pred",  default="houses.csv")
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument("--no-detail",   action="store_true")
    args = parser.parse_args()

    truth_df = pd.read_csv(args.truth)
    pred_df  = pd.read_csv(args.pred)

    truth = truth_df["Hogwarts House"].str.strip().values
    pred  = pred_df["Hogwarts House"].str.strip().values

    if len(truth) != len(pred):
        print(f"⚠ Tailles différentes : vérité={len(truth)}, prédictions={len(pred)}")
        n = min(len(truth), len(pred))
        truth, pred = truth[:n], pred[:n]

    # ── Tableau ligne par ligne ────────────────────────────────────────────────
    if not args.no_detail:
        fn_col = next((c for c in truth_df.columns if "first" in c.lower()), None)
        ln_col = next((c for c in truth_df.columns if "last"  in c.lower()), None)

        print(f"{'#':<5} {'Prénom':<14} {'Nom':<16} {'Réel':<14} {'Prédit':<14} {'OK?'}")
        print("─" * 70)
        for i, (t, p) in enumerate(zip(truth, pred)):
            if args.show_errors and t == p:
                continue
            ok = f"{GREEN}✓{RESET}" if t == p else f"{RED}✗{RESET}"
            fn = str(truth_df[fn_col].iloc[i]) if fn_col else "—"
            ln = str(truth_df[ln_col].iloc[i]) if ln_col else "—"
            idx = truth_df["Index"].iloc[i] if "Index" in truth_df.columns else i
            print(f"{str(idx):<5} {fn:<14} {ln:<16} {ch(t):<23} {ch(p):<23} {ok}")

    # ── Accuracy globale ───────────────────────────────────────────────────────
    correct  = sum(t == p for t, p in zip(truth, pred))
    total    = len(truth)
    accuracy = correct / total * 100
    acc_color = GREEN if accuracy >= 95 else ("\033[93m" if accuracy >= 80 else RED)

    print(f"\n{'─'*55}")
    print(f"{BOLD}Accuracy : {correct}/{total}  →  {acc_color}{accuracy:.2f}%{RESET}")

    # ── Score par maison ───────────────────────────────────────────────────────
    print(f"\n{'Maison':<14} {'Correct':>8} {'Total':>8} {'Précision':>10}  {'':}")
    print("─" * 55)
    for house in HOUSES:
        h_total   = sum(t == house for t in truth)
        h_correct = sum(t == p == house for t, p in zip(truth, pred))
        h_acc     = (h_correct / h_total * 100) if h_total > 0 else 0
        bar = "█" * int(h_acc / 5)
        print(f"{ch(house):<23} {h_correct:>8} {h_total:>8}  {h_acc:>6.1f}%  {bar}")

    # ── Matrice de confusion ───────────────────────────────────────────────────
    confusion = defaultdict(lambda: defaultdict(int))
    for t, p in zip(truth, pred):
        confusion[t][p] += 1

    col_w = 11
    print(f"\nMatrice de confusion (lignes=réel, colonnes=prédit)")
    print(f"{'':14}", end="")
    for h in HOUSES:
        print(f"{h[:9]:>{col_w}}", end="")
    print()
    print("─" * (14 + col_w * len(HOUSES)))
    for actual in HOUSES:
        print(f"{actual:<14}", end="")
        for predicted in HOUSES:
            val  = confusion[actual][predicted]
            cell = f"{val:>{col_w}}"
            if actual == predicted:
                cell = f"{GREEN}{val:>{col_w}}{RESET}"
            elif val > 0:
                cell = f"{RED}{val:>{col_w}}{RESET}"
            print(cell, end="")
        print()

if __name__ == "__main__":
    main()
