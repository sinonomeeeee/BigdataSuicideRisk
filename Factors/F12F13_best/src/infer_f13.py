import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

OUT = Path("outputs/factor/F13_evidence_taxonomy_verifier")
LBOUT = OUT / "leaderboard"
LBOUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 320
BATCH = 96

LABELS = [
    "hopelessness","low self-esteem","emotion dysregulation","poor social support",
    "coping strategy","psychological capital","suicide means (with access)",
    "interpersonal difficulty","mental health issues",
    "prior self-harm or suicidal thought/attempt","stressful life event",
    "dysfunctional family","social support","interpersonal violence",
    "physical health/characteristic","sense of responsibility",
    "low socio-economic status","traumatic experience","meaning in life",
    "cognitive deficits","substance use","exposure to others' suicide",
    "poor school performance","sexual orientation related issues",
]

DEFS = {
    "mental health issues": "Mental illness, psychiatric symptoms, depression, anxiety, PTSD, bipolar disorder, psychosis, diagnosis, therapy, medication, hospitalization, or explicit mental health problems.",
    "physical health/characteristic": "Physical illness, chronic pain, disability, injury, body-related condition, medical disease, or physical characteristic contributing to distress.",
    "substance use": "Use, abuse, dependence, intoxication, withdrawal, relapse, or harmful effects of alcohol, drugs, pills, or other substances.",
    "hopelessness": "Feeling hopeless, trapped, no future, no way out, nothing will improve, despair, or expecting continued suffering.",
    "emotion dysregulation": "Overwhelming emotions, inability to control feelings, panic, rage, numbness, breakdown, emotional instability, or intense affective distress.",
    "low self-esteem": "Self-hatred, worthlessness, shame, feeling like a burden, uselessness, failure as a person, or negative self-evaluation.",
    "poor school performance": "Academic failure, failing grades, exams, schoolwork problems, inability to study, dropping out, or poor performance at school or college; not merely mentioning school.",
    "low socio-economic status": "Poverty, unemployment, financial hardship, homelessness, debt, inability to afford necessities, or economic insecurity.",
    "interpersonal violence": "Abuse, assault, bullying, harassment, domestic violence, coercion, threats, or victimization by others.",
    "prior self-harm or suicidal thought/attempt": "Past or current self-harm, suicide attempt, suicidal ideation, wanting to die, planning suicide, cutting, overdose, or previous suicidal behavior.",
    "poor social support": "Loneliness, isolation, lack of friends or family support, abandonment, nobody to talk to, or feeling unsupported.",
    "interpersonal difficulty": "Conflict, breakup, rejection, relationship problems, arguments, social difficulties, or trouble with family, friends, partners, or others.",
    "dysfunctional family": "Family conflict, neglect, abuse, divorce, toxic household, parental problems, lack of family care, or harmful family environment.",
    "exposure to others' suicide": "Exposure to another person's suicide, suicide death, suicide attempt, or suicidal behavior; not the author's own suicidal thoughts.",
    "stressful life event": "Major stressful events such as loss, bereavement, job loss, exam pressure, legal trouble, moving, crisis, or acute life stress.",
    "traumatic experience": "Trauma, traumatic memory, severe past adverse experience, childhood trauma, sexual assault, abuse trauma, war, accident, or PTSD-related trauma.",
    "cognitive deficits": "Impaired thinking, concentration, memory, decision-making, confusion, rumination, intrusive thoughts, or inability to think clearly.",
    "suicide means (with access)": "Access to or mention of concrete suicide means such as gun, rope, pills, bridge, train, knife, poison, overdose materials, or method availability.",
    "sexual orientation related issues": "Distress, discrimination, rejection, identity conflict, bullying, family conflict, or safety issues related to sexual orientation or gender identity.",
    "social support": "Presence of supportive friends, family, community, partner, therapist, crisis line, help-seeking support, or someone caring.",
    "coping strategy": "Adaptive coping, seeking help, therapy, distraction, exercise, journaling, safety planning, grounding, talking to others, or trying to manage distress.",
    "psychological capital": "Hope, optimism, resilience, self-efficacy, confidence, perseverance, or belief in ability to recover.",
    "sense of responsibility": "Feeling responsible for others, duty, obligation, caring for family/pets/children, or reasons to stay alive because others depend on them.",
    "meaning in life": "Purpose, meaning, reasons for living, life goals, values, future plans, or loss/absence of life meaning.",
}

PROTECTIVE = {"social support","coping strategy","psychological capital","sense of responsibility","meaning in life"}

def split_units(text):
    text = str(text).replace("\n", " ")
    parts = []
    cur = []
    for ch in text:
        cur.append(ch)
        if ch in ".!?;。！？":
            s = "".join(cur).strip()
            if s:
                parts.append(s)
            cur = []
    rest = "".join(cur).strip()
    if rest:
        parts.append(rest)
    if not parts:
        parts = [text.strip()]
    out = []
    for p in parts:
        p = " ".join(p.split())
        if p:
            out.append(p[:500])
    return out[:12]

def label_text(unit, post, factor):
    group = "protective factor" if factor in PROTECTIVE else "risk factor"
    return (
        f"[UNIT] {unit}\n"
        f"[POST] {str(post)[:900]}\n"
        f"[FACTOR] {factor}\n"
        f"[TYPE] {group}\n"
        f"[DEFINITION] {DEFS[factor]}\n"
        f"[QUESTION] Is this factor clearly present in the unit or post?"
    )

class PairDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        return self.rows[i]

class Collator:
    def __init__(self, tok):
        self.tok = tok
    def __call__(self, batch):
        texts = [x["text"] for x in batch]
        enc = self.tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "row_id": [x["row_id"] for x in batch],
            "factor": [x["factor"] for x in batch],
            "unit_index": [
                int(x["unit_index"])
                for x in batch
            ],

        }

class Verifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        h = self.enc.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h // 2, 1),
        )
    def forward(self, input_ids, attention_mask):
        o = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        cls = o.last_hidden_state[:, 0, :].float()
        return self.head(cls).squeeze(-1)

def find_test_posts():
    candidates = [
        "data/test.csv",
        "data/test_posts.csv",
        "data/test_data.csv",
        "data/Test.csv",
        "data/leaderboard.csv",
        "data/leaderboard_posts.csv",
        "data/dev/test.csv",
        "data/raw/test.csv",
    ]
    for p in candidates:
        path = Path(p)
        if path.exists():
            df = pd.read_csv(path)
            if "post" in df.columns and "row_id" in df.columns:
                print("using test posts:", path)
                return df[["row_id","post"]].copy()

    # fallback: infer from current submission row_ids and any csv containing posts
    template = Path("submissions/SatAndSeriousTokyoScript.csv")
    if template.exists():
        sub = pd.read_csv(template)
        row_ids = set(sub["row_id"].astype(str))
        for path in Path("data").rglob("*.csv"):
            try:
                df = pd.read_csv(path, nrows=5)
            except Exception:
                continue
            if "row_id" in df.columns and "post" in df.columns:
                full = pd.read_csv(path)
                hit = full[full["row_id"].astype(str).isin(row_ids)]
                if len(hit) == len(row_ids):
                    print("using inferred posts:", path)
                    return hit[["row_id","post"]].copy()

    raise SystemExit("Cannot find test posts CSV with row_id and post columns.")

def build_pairs(test):
    rows = []
    for _, r in test.iterrows():
        rid = str(r["row_id"])
        post = str(r["post"])
        units = split_units(post)
        for factor in LABELS:
            for unit_index, unit in enumerate(units):
                rows.append({
                    "row_id": rid,
                    "factor": factor,
                    "unit_index": unit_index,
                    "text": label_text(unit, post, factor),
                })
    return rows

def load_thresholds():
    ths_by_fold = []
    for f in range(5):
        p = OUT / f"F13_evidence_taxonomy_verifier_fold{f}_summary.json"
        js = json.loads(p.read_text())
        ths_by_fold.append(js["thresholds"])
    mean_th = {lab: float(np.mean([ths[lab] for ths in ths_by_fold])) for lab in LABELS}
    return ths_by_fold, mean_th

def infer_one_fold(fold, loader, tok, device):
    ckpt_path = OUT / "checkpoints" / f"F13_evidence_taxonomy_verifier_fold{fold}_best.pt"
    print("loading", ckpt_path)
    model = Verifier().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    recs = []
    with torch.no_grad():
        for step, b in enumerate(loader, start=1):
            input_ids = b["input_ids"].to(device)
            attention_mask = b["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            for rid, fac, unit_index, pr in zip(
                b["row_id"],
                b["factor"],
                b["unit_index"],
                probs,
            ):
                recs.append({
                    "row_id": str(rid),
                    "factor": str(fac),
                    "unit_index": int(unit_index),
                    "prob_unit": float(pr),
                })

            if step % 200 == 0:
                print(f"fold {fold} infer step {step}/{len(loader)}", flush=True)

    unit_df = pd.DataFrame(recs)
    unit_df["fold"] = int(fold)

    unit_path = (
        LBOUT
        / f"F13_leaderboard_fold{fold}_unit_probs.csv"
    )

    unit_df.to_csv(
        unit_path,
        index=False,
    )

    post_probs = (
        unit_df.groupby(
            ["row_id", "factor"],
            as_index=False,
        )["prob_unit"]
        .max()
    )

    post_probs["fold"] = int(fold)

    outp = (
        LBOUT
        / f"F13_leaderboard_fold{fold}_post_probs_long.csv"
    )

    post_probs.to_csv(
        outp,
        index=False,
    )

    print("wrote", unit_path)
    print("wrote", outp)

    return post_probs, unit_df

def factors_to_string(labs):
    labs = [x for x in labs if x]
    if not labs:
        return "[]"
    return str(labs)

test = find_test_posts()
test["row_id"] = test["row_id"].astype(str)
print("test posts:", len(test))

pairs = build_pairs(test)
print("inference pairs:", len(pairs), "avg pairs/post:", len(pairs) / max(len(test), 1))

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
loader = DataLoader(PairDataset(pairs), batch_size=BATCH, shuffle=False, collate_fn=Collator(tok))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "batch:", BATCH, "steps/fold:", len(loader))

ths_by_fold, mean_th = load_thresholds()
print("mean thresholds:")
for lab in LABELS:
    print(lab, mean_th[lab])

all_fold_probs = []
all_fold_units = []

for f in range(5):
    fold_post, fold_unit = infer_one_fold(
        f,
        loader,
        tok,
        device,
    )

    all_fold_probs.append(fold_post)
    all_fold_units.append(fold_unit)

long = pd.concat(
    all_fold_probs,
    ignore_index=True,
)

unit_long = pd.concat(
    all_fold_units,
    ignore_index=True,
)

unit_long.to_csv(
    LBOUT
    / "F13_leaderboard_5fold_unit_probs_long.csv",
    index=False,
)

unit_ensemble = (
    unit_long.groupby(
        [
            "row_id",
            "factor",
            "unit_index",
        ],
        as_index=False,
    )["prob_unit"]
    .mean()
)

unit_ensemble.to_csv(
    LBOUT
    / "F13_leaderboard_ensemble_unit_probs.csv",
    index=False,
)
avg = long.groupby(["row_id","factor"])["prob_unit"].mean().reset_index(name="prob")
avg.to_csv(LBOUT / "F13_leaderboard_5fold_avg_probs_long.csv", index=False)

wide = avg.pivot(index="row_id", columns="factor", values="prob").fillna(0.0)
wide = wide.reindex(test["row_id"].tolist())
wide.to_csv(LBOUT / "F13_leaderboard_5fold_avg_probs_wide.csv")

# Candidate A: average probability + mean threshold
pred_rows = []
for rid in test["row_id"].tolist():
    labs = []
    for lab in LABELS:
        if float(wide.loc[rid, lab]) >= mean_th[lab]:
            labs.append(lab)
    pred_rows.append({"row_id": rid, "factors": factors_to_string(labs), "n_factors": len(labs)})
pred_a = pd.DataFrame(pred_rows)
pred_a.to_csv(LBOUT / "F13_leaderboard_pred_mean_threshold.csv", index=False)

# Candidate B: fold voting with each fold threshold, need at least 2/5 votes
vote_df = long.copy()
vote_map = {}
for f in range(5):
    sub = vote_df[vote_df["fold"] == f].copy()
    ths = ths_by_fold[f]
    sub["vote"] = [int(p >= ths[lab]) for p, lab in zip(sub["prob_unit"], sub["factor"])]
    for _, r in sub.iterrows():
        key = (r["row_id"], r["factor"])
        vote_map[key] = vote_map.get(key, 0) + int(r["vote"])

pred_rows = []
for rid in test["row_id"].tolist():
    labs = []
    for lab in LABELS:
        if vote_map.get((rid, lab), 0) >= 2:
            labs.append(lab)
    pred_rows.append({"row_id": rid, "factors": factors_to_string(labs), "n_factors": len(labs)})
pred_b = pd.DataFrame(pred_rows)
pred_b.to_csv(LBOUT / "F13_leaderboard_pred_vote2.csv", index=False)

# Candidate C: conservative mean threshold + 0.05 for noisy labels
NOISY = {
    "meaning in life",
    "sense of responsibility",
    "physical health/characteristic",
    "social support",
    "traumatic experience",
    "interpersonal difficulty",
    "psychological capital",
}
pred_rows = []
for rid in test["row_id"].tolist():
    labs = []
    for lab in LABELS:
        th = mean_th[lab] + (0.05 if lab in NOISY else 0.0)
        if float(wide.loc[rid, lab]) >= th:
            labs.append(lab)
    pred_rows.append({"row_id": rid, "factors": factors_to_string(labs), "n_factors": len(labs)})
pred_c = pd.DataFrame(pred_rows)
pred_c.to_csv(LBOUT / "F13_leaderboard_pred_conservative.csv", index=False)

# Write submissions based on current best/template
template_path = Path("submissions/SatAndSeriousTokyoScript.csv")
if not template_path.exists():
    raise SystemExit("Missing submissions/SatAndSeriousTokyoScript.csv template")

template = pd.read_csv(template_path)
template["row_id"] = template["row_id"].astype(str)

def write_submission(pred_df, name):
    out = template.copy()
    mp = dict(zip(pred_df["row_id"].astype(str), pred_df["factors"]))
    if "factors" not in out.columns:
        raise SystemExit("template has no factors column")
    out["factors"] = out["row_id"].map(mp).fillna("[]")
    path = Path("submissions") / name
    out.to_csv(path, index=False)

    # diagnostics
    counts = pred_df["n_factors"].describe()
    print("\n", name)
    print(counts)
    print("wrote", path)
    return path

write_submission(pred_a, "SatAndSeriousTokyoScript_F13_mean_threshold.csv")
write_submission(pred_b, "SatAndSeriousTokyoScript_F13_vote2.csv")
write_submission(pred_c, "SatAndSeriousTokyoScript_F13_conservative.csv")

print("\nDONE")
print("Main candidate: submissions/SatAndSeriousTokyoScript_F13_mean_threshold.csv")
print("Second candidate: submissions/SatAndSeriousTokyoScript_F13_vote2.csv")
print("Conservative: submissions/SatAndSeriousTokyoScript_F13_conservative.csv")
