import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


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


def load_input(input_path):
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)

    required = {"row_id", "post"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing)}"
        )

    if df["row_id"].isna().any():
        raise ValueError("row_id must not contain missing values.")

    if df["post"].isna().any():
        raise ValueError("post must not contain missing values.")

    df = df[["row_id", "post"]].copy()
    df["row_id"] = df["row_id"].astype(str)

    if df["row_id"].duplicated().any():
        raise ValueError("row_id values must be unique.")

    return df

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


def load_thresholds(model_dir):
    model_dir = Path(model_dir)

    ths_by_fold = []

    for fold in range(5):
        path = model_dir / f"fold{fold}_summary.json"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing threshold summary: {path}"
            )

        js = json.loads(path.read_text())

        if "thresholds" not in js:
            raise ValueError(
                f"No 'thresholds' field in {path}"
            )

        ths_by_fold.append(js["thresholds"])

    mean_th = {
        label: float(
            np.mean([
                thresholds[label]
                for thresholds in ths_by_fold
            ])
        )
        for label in LABELS
    }

    return ths_by_fold, mean_th


def infer_one_fold(
    fold,
    loader,
    device,
    model_dir,
):
    model_dir = Path(model_dir)
    ckpt_path = model_dir / f"fold{fold}.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {ckpt_path}"
        )

    print("loading", ckpt_path)

    model = Verifier().to(device)

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    recs = []

    with torch.no_grad():
        for step, batch in enumerate(
            loader,
            start=1,
        ):
            input_ids = (
                batch["input_ids"]
                .to(device)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(device)
            )

            logits = model(
                input_ids,
                attention_mask,
            )

            probs = (
                torch.sigmoid(logits)
                .detach()
                .cpu()
                .numpy()
            )

            for (
                rid,
                factor,
                unit_index,
                prob,
            ) in zip(
                batch["row_id"],
                batch["factor"],
                batch["unit_index"],
                probs,
            ):
                recs.append({
                    "row_id": str(rid),
                    "factor": str(factor),
                    "unit_index": int(unit_index),
                    "prob_unit": float(prob),
                })

            if step % 200 == 0:
                print(
                    f"fold {fold} infer "
                    f"step {step}/{len(loader)}",
                    flush=True,
                )

    unit_df = pd.DataFrame(recs)

    post_probs = (
        unit_df
        .groupby(
            ["row_id", "factor"],
            as_index=False,
        )["prob_unit"]
        .max()
    )

    post_probs["fold"] = fold

    return post_probs

def factors_to_string(labs):

    labs = [x for x in labs if x]
    if not labs:
        return "[]"
    return str(labs)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run five-fold Evidence-aware "
            "Taxonomy Verifier inference."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="CSV containing row_id and post.",
    )

    parser.add_argument(
        "--model-dir",
        required=True,
        help=(
            "Directory containing fold0.pt ... "
            "fold4.pt and fold*_summary.json."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output long-format probability CSV.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH,
    )

    args = parser.parse_args()

    test = load_input(args.input)

    print("test posts:", len(test))

    pairs = build_pairs(test)

    print(
        "inference pairs:",
        len(pairs),
        "avg pairs/post:",
        len(pairs) / max(len(test), 1),
    )

    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    loader = DataLoader(
        PairDataset(pairs),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=Collator(tok),
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device:",
        device,
        "batch:",
        args.batch_size,
        "steps/fold:",
        len(loader),
    )

    _, mean_th = load_thresholds(
        args.model_dir
    )

    print("mean thresholds:")

    for label in LABELS:
        print(label, mean_th[label])

    all_fold_probs = []

    for fold in range(5):
        all_fold_probs.append(
            infer_one_fold(
                fold,
                loader,
                device,
                args.model_dir,
            )
        )

    long = pd.concat(
        all_fold_probs,
        ignore_index=True,
    )

    avg = (
        long
        .groupby(
            ["row_id", "factor"],
            as_index=False,
        )["prob_unit"]
        .mean()
        .rename(
            columns={
                "prob_unit": "prob"
            }
        )
    )

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    avg.to_csv(
        output,
        index=False,
    )

    print("wrote:", output)
    print("rows:", len(avg))


if __name__ == "__main__":
    main()
