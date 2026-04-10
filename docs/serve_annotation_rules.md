# Serve annotation rules (timestamps & short clips)

This document defines **what to label as a serve** and **what to use as not-serve** when you mark **anchor times** for clip export (e.g. `serve_clip_export_from_video_colab.ipynb`). It matches the pipeline where each clip’s frames are taken from **[anchor − WINDOW_BEFORE, anchor]** with the anchor as the **end** of that interval.

---

## 1. What the anchor time means

- Record **one time in seconds** per serve: the **serve anchor**.
- **Default rule:** the anchor is the **instant of racket–ball contact** on the serve (or the **first clear contact** if motion blur spans several frames).  
  Do **not** move the anchor 1–2 s earlier “to include wind-up”: **`WINDOW_BEFORE_SEC`** already covers the seconds *before* the anchor.
- **Consistency beats perfection:** pick a rule (e.g. “first frame where ball clearly leaves racket toward the service box”) and apply it the same way for every clip in the project.

---

## 2. Count as **serve** (positive)

Label **serve** when, in the window ending at the anchor, the player is performing a **legal or attempted legal serve** for a live point:

- **Standard serve:** underhand motion, contact below waist (or your league’s rule), ball struck to start the rally.
- **Second serve** after a fault: still **serve** (same anchor rule at contact).
- **Service motion that ends in a fault** (net, long, wide): still **serve** if the anchor is on **that** serve attempt’s contact (or clear attempted contact).
- **Different players / angles:** same definition; camera cut does not change the label if the action is still a serve.

If you are unsure whether the motion is a serve vs a similar stroke, use **§4** and prefer **uncertain / skip** until rules are clarified.

---

## 3. Count as **not-serve** (negative)

Use **not-serve** when the clip is **not** a serve attempt as defined above. Prefer clips where the **last part of the window** (near the anchor) is clearly **not** serve contact.

**Typical good negatives**

- **Rally:** groundstrokes, volleys, dinks, overheads **during** a point (after the serve has already happened).
- **Between points:** walking, toweling, ball pickup, discussion, change of ends.
- **Warm-up / practice** before the match or casual feeds that are **not** formal serves for a scored point (if your project excludes those from “serve”).
- **Receiver waiting** while the server is not in a clear serve motion ending at your anchor.
- **Another court / background** action that could confuse a detector (if your crop still shows ambiguous motion, treat as hard negative only if you are sure it is not a serve).

**Hard negatives (especially valuable)**

- **False positives from your proposer:** whatever your cheap stage flagged but is **not** a serve—label those moments as **not-serve** and export windows around them.
- **Serve-like posture without a serve:** practice toss with no hit, aborted motion, ball dropped and re-tossed before any serve contact.
- **Overhead smashes or high volleys** that can look like a serving pose from a distance.
- **Slow motion / replay** of a serve: usually **not-serve** if your model should only see **live** footage (state this in your project scope).

---

## 4. Edge cases (decide once, write down your choice)

| Situation | Recommended default | Notes |
|-----------|---------------------|--------|
| **Foot fault / illegal motion** | **Serve** if anchor is on contact | Model learns “serve motion”; legality can be a separate task. |
| **Let / replay** | **Serve** on the **actual** contact you anchor | If you only label one anchor per logical serve, pick the one you want the model to learn. |
| **Ball toss only, no hit** | **Not-serve** | Anchor should not be placed at contact if there is none; pick another negative window or skip. |
| **Drop serve / bounce serve** | **Serve** at contact with ball after allowed bounce | Align with your league rules; stay consistent. |
| **Singles vs doubles** | **Serve** same rule | Server identity does not change the definition. |
| **Kids / training feeds** | Project policy | Either include as serve or exclude entirely; do not mix inconsistent policies in one dataset. |
| **Occluded contact** | **Serve** if expert consensus says contact is a serve | Otherwise skip or mark uncertain. |
| **Clip ends before contact** | **Do not** use as serve example | Bad alignment with anchor semantics. |
| **Slow motion replay** | **Not-serve** (if live-game scope) | Avoid teaching the model replay graphics. |

Keep a **one-page project addendum** (e.g. “we include second serves; we exclude replays”) for anything you customize.

---

## 5. Not-serve anchor placement (when you choose times manually)

- For **not-serve**, the same math applies: frames span **[anchor − WINDOW_BEFORE, anchor]**.
- Put the anchor on a moment that is **clearly not serve contact**—often **mid-rally** or **between points**—so the **end** of the clip is not accidentally a serve hit.
- Avoid anchors **immediately adjacent** to a serve contact (within ~1–2 s) unless you intentionally want a **hard** “almost serve” negative; if you do, document it.

---

## 6. Balance and quality

- Include enough **hard negatives** (§3) so the model learns to **suppress** your real false positives.
- **Balance** serve vs not-serve roughly for training stability; exact ratio can be tuned later.
- **One match / one camera session** should not appear in both train and val if you split by `video_id` in the fine-tune notebook.

---

## 7. Quick checklist before export

- [ ] Anchor on **contact** (or your written equivalent), not seconds before.
- [ ] `WINDOW_BEFORE_SEC` matches how you think about “context before the hit.”
- [ ] Second serves and faults handled per §4.
- [ ] Replays / warm-up policy is explicit.
- [ ] Hard negatives from your detector are included where possible.

For general folder-based labeling and tooling, see [dataset_labeling.md](./dataset_labeling.md).
