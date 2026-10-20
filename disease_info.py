"""
Turns the pretrained model's output into a readable report.

The model is multi-label (a patient can have several conditions), so the
report works from per-class probabilities AND the model's thresholded
predictions, not from a single argmax.

Educational text only. Not a diagnosis.
"""

# Self-reported F1 per class on the model's ODIR-5K validation split
# (from the model card). Shown so users know how far to trust each class.
F1 = {"N": 0.659, "D": 0.717, "G": 0.649, "C": 0.857,
      "A": 0.659, "H": 0.306, "M": 0.889, "O": 0.518}

INFO = {
    "N": {
        "name": "Normal",
        "what": "No features the model associates with the conditions it knows.",
        "signs": "Clear optic disc margins, healthy rim, uniform macula, normal vessels.",
        "risk": "Not a clearance: the model only knows 8 categories and can miss early or rare disease.",
        "next": "Keep routine eye exams, especially with diabetes, high blood pressure, "
                "family history of glaucoma, or age over 50.",
    },
    "D": {
        "name": "Diabetic eye disease (diabetic retinopathy)",
        "what": "Damage to the small retinal blood vessels caused by long-standing high blood sugar.",
        "signs": "Microaneurysms (tiny red dots), dot/blot hemorrhages, hard exudates (yellow deposits), "
                 "cotton-wool spots, and in advanced stages abnormal new vessels.",
        "risk": "Often painless until late; a leading cause of preventable blindness.",
        "next": "Dilated eye exam by an ophthalmologist, blood sugar and blood pressure control, "
                "regular screening.",
    },
    "G": {
        "name": "Glaucoma",
        "what": "Progressive damage to the optic nerve, often linked to raised eye pressure.",
        "signs": "Enlarged optic cup (high cup-to-disc ratio), thinning neuroretinal rim, "
                 "nerve fibre layer defects, disc hemorrhages.",
        "risk": "Usually painless; peripheral vision is lost first and the damage is irreversible.",
        "next": "Eye pressure measurement, visual field test, OCT of the optic nerve, specialist follow-up.",
    },
    "C": {
        "name": "Cataract",
        "what": "Clouding of the eye's lens that scatters light.",
        "signs": "In a fundus photo it shows up as a hazy, blurred, low-contrast image "
                 "with poorly visible vessels and disc.",
        "risk": "Gradual blurry or dim vision, glare; the most common treatable cause of vision loss.",
        "next": "Slit-lamp exam by an eye doctor; surgery is effective when vision is affected.",
    },
    "A": {
        "name": "Age-related Macular Degeneration (AMD)",
        "what": "Degeneration of the macula, the central retina used for sharp vision.",
        "signs": "Drusen (yellowish deposits), pigment changes, and in advanced forms "
                 "geographic atrophy or bleeding from abnormal vessels (wet AMD).",
        "risk": "Central vision loss (blur, distorted straight lines); more common after 50.",
        "next": "Retina specialist review with OCT; wet AMD needs prompt treatment.",
    },
    "H": {
        "name": "Hypertensive retinopathy",
        "what": "Retinal vessel changes caused by long-term high blood pressure.",
        "signs": "Narrowed arterioles, arteriovenous nicking, flame hemorrhages, "
                 "cotton-wool spots, and in severe cases optic disc swelling.",
        "risk": "Signals blood-vessel damage elsewhere in the body as well.",
        "next": "Blood pressure check and medical review, plus a dilated eye exam.",
    },
    "M": {
        "name": "Pathological myopia",
        "what": "Severe short-sightedness with stretching of the eyeball and retina.",
        "signs": "Tilted disc, peripapillary atrophy, a pale 'tessellated' fundus, "
                 "and sometimes chorioretinal atrophy at the posterior pole.",
        "risk": "Raises the risk of retinal tears, detachment, and macular damage.",
        "next": "Specialist retinal exam; prompt care if flashes, floaters or a curtain over vision appear.",
    },
    "O": {
        "name": "Other abnormality",
        "what": "A catch-all for findings outside the other seven categories.",
        "signs": "Varies widely; the model cannot say which condition.",
        "risk": "Unspecific by nature, so this label is the least informative.",
        "next": "An eye doctor should review the image to identify what was flagged.",
    },
}


def build_report(probs, preds, single_eye=False):
    """probs: {code: prob}; preds: {code: bool} (per-class tuned thresholds)."""
    ranked = sorted(probs.items(), key=lambda t: -t[1])
    flagged = [c for c, _ in ranked if preds.get(c) and c != "N"]
    normal = bool(preds.get("N"))

    L = ["=" * 64, "RESULT", "=" * 64, "Class probabilities:"]
    for c, p in ranked:
        mark = "  <- flagged" if preds.get(c) else ""
        L.append(f"  {c}  {INFO[c]['name']:<44} {p:.3f}{mark}")

    if flagged:
        L.append("\nVERDICT: possible disease detected -> "
                 + ", ".join(INFO[c]["name"] for c in flagged))
        if normal:
            L.append("Note: 'Normal' was also flagged, so the model is internally conflicted. "
                     "Treat this result with extra caution.")
    elif normal:
        L.append("\nVERDICT: no disease detected by the model.")
    else:
        top = ranked[0]
        L.append(f"\nVERDICT: inconclusive. No class passed its threshold "
                 f"(highest: {INFO[top[0]]['name']} at {top[1]:.2f}).")

    for c in (flagged if flagged else (["N"] if normal else [])):
        i = INFO[c]
        L += [
            "\n" + "-" * 64,
            f"{i['name']}  (model reliability for this class: F1 {F1[c]:.2f})",
            "-" * 64,
            f"What it is:      {i['what']}",
            f"Typical signs:   {i['signs']}",
            f"Why it matters:  {i['risk']}",
            f"Suggested next:  {i['next']}",
        ]
        if F1[c] < 0.5:
            L.append("Warning:         this class is unreliable in the model's own validation results.")

    if single_eye:
        L.append("\nNote: this model expects BOTH eyes. With one image, the same image was used "
                 "for both eyes, so accuracy may be lower.")
    L.append("\nDisclaimer: educational output from a research model that is not clinically "
             "validated. It is not a diagnosis. See an eye doctor for confirmation.")
    return "\n".join(L)