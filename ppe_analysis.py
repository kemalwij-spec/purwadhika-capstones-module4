"""
Logic analisis kelengkapan APD (Alat Pelindung Diri).

Mengambil hasil deteksi mentah YOLO untuk sebuah gambar (box person, helmet,
no-helmet, vest, no-vest) dan menentukan status kelengkapan APD tiap pekerja
yang terdeteksi.

Aturan penting: pekerja yang tidak punya box helmet/no-helmet yang cocok
ditandai "unclear", bukan otomatis dianggap sudah pakai
"""

import os
import glob
from dataclasses import dataclass
from typing import List, Optional
from ultralytics import YOLO
from PIL import Image as PILImage, ImageDraw

OVERLAP_THRESHOLD = 0.5  # persentase area box gear yang harus ada di dalam
                          # box person supaya dianggap "milik" orang itu


@dataclass
class Detection:
    cls_name: str
    conf: float
    xyxy: tuple  # (x1, y1, x2, y2)


@dataclass
class PersonStatus:
    box: tuple
    head_status: str            # "helmet" | "no-helmet" | "unclear"
    body_status: str            # "vest" | "no-vest" | "unclear"
    head_conf: Optional[float] = None
    body_conf: Optional[float] = None

    @property
    def fully_equipped(self):
        return self.head_status == "helmet" and self.body_status == "vest"

    @property
    def confirmed_missing(self):
        return self.head_status == "no-helmet" or self.body_status == "no-vest"

    @property
    def needs_review(self):
        return self.head_status == "unclear" or self.body_status == "unclear"


def box_overlap_ratio(gear_box, person_box):
    """Fraction area box GEAR yang masuk ke dalam box PERSON. Pakai luas box
    gear sendiri (bukan IoU) karena box helmet yang kecil di atas box person
    yang tinggi akan punya IoU rendah."""
    gx1, gy1, gx2, gy2 = gear_box
    px1, py1, px2, py2 = person_box
    ix1, iy1 = max(gx1, px1), max(gy1, py1)
    ix2, iy2 = min(gx2, px2), min(gy2, py2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter_area = (ix2 - ix1) * (iy2 - iy1)
    gear_area = (gx2 - gx1) * (gy2 - gy1)
    return inter_area / gear_area if gear_area > 0 else 0.0


def assign_gear_to_persons(persons, gear_detections, overlap_threshold=OVERLAP_THRESHOLD):
    """Tiap box gear di-assign ke box person dengan overlap terbesar, kalau
    overlap-nya lolos threshold. Return {index_person: [gear_detections]}."""
    assignments = {i: [] for i in range(len(persons))}
    for gear in gear_detections:
        best_idx, best_ratio = None, 0.0
        for i, person in enumerate(persons):
            ratio = box_overlap_ratio(gear.xyxy, person.xyxy)
            if ratio > best_ratio:
                best_idx, best_ratio = i, ratio
        if best_idx is not None and best_ratio >= overlap_threshold:
            assignments[best_idx].append(gear)
    return assignments


def resolve_status(gear_list, positive_label, negative_label):
    """Tentukan status head/body dari gear yang sudah di-assign ke satu
    orang. Kalau label positif & negatif sama-sama muncul (jarang terjadi,
    biasanya di foto ramai), pilih yang confidence-nya lebih tinggi."""
    candidates = [g for g in gear_list if g.cls_name in (positive_label, negative_label)]
    if not candidates:
        return "unclear", None
    best = max(candidates, key=lambda g: g.conf)
    return best.cls_name, best.conf


def analyze_detections(detections: List[Detection], overlap_threshold=OVERLAP_THRESHOLD):
    persons = [d for d in detections if d.cls_name == "person"]
    gear = [d for d in detections if d.cls_name != "person"]
    assignments = assign_gear_to_persons(persons, gear, overlap_threshold)

    statuses = []
    for i, person in enumerate(persons):
        assigned = assignments[i]
        head_status, head_conf = resolve_status(assigned, "helmet", "no-helmet")
        body_status, body_conf = resolve_status(assigned, "vest", "no-vest")
        statuses.append(PersonStatus(person.xyxy, head_status, body_status, head_conf, body_conf))
    return statuses


def summarize(statuses: List[PersonStatus]) -> dict:
    total = len(statuses)
    return {
        "total_workers": total,
        "fully_equipped": sum(s.fully_equipped for s in statuses),
        "confirmed_missing_gear": sum(s.confirmed_missing for s in statuses),
        "needs_review": sum(s.needs_review and not s.confirmed_missing for s in statuses),
        "missing_helmet_count": sum(s.head_status == "no-helmet" for s in statuses),
        "missing_vest_count": sum(s.body_status == "no-vest" for s in statuses),
    }


def summary_text(summary: dict) -> str:
    if summary["total_workers"] == 0:
        return "Tidak ada pekerja terdeteksi di gambar ini."
    parts = [f"{summary['total_workers']} pekerja terdeteksi"]
    if summary["confirmed_missing_gear"]:
        parts.append(f"{summary['confirmed_missing_gear']} APD Tidak Lengkap")
    if summary["needs_review"]:
        parts.append(f"{summary['needs_review']} belum jelas (perlu dicek manual)")
    if summary["fully_equipped"] == summary["total_workers"]:
        parts.append("semua sudah lengkap APD-nya")
    return ", ".join(parts) + "."


def run_on_image(model_path, image_path, conf=0.25, overlap_threshold=OVERLAP_THRESHOLD):
    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf, verbose=False)[0]
    names = results.names
    detections = [
        Detection(names[int(cls)], float(c), tuple(map(float, box)))
        for box, cls, c in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf)
    ]
    statuses = analyze_detections(detections, overlap_threshold)
    summary = summarize(statuses)
    return statuses, summary, results


STATUS_COLORS = {
    "fully_equipped": (0, 170, 0),      # hijau
    "confirmed_missing": (220, 30, 30), # merah
    "needs_review": (230, 170, 0),      # kuning/amber
}


def person_status_category(s: PersonStatus) -> str:
    if s.confirmed_missing:
        return "confirmed_missing"
    if s.needs_review:
        return "needs_review"
    return "fully_equipped"


def draw_annotated(image, statuses: List[PersonStatus], out_path=None):
    """Gambar box tiap orang dengan warna sesuai status: hijau = lengkap,
    merah = APD tidak lengkap, kuning = belum jelas / perlu dicek manual.
    `image` boleh berupa path file (str) atau objek PIL Image yang sudah dibuka."""
    img = PILImage.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for s in statuses:
        color = STATUS_COLORS[person_status_category(s)]
        x1, y1, x2, y2 = s.box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{s.head_status}/{s.body_status}"
        draw.text((x1, max(0, y1 - 12)), label, fill=color)
    if out_path:
        img.save(out_path)
    return img


def batch_test(model_path, image_dir, out_dir, conf=0.25, overlap_threshold=OVERLAP_THRESHOLD, limit=30):
    """Jalankan pipeline lengkap ke banyak gambar sekaligus: print ringkasan
    per gambar dan simpan copy yang sudah dianotasi supaya gampang direview.
    Gambar dengan pelanggaran/unclear ditandai di output supaya bisa langsung
    dicek tanpa scroll semua gambar satu per satu."""
    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(model_path)
    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    )[:limit]

    for img_path in image_paths:
        results = model.predict(img_path, conf=conf, verbose=False)[0]
        names = results.names
        detections = [
            Detection(names[int(cls)], float(c), tuple(map(float, box)))
            for box, cls, c in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf)
        ]
        statuses = analyze_detections(detections, overlap_threshold)
        summary = summarize(statuses)

        fname = os.path.basename(img_path)
        flag = "  <-- REVIEW" if (summary["confirmed_missing_gear"] or summary["needs_review"]) else ""
        print(f"{fname}: {summary_text(summary)}{flag}")

        draw_annotated(img_path, statuses, os.path.join(out_dir, fname))

    print(f"\nGambar hasil anotasi disimpan di: {out_dir}")


def debug_image(model_path, image_path, conf=0.25):
    """Print semua deteksi mentah dan, untuk tiap person, overlap ratio ke
    tiap box gear di gambar itu -- dipakai untuk membedakan 'memang tidak
    ada gear terdeteksi' vs 'gear terdeteksi tapi overlap-nya di bawah
    threshold'."""
    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf, verbose=False)[0]
    names = results.names
    detections = [
        Detection(names[int(cls)], float(c), tuple(map(float, box)))
        for box, cls, c in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf)
    ]

    print(f"Total deteksi mentah: {len(detections)}")
    for d in detections:
        print(f"  {d.cls_name:12s} conf={d.conf:.2f}  box={tuple(round(v) for v in d.xyxy)}")

    persons = [d for d in detections if d.cls_name == "person"]
    gear = [d for d in detections if d.cls_name != "person"]
    print(f"\n{len(persons)} box person, {len(gear)} box gear.")

    for i, person in enumerate(persons):
        print(f"\nPerson {i} box={tuple(round(v) for v in person.xyxy)}:")
        if not gear:
            print("    (tidak ada gear terdeteksi sama sekali di gambar ini)")
        for g in gear:
            ratio = box_overlap_ratio(g.xyxy, person.xyxy)
            flag = "  <-- cocok (>=0.5)" if ratio >= OVERLAP_THRESHOLD else ""
            print(f"    vs {g.cls_name:12s} conf={g.conf:.2f}  overlap_ratio={ratio:.2f}{flag}")

