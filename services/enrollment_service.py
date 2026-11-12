"""Enrol people from photos.

Directory layout (one folder per person, folder name = person name):
    data/enrollment/
        Alice Smith/  img1.jpg img2.jpg ...
        Bob/          bob.png
Use 3-10 clear, frontal-ish photos per person, ideally from the real cameras.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from constants import ENROLLMENT_DIR, IMAGE_EXTENSIONS
from database.embedding_repository import EmbeddingRepository
from database.person_repository import PersonRepository
from inference.inference_manager import InferenceManager
from processing.face_alignment import norm_crop
from processing.face_detection import faces_from_detections
from processing.face_quality import QualityConfig, assess_quality
from utils.image_utils import load_image, resize_max

log = logging.getLogger(__name__)


@dataclass
class EnrollResult:
    name: str
    person_id: Optional[int] = None
    added: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)


class EnrollmentService:
    def __init__(self, inference: InferenceManager, persons: PersonRepository, embeddings: EmbeddingRepository,
                 min_face_size: int = 60, blur_threshold: float = 20.0) -> None:
        self.inference = inference
        self.persons = persons
        self.embeddings = embeddings
        self.quality_cfg = QualityConfig(min_face_size=min_face_size, blur_threshold=blur_threshold,
                                         max_yaw_ratio=0.6, min_quality=0.1)

    def _best_face_crop(self, img: np.ndarray):
        dets, kps = self.inference.detect([img])[0]
        faces = faces_from_detections(dets, kps)
        if not faces:
            return None, "no face"
        if len(faces) > 1:
            faces.sort(key=lambda f: f.area, reverse=True)
            if faces[1].area > 0.5 * faces[0].area:
                return None, "multiple faces of similar size"
        face = faces[0]
        crop = norm_crop(img, face.kps)
        q = assess_quality(face, crop, self.quality_cfg)
        if not q.passed:
            return None, f"low quality ({q.reason})"
        return (crop, q.score), ""

    def enroll_images(self, name: str, images: Iterable[Path], external_id: Optional[str] = None,
                      replace: bool = False) -> EnrollResult:
        res = EnrollResult(name=name)
        pid = self.persons.get_or_create(name, external_id)
        res.person_id = pid
        if replace:
            self.embeddings.delete_for_person(pid)

        crops, meta = [], []
        for path in images:
            path = Path(path)
            src = str(path.resolve())
            if not replace and self.embeddings.has_source(pid, src):
                res.skipped += 1
                continue
            img = load_image(path)
            if img is None:
                res.errors.append(f"{path.name}: unreadable")
                continue
            img = resize_max(img, 1920)
            found, err = self._best_face_crop(img)
            if found is None:
                res.errors.append(f"{path.name}: {err}")
                continue
            crops.append(found[0])
            meta.append((found[1], src))

        if crops:
            emb = self.inference.embed(crops)
            res.added = self.embeddings.add_many((pid, emb[i], meta[i][0], meta[i][1]) for i in range(len(crops)))
        log.info("Enrolled %s (id=%s): +%d embeddings, %d skipped, %d errors",
                 name, pid, res.added, res.skipped, len(res.errors))
        for e in res.errors:
            log.warning("  %s: %s", name, e)
        return res

    def enroll_directory(self, root: Path = ENROLLMENT_DIR, replace: bool = False) -> List[EnrollResult]:
        root = Path(root)
        results = []
        if not root.exists():
            log.warning("Enrolment directory %s does not exist", root)
            return results
        for person_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            images = sorted(p for p in person_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
            if images:
                results.append(self.enroll_images(person_dir.name, images, replace=replace))
        return results

    def delete_person(self, person_id: int) -> bool:
        return self.persons.delete(person_id)
