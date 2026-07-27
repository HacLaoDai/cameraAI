"""
⚠️ DEPRECATED - KHÔNG còn được dùng trong pipeline chính (multi_main.py).

multi_main.py hiện dùng recognition/person_db_recognizer.py (insightface +
faiss, đọc trực tiếp từ MongoDB persons) để nhận diện, KHÔNG dùng
ArcFaceRecognizer (DeepFace + faiss trên file ảnh) ở file này nữa.

File này (cùng với face_identifier.py) là cách tiếp cận CŨ, giữ lại chỉ để
tham khảo. Nếu không còn dùng, nên xoá hẳn để tránh nhầm lẫn khi bảo trì
sau này (2 pipeline nhận diện // 2 nguồn embedding khác nhau rất dễ gây lỗi
nếu ai đó vô tình import nhầm file).
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # ép chạy CPU tránh lỗi CUDA
import pickle

import faiss
import numpy as np

from deepface import DeepFace


class ArcFaceRecognizer:

    def __init__(
        self,
        db_folder,
        model_name="ArcFace",
        detector="retinaface",
        threshold=0.62,
        index_file="faces.index",
        path_file="paths.pkl",
    ):

        self.db_folder = db_folder

        self.model_name = model_name
        self.detector = detector
        self.threshold = threshold

        self.index_file = index_file
        self.path_file = path_file

        self.index = None
        self.paths = []

    # ----------------------------------
    # Build database
    # ----------------------------------
    def build_index(self):

        embeddings = []
        paths = []

        for file in os.listdir(self.db_folder):

            if not file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            img_path = os.path.join(
                self.db_folder,
                file,
            )

            try:

                rep = DeepFace.represent(
                    img_path=img_path,
                    model_name=self.model_name,
                    detector_backend=self.detector,
                    enforce_detection=False,
                )

                embedding = rep[0]["embedding"]

                embeddings.append(embedding)
                paths.append(img_path)

                print("Indexed:", img_path)

            except Exception as e:

                print("Skip:", img_path, e)

        if len(embeddings) == 0:
            raise RuntimeError("Không tìm thấy embedding nào.")

        embeddings = np.array(embeddings, dtype=np.float32)

        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])

        index.add(embeddings)

        faiss.write_index(index, self.index_file)

        pickle.dump(paths, open(self.path_file, "wb"))

        self.index = index
        self.paths = paths

        print(f"Build xong {len(paths)} ảnh")

    # ----------------------------------
    # Load index
    # ----------------------------------
    def load_index(self):

        self.index = faiss.read_index(self.index_file)

        self.paths = pickle.load(open(self.path_file, "rb"))

    # ----------------------------------
    # Extract embedding
    # ----------------------------------
    def extract_embedding(self, img_path):

        rep = DeepFace.represent(
            img_path=img_path,
            model_name=self.model_name,
            detector_backend=self.detector,
            enforce_detection=False,
        )

        embedding = np.array(rep[0]["embedding"], dtype=np.float32)

        return embedding

    # ----------------------------------
    # Search
    # ----------------------------------
    def search(self, query_img, k=1):

        if self.index is None:
            self.load_index()

        embedding = self.extract_embedding(query_img)

        query = np.array([embedding], dtype=np.float32)

        faiss.normalize_L2(query)

        D, I = self.index.search(query, k)

        score = float(D[0][0])
        idx = int(I[0][0])

        result = {
            "matched": score >= self.threshold,
            "score": score,
            "path": self.paths[idx],
        }

        return result


# ==========================================================
# CHỈ chạy khi gọi trực tiếp file này (python build_system_regconition.py)
# Import module này ở nơi khác (vd: face_identifier.py) sẽ KHÔNG
# tự động chạy đoạn test bên dưới -> an toàn khi ghép vào pipeline.
# ==========================================================
if __name__ == "__main__":

    recognizer = ArcFaceRecognizer(
        db_folder="/home/lychien/Desktop/img"
    )

    # Chạy 1 lần duy nhất để build index, sau đó comment lại
    # recognizer.build_index()

    recognizer.load_index()

    result = recognizer.search(
        "/home/lychien/Desktop/Project_new/lisa2.png"
    )

    print(result)