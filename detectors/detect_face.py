from insightface.app import FaceAnalysis
import cv2
import os


class ArcFaceExtractor:
    def __init__(self, ctx_id=0, det_size=(640, 640), allowed_modules=("detection", "recognition")):
        """
        ctx_id = 0: GPU
        ctx_id = -1: CPU

        allowed_modules: CHỈ load/chạy đúng những model cần thiết. Mặc định
        FaceAnalysis() load ĐỦ 5 model (detection, recognition, landmark_3d_68,
        landmark_2d_106, genderage) và app.get() chạy TẤT CẢ mỗi lần gọi, dù
        pipeline hiện tại chỉ dùng bbox (detection) + embedding (recognition)
        để so khớp. 3 model landmark/genderage hoàn toàn lãng phí compute -
        đây là nguyên nhân chính khiến mỗi lần detect mất 900ms-1.1s dù đã
        giảm det_size. Giới hạn allowed_modules giúp giảm đáng kể thời gian
        này (chỉ còn chạy 2/5 model).
        """
        self.app = FaceAnalysis(allowed_modules=list(allowed_modules))
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect_faces(self, img):
        """
        Nhận ảnh (numpy array) và trả về danh sách face.
        """
        return self.app.get(img)

    def extract_embeddings(self, img):
        """
        Trả về:
            embeddings: list embedding
            bboxes: list bounding box
        """
        faces = self.app.get(img)

        embeddings = []
        bboxes = []

        for face in faces:
            embeddings.append(face.embedding)
            bboxes.append(face.bbox.astype(int))

        return embeddings, bboxes

    def save_faces(self, img, output_dir="faces"):
        """
        Cắt từng khuôn mặt và lưu ra thư mục.
        """
        os.makedirs(output_dir, exist_ok=True)

        faces = self.app.get(img)

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)

            crop = img[y1:y2, x1:x2]

            cv2.imwrite(
                os.path.join(output_dir, f"face_{i}.jpg"),
                crop
            )

        return len(faces)

    def draw_faces(self, img):
        """
        Vẽ bounding box lên ảnh.
        """
        result = img.copy()

        faces = self.app.get(result)

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)

            cv2.rectangle(
                result,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        return result
    
# extractor = ArcFaceExtractor(ctx_id=0)

# img = cv2.imread("/home/lychien/Desktop/test/captures/loptest.jpg")

# embeddings, bboxes = extractor.extract_embeddings(img)

# print("Số khuôn mặt:", len(embeddings))

# if len(embeddings) > 0:
#     print("Kích thước embedding:", len(embeddings[0]))

# extractor.save_faces(img)

# result = extractor.draw_faces(img)

# cv2.imwrite("result.jpg", result)