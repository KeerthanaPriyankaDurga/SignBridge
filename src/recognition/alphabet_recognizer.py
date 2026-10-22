import cv2
import mediapipe as mp


class AlphabetRecognizer:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

    def extract_landmarks(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        features = []
        for hand in result.multi_hand_landmarks[:2]:
            for point in hand.landmark:
                features.extend([point.x, point.y, point.z])

        return features

    def close(self):
        self.hands.close()