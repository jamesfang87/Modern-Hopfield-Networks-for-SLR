import numpy as np
import cv2
import mediapipe as mp


class VideoReader:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic

    def read_video(self, video_path: str, show_video: bool = False) -> np.ndarray:
        with self.mp_holistic.Holistic() as holistic:
            all_results = [] # list of results from holistic.process(frame) for all frames
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # mark frame as not writeable for performance increase
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # generate results from holistic model
                results = holistic.process(frame)
                all_results.append(results)

                # if show_video is true and we have results
                if show_video and results is not None:
                    # mark as writeable again and convert back to RGB before
                    # drawing landmarks
                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = self.draw_landmarks(frame, results)
                    # show flipped image
                    video_name = video_path.split("/")[-1]
                    cv2.imshow(f"{video_name}", cv2.flip(frame, 1))

                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()

        return np.ndarray(all_results)  # this don't work

    def extract_important_landmarks(self, results):
        pass

    def draw_landmarks(self, frame, results):
        self.mp_drawing.draw_landmarks(
                    frame,
                    results.face_landmarks,
                    self.mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_contours_style())
        self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles
                    .get_default_pose_landmarks_style())
        self.mp_drawing.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
        self.mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
        return frame
    
    def write_data(self, output_path: str) -> None:
        pass