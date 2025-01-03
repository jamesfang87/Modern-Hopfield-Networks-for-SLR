import cv2
import mediapipe as mp
import numpy as np


class VideoReader:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic

        # list of results from holistic.process(frame) for all frames
        self.video_results = None  # instantiated to None for now

    def read_video(self, video_path: str, show_video: bool = False) -> None:
        """
        Reads in and returns the 3D locations of important hand and pose landmarks from a video of a sign.

        :param video_path: path to the video
        :param show_video: whether to display the video or not
        """
        with self.mp_holistic.Holistic() as holistic:
            cap = cv2.VideoCapture(video_path)

            # change video_results so that it is np.zeros with shape T x num_landmarks x spatial dimensions (3D)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.video_results = np.zeros((int(total_frames), 21 + 21 + 17, 3))

            frame_num = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # mark frame as not writeable for performance increase
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # generate results from holistic model
                results = holistic.process(frame)
                self.extract_important_landmarks(results, frame_num)

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

                # increment frame number
                frame_num += 1

                if cv2.waitKey(5) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()

    def extract_important_landmarks(self, results, frame_num: int) -> None:
        """
        Extracts all hand landmarks and pose landmarks numbered 0-16 from a results object
        the x, y, z coordinates of these landmarks are stored into self.video_results. Left hand
        landmarks are stored in the first 21 indices, right hand landmarks are stored in the next 21,
        and pose landmarks in the last 17.
        
        Precondition: results and self.video_results is not None, 0 <= frame_num < max number of frames

        :param results: results returned from the mediapipe holistic model
        :param frame_num: the current frame number
        """
        # extract all from left hand
        for i in range(21):
            if results.left_hand_landmarks:
                temp = results.left_hand_landmarks.landmark[i]
                x, y, z = temp.x, temp.y, temp.z
                self.video_results[frame_num, i] = np.array([x, y, z])

        # extract all from right hand
        for i in range(21):
            if results.right_hand_landmarks:
                temp = results.right_hand_landmarks.landmark[i]
                x, y, z = temp.x, temp.y, temp.z
                self.video_results[frame_num, 21 + i] = np.array([x, y, z])

        # extract landmarks 0-16 from pose
        for i in range(17):
            if results.pose_landmarks:
                temp = results.pose_landmarks.landmark[i]
                x, y, z = temp.x, temp.y, temp.z
                self.video_results[frame_num, 42 + i] = np.array([x, y, z])

    def draw_landmarks(self, frame, results):
        """
        Draws results from mediapipe holistic model on the frame and returns the new frame
        """
        self.mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            self.mp_holistic.FACEMESH_CONTOURS,
            None,
            self.mp_drawing_styles.get_default_face_mesh_contours_style())
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            self.mp_drawing_styles.get_default_pose_landmarks_style())
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

    def write_data(self, file_name: str, output_path: str) -> None:
        """
        Saves the 3D locations read from the video into output_path/file_name.npy (as a .npy file)

        :param file_name: the name of the .npy file to save the asl_citizen as
        :param output_path: the path to put the .npy file into
        :return: None
        """
        np.save(output_path + "/" + file_name, self.video_results)
