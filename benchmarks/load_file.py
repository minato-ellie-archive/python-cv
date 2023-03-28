import timeit
import cv2

from pythoncv.io import read_video_from_file


def baseline():
    cap = cv2.VideoCapture('demos/sample.mp4')
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"Frame {idx}: {frame.shape}")
        idx += 1

    cap.release()


def ours():
    video = read_video_from_file('demos/sample.mp4')
    idx = 0
    for frame in video:
        print(f"Frame {idx}: {frame.shape}")
        idx += 1


def main():
    ours_time = timeit.timeit(ours, number=10) / 10.
    baseline_time = timeit.timeit(baseline, number=10) / 10.

    print("================ Results ================")
    print("Baseline:")
    print(f"Time: {baseline_time} s")  # Time: 6.98786884 s
    print("Ours:")
    print(f"Time: {ours_time} s")  # Time: 6.95311476 s


if __name__ == "__main__":
    main()
