import os
import cv2

while True:
    file_name_without_extension = input("Please enter the file name of the fingerprint image (or type 'exit' to quit): ")

    if file_name_without_extension.lower() == 'exit':
        break

    file_extension = ".BMP"
    sample_path = os.path.join("archive/Altered-Real", file_name_without_extension + file_extension)

    if not os.path.isfile(sample_path):
        print("File could not be found.")
    else:
        sample = cv2.imread(sample_path)
        if sample is None:
            print("File could not be read.")
        else:
            best_score = 0
            filename = None
            image = None
            kp1, kp2, mp = None, None, None

            sift = cv2.SIFT_create()

            keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)

            for file in os.listdir("archive/Real/")[:500]:
                fingerprint_image = cv2.imread("archive/Real/" + file)

                if fingerprint_image is None:
                    continue

                keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

                flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {})
                matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

                match_points = []
                for match in matches:
                    if len(match) == 2:
                        p, q = match
                        if p.distance < 0.1 * q.distance:
                            match_points.append(p)

                keypoints = min(len(keypoints_1), len(keypoints_2))
                if keypoints > 0:
                    score = len(match_points) / keypoints * 100

                    if score > best_score:
                        best_score = score
                        filename = file
                        image = fingerprint_image
                        kp1, kp2, mp = keypoints_1, keypoints_2, match_points

            print("BEST MATCH: " + filename)
            print("ACCURACY: " + str(best_score))

            result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
            result = cv2.resize(result, None, fx=4, fy=4)
            cv2.imshow("Result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()                   
            
