import cv2, time
import interface
import annotation


if __name__ == "__main__":
    xml_path = '../test/2_77_00.mp4_00001.xml'
    image_path = "../test/2_77_00.mp4_00001.jpg"
    k1 = 0.0000007
    k2 = 0.00000000005
    k3 = 0.7
    angle = 0.0
    map_width = 1920
    map_height = 1080

    import os
    print(os.path.exists(xml_path))

    src_ndarray = cv2.imread(image_path)
    h, w, c = src_ndarray.shape


    e = time.time()
    fisheye_coord_transform = annotation.Regular2Fisheye(map_width, map_height, angle, k1, k2, k3)
    f = time.time()
    dst_ndarray = fisheye_coord_transform.transform_img(src_ndarray)
    g = time.time()
    ground_truths = fisheye_coord_transform.transform_xml(xml_path)
    h = time.time()

    print("transform regular coordinates to fisheye coordinates/image (initialization):", f - e)
    print("transform regular image to fisheye image (execution):", g - f)
    print("transform regular coordinates to fisheye coordinates (execution):", h - g)


    for index in range(len(ground_truths)):
        x_min, y_min, x_max, y_max, name = ground_truths[index]
        cv2.rectangle(dst_ndarray, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    print(dst_ndarray.shape)
    cv2.imshow("dst_ndarray", dst_ndarray)
    cv2.waitKey(0)
