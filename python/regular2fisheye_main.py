import cv2, time
import interface
import annotation


if __name__ == "__main__":
    xml_path = '../test/FILE3567_04.mp4_01716.xml'
    image_path = "../test/FILE3567_04.mp4_01716.jpg"
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

    a = time.time()
    src_tensor = interface.nd2tensor(src_ndarray, interface.HWCN)
    b = time.time()
    dst_tensor = interface.regular2fisheye(src_tensor, angle, k1, k2, k3)
    c = time.time()
    dst_ndarray = interface.tensor2nd(dst_tensor, interface.HWCN)
    d = time.time()

    print("nd2tensor time:", b - a)
    print("regular2fisheye time:", c - b)
    print("tensor2nd time:", d - c)
    print("total time:", d - a)


    e = time.time()
    fisheye_coord_transform = annotation.Regular2Fisheye(map_width, map_height, angle, k1, k2, k3)
    f = time.time()
    ground_truths = fisheye_coord_transform.transform_xml(xml_path)
    g = time.time()

    print("transform regular coordinates to fisheye coordinates (initialization):", f - e)
    print("transform regular coordinates to fisheye coordinates (execution):", g - f)


    for index in range(len(ground_truths)):
        x_min, y_min, x_max, y_max, name = ground_truths[index]
        cv2.rectangle(dst_ndarray[:, :, :, 0], (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    print(dst_ndarray.shape)
    cv2.imshow("dst_ndarray", dst_ndarray[:, :, :, 0])
    cv2.waitKey(0)
