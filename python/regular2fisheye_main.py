import cv2, time
import interface
import annotation


if __name__ == "__main__":
    xml_path = '../test/2_78_02.mp4_00001.xml'
    image_path = "../test/2_78_02.mp4_00001.jpg"
    k1 = 0.0000007
    k2 = 0.00000000005
    k3 = 0.7
    angle = 0.0

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


    
    ground_truths = annotation.fisheye_transform_xml(xml_path, src_ndarray.shape, angle, k1, k2, k3)

    for index in range(len(ground_truths)):
        x_min, y_min, x_max, y_max, name = ground_truths[index]
        cv2.rectangle(dst_ndarray[:, :, :, 0], (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    print(dst_ndarray.shape)
    cv2.imshow("dst_ndarray", dst_ndarray[:, :, :, 0])
    cv2.waitKey(0)
