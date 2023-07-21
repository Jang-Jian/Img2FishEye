import cv2, time
import interface


if __name__ == "__main__":
    image_path = "../test/2_77_00.mp4_00001.jpg"
    k1 = 0.0000007
    k2 = 0.00000000005
    k3 = 0.7
    angle = 0.0

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
    
    print(dst_ndarray.shape)
    cv2.imshow("dst_ndarray", dst_ndarray[:, :, :, 0])
    cv2.waitKey(0)