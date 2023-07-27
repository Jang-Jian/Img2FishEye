import numpy as np
import xml.etree.ElementTree as ET
import interface


class Regular2Fisheye():
    def __init__ (self, map_width: int, map_height: int, 
                  angle: float = 0.0, k1: float = 0.0000007, 
                  k2: float = 0.00000000005, k3: float = 0.7):
        self.cvtptrl2fe = interface.Cvtptrl2fe(map_width, map_height, angle, k1, k2, k3)
        self.map_width = map_width
        self.map_height = map_height
        self.k1 = k1 
        self.k2 = k2 
        self.k3 = k3

    def transform_xml(self, xml_path: str) -> list:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 更新圖片寬度和高度
        #size = root.find('size')
        #size.find('width').text = str(image_shape[1])
        #size.find('height').text = str(image_shape[0])

        # 更新物件標記框座標
        ground_truths = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            name = str(obj.find('name').text)
            x_min = int(bndbox.find('xmin').text)
            y_min = int(bndbox.find('ymin').text)
            x_max = int(bndbox.find('xmax').text)
            y_max = int(bndbox.find('ymax').text)

            #print(name)

            # 執行魚眼轉換
            if self.k1 > 0 and self.k2 > 0 and self.k3 > 0:
                xy_min = self.cvtptrl2fe.cvtCoord(x_min, y_min)
                x_min = int(max(0, xy_min.x))
                y_min = int(max(0, xy_min.y))
                #print(x_min_transformed, y_min_transformed)

                #print(x_max, y_max)
                xy_max = self.cvtptrl2fe.cvtCoord(x_max, y_max)
                x_max = int(min(self.map_width, xy_max.x))
                y_max = int(min(self.map_height, xy_max.y))

            # 更新標記框座標
            #bndbox.find('xmin').text = str(x_min)
            #bndbox.find('ymin').text = str(y_min)
            #bndbox.find('xmax').text = str(x_max)
            #bndbox.find('ymax').text = str(y_max)
            ground_truths.append((x_min, y_min, x_max, y_max, name))

        return ground_truths
    
    def transform_img(self, src_ndarray: np.ndarray) -> np.ndarray:
        if self.k1 > 0 and self.k2 > 0 and self.k3 > 0:
            src_tensor = interface.nd2tensor(src_ndarray, interface.HWCN)
            dst_tensor = interface.Tensor()
            self.cvtptrl2fe.cvtImage(src_tensor, dst_tensor)
            dst_ndarray = interface.tensor2nd(dst_tensor, interface.HWCN)
            return dst_ndarray[:, :, :, 0]
        return src_ndarray