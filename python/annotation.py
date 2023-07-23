import xml.etree.ElementTree as ET
import interface


def fisheye_transform_bbox(x_min, y_min, x_max, y_max, image_shape,
                           angle: float = 0.0, k1: float = 0.0000007, 
                           k2: float = 0.00000000005, k3: float = 0.7):
    # 執行魚眼轉換
    # 在這裡，你可以根據需要修改座標的轉換方法，這是一個示範
    # 注意：請確保轉換後的座標在圖片範圍內
    # 這裡只是一個簡單的示範，你可能需要根據實際需求進行更複雜的轉換
    #center_x = image_shape[1] // 2
    #center_y = image_shape[0] // 2
    #print(image_shape[1], image_shape[0])
    #print(center_x, center_y)
    
    #k1 = 0.0000007
    #k2 = 0.00000000005
    #k3 = 0.7
    #angle = 0.0

    #print(x_min, y_min)
    x_transformed, y_transformed = interface.cvtptrl2fe(x_min, y_min, image_shape[1], image_shape[0], angle, k1, k2, k3)
    x_min_transformed = int(max(0, x_transformed))
    y_min_transformed = int(max(0, y_transformed))
    #print(x_min_transformed, y_min_transformed)

    #print(x_max, y_max)
    x_transformed, y_transformed = interface.cvtptrl2fe(x_max, y_max, image_shape[1], image_shape[0], angle, k1, k2, k3)
    x_max_transformed = int(min(image_shape[1], x_transformed))
    y_max_transformed = int(min(image_shape[0], y_transformed))
    #print(x_max_transformed, y_max_transformed)

    #print(x_min, y_min, x_max, y_max)

    return x_min_transformed, y_min_transformed, x_max_transformed, y_max_transformed


def fisheye_transform_xml(xml_path, image_shape,
                          angle: float = 0.0, k1: float = 0.0000007, 
                          k2: float = 0.00000000005, k3: float = 0.7):
    # 讀取 XML 標記檔案
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
        x_min, y_min, x_max, y_max = fisheye_transform_bbox(x_min, y_min, x_max, y_max, image_shape,
                                                            angle, k1, k2, k3)

        # 更新標記框座標
        #bndbox.find('xmin').text = str(x_min)
        #bndbox.find('ymin').text = str(y_min)
        #bndbox.find('xmax').text = str(x_max)
        #bndbox.find('ymax').text = str(y_max)
        ground_truths.append((x_min, y_min, x_max, y_max, name))

    return ground_truths
    # 儲存更新後的 XML 標記檔案
    #tree.write(xml_path)
