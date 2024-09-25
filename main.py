# Họ và tên: Võ Đại Lượng
# Mã số sinh viên: 22520834


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from scipy.stats import skew
import numpy as np
import warnings
import cv2
import os

warnings.filterwarnings("ignore")

def read_image(image_path):
    image = cv2.imread(image_path, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    return image

# Đặc trưng Histogram
def color_histogram(image):
    channel = image.shape[2]
    
    feature = []
    for k in range(channel):
        histogram = cv2.calcHist([image], [k], None, [256], [0, 256])
        feature.append(histogram)
        
    #chuyển đổi vector đơn vị
    feature_vector = np.array(feature).flatten()

    norm = np.linalg.norm(feature_vector)
    if norm > 0: 
        feature_vector /= norm
    
    return feature_vector

# Đặc trưng Color moment
def color_moment(image):
    moments = []
    for channel in cv2.split(image):
        mean = np.mean(channel)
        std = np.std(channel)
        skewness = skew(channel, axis=None)
        
        moments.extend([mean, std, skewness])
    
    #chuyển đổi vector đơn vị
    feature_vector = np.array(moments).flatten()

    norm = np.linalg.norm(feature_vector)
    if norm > 0: 
        feature_vector /= norm
    
    return feature_vector

#Đặc trưng DCD
def DCD(image, k= 1):
    pixels = image.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    dominant_colors = kmeans.cluster_centers_

    #chuyển đổi vector đơn vị
    feature_vector = np.array(dominant_colors).flatten()

    norm = np.linalg.norm(feature_vector)
    if norm > 0: 
        feature_vector /= norm
    
    return feature_vector

# Đặc trưng CCV
def CCV(image, tau=0.1):
    ccv = np.zeros((256, 2))

    threshold = tau * np.max(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            pixel = image[y, x]

            coherence = 1 if np.linalg.norm(pixel) > threshold else 0

            avg_color = int(np.mean(pixel))
            ccv[avg_color][coherence] += 1

    #chuyển đổi vector đơn vị
    feature_vector = np.array(ccv).flatten()

    norm = np.linalg.norm(feature_vector)
    if norm > 0: 
        feature_vector /= norm
    
    return feature_vector

#Các độ đo khoảng cách dùng cho knn
def euclidean_distance(u, v):
    return np.sqrt(np.sum((u - v) ** 2))

def correlation_distance(u, v):
    return 1 - np.corrcoef(u, v)[0,1]

def chi_square_distance(u, v):
    return 0.5 * np.sum((u - v) ** 2 / (u + v) + 1e-10)

def intersection_distance(u, v):
    return np.sum(np.minimum(u, v))

def bhattacharyya_distance(u, v):
    return -np.log(np.sum(np.sqrt(u * v)))

# Hàm tạo dữ liệu
def create_train_data(color_feature, train_path):
    train_data = []
    train_label = []

    color_list = os.listdir(train_path)

    for index, color_name in enumerate(color_list):
        path = os.path.join(train_path, color_name)

        image_list = os.listdir(os.path.join(path))

        for image_name in image_list:
            image = read_image(os.path.join(path, image_name))

            color_features = color_feature(image)

            train_data.append(color_features)
            train_label.append(index)

    train_data = SimpleImputer(strategy = 'mean').fit_transform(train_data)
    return np.array(train_data), np.array(train_label)


def create_test_data(color_feature, test_path):
    test_data = []
    test_label = []

    color_list = os.listdir(test_path)

    for index, color_name in enumerate(color_list):
        path = os.path.join(test_path, color_name)

        image_list = os.listdir(os.path.join(path))

        for image_name in image_list:
            image = read_image(os.path.join(path, image_name))

            color_features = color_feature(image)

            test_data.append(color_features)
            test_label.append(index)

    test_data = SimpleImputer(strategy = 'mean').fit_transform(test_data)
    return np.array(test_data), np.array(test_label)


# Main code

color_features = {
    'Histogram': color_histogram,
    'Moment': color_moment,
    'DCD': DCD,
    'CCV': CCV,
}

metrics = {
    'Euclidiean': euclidean_distance,
    'Chi_square': chi_square_distance,
    'Correlation': correlation_distance,
    'Intersection': intersection_distance,
    'Bhattacharyya': bhattacharyya_distance,
}

for num_neighbors in [1, 5]:
    print()
    for metric_ in [ 'Euclidiean', 'Chi_square', 'Correlation', 'Intersection', 'Bhattacharyya']:
        print()
        for color_feature in ['Histogram', 'Moment', 'DCD', 'CCV']:
            model = KNeighborsClassifier(n_neighbors= num_neighbors, metric = metrics[metric_])

            train_data, train_label = create_train_data(color_features[color_feature], 'train')
            test_data, test_label = create_test_data(color_features[color_feature], 'test')

            model.fit(train_data, train_label)

            predictions = model.predict(test_data)

            accuracy = accuracy_score(test_label, predictions)
            f1 = f1_score(test_label, predictions, average='weighted')

            print(f"With {num_neighbors} neighbors, the {metric_} metric and {color_feature} feature, the results are: Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")


