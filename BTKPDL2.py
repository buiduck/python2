import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Đọc dữ liệu từ file
red_wine_data = pd.read_csv("winequality-red.csv")
white_wine_data = pd.read_csv("winequality-white.csv")
""""""
# Tạo vòng lặp để phân chia và xuất tập huấn luyện và tập kiểm tra ba lần
for i in range(3):
    # Phân chia dữ liệu cho rượu đỏ
    red_train, red_test = train_test_split(red_wine_data, test_size=0.1, random_state=i)
    # Xuất tập huấn luyện và tập kiểm tra cho rượu đỏ
    red_train.to_csv(f"red_wine_train_{i}.csv", index=False)
    red_test.to_csv(f"red_wine_test_{i}.csv", index=False)

    # Phân chia dữ liệu cho rượu trắng
    white_train, white_test = train_test_split(white_wine_data, test_size=0.1, random_state=i)
    # Xuất tập huấn luyện và tập kiểm tra cho rượu trắng
    white_train.to_csv(f"white_wine_train_{i}.csv", index=False)
    white_test.to_csv(f"white_wine_test_{i}.csv", index=False)

#Kmeans Cau 2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
for i in range(3):
    red_train = f"red_train_{i}.csv"
    red_test = f"red_test_{i}.csv"
    white_train = f"white_train_{i}.csv"
    white_test = f"white_test_{i}.csv"

    red_train_data = pd.read_csv(red_train)
    red_test_data = pd.read_csv(red_test)
    white_train_data = pd.read_csv(white_train)
    white_test_data = pd.read_csv(white_test)

    # Huấn luyện mô hình k-means cho dữ liệu đỏ
    kmeans_red = KMeans(n_clusters=5, random_state=0, n_init=10).fit(red_train_data)
    # Lấy các trọng tâm của các cụm
    cluster_centers_red = kmeans_red.cluster_centers_
    print(f"Cluster centers for red wine - iteration {i}:\n", cluster_centers_red)
    # Dự đoán nhãn cho tập kiểm tra dữ liệu đỏ
    red_pred = kmeans_red.predict(red_test_data)
    # Tính các chỉ số đánh giá cho dữ liệu đỏ
    cm_red = confusion_matrix(red_test_data['quality'], red_pred)
    acc_red = accuracy_score(red_test_data['quality'], red_pred)
    precision_red = precision_score(red_test_data['quality'], red_pred, average='macro', zero_division=1)
    recall_red = recall_score(red_test_data['quality'], red_pred, average='macro', zero_division=1)
    f1_red = f1_score(red_test_data['quality'], red_pred, average='macro')
    # Tính false negative rate
    if (cm_red[1, 0] + cm_red[1, 1]) != 0:
        false_negative_rate_red = cm_red[1, 0] / (cm_red[1, 0] + cm_red[1, 1])
    else:
        false_negative_rate_red = 0

    print(f'Red Wine - Confusion matrix for iteration {i}:\n', cm_red)
    print(f'Red Wine - Accuracy for iteration {i}: ', acc_red)
    print(f'Red Wine - Precision (Positive Predictive Value) for iteration {i}: ', precision_red)
    print(f'Red Wine - False Negative Rate for iteration {i}: ', false_negative_rate_red)
    print(f'Red Wine - Recall (True Positive Rate) for iteration {i}: ', recall_red)
    print(f'Red Wine - F1 Score for iteration {i}: ', f1_red)


    # Huấn luyện mô hình k-means cho dữ liệu trắng
    kmeans_white = KMeans(n_clusters=5, random_state=0, n_init=10).fit(white_train_data)
    # Lấy các trọng tâm của các cụm
    cluster_centers_white = kmeans_white.cluster_centers_
    print(f"Cluster centers for white wine - iteration {i}:\n", cluster_centers_white)
    # Dự đoán nhãn cho tập kiểm tra dữ liệu trắng
    white_pred = kmeans_white.predict(white_test_data)
    # Tính các chỉ số đánh giá cho dữ liệu trắng
    cm_white = confusion_matrix(white_test_data['quality'], white_pred)
    acc_white = accuracy_score(white_test_data['quality'], white_pred)
    precision_white = precision_score(white_test_data['quality'], white_pred, average='macro', zero_division=1)
    recall_white = recall_score(white_test_data['quality'], white_pred, average='macro', zero_division=1)
    f1_white = f1_score(white_test_data['quality'], white_pred, average='macro')
    # Tính false negative rate
    if (cm_white[1, 0] + cm_white[1, 1]) != 0:
        false_negative_rate_white = cm_white[1, 0] / (cm_white[1, 0] + cm_white[1, 1])
    else:
        false_negative_rate_white = 0

    print(f'White Wine - Confusion matrix for iteration {i}:\n', cm_white)
    print(f'White Wine - Accuracy for iteration {i}: ', acc_white)
    print(f'White Wine - Precision (Positive Predictive Value) for iteration {i}: ', precision_white)
    print(f'White Wine - False Negative Rate for iteration {i}: ', false_negative_rate_white)
    print(f'White Wine - Recall (True Positive Rate) for iteration {i}: ', recall_white)
    print(f'White Wine - F1 Score for iteration {i}: ', f1_white)


  # Khởi tạo các biến để tính tổng các chỉ số
total_acc_red = 0
total_precision_red = 0
total_recall_red = 0
total_f1_red = 0

total_acc_white = 0
total_precision_white = 0
total_recall_white = 0
total_f1_white = 0

# Lặp qua ba lượt lặp
for i in range(3):
    # Cập nhật tổng các chỉ số cho dữ liệu đỏ
    total_acc_red += acc_red
    total_precision_red += precision_red
    total_recall_red += recall_red
    total_f1_red += f1_red

    # Cập nhật tổng các chỉ số cho dữ liệu trắng
    total_acc_white += acc_white
    total_precision_white += precision_white
    total_recall_white += recall_white
    total_f1_white += f1_white

# Tính giá trị trung bình cho dữ liệu đỏ
avg_acc_red = total_acc_red / 3
avg_precision_red = total_precision_red / 3
avg_recall_red = total_recall_red / 3
avg_f1_red = total_f1_red / 3

# Tính giá trị trung bình cho dữ liệu trắng
avg_acc_white = total_acc_white / 3
avg_precision_white = total_precision_white / 3
avg_recall_white = total_recall_white / 3
avg_f1_white = total_f1_white / 3

# In giá trị trung bình
print("Average performance metrics for Red Wine:")
print("Accuracy:", avg_acc_red)
print("Precision:", avg_precision_red)
print("Recall:", avg_recall_red)
print("F1 Score:", avg_f1_red)

print("\nAverage performance metrics for White Wine:")
print("Accuracy:", avg_acc_white)
print("Precision:", avg_precision_white)
print("Recall:", avg_recall_white)
print("F1 Score:", avg_f1_white)




