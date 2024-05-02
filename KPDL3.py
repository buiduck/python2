import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import seaborn as sns

# Đọc dữ liệu từ tập tin CSV
data = pd.read_csv("dataset.csv")
print(data.head())

# Loại bỏ các dòng có giá trị âm trong cột "Occupancy"
data = data[data['Occupancy'] >= 0]
# Nhóm dữ liệu theo tên bãi đậu xe và tính tỷ lệ trung bình của sức chứa và tỷ lệ chiếm dụng
grouped_data = data.groupby('SystemCodeNumber').agg({'Capacity': 'mean', 'Occupancy': 'mean'}).reset_index()
# Sắp xếp dữ liệu theo tỷ lệ chiếm dụng để biểu đồ dễ đọc hơn
sorted_data = grouped_data.sort_values(by='Occupancy', ascending=False)
# Trực quan hóa tỷ lệ chiếm dụng và sức chứa của các bãi đậu xe
plt.figure(figsize=(12, 8))
bar_width = 0.35
index = range(len(sorted_data))
plt.bar(index, sorted_data['Capacity'], bar_width, label='Sức chứa', color='skyblue')
plt.bar([i + bar_width for i in index], sorted_data['Occupancy'], bar_width, label='Tỷ lệ chiếm dụng', color='orange')
plt.xlabel('Tên bãi đậu xe')
plt.ylabel('Sức chứa và tỷ lệ chiếm dụng')
plt.title('Tỷ lệ chiếm dụng của các bãi đậu xe')
plt.xticks([i + bar_width / 2 for i in index], sorted_data['SystemCodeNumber'], rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Kết quả tương quan giữa hai bãi đỗ xe
# Lấy tên các bãi đậu xe duy nhất
unique_parking_lots = data['SystemCodeNumber'].unique()
# Chọn hai bãi đậu xe cụ thể
selected_data = data[data['SystemCodeNumber'].isin(['BHMBCCMKT01', 'BHMBCCPST01'])]
# Tính toán hệ số tương quan Pearson
correlation = selected_data['Occupancy'].corr(selected_data['Capacity'])
print("Correlation between two selected parking lots:", correlation)
# Vẽ biểu đồ scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(selected_data['Capacity'], selected_data['Occupancy'])
plt.xlabel('Sức chứa')
plt.ylabel('Tỷ lệ chiếm dụng (%)')
plt.title('Biểu đồ scatter plot giữa sức chứa và tỷ lệ chiếm dụng')
plt.tight_layout()
plt.show()

# 3. Xây dựng mô hình dự đoán tỷ lệ chiếm chỗ
# Xử lý dữ liệu thiếu sót
imputer = SimpleImputer(strategy='mean')
selected_data[['Occupancy', 'Capacity']] = imputer.fit_transform(selected_data[['Occupancy', 'Capacity']])

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X = selected_data[['Capacity']]  # Sử dụng sức chứa làm đặc trưng
y = selected_data['Occupancy']   # Tỷ lệ chiếm chỗ là biến mục tiêu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Xây dựng mô hình Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Xây dựng mô hình Random Forest Regression
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Dự đoán và đánh giá mô hình bằng RMSE và MAE
lr_predictions = lr_model.predict(X_test_scaled)
rf_predictions = rf_model.predict(X_test_scaled)
lr_mae = mean_absolute_error(y_test, lr_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
print("Dự đoán tỷ lệ chiếm dụng của BHMBCCPST01 bằng Linear Regression:", lr_predictions[0])
print("Dự đoán tỷ lệ chiếm dụng của BHMBCCPST01 bằng Random Forest Regression:", rf_predictions[0])
print("Dự đoán tỷ lệ chiếm dụng của BHMBCCMKT01 bằng Linear Regression:", lr_predictions[1])
print("Dự đoán tỷ lệ chiếm dụng của BHMBCCMKT01 bằng Random Forest Regression:", rf_predictions[1])
print("Linear Regression MAE:", lr_mae)
print("Random Forest Regression MAE:", rf_mae)
print("Linear Regression RMSE:", lr_rmse)
print("Random Forest Regression RMSE:", rf_rmse)

