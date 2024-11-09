import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Tải dữ liệu
# Dữ liệu Boston có sẵn trong thư viện scikit-learn
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)  # Dữ liệu đầu vào (đặc trưng)
y = pd.Series(boston.target)  # Mục tiêu (giá nhà trung bình)

# 2. Xử lý dữ liệu
## 2.1 Kiểm tra dữ liệu rỗng
print("Kiểm tra dữ liệu rỗng:\n", X.isnull().sum())

# Nếu có dữ liệu rỗng, loại bỏ hoặc thay thế bằng giá trị trung bình (median)
if X.isnull().sum().sum() > 0:
    X.fillna(X.median(), inplace=True)

## 2.2 Kiểm tra và loại bỏ dữ liệu trùng lặp
print("\nSố lượng dòng trùng lặp trước khi loại bỏ:", X.duplicated().sum())
X.drop_duplicates(inplace=True)
print("Số lượng dòng trùng lặp sau khi loại bỏ:", X.duplicated().sum())

# 2.3 Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu thành 80% tập huấn luyện và 20% tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Áp dụng mô hình hồi quy tuyến tính
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# 4. Đánh giá mô hình hồi quy tuyến tính
y_pred_lr = lin_reg.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\nMô hình hồi quy tuyến tính:")
print(f"MAE: {mae_lr:.2f}")
print(f"MSE: {mse_lr:.2f}")
print(f"R²: {r2_lr:.2f}")

# 5. Áp dụng mô hình Decision Tree Regression
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)
y_pred_tree = tree_reg.predict(X_test)

mae_tree = mean_absolute_error(y_test, y_pred_tree)
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print("\nMô hình Decision Tree Regression:")
print(f"MAE: {mae_tree:.2f}")
print(f"MSE: {mse_tree:.2f}")
print(f"R²: {r2_tree:.2f}")


# 6. Vẽ biểu đồ dự đoán của Linear Regression so với giá trị thực tế
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, color='blue', label='Linear Regression')
plt.scatter(y_test, y_pred_tree, color='green', label='Decision Tree Regression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.legend()
plt.title('So sánh dự đoán giá nhà')
plt.show()