import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
import string
import re

# 1. Tải dữ liệu
# Giả sử chúng ta có tập dữ liệu đánh giá sản phẩm trong file CSV với hai cột: 'review' và 'sentiment'
# 'review' chứa văn bản đánh giá sản phẩm, 'sentiment' là nhãn (1: tích cực, 0: tiêu cực)
data = pd.read_csv('product_reviews.csv')

# Chia dữ liệu thành 80% tập huấn luyện và 20% tập kiểm tra
X = data['review']  # Đánh giá sản phẩm
y = data['sentiment']  # Nhãn cảm xúc (1: tích cực, 0: tiêu cực)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Tiền xử lý văn bản
def preprocess_text(text):
    # Chuyển thành chữ thường
    text = text.lower()
    # Xóa dấu câu
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Xóa số và các ký tự không mong muốn
    text = re.sub(r'\d+', '', text)
    return text

# Áp dụng hàm tiền xử lý cho dữ liệu huấn luyện và kiểm tra
X_train = X_train.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

# 3. Sử dụng TF-IDF để chuyển đổi văn bản thành dạng số
tfidf = TfidfVectorizer(max_features=5000)  # Lấy 5000 từ quan trọng nhất
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 4. Áp dụng thuật toán Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# 5. Đánh giá mô hình
y_pred = nb_model.predict(X_test_tfidf)

# Độ chính xác
accuracy = accuracy_score(y_test, y_pred)
# F1-score
f1 = f1_score(y_test, y_pred)

print(f"Độ chính xác của mô hình: {accuracy:.2f}")
print(f"F1-score của mô hình: {f1:.2f}")

# 6. Đề xuất phương pháp cải thiện dự đoán

# 6.1. Sử dụng các thuật toán phức tạp hơn như SVM, Random Forest, hoặc Logistic Regression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Thử nghiệm với Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)

# Đánh giá Logistic Regression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print(f"Độ chính xác của Logistic Regression: {accuracy_lr:.2f}")
print(f"F1-score của Logistic Regression: {f1_lr:.2f}")

# 6.2. Sử dụng GridSearchCV để tìm tham số tối ưu cho Naive Bayes
from sklearn.model_selection import GridSearchCV

# Thử nghiệm với Multinomial Naive Bayes và GridSearchCV
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}  # Thử nghiệm các giá trị alpha khác nhau
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

# Tìm alpha tốt nhất
best_nb_model = grid_search.best_estimator_
y_pred_best_nb = best_nb_model.predict(X_test_tfidf)

# Đánh giá mô hình tốt nhất
accuracy_best_nb = accuracy_score(y_test, y_pred_best_nb)
f1_best_nb = f1_score(y_test, y_pred_best_nb)

print(f"Độ chính xác của Naive Bayes tốt nhất (GridSearchCV): {accuracy_best_nb:.2f}")
print(f"F1-score của Naive Bayes tốt nhất (GridSearchCV): {f1_best_nb:.2f}")