import time
from main import prepare_data, train_and_evaluate

# データの準備
X_train, X_test, y_train, y_test = prepare_data()

# モデルの学習（推論に必要）
model, _ = train_and_evaluate(X_train, X_test, y_train, y_test)

# 推論時間の計測
start = time.time()
_ = model.predict(X_test)
elapsed = time.time() - start

# しきい値（秒）を設定
threshold = 1.0
print(f"Inference time: {elapsed:.3f} seconds")

# 検証
if elapsed > threshold:
    raise RuntimeError(f"Inference too slow: {elapsed:.3f} > {threshold}")
