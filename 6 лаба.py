from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

app = FastAPI()

# Загрузка данных ириса
iris = load_iris()
X = iris.data
y = iris.target

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели логистической регрессии
model = LogisticRegression()
model.fit(X_train, y_train)

# Сохранение модели в файл
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Загрузка модели из файла
with open('iris_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

@app.get("/")
async def index():
    return HTMLResponse(
        """
        <html>
        <head>
            <title>Классификация Ирисов</title>
        </head>
        <body>
            <h1>Классификация Ирисов</h1>
            <form method="post" action="/predict">
                <label for="sepal_length">Длина чашелистика:</label>
                <input type="number" id="sepal_length" name="sepal_length"><br><br>
                <label for="sepal_width">Ширина чашелистика:</label>
                <input type="number" id="sepal_width" name="sepal_width"><br><br>
                <label for="petal_length">Длина лепестка:</label>
                <input type="number" id="petal_length" name="petal_length"><br><br>
                <label for="petal_width">Ширина лепестка:</label>
                <input type="number" id="petal_width" name="petal_width"><br><br>
                <input type="submit" value="Отправить">
            </form>
        </body>
        </html>
        """
    )

@app.post("/predict")
async def predict(sepal_length: float = Form(...),
                  sepal_width: float = Form(...),
                  petal_length: float = Form(...),
                  petal_width: float = Form(...)):
    # Формируем входные данные
    data = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Предсказание
    prediction = loaded_model.predict(data)[0]

    # Имя цветка
    iris_names = ['setosa', 'versicolor', 'virginica']
    predicted_flower = iris_names[prediction]

    return HTMLResponse(
        f"""
        <html>
        <head>
            <title>Результат</title>
        </head>
        <body>
            <h1>Результат классификации</h1>
            <p>Предсказанный цветок: {predicted_flower}</p>
        </body>
        </html>
        """
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
